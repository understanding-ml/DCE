import torch
import numpy as np
import math
from .ot_guidance_mixin import OTGuidanceMixin, CategoricalEmbedding

class SimulatedAnnealingStrategy(OTGuidanceMixin):
    def __init__(
        self,
        explainer,
        T0: float = 1.0,
        T_final: float = 0.001,
        temp_decay: float = None,
        random_state: int = None,
        cone_angle=math.pi/4,
        use_cone_sampling_categorical=True,
        use_cone_sampling_continuous=True,
        categorical_step=1.2,
        continuous_step=0.1,
        temperature=2.0
    ):
        self.explainer = explainer
        
        self.continuous_global_ranges = {}
        self.categorical_global_ranges = {}
        
        OTGuidanceMixin.__init__(self, explainer, cone_angle=cone_angle, random_state=random_state, 
                                      use_cone_sampling_categorical=use_cone_sampling_categorical, 
                                      use_cone_sampling_continuous=use_cone_sampling_continuous)
        self.T0 = T0
        self.T_final = T_final
        self.temp_decay = temp_decay
        self.random_state = random_state

        self._np_rng = np.random.RandomState(random_state)
        self._torch_rng = torch.Generator(device=explainer.device).manual_seed(random_state or 0)
        self._rng = self._np_rng  
        
        self.categorical_step = categorical_step
        self.continuous_step = continuous_step
        self.temperature = temperature
        
        self._embedding_cache = {}

    def set_global_ranges(self, continuous_ranges_by_index, categorical_ranges_by_index):
        self.continuous_global_ranges = continuous_ranges_by_index
        self.categorical_global_ranges = categorical_ranges_by_index

    def generate_new_X(self, eta: float, num_trials: int, top_k: int = 1) -> torch.Tensor:
        explainer = self.explainer
        best_X = explainer.X.clone()
        best_overall_Q = float("inf")
        T = self.T0

        alpha_T = (
            self.temp_decay if self.temp_decay is not None
            else (self.T_final / self.T0) ** (1 / num_trials)
        )

        with torch.no_grad():
            X_s = explainer.X[:, explainer.explain_indices] * explainer.costs_vector_reshaped
            X_t = explainer.X_prime[:, explainer.explain_indices] * explainer.costs_vector_reshaped
            y = explainer.y.view(-1)
            y_prime = explainer.y_prime.view(-1)
            mu_list = explainer.swd.mu_list
            nu = explainer.wd.nu
            candidate_indices = explainer._get_topk_Q_indices(
                X_s, X_t, y, y_prime, mu_list, nu, eta, top_k
            )

            for idx in candidate_indices:
                current_candidate = explainer.X[idx].clone()
                X_temp = explainer._update_row(explainer.X.clone(), idx, current_candidate)
                y_temp = explainer.model(X_temp)
                current_Q = explainer.evaluate_Q(X_temp, y_temp, eta)[0]

                for _ in range(num_trials):
                    new_candidate = current_candidate.clone()

                    temp_cand = explainer.X.clone()
                    temp_cand[idx] = new_candidate
                    ref_idx = self._np_rng.randint(explainer.X_prime.shape[0])
                    
                    ot_term1 = self._compute_term1_ot_direction(temp_cand, eta)
                    
                    for feat in explainer.explain_indices:
                        if feat in explainer.categorical_indices:
                            if self._np_rng.rand() < T:
                                if not self.use_cone_sampling_categorical:
                                    if feat in self.categorical_global_ranges:
                                        global_vals = self.categorical_global_ranges[feat]
                                        unique_vals = torch.tensor(global_vals, device=explainer.device)
                                    else:
                                        unique_vals = torch.unique(explainer.X_prime[:, feat])
                                    sampled_val = unique_vals[self._np_rng.randint(len(unique_vals))]
                                    temp_cand[idx, feat] = (1 - eta) * explainer.X_prime[ref_idx, feat] + eta * sampled_val
                                    continue

                                if feat in self.categorical_global_ranges:
                                    global_vals = self.categorical_global_ranges[feat]
                                    unique_vals = torch.tensor(global_vals, device=explainer.device)
                                else:
                                    unique_vals = torch.unique(explainer.X_prime[:, feat])
                                
                                cache_key = f"feat_{feat}_{len(unique_vals)}"
                                if cache_key not in self._embedding_cache:
                                    embedding_layer = CategoricalEmbedding(unique_vals, embed_dim=None, random_state=self.random_state)
                                    self._embedding_cache[cache_key] = embedding_layer
                                else:
                                    embedding_layer = self._embedding_cache[cache_key]
                                
                                current_val = embedding_layer.map_to_nearest_category(temp_cand[idx, feat])
                                current_embedding = embedding_layer.encode(current_val)

                                
                                feat_pos = explainer.explain_indices.index(feat)
                                guide_scalar = ot_term1[idx, feat_pos].item()
                                
                                sorted_vals = torch.sort(unique_vals)[0]
                                current_idx = torch.where(sorted_vals == current_val)[0][0]
                                
                                direction_vectors = []
                                if current_idx < len(sorted_vals) - 1:  
                                    higher_embed = embedding_layer.encode(sorted_vals[current_idx + 1])
                                    direction_vectors.append(higher_embed - current_embedding)
                                if current_idx > 0: 
                                    lower_embed = embedding_layer.encode(sorted_vals[current_idx - 1])
                                    direction_vectors.append(current_embedding - lower_embed)
                                
                                if direction_vectors:
                                    value_direction = torch.stack(direction_vectors).mean(0)
                                    value_direction = value_direction / (torch.norm(value_direction) + 1e-8)
                                else:
                                    value_direction = torch.zeros(embedding_layer.embed_dim)
                                
                                if guide_scalar < 0:
                                    guide_direction = value_direction
                                else:
                                    guide_direction = -value_direction
                                
                                if torch.norm(guide_direction) < 1e-8:
                                    guide_direction = torch.randn(embedding_layer.embed_dim, generator=self._torch_rng)
                                    guide_direction = guide_direction / (torch.norm(guide_direction) + 1e-8)
                                
                                cone_dir = self._sample_in_cone(guide_direction.unsqueeze(0), self.cone_angle, embedding_layer.embed_dim)[0]
                                
                                base_step_size = 1.2
                                step_size = torch.rand(1, generator=self._torch_rng) * base_step_size
                                perturbation = cone_dir * step_size
                                new_embedding = current_embedding + perturbation
                                
                                temperature = 2.0  
                                
                                new_categorical_val = embedding_layer.decode(new_embedding, temperature=temperature, random_generator=self._torch_rng)
                                temp_cand[idx, feat] = new_categorical_val

                    if self.use_cone_sampling_continuous:
                        guide_direction = -ot_term1[:, explainer.explain_indices]
                        cone_dir = self._sample_in_cone(guide_direction, self.cone_angle, len(explainer.explain_indices))

                        for feat_pos, feat in enumerate(explainer.explain_indices):
                            if feat in explainer.continuous_indices:
                                explain_pos = explainer.explain_indices.index(feat)
                                if feat in self.continuous_global_ranges:
                                    min_val = torch.tensor(self.continuous_global_ranges[feat]['min'], device=explainer.device)
                                    max_val = torch.tensor(self.continuous_global_ranges[feat]['max'], device=explainer.device)
                                else:
                                    min_val = explainer.X_prime[:, feat].min()
                                    max_val = explainer.X_prime[:, feat].max()
                                direction = cone_dir[idx, explain_pos]
                                step = torch.rand(1, generator=self._torch_rng, device=explainer.device) * self.continuous_step * T
                                perturbation = direction * step * (max_val - min_val)
                                temp_cand[idx, feat] = torch.clamp(explainer.X_prime[ref_idx, feat] + perturbation, min_val, max_val)
                    else:
                        for feat in explainer.continuous_indices:
                                if feat in self.continuous_global_ranges:
                                    min_val = torch.tensor(self.continuous_global_ranges[feat]['min'], device=explainer.device)
                                    max_val = torch.tensor(self.continuous_global_ranges[feat]['max'], device=explainer.device)
                                else:
                                    min_val = explainer.X_prime[:, feat].min()
                                    max_val = explainer.X_prime[:, feat].max()
                                rand_val = torch.rand(1, generator=self._torch_rng, device=explainer.device)
                                sampled_val = min_val + rand_val * (max_val - min_val)
                                temp_cand[idx, feat] = sampled_val
                    
                    new_candidate = temp_cand[idx]

                    X_new_candidate = explainer._update_row(explainer.X.clone(), idx, new_candidate)
                    y_new_candidate = explainer.model(X_new_candidate)
                    new_Q = explainer.evaluate_Q(X_new_candidate, y_new_candidate, eta)[0]
                    delta_Q = new_Q - current_Q

                    if delta_Q < 0 or self._np_rng.rand() < np.exp(-delta_Q.item() / T):
                        current_candidate = new_candidate.clone()
                        current_Q = new_Q

                    T *= alpha_T

                if current_Q < best_overall_Q:
                    best_overall_Q = current_Q
                    best_X = explainer._update_row(explainer.X.clone(), idx, current_candidate)

        return best_X
