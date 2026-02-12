import torch
import numpy as np
import math
from .ot_guidance_mixin import OTGuidanceMixin, CategoricalEmbedding

class MonteCarloStrategy(OTGuidanceMixin):
    def __init__(self, explainer, random_state=None, cone_angle=math.pi/4, use_cone_sampling_categorical=True, use_cone_sampling_continuous=True, categorical_step=1.2, continuous_step=0.1, temperature=2.0, h=2):
        self.explainer = explainer
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)
        self._torch_rng = torch.Generator(device=explainer.device).manual_seed(random_state or 0)

        self.categorical_step = categorical_step
        self.continuous_step = continuous_step
        self.temperature = temperature

        self.h = h
        
        self.continuous_global_ranges = {}
        self.categorical_global_ranges = {}
        
        OTGuidanceMixin.__init__(self, explainer, cone_angle=cone_angle, random_state=random_state, 
                                     use_cone_sampling_categorical=use_cone_sampling_categorical, 
                                     use_cone_sampling_continuous=use_cone_sampling_continuous)
        
        self._embedding_cache = {}

    def set_global_ranges(self, continuous_ranges_by_index, categorical_ranges_by_index):
        self.continuous_global_ranges = continuous_ranges_by_index
        self.categorical_global_ranges = categorical_ranges_by_index

    def generate_new_X(self, eta: float, num_trials: int, top_k: int = 1) -> torch.Tensor:
        explainer = self.explainer
        X_best = None
        best_Q = float("inf")

        with torch.no_grad():
            X_s = explainer.X[:, explainer.explain_indices] * explainer.costs_vector_reshaped
            X_t = explainer.X_prime[:, explainer.explain_indices] * explainer.costs_vector_reshaped
            y = explainer.y.view(-1)
            y_prime = explainer.y_prime.view(-1)
            mu_list = explainer.swd.mu_list
            nu = explainer.wd.nu
            candidate_indices = explainer._get_topk_Q_indices(X_s, X_t, y, y_prime, mu_list, nu, eta, top_k=top_k)

        for trial in range(num_trials):
            cand = explainer.X.clone()
            for idx in candidate_indices:
                if idx >= explainer.X.shape[0]:
                    continue
                ref_idx = self._rng.randint(explainer.X_prime.shape[0])

                mutate_features = self._rng.choice(
                    explainer.explain_indices,
                    size=min(self.h, len(explainer.explain_indices)),
                    replace=False
                ).tolist()

                ot_term1 = self._compute_term1_ot_direction(cand, eta)

                for idx_feat in explainer.categorical_indices:
                    if idx_feat not in explainer.explain_indices:
                        continue

                    if idx_feat not in mutate_features:
                        continue

                    if not self.use_cone_sampling_categorical:
                        if idx_feat in self.categorical_global_ranges:
                            global_vals = self.categorical_global_ranges[idx_feat]
                            unique_vals = torch.tensor(global_vals, device=explainer.device)
                        else:
                            unique_vals = torch.unique(explainer.X_prime[:, idx_feat])
                        sampled_val = unique_vals[self._rng.randint(len(unique_vals))]
                        cand[idx, idx_feat] = (1 - eta) * explainer.X_prime[ref_idx, idx_feat] + eta * sampled_val
                        continue

                    if idx_feat in self.categorical_global_ranges:
                        global_vals = self.categorical_global_ranges[idx_feat]
                        unique_vals = torch.tensor(global_vals, device=explainer.device)
                    else:
                        unique_vals = torch.unique(explainer.X_prime[:, idx_feat])
                    
                    cache_key = f"feat_{idx_feat}_{len(unique_vals)}"
                    if cache_key not in self._embedding_cache:
                        embedding_layer = CategoricalEmbedding(unique_vals, embed_dim=None, random_state=self.random_state)
                        self._embedding_cache[cache_key] = embedding_layer
                    else:
                        embedding_layer = self._embedding_cache[cache_key]
                    
                    current_val = embedding_layer.map_to_nearest_category(cand[idx, idx_feat])
                    current_embedding = embedding_layer.encode(current_val)

                    
                    feat_pos = explainer.explain_indices.index(idx_feat)
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
                    
                    step_size = torch.rand(1, generator=self._torch_rng) * self.categorical_step
                    perturbation = cone_dir * step_size
                    new_embedding = current_embedding + perturbation
                    
                    new_categorical_val = embedding_layer.decode(new_embedding, temperature=self.temperature, random_generator=self._torch_rng)
                    cand[idx, idx_feat] = new_categorical_val

                if self.use_cone_sampling_continuous:
                    guide_direction = -ot_term1[:, explainer.explain_indices]
                    cone_dir = self._sample_in_cone(guide_direction, self.cone_angle, len(explainer.explain_indices))

                    for idx_feat in explainer.continuous_indices:
                        if idx_feat not in mutate_features:
                            continue

                        explain_pos = explainer.explain_indices.index(idx_feat)

                        if idx_feat in self.continuous_global_ranges:
                            min_val = torch.tensor(self.continuous_global_ranges[idx_feat]['min'], device=explainer.device)
                            max_val = torch.tensor(self.continuous_global_ranges[idx_feat]['max'], device=explainer.device)
                        else:
                            min_val = explainer.X_prime[:, idx_feat].min()
                            max_val = explainer.X_prime[:, idx_feat].max()
                            
                        direction = cone_dir[idx, explain_pos]
                        step = torch.rand(1, generator=self._torch_rng, device=explainer.device) * self.continuous_step
                        perturbation = direction * step * (max_val - min_val)
                        cand[idx, idx_feat] = torch.clamp(explainer.X_prime[ref_idx, idx_feat] + perturbation, min_val, max_val)
                else:
                    for idx_feat in explainer.continuous_indices:
                        if idx_feat not in mutate_features:
                            continue

                        if idx_feat in self.continuous_global_ranges:
                            min_val = torch.tensor(self.continuous_global_ranges[idx_feat]['min'], device=explainer.device)
                            max_val = torch.tensor(self.continuous_global_ranges[idx_feat]['max'], device=explainer.device)
                        else:
                            min_val = explainer.X_prime[:, idx_feat].min()
                            max_val = explainer.X_prime[:, idx_feat].max()
                            
                        rand_val = torch.rand(1, generator=self._torch_rng, device=explainer.device)
                        sampled_val = min_val + rand_val * (max_val - min_val)
                        cand[idx, idx_feat] = sampled_val

            y_cand = explainer.model(cand)
            current_Q, *_ = explainer.evaluate_Q(cand, y_cand, eta)
            if current_Q < best_Q:
                best_Q = current_Q
                X_best = cand.clone().detach()

        return X_best
