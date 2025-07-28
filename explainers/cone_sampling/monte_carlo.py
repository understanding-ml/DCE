import torch
import numpy as np
import math
from .gradient_guidance_mixin import GradientGuidanceMixin

class MonteCarloStrategy(GradientGuidanceMixin):
    def __init__(self, explainer, random_state=None, use_gradient_guidance=False, cone_angle=math.pi/4):
        # Initialize base attributes first
        self.explainer = explainer
        
        # Initialize gradient guidance mixin
        GradientGuidanceMixin.__init__(self, explainer, use_gradient_guidance=use_gradient_guidance, cone_angle=cone_angle, random_state=random_state)
        
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)
        self._torch_rng = torch.Generator(device=explainer.device).manual_seed(random_state or 0)

    def reseed(self, seed):
        self.random_state = seed
        self._rng = np.random.RandomState(seed)
        self._torch_rng = torch.Generator(device=self.explainer.device).manual_seed(seed)
    

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

        for _ in range(num_trials):
            cand = explainer.X.clone()
            for idx in candidate_indices:
                # Ensure idx is within bounds for X (current candidates)
                if idx >= explainer.X.shape[0]:
                    continue
                    
                # Use random sampling from X_prime for perturbation since X_prime contains factual data
                # Select a random row from X_prime for reference
                ref_idx = self._rng.randint(explainer.X_prime.shape[0])
                    
                # Categorical perturbation with gradient guidance
                categorical_guidance_applied = False
                if self.use_gradient_guidance:
                    categorical_guidance_applied = self._apply_gradient_guidance_to_categorical_features(cand, eta, ref_idx, idx)
                
                if not categorical_guidance_applied:
                    # Original categorical perturbation (fallback)
                    for idx_feat in explainer.categorical_indices:
                        unique_vals = torch.unique(explainer.X_prime[:, idx_feat])
                        sampled_val = unique_vals[self._rng.randint(len(unique_vals))]  # numpy sampling
                        cand[idx, idx_feat] = (1 - eta) * explainer.X_prime[ref_idx, idx_feat] + eta * sampled_val

                # Continuous perturbation with optional gradient guidance
                if self.use_gradient_guidance:
                    # Use mixin's gradient guidance method for continuous features
                    guidance_applied = self._apply_gradient_guidance_to_continuous_features(cand, eta, ref_idx, idx)
                    if not guidance_applied:
                        # Fallback to random sampling if gradient guidance fails
                        for idx_feat in explainer.continuous_indices:
                            self._apply_random_sampling_to_feature(cand, eta, ref_idx, idx, idx_feat)
                else:
                    # Original random sampling using mixin method
                    for idx_feat in explainer.continuous_indices:
                        self._apply_random_sampling_to_feature(cand, eta, ref_idx, idx, idx_feat)

            y_cand = explainer.model(cand)
            current_Q, *_ = explainer.evaluate_Q(cand, y_cand, eta)
            if current_Q < best_Q:
                best_Q = current_Q
                X_best = cand.clone().detach()

        return X_best
