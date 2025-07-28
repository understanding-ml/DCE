import torch
import numpy as np
import math
from .gradient_guidance_mixin import GradientGuidanceMixin

class SimulatedAnnealingStrategy(GradientGuidanceMixin):
    def __init__(
        self,
        explainer,
        T0: float = 1.0,
        T_final: float = 0.001,
        temp_decay: float = None,
        random_state: int = None,
        use_gradient_guidance=False,
        weight_alpha=0.5
    ):
        # Initialize base attributes first
        self.explainer = explainer
        
        # Initialize gradient guidance mixin
        GradientGuidanceMixin.__init__(self, explainer, use_gradient_guidance=use_gradient_guidance, weight_alpha=weight_alpha, random_state=random_state)
        self.T0 = T0
        self.T_final = T_final
        self.temp_decay = temp_decay
        self.random_state = random_state

        # Setup reproducible RNGs
        self._np_rng = np.random.RandomState(random_state)
        self._torch_rng = torch.Generator(device=explainer.device).manual_seed(random_state or 0)
        self._rng = self._np_rng  # Alias for mixin compatibility

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

                    for feat in explainer.explain_indices:
                        if feat in explainer.categorical_indices:
                            unique_vals = torch.unique(explainer.X_prime[:, feat])
                            if self._np_rng.rand() < T:
                                sampled = unique_vals[
                                    torch.randint(len(unique_vals), (1,), generator=self._torch_rng, device=explainer.device)
                                ]
                                new_candidate[feat] = (1 - eta) * explainer.X_prime[idx, feat] + eta * sampled

                        elif feat in explainer.continuous_indices:
                            # Use gradient guidance if enabled, otherwise use standard perturbation
                            if self.use_gradient_guidance:
                                # Create a temporary candidate for gradient computation
                                temp_cand = explainer.X.clone()
                                temp_cand[idx] = new_candidate
                                guidance_applied = self._apply_gradient_guidance_to_continuous_features(temp_cand, eta, idx, idx)
                                if guidance_applied:
                                    new_candidate = temp_cand[idx]
                                else:
                                    # Fallback to standard perturbation
                                    min_val = explainer.X_prime[:, feat].min()
                                    max_val = explainer.X_prime[:, feat].max()
                                    noise = torch.randn(1, generator=self._torch_rng, device=explainer.device) * 0.1 * T * (max_val - min_val)
                                    perturbed = current_candidate[feat] + noise
                                    new_candidate[feat] = (1 - eta) * explainer.X_prime[idx, feat] + eta * perturbed
                            else:
                                # Standard perturbation
                                min_val = explainer.X_prime[:, feat].min()
                                max_val = explainer.X_prime[:, feat].max()
                                noise = torch.randn(1, generator=self._torch_rng, device=explainer.device) * 0.1 * T * (max_val - min_val)
                                perturbed = current_candidate[feat] + noise
                                new_candidate[feat] = (1 - eta) * explainer.X_prime[idx, feat] + eta * perturbed

                    X_new_candidate = explainer._update_row(explainer.X.clone(), idx, new_candidate)
                    y_new_candidate = explainer.model(X_new_candidate)
                    new_Q = explainer.evaluate_Q(X_new_candidate, y_new_candidate, eta)[0]
                    delta_Q = new_Q - current_Q

                    # Acceptance condition
                    if delta_Q < 0 or self._np_rng.rand() < np.exp(-delta_Q.item() / T):
                        current_candidate = new_candidate.clone()
                        current_Q = new_Q

                    T *= alpha_T

                if current_Q < best_overall_Q:
                    best_overall_Q = current_Q
                    best_X = explainer._update_row(explainer.X.clone(), idx, current_candidate)

        return best_X
