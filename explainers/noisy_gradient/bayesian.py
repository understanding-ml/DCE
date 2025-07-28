import torch
import numpy as np
import math
from skopt import gp_minimize
from skopt.space import Real
from .gradient_guidance_mixin import GradientGuidanceMixin

class BayesianStrategy(GradientGuidanceMixin):
    def __init__(self, explainer, use_gradient_guidance=False, noise_beta=0.1, random_state=None):
        # Initialize base attributes first
        self.explainer = explainer
        
        # Initialize gradient guidance mixin
        GradientGuidanceMixin.__init__(self, explainer, use_gradient_guidance=use_gradient_guidance, noise_beta=noise_beta, random_state=random_state)
        self.random_state = random_state if random_state is not None else 0
        self._rng = np.random.RandomState(random_state)
        self._torch_rng = torch.Generator(device=explainer.device).manual_seed(random_state or 0)

    def generate_new_X(self, eta, num_trials, top_k=1):
        explainer = self.explainer
        best_X = explainer.X.clone()
        best_overall_Q = float("inf")

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
                bounds = [
                    Real(float(explainer.X_prime[:, feat].min()), float(explainer.X_prime[:, feat].max()), name=f"x_{feat}")
                    for feat in explainer.explain_indices
                ]

                def obj(x):
                    row = explainer.X[idx].clone()
                    for j, feat in enumerate(explainer.explain_indices):
                        row[feat] = x[j]
                    
                    # Apply gradient guidance for continuous features if enabled
                    X_candidate = explainer._update_row(explainer.X.clone(), idx, row)
                    if self.use_gradient_guidance:
                        ref_idx = 0  # Use first reference point for Bayesian optimization
                        self._apply_gradient_guidance_to_continuous_features(X_candidate, eta, ref_idx, idx)
                    
                    X_temp = X_candidate
                    y_temp = explainer.model(X_temp)
                    return explainer.evaluate_Q(X_temp, y_temp, eta)[0].item()

                res = gp_minimize(obj, bounds, n_calls=num_trials, random_state=self.random_state)
                best_vals = res.x

                row = explainer.X[idx].clone()
                for j, feat in enumerate(explainer.explain_indices):
                    row[feat] = best_vals[j]
                best_X = explainer._update_row(explainer.X.clone(), idx, row)

        return best_X
