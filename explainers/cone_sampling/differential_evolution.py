import torch
import numpy as np
import math
from .gradient_guidance_mixin import GradientGuidanceMixin

class DifferentialEvolutionStrategy(GradientGuidanceMixin):
    def __init__(self, explainer, F=0.5, CR=0.9, use_gradient_guidance=False, cone_angle=math.pi/4, random_state=None):
        # Initialize base attributes first
        self.explainer = explainer
        
        # Initialize gradient guidance mixin
        GradientGuidanceMixin.__init__(self, explainer, use_gradient_guidance=use_gradient_guidance, cone_angle=cone_angle, random_state=random_state)
        self.F = F  # 差分缩放因子
        self.CR = CR  # 交叉概率
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)
        self._torch_rng = torch.Generator(device=explainer.device).manual_seed(random_state or 0)

    def reseed(self, seed):
        self.random_state = seed
        self._rng = np.random.RandomState(seed)
        self._torch_rng = torch.Generator(device=self.explainer.device).manual_seed(seed)

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
                X_s, X_t, y, y_prime, mu_list, nu, eta, top_k=top_k
            )

            for idx in candidate_indices:
                current_candidate = explainer.X[idx].clone()
                X_temp = explainer._update_row(explainer.X.clone(), idx, current_candidate)
                y_temp = explainer.model(X_temp)
                best_Q_candidate = explainer.evaluate_Q(X_temp, y_temp, eta)[0]

                for _ in range(num_trials):
                    candidates = list(range(explainer.X.size(0)))
                    candidates.remove(idx)
                    r1, r2 = self._rng.choice(candidates, 2, replace=False)
                    x_r1 = explainer.X[r1, explainer.explain_indices]
                    x_r2 = explainer.X[r2, explainer.explain_indices]
                    current_vec = current_candidate[explainer.explain_indices]
                    mutant = current_vec + self.F * (x_r1 - x_r2)

                    trial = current_vec.clone()
                    for j in range(len(explainer.explain_indices)):
                        if self._rng.rand() < self.CR:
                            trial[j] = mutant[j]

                    # Apply gradient guidance for continuous features if enabled
                    if self.use_gradient_guidance:
                        X_candidate = explainer.X.clone()
                        new_candidate_temp = current_candidate.clone()
                        for pos, feat in enumerate(explainer.explain_indices):
                            new_candidate_temp[feat] = trial[pos]
                        X_candidate = explainer._update_row(X_candidate, idx, new_candidate_temp)
                        ref_idx = 0  # Use first reference point
                        self._apply_gradient_guidance_to_continuous_features(X_candidate, eta, ref_idx, idx)
                        # Update trial with guided values for continuous features
                        for pos, feat in enumerate(explainer.explain_indices):
                            if feat in explainer.continuous_indices:
                                trial[pos] = X_candidate[idx, feat]

                    for feat in explainer.categorical_indices:
                        if feat in explainer.explain_indices:
                            pos = explainer.explain_indices.index(feat)
                            vals = torch.unique(explainer.X_prime[:, feat])
                            if trial[pos] not in vals:
                                rand_idx = torch.randint(0, len(vals), (1,), generator=self._torch_rng, device=explainer.device)
                                trial[pos] = vals[rand_idx]

                    new_candidate = current_candidate.clone()
                    for pos, feat in enumerate(explainer.explain_indices):
                        new_candidate[feat] = trial[pos]

                    X_new_candidate = explainer._update_row(explainer.X.clone(), idx, new_candidate)
                    y_new_candidate = explainer.model(X_new_candidate)
                    Q_trial = explainer.evaluate_Q(X_new_candidate, y_new_candidate, eta)[0]

                    if Q_trial < best_Q_candidate:
                        current_candidate = new_candidate.clone()
                        best_Q_candidate = Q_trial

                if best_Q_candidate < best_overall_Q:
                    best_overall_Q = best_Q_candidate
                    best_X = explainer._update_row(explainer.X.clone(), idx, current_candidate)

        return best_X
