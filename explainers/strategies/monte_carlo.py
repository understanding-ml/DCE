import torch
import numpy as np

class MonteCarloStrategy:
    def __init__(self, explainer, random_state=None):
        self.explainer = explainer
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
                    
                # Categorical perturbation
                for idx_feat in explainer.categorical_indices:
                    unique_vals = torch.unique(explainer.X_prime[:, idx_feat])
                    sampled_val = unique_vals[self._rng.randint(len(unique_vals))]  # numpy sampling
                    cand[idx, idx_feat] = (1 - eta) * explainer.X_prime[ref_idx, idx_feat] + eta * sampled_val

                # Continuous perturbation
                for idx_feat in explainer.continuous_indices:
                    min_val = explainer.X_prime[:, idx_feat].min()
                    max_val = explainer.X_prime[:, idx_feat].max()
                    rand_val = torch.rand(1, generator=self._torch_rng, device=explainer.device)
                    sampled_val = min_val + rand_val * (max_val - min_val)
                    cand[idx, idx_feat] = (1 - eta) * explainer.X_prime[ref_idx, idx_feat] + eta * sampled_val

            y_cand = explainer.model(cand)
            current_Q, *_ = explainer.evaluate_Q(cand, y_cand, eta)
            if current_Q < best_Q:
                best_Q = current_Q
                X_best = cand.clone().detach()

        return X_best
