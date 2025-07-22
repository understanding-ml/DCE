import torch
import numpy as np

class CovarianceMatrixAdaptationEvolutionStrategy:
    def __init__(self, explainer, population_size=10, sigma_decay=0.9, random_state=None):
        self.explainer = explainer
        self.population_size = population_size
        self.sigma_decay = sigma_decay
        self.random_state = random_state
        self._rng = torch.Generator(device=explainer.device)
        if random_state is not None:
            self._rng.manual_seed(random_state)

    def generate_new_X(self, eta, num_trials, top_k=1):
        explainer = self.explainer
        best_X = explainer.X.clone()
        best_overall_Q = float('inf')

        with torch.no_grad():
            X_s = explainer.X[:, explainer.explain_indices] * explainer.costs_vector_reshaped
            X_t = explainer.X_prime[:, explainer.explain_indices] * explainer.costs_vector_reshaped
            y = explainer.y.view(-1)
            y_prime = explainer.y_prime.view(-1)
            mu_list = explainer.swd.mu_list
            nu = explainer.wd.nu
            candidate_indices = explainer._get_topk_Q_indices(X_s, X_t, y, y_prime, mu_list, nu, eta, top_k)

            for idx in candidate_indices:
                x0 = explainer.X[idx, explainer.explain_indices].clone()
                mean = x0.clone()
                sigma = 0.1

                for _ in range(num_trials):
                    pop = [
                        mean + sigma * torch.randn(mean.shape, generator=self._rng, device=mean.device)
                        for _ in range(self.population_size)
                    ]
                    scores = []
                    for p in pop:
                        row = explainer.X[idx].clone()
                        row[explainer.explain_indices] = p
                        X_temp = explainer._update_row(explainer.X.clone(), idx, row)
                        Q = explainer.evaluate_Q(X_temp, explainer.model(X_temp), eta)[0]
                        scores.append((Q, p.clone()))
                    scores.sort(key=lambda x: x[0])
                    mean = scores[0][1].clone()
                    sigma *= self.sigma_decay

                best_row = explainer.X[idx].clone()
                best_row[explainer.explain_indices] = mean
                best_X = explainer._update_row(explainer.X.clone(), idx, best_row)

        return best_X
