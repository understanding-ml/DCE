import torch
import numpy as np

class GeneticStrategy:
    def __init__(
        self,
        explainer,
        crossover_prob=0.8,
        gene_swap_prob=0.5,
        mutation_prob_cat=0.3,
        mutation_prob_cont=0.8,
        mutation_noise_scale=0.1,
        random_state=None
    ):
        self.explainer = explainer
        self.crossover_prob = crossover_prob
        self.gene_swap_prob = gene_swap_prob
        self.mutation_prob_cat = mutation_prob_cat
        self.mutation_prob_cont = mutation_prob_cont
        self.mutation_noise_scale = mutation_noise_scale

        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)
        self._torch_rng = torch.Generator(device=explainer.device).manual_seed(random_state or 0)

    def reseed(self, seed):
        self.random_state = seed
        self._rng = np.random.RandomState(seed)
        self._torch_rng = torch.Generator(device=self.explainer.device).manual_seed(seed)

    def generate_new_X(self, eta: float, num_trials: int, top_k: int = 1) -> torch.Tensor:
        explainer = self.explainer

        with torch.no_grad():
            X_s = explainer.X[:, explainer.explain_indices] * explainer.costs_vector_reshaped
            X_t = explainer.X_prime[:, explainer.explain_indices] * explainer.costs_vector_reshaped
            y = explainer.y.view(-1)
            y_prime = explainer.y_prime.view(-1)
            mu_list = explainer.swd.mu_list
            nu = explainer.wd.nu
            top_indices = explainer._get_topk_Q_indices(X_s, X_t, y, y_prime, mu_list, nu, eta, top_k)

        population = []
        y_candidates = []

        for _ in range(num_trials):
            cand = explainer.X.clone()
            for idx in top_indices:
                parent1 = explainer.X[idx].clone()

                # Crossover
                if torch.rand(1, generator=self._torch_rng, device=explainer.device).item() < self.crossover_prob:
                    idx_partner = self._rng.randint(explainer.X.shape[0])
                    parent2 = explainer.X[idx_partner]
                    for feat_idx in explainer.explain_indices:
                        if torch.rand(1, generator=self._torch_rng, device=explainer.device).item() < self.gene_swap_prob:
                            parent1[feat_idx] = parent2[feat_idx]

                # Categorical mutation
                for feat_idx in explainer.categorical_indices:
                    if torch.rand(1, generator=self._torch_rng, device=explainer.device).item() < self.mutation_prob_cat:
                        unique_vals = torch.unique(explainer.X_prime[:, feat_idx])
                        sampled_val = unique_vals[self._rng.randint(len(unique_vals))]
                        parent1[feat_idx] = (1 - eta) * explainer.X_prime[idx, feat_idx] + eta * sampled_val

                # Continuous mutation
                for feat_idx in explainer.continuous_indices:
                    if torch.rand(1, generator=self._torch_rng, device=explainer.device).item() < self.mutation_prob_cont:
                        min_val = explainer.X_prime[:, feat_idx].min()
                        max_val = explainer.X_prime[:, feat_idx].max()
                        noise = torch.randn(1, generator=self._torch_rng, device=explainer.device) * self.mutation_noise_scale
                        mutated_val = parent1[feat_idx] + noise * (max_val - min_val)
                        parent1[feat_idx] = (1 - eta) * parent1[feat_idx] + eta * mutated_val

                cand[idx] = parent1

            y_cand = explainer.model(cand)
            population.append(cand)
            y_candidates.append(y_cand)

        Q_values = [explainer.evaluate_Q(population[i], y_candidates[i], eta)[0] for i in range(num_trials)]
        best_idx = torch.argmin(torch.tensor(Q_values))
        return population[best_idx]
