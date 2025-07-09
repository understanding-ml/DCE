from abc import ABC, abstractmethod
import torch
import pandas as pd
import numpy as np
from typing import Optional
import torch.optim as optim
from explainers.distances import SlicedWassersteinDivergence, WassersteinDivergence
from explainers.base_explainer import BaseExplainer
from ml_model_interface import Model

class DCEExplainerGradient(BaseExplainer):
    def __init__(self, model: Model, random_state=None):
        super().__init__(model=model, data=None)  # data
        self.random_state = random_state 
        if self.model.backend == 'sklearn':
            self.model.backend = 'sklearn'
        elif self.model.backend == 'pytorch':
            self.model.backend = 'PYT'

        self.model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = self.model.device if hasattr(self.model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.X = None
        self.X_prime = None
        self.y_prime = None
        self.optimizer = None
        self.best_X = None
        self.Qx_grads = None
        self.best_Q = float("inf")

        self.swd = None
        self.wd = WassersteinDivergence(random_state=self.random_state)
        self.Q = torch.tensor(torch.inf, dtype=torch.float)
        self.delta = 0.1

    def explain(
        self,
        df_factual: pd.DataFrame,
        explain_columns,
        categorical_columns,
        continuous_columns,
        y_target,
        X_init=None,
        lr=1e-1,
        n_proj=50,
        delta=0.1,
        U_1=0.5,
        U_2=0.3,
        alpha=0.05,
        l=0.2,
        r=1.0,
        kappa=0.05,
        max_iter=50,
        tau=10,
        tol=1e-6,
        bootstrap=True,
        callback=None,
        random_state=None
    ):
        # Store the random_state for this explanation session
        if random_state is not None:
            self.random_state = random_state
        self.explain_columns = explain_columns
        self.explain_indices = [df_factual.columns.get_loc(col) for col in explain_columns]
        self.categorical_indices = [df_factual.columns.get_loc(col) for col in categorical_columns]
        self.continuous_indices = [df_factual.columns.get_loc(col) for col in continuous_columns]
        
        self.X = df_factual.values
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X = torch.from_numpy(self.X).float().to(self.device)

        self.X_prime = self.X.clone()
        if X_init is None:
            noise = torch.randn_like(self.X_prime[:, self.explain_indices]) * 0.01
            self.X = self.X.clone()
            self.X[:, self.explain_indices] = (
                self.X_prime[:, self.explain_indices] + noise
            ).to(self.device)
        else:
            self.X = X_init.clone().to(self.device)


        self.X.requires_grad_(True).retain_grad()
        self.optimizer = optim.SGD([self.X], lr=lr)

        self.y_prime = y_target.to(self.device)
        self.y = self.model(self.X)

        self.swd = SlicedWassersteinDivergence(len(self.explain_indices), n_proj=n_proj, random_state=self.random_state)
        self.costs_vector = torch.ones(len(self.explain_indices)).float().to(self.device)
        self.costs_vector_reshaped = self.costs_vector.view(1, -1)

    
        self.interval_left = l
        self.interval_right = r
        past_Qs = [float("inf")] * 5

        print("Optimization started")
        for i in range(0, max_iter + 1):
            print(f"\n--- Iteration {i} ---")

            if i > 0:
                self.optimizer.zero_grad()
                self._update_X_grads(mu_list, nu, eta, tau)
                self.optimizer.step()
                self.y = self.model(self.X)

            X_s = self.X[:, self.explain_indices] * self.costs_vector_reshaped
            X_t = self.X_prime[:, self.explain_indices] * self.costs_vector_reshaped
            y_s = self.model(self.X).detach().clone()
            y_t = self.y_prime.detach().clone()

            swd_dist, _ = self.swd.distance(X_s, X_t, delta=self.delta)
            wd_dist, _ = self.wd.distance(y_s, y_t, delta=self.delta)

            Qv_lower, Qv_upper = self.wd.distance_interval(y_s, y_t, delta=self.delta, alpha=alpha, bootstrap=bootstrap)
            Qu_lower, Qu_upper = self.swd.distance_interval(X_s, X_t, delta=self.delta, alpha=alpha, bootstrap=False)

            if not torch.isfinite(torch.tensor(Qu_upper)):
                Qu_upper = swd_dist
            if not torch.isfinite(torch.tensor(Qv_upper)):
                Qv_upper = wd_dist

            eta, self.interval_left, self.interval_right = self._get_eta_interval_narrowing(
                U_1, U_2, Qu_upper, Qv_upper, self.interval_left, self.interval_right, kappa
            )

            self.swd.distance(X_s, X_t, delta=self.delta)
            self.wd.distance(y_s, y_t, delta=self.delta)
            mu_list = self.swd.mu_list
            nu = self.wd.nu
            Q, term1, term2 = self._update_Q(mu_list, nu, eta)

            print(f"U_1 - Qu_upper = {U_1 - Qu_upper:.6f}, U_2 - Qv_upper = {U_2 - Qv_upper:.6f}")
            print(f"eta = {eta:.4f}, interval_left = {self.interval_left:.4f}, interval_right = {self.interval_right:.4f}")
            print(f"Q = {Q.item():.6f}, term1 = {term1.item():.6f}, term2 = {term2.item():.6f}")


            past_Qs.pop(0)
            past_Qs.append(Q.item())
            avg_Q_change = (past_Qs[-1] - past_Qs[0]) / 5

            if Q.item() < self.best_Q:
                self.best_Q = Q.item()
                self.best_X = self.X.clone().detach()
                self.best_y = self.model(self.best_X).detach()
                self.best_iter = i

            self.final_X = self.X.clone().detach()
            self.final_y = self.model(self.final_X).detach()
            self.final_Q = Q.item()

            if callback:
                callback(i)

        print("Optimization finished.")

        return pd.DataFrame(
            self.best_X.detach().cpu().numpy(),
            columns=df_factual.columns
        )


    def _update_Q(self, mu_list, nu, eta):
        n, m = (
            self.X[:, self.explain_indices].shape[0],
            self.X_prime[:, self.explain_indices].shape[0],
        )

        thetas = [
            torch.from_numpy(theta).float().to(self.device) for theta in self.swd.thetas
        ]

        # Compute the first term
        self.term1 = torch.tensor(0.0, dtype=torch.float).to(self.device)
        for k, theta in enumerate(thetas):
            mu = mu_list[k]
            mu = mu.to(self.device)
            for i in range(n):
                for j in range(m):
                    # Apply the costs to the features of X and X_prime
                    weighted_X = (
                        self.X[:, self.explain_indices] * self.costs_vector_reshaped
                    )
                    weighted_X_prime = (
                        self.X_prime[:, self.explain_indices]
                        * self.costs_vector_reshaped
                    )

                    self.term1 += (
                        mu[i, j]
                        * (
                            torch.dot(theta, weighted_X[i])
                            - torch.dot(theta, weighted_X_prime[j])
                        )
                        ** 2
                    )
        self.term1 /= torch.tensor(
            self.swd.n_proj, dtype=torch.float, device=self.device
        )

        # Compute the second term
        self.term2 = torch.tensor(0.0, dtype=torch.float)
        for i in range(n):
            for j in range(m):
                self.term2 += (
                    nu[i, j] * (self.model(self.X[i]) - self.y_prime[j]) ** 2
                ).item()

        self.Q = (1 - eta) * self.term1 + eta * self.term2
        return self.Q, self.term1, self.term2

    def _update_X_grads(self, mu_list, nu, eta, tau):
        n, m = (
            self.X[:, self.explain_indices].shape[0],
            self.X_prime[:, self.explain_indices].shape[0],
        )
        thetas = [
            torch.from_numpy(theta).float().to(self.device) for theta in self.swd.thetas
        ]

        # Obtain model gradients with a dummy backward pass
        outputs = self.model(self.X)
        loss = outputs.sum()

        # Ensure gradients are zeroed out before backward pass
        self.X.grad = None
        loss.backward()
        model_grads = self.X.grad[
            :, self.explain_indices
        ].clone()  # Store the gradients

        # Weights applied to the features of X and X_prime
        weighted_X = self.X[:, self.explain_indices] * self.costs_vector_reshaped
        weighted_X_prime = (
            self.X_prime[:, self.explain_indices] * self.costs_vector_reshaped
        )

        # Compute the projections with the weighted features
        X_proj = torch.stack(
            [torch.matmul(weighted_X, theta) for theta in thetas],
            dim=1,
        )  # Shape: [n, num_thetas]
        X_prime_proj = torch.stack(
            [torch.matmul(weighted_X_prime, theta) for theta in thetas],
            dim=1,
        )  # Shape: [m, num_thetas]

        # Use broadcasting to compute differences for all i, j
        differences = (
            X_proj[:, :, None] - X_prime_proj.T[None, :, :]
        )  # Shape: [n, num_thetas, m]

        # Multiply by mu and sum over j
        gradient_term1_matrix = torch.stack(
            [mu.to(self.device) * differences[:, k, :] for k, mu in enumerate(mu_list)],
            dim=1,
        )  # [n, num_thetas, m]
        gradient_term1 = torch.sum(
            gradient_term1_matrix, dim=2
        )  # Shape [n, num_thetas]

        # Weight by theta to get the gradient
        gradient_term1 = torch.matmul(
            gradient_term1, torch.stack(thetas)
        )  # Shape [n, d]

        # Compute the second term
        diff_model = self.model(self.X).unsqueeze(1) - self.y_prime.reshape(
            len(self.y_prime), 1
        )
        nu = nu.to(self.device)

        self.nu = nu
        self.diff_model = diff_model
        self.model_grads = model_grads
        # print("nu.unsqueeze(-1).shape:", nu.unsqueeze(-1).shape)
        # print("diff_model.shape:", diff_model.shape)
        # print("diff_model.unsqueeze(-1).shape:", diff_model.unsqueeze(-1).shape)
        # print("model_grads.unsqueeze(1).shape:", model_grads.unsqueeze(1).shape)


        gradient_term2 = (nu.unsqueeze(-1) * diff_model * model_grads.unsqueeze(1)).sum(
            dim=1
        )

        self.Qx_grads = (1 - eta) * gradient_term1 + eta * gradient_term2
        # self.Qx_grads = gradient_term2
        self.X.grad.zero_()
        self.X.grad[:, self.explain_indices] = self.Qx_grads * tau


    def _get_eta_interval_narrowing(self, U_1, U_2, Qu_upper, Qv_upper, l, r, kappa):
        eta = self.__choose_eta_within_interval(a=U_1 - Qu_upper, b=U_2 - Qv_upper, l=l, r=r)
        if eta > (l + r) / 2:
            l = l + kappa * (r - l)
        else:
            r = r - kappa * (r - l)
        return eta, l, r

    def __choose_eta_within_interval(self, a, b, l, r):
        if (a < 0 < b) or (a > 0 > b):
            return l if a < 0 else r
        eta_proportion = b / (a + b) if a < 0 else a / (a + b)
        return l + eta_proportion * (r - l)
