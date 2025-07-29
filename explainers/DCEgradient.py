import torch
import numpy as np
import pandas as pd
from typing import Optional
from explainers.base_explainer import BaseExplainer
from explainers.distances import SlicedWassersteinDivergence, WassersteinDivergence
from explainers.model import Model
from explainers.visualization import CallbackVisualizer
from explainers.result_saver import ResultSaver
import torch.optim as optim
import math
import time


class DCEExplainerGradient(BaseExplainer):
    def __init__(self, model: Optional[Model] = None, data=None, model_name=None):
        if model is None:
            if model_name is None:
                raise ValueError("You must provide either `model` or `model_name`.")
            if data is None:
                raise ValueError("`data` must be provided when using `model_name` to auto-train.")
            model = Model(model_name=model_name, X_train=data.X_train, y_train=data.y_train)

        super().__init__(model=model, data=data)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.data = data
        self.logger_data = []

    def explain(
        self,
        df_factual: pd.DataFrame,
        explain_columns=None,
        categorical_columns=None,
        continuous_columns=None,
        y_target=None,
        X_init=True,
        lr=0.1,
        n_proj=50,
        delta=0.1,
        U_1=0.5,
        U_2=0.3,
        alpha=0.05,
        l=0.2,
        r=1.0,
        kappa=0.05,
        max_iter=100,
        tau=1e3,
        tol=1e-6,
        bootstrap=True,
        callback=None,
        save_results=True,
        dataset_name=None,
        model_name=None,
        seed=None,
        random_state=None
    ):
        # Auto-fetch if omitted
        if explain_columns is None:
            explain_columns = self.data.explain_columns
        if categorical_columns is None:
            categorical_columns = self.data.categorical_columns
        if continuous_columns is None:
            continuous_columns = self.data.continuous_columns

        # Store the random_state for this explanation session
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = seed

        # Handle X_init as boolean to control initial perturbation
        if X_init:
            # Use perturbed initial data (with noise)
            X_init_tensor = self.data.get_X_init()
            print("🎲 Using perturbed initial data (X_init=True)")
        else:
            # Use original factual data as starting point (no noise)
            X_init_tensor = torch.from_numpy(df_factual.values).float()
            print("📍 Using original factual data as starting point (X_init=False)")
            
        y_target = self.data.get_y_target()
        
        # Store X_init for saving
        self.X_init = X_init_tensor

        # Setup save directory FIRST if save_results is True
        if save_results:
            self.result_saver = ResultSaver()
            # Create a mock strategy object for the result saver
            class GradientStrategy:
                def __init__(self):
                    pass
            
            mock_strategy = GradientStrategy()
            mock_strategy.__class__.__name__ = "GradientStrategy"
            
            # self.save_dir = self.result_saver.setup_save_directory(
            #     dataset_name, model_name, mock_strategy, n_proj, delta, U_1, U_2, l, r, max_iter, 1, seed
            # )
            self.save_dir = self.result_saver.setup_save_directory(
                dataset_name, model_name, mock_strategy, n_proj, delta, U_1, U_2, l, r,
                max_iter, 1, seed,
                gradient_method=""
            )

            # Override the DCE_ prefix to DCE_gradient_
            import os
            old_save_dir = self.save_dir
            new_save_dir = old_save_dir.replace("/DCE_", "/DCE_gradient_")
            if old_save_dir != new_save_dir:
                # Create the new directory and update
                os.makedirs(os.path.dirname(new_save_dir), exist_ok=True)
                if os.path.exists(old_save_dir):
                    import shutil
                    shutil.move(old_save_dir, new_save_dir)
                self.save_dir = new_save_dir
                self.result_saver.save_dir = new_save_dir
                print(f"Updated save directory to: {self.save_dir}")
        else:
            self.result_saver = None
            self.save_dir = None

        # Set up callback visualizer (mode = "off" | "final_only" | "full")
        if callback in [True, "final_only"]:
            callback = CallbackVisualizer(mode="final_only", model=self.model, data=self.data,
                                        explain_columns=explain_columns, y_target=y_target, max_iter=max_iter,
                                        save_dir=self.save_dir)
        elif callback == "full":
            callback = CallbackVisualizer(mode="full", model=self.model, data=self.data,
                                        explain_columns=explain_columns, y_target=y_target, max_iter=max_iter,
                                        save_dir=self.save_dir)
        else:
            callback = None

        self.explain_columns = explain_columns
        self.explain_indices = [df_factual.columns.get_loc(col) for col in explain_columns]
        self.categorical_indices = [df_factual.columns.get_loc(col) for col in categorical_columns]
        self.continuous_indices = [df_factual.columns.get_loc(col) for col in continuous_columns]
        
        self.X = X_init_tensor.clone().to(self.device)
        self.X_prime = torch.from_numpy(df_factual.values).float().to(self.device)

        self.X.requires_grad_(True).retain_grad()
        self.optimizer = optim.SGD([self.X], lr=lr)

        self.y_prime = y_target.to(self.device)
        self.y = self.model(self.X)

        # Fixed seed SWD computation like nondifferentiable.py
        self.swd = SlicedWassersteinDivergence(len(self.explain_indices), n_proj=n_proj, random_state=seed)
        self.wd = WassersteinDivergence(random_state=seed)
        
        self.costs_vector = torch.ones(len(self.explain_indices)).float().to(self.device)
        self.costs_vector_reshaped = self.costs_vector.view(1, -1)

        self.interval_left = l
        self.interval_right = r
        past_Qs = [float("inf")] * 5
        
        self.best_Q = float("inf")
        self.best_X = None
        self.best_y = None
        self.best_iter = 0
        self.found_feasible_solution = False

        # Initialize Q and terms
        self.Q = torch.tensor(torch.inf, dtype=torch.float, device=self.device)
        self.term1 = torch.tensor(0.0, dtype=torch.float, device=self.device)
        self.term2 = torch.tensor(0.0, dtype=torch.float, device=self.device)
        self.X_prev = self.X.clone().detach()  # For tracking changes

        # Save initial data files (x_true, y_true, y_target)
        if save_results and self.result_saver:
            self.result_saver.save_initial_data(self, df_factual, y_target)

        print("Optimization started")
        for i in range(max_iter):
            print(f"\n--- Iteration {i} ---")

            # Calculate distances first
            X_s = self.X[:, self.explain_indices] * self.costs_vector_reshaped
            X_t = self.X_prime[:, self.explain_indices] * self.costs_vector_reshaped
            y_s = self.y
            y_t = self.y_prime

            swd_dist, _ = self.swd.distance(X_s, X_t, delta=delta)
            wd_dist, _ = self.wd.distance(y_s, y_t, delta=delta)

            Qv_lower, Qv_upper = self.wd.distance_interval(y_s, y_t, delta=delta, alpha=alpha, bootstrap=bootstrap)
            Qu_lower, Qu_upper = self.swd.distance_interval(X_s, X_t, delta=delta, alpha=alpha, bootstrap=False)

            if not torch.isfinite(torch.tensor(Qu_upper)):
                Qu_upper = swd_dist
            if not torch.isfinite(torch.tensor(Qv_upper)):
                Qv_upper = wd_dist
                
            # Store for logging (matching nondifferentiable.py)
            self.Qu_upper = Qu_upper
            self.Qv_upper = Qv_upper

            eta, self.interval_left, self.interval_right = self._get_eta_interval_narrowing(
                U_1, U_2, Qu_upper, Qv_upper, self.interval_left, self.interval_right, kappa
            )

            print(f"U_1 - Qu_upper = {U_1 - Qu_upper:.6f}, U_2 - Qv_upper = {U_2 - Qv_upper:.6f}")
            print(f"eta = {eta:.4f}, interval_left = {self.interval_left:.4f}, interval_right = {self.interval_right:.4f}")

            # Perform SGD step (includes Q calculation)
            avg_Q_change = self.__perform_SGD(past_Qs, eta=eta, tau=tau)

            print(f"Q = {self.Q.item():.6f}, term1 = {self.term1.item():.6f}, term2 = {self.term2.item():.6f}")

            # Check feasibility and update best solution
            if (U_1 - Qu_upper) < 0 or (U_2 - Qv_upper) < 0:
                gap = np.inf
                is_feasible = False
            else:
                gap = (U_1 - Qu_upper) + (U_2 - Qv_upper)
                self.found_feasible_solution = True
                is_feasible = True

            if is_feasible and self.Q.item() < getattr(self, "best_Q", float("inf")):
                self.best_Q = self.Q.item()
                self.best_X = self.X.clone().detach()
                self.best_y = self.model(self.best_X).detach()
                self.best_iter = i
                self.found_feasible_solution = True
                print(f"🌟 New best Q found: {self.best_Q:.6f} at iter {i}")

            self.final_X = self.X.clone().detach()
            self.final_y = self.model(self.final_X).detach()
            self.final_Q = self.Q.item()
            print(f"best_Q = {getattr(self, 'best_Q', None)}, final_Q = {self.final_Q:.6f}")

            # Log data for saving (matching nondifferentiable.py exactly)
            if save_results:
                self.logger_data.append({
                    'iteration': i,
                    'Q': self.Q.item(),
                    'term1': self.term1.item(),
                    'term2': self.term2.item(),
                    'eta': eta,
                    'Qu_upper': self.Qu_upper,
                    'Qv_upper': self.Qv_upper,
                    'U1_minus_Qu': U_1 - self.Qu_upper,
                    'U2_minus_Qv': U_2 - self.Qv_upper,
                    'is_feasible': is_feasible,
                    'interval_left': self.interval_left,
                    'interval_right': self.interval_right
                })

            if callback:
                callback(self, i)

            if abs(avg_Q_change) < tol:
                print(f"Converged at iteration {i}")
                break

        print("Optimization End")

        # Save results if requested
        if save_results and self.result_saver:
            self.result_saver.save_results(self, df_factual.columns)
            
        # Return recovered (denormalized) data for better readability
        result_X = self.best_X if self.found_feasible_solution else self.final_X
        df_result = pd.DataFrame(result_X.detach().cpu().numpy(), columns=df_factual.columns)
        
        # Apply data recovery if possible
        if hasattr(self.data, 'mean') and hasattr(self.data, 'std') and self.result_saver:
            df_result = self.result_saver.recover_data(df_result, df_factual.columns, self.data)
            
        return df_result

    def _update_Q(self, mu_list, nu, eta):
        n, m = (
            self.X[:, self.explain_indices].shape[0],
            self.X_prime[:, self.explain_indices].shape[0],
        )

        thetas = [
            torch.from_numpy(theta).float().to(self.device) for theta in self.swd.thetas
        ]

        # Compute the first term
        term1 = torch.tensor(0.0, dtype=torch.float).to(self.device)
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

                    term1 += (
                        mu[i, j]
                        * (
                            torch.dot(theta, weighted_X[i])
                            - torch.dot(theta, weighted_X_prime[j])
                        )
                        ** 2
                    )
        term1 /= torch.tensor(
            self.swd.n_proj, dtype=torch.float, device=self.device
        )

        # Compute the second term
        term2 = torch.tensor(0.0, dtype=torch.float)
        for i in range(n):
            for j in range(m):
                term2 += (
                    nu[i, j] * (self.model(self.X[i]) - self.y_prime[j]) ** 2
                ).item()

        Q = (1 - eta) * term1 + eta * term2
        
        # Store as instance attributes
        self.Q = Q
        self.term1 = term1
        self.term2 = term2
        
        return Q, term1, term2

    def __perform_SGD(self, past_Qs, eta, tau):
        # Reset the gradients
        self.optimizer.zero_grad()

        # Compute the gradients for self.X[:, self.explain_indices]
        self._update_X_grads(
            mu_list=self.swd.mu_list,
            nu=self.wd.nu,
            eta=eta,
            tau=tau,
        )

        # Perform an optimization step
        self.optimizer.step()

        # Update the Q value, X_all, and y by the newly optimized X
        self._update_Q(mu_list=self.swd.mu_list, nu=self.wd.nu, eta=eta)
        self.y = self.model(self.X)

        # Check for convergence using moving average of past Q changes
        past_Qs.pop(0)
        past_Qs.append(self.Q.item())
        avg_Q_change = (past_Qs[-1] - past_Qs[0]) / 5
        return avg_Q_change

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

        gradient_term2 = (nu.unsqueeze(-1) * diff_model * model_grads.unsqueeze(1)).sum(
            dim=1
        )

        self.Qx_grads = (1 - eta) * gradient_term1 + eta * gradient_term2
        self.X.grad.zero_()
        self.X.grad[:, self.explain_indices] = self.Qx_grads * tau

    def _get_eta_interval_narrowing(self, U_1, U_2, Qu_upper, Qv_upper, l, r, kappa):
        """
        Implements the interval narrowing algorithm.
        """
        if not math.isfinite(Qv_upper):
            return l, l, r

        if not math.isfinite(Qu_upper):
            return r, l, r

        eta = self.__choose_eta_within_interval(
            a=U_1 - Qu_upper, b=U_2 - Qv_upper, l=l, r=r
        )

        # Narrow the interval
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
    
