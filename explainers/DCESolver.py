import torch
import numpy as np
import pandas as pd
import time
from typing import Optional
from abc import ABC, abstractmethod
from explainers.base_explainer import BaseExplainer
from explainers.distances import SlicedWassersteinDivergence, WassersteinDivergence
from explainers.model import Model
from explainers.visualization import CallbackVisualizer
from explainers.result_saver import ResultSaver
import math


class DCESolver(BaseExplainer):
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
        strategy=None,
        X_init=True,
        n_proj=10,
        delta=0.1,
        costs_vector=None,
        U_1=0.5,
        U_2=0.3,
        alpha=0.05,
        l=0.2,
        r=1.0,
        kappa=0.05,
        max_iter=100,
        num_trials=10,
        bootstrap=True,
        callback=None,
        top_k=1,
        save_results=True,
        dataset_name=None,
        model_name=None,
        seed=None,
        g_method="cone_sampling",
        results_dir=None,
        use_global_ranges=True,
        save_iterations=True,
        coverage=None,
        target_ot_y=None
    ):
        if not hasattr(strategy, "generate_new_X"):
            raise ValueError("The strategy must implement method `generate_new_X(eta, num_trials, top_k)`")

        if explain_columns is None:
            explain_columns = self.data.explain_columns
        if categorical_columns is None:
            categorical_columns = self.data.categorical_columns
        if continuous_columns is None:
            continuous_columns = self.data.continuous_columns


        if X_init:
            X_init_tensor = self.data.get_X_init()
            print("🎲 Using perturbed initial data (X_init=True)")
        else:
            X_init_tensor = torch.from_numpy(df_factual.values).float()
            print("📍 Using original factual data as starting point (X_init=False)")
            
        y_target = self.data.get_y_target()
        
        # Store X_init for saving
        self.X_init = X_init_tensor

        # Setup save directory FIRST if save_results is True
        if save_results:
            self.result_saver = ResultSaver()
            self.save_dir = self.result_saver.setup_save_directory(
                dataset_name, model_name, strategy, n_proj, delta, U_1, U_2, l, r, max_iter, top_k, seed, g_method, num_trials, results_dir
            )
        else:
            self.result_saver = None
            self.save_dir = None

        # if callback in [True, "final_only"]:
        #     callback = CallbackVisualizer(mode="final_only", model=self.model, data=self.data,
        #                                 explain_columns=explain_columns, y_target=y_target, max_iter=max_iter,
        #                                 save_dir=self.save_dir)
        # elif callback == "full":
        #     callback = CallbackVisualizer(mode="full", model=self.model, data=self.data,
        #                                 explain_columns=explain_columns, y_target=y_target, max_iter=max_iter,
        #                                 save_dir=self.save_dir)
        # else:
        #     callback = None
        callback = None
        self.X = df_factual.values
        self.explain_indices = [df_factual.columns.get_loc(col) for col in explain_columns]
        self.categorical_indices = [df_factual.columns.get_loc(col) for col in categorical_columns]
        self.continuous_indices = [df_factual.columns.get_loc(col) for col in continuous_columns]
        self.explain_columns = explain_columns

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X = torch.from_numpy(self.X).float().to(self.device)
        self.X_prime = self.X.clone()
        self.best_iter = 0

        self.X = X_init_tensor.clone().to(self.device)

        self.best_X = self.X.clone()
        self.final_X = self.X.clone()
        self.y = self.model(self.X)
        self.y_prime = y_target.clone().to(self.device)
        self.best_y = self.y.clone()
        self.final_y = self.y.clone()

        self.swd = SlicedWassersteinDivergence(
            self.X_prime[:, self.explain_indices].shape[1], n_proj=n_proj, random_state=seed
        )
        self.wd = WassersteinDivergence(random_state=seed)

        self.Q = torch.tensor(torch.inf, dtype=torch.float, device=self.device)
        self.best_Q = torch.tensor(torch.inf, dtype=torch.float, device=self.device)
        self.final_Q = torch.tensor(torch.inf, dtype=torch.float, device=self.device)

        self._eta_first_done = False


        self.delta = delta
        self.found_feasible_solution = False

        if costs_vector is None:
            self.costs_vector = torch.ones(len(self.explain_indices)).float()
        else:
            self.costs_vector = torch.tensor(costs_vector).float()
        self.costs_vector_reshaped = self.costs_vector.reshape(1, -1)

        self.interval_left = l
        self.interval_right = r
        self.l_init = l

        # Bridge range information to sampling strategy
        if use_global_ranges and hasattr(self.data, 'continuous_ranges') and hasattr(self.data, 'categorical_ranges'):
            # Use global ranges
            continuous_global_ranges = getattr(self.data, 'continuous_ranges', {})
            categorical_global_ranges = getattr(self.data, 'categorical_ranges', {})
            
            continuous_ranges_by_index = {}
            categorical_ranges_by_index = {}

            for col in continuous_columns:
                if col in continuous_global_ranges:
                    col_idx = df_factual.columns.get_loc(col)
                    continuous_ranges_by_index[col_idx] = continuous_global_ranges[col]
            
            for col in categorical_columns:
                if col in categorical_global_ranges:
                    col_idx = df_factual.columns.get_loc(col)
                    categorical_ranges_by_index[col_idx] = categorical_global_ranges[col]
            
            if hasattr(strategy, 'set_global_ranges'):
                strategy.set_global_ranges(continuous_ranges_by_index, categorical_ranges_by_index)
            else:
                strategy.continuous_global_ranges = continuous_ranges_by_index
                strategy.categorical_global_ranges = categorical_ranges_by_index

            print(f"🌐 Applied global ranges: {len(continuous_ranges_by_index)} continuous, {len(categorical_ranges_by_index)} categorical")
        else:
            if hasattr(strategy, 'set_global_ranges'):
                strategy.set_global_ranges({}, {})
            else:
                strategy.continuous_global_ranges = {}
                strategy.categorical_global_ranges = {}
            
            if not use_global_ranges:
                print("Using original X_prime ranges (local sampling)")
            else:
                print("Use X_prime ranges")
        
        self.logger_data = []

        # Initialize containers to store X and y for each iteration
        self.iteration_X_list = []  # Store X for each iteration 
        self.iteration_y_matrix = []  # Store y values for each iteration

        # Save initial data before optimization (x_true, y_true, y_target)
        if save_results and self.result_saver:
            self.result_saver.save_initial_data(self, df_factual, y_target)

        print("Training Start")
        print("Optimization Start")

        optimization_start_time = time.time()

        for i in range(max_iter):
            print(f"\n--- Iteration {i} ---")

            if i > 0:
                X_new = strategy.generate_new_X(eta=eta, num_trials=num_trials, top_k=top_k)
                self.X.data = X_new.data
                self.y = self.model(self.X)

            swd_dist, _ = self.swd.distance(
                X_s=self.X[:, self.explain_indices] * self.costs_vector_reshaped,
                X_t=self.X_prime[:, self.explain_indices] * self.costs_vector_reshaped,
                delta=self.delta,
            )
            wd_dist, _ = self.wd.distance(
                y_s=self.y,
                y_t=self.y_prime,
                delta=self.delta,
            )
            self.Qv_lower, self.Qv_upper = self.wd.distance_interval(
                self.y, self.y_prime, delta=self.delta, alpha=alpha, bootstrap=bootstrap
            )
            self.Qu_lower, self.Qu_upper = self.swd.distance_interval(
                self.X[:, self.explain_indices] * self.costs_vector_reshaped,
                self.X_prime[:, self.explain_indices] * self.costs_vector_reshaped,
                delta=self.delta,
                alpha=alpha,
                bootstrap=False,
            )
            if not self.Qu_upper >= 0:
                self.Qu_upper = swd_dist
            if not self.Qv_upper >= 0:
                self.Qv_upper = wd_dist

            eta, l, r = self._get_eta_interval_narrowing(U_1, U_2, self.Qu_upper, self.Qv_upper, l, r, kappa)

            Q, term1, term2, mu_list, nu = self.evaluate_Q(self.X, self.y, eta, inplace=True)
            self.swd.mu_list = mu_list
            self.wd.nu = nu

            print(f"U1 - Qu_upper = {U_1 - self.Qu_upper:.6f}, U2 - Qv_upper = {U_2 - self.Qv_upper:.6f}")
            print(f"eta = {eta:.4f}, interval_left = {l:.4f}, interval_right = {r:.4f}")
            print(f"Q = {Q.item():.6f}, term1 = {term1.item():.6f}, term2 = {term2.item():.6f}")

            is_feasible = (U_1 - self.Qu_upper) > 0 and (U_2 - self.Qv_upper) > 0

            # Calculate OT distances (X and Y) 
            ot_distance_x = None
            ot_distance_y = None

            if save_results:
                def compute_distance(X_s, X_t):
                    if type(X_s) == pd.DataFrame:
                        X_s = torch.FloatTensor(X_s.values)
                    if type(X_t) == pd.DataFrame:
                        X_t = torch.FloatTensor(X_t.values)
                    if type(X_s) == np.ndarray:
                        X_s = torch.FloatTensor(X_s)
                    if type(X_t) == np.ndarray:
                        X_t = torch.FloatTensor(X_t)

                    if isinstance(X_s, torch.Tensor):
                        X_s = X_s.float()
                    if isinstance(X_t, torch.Tensor):
                        X_t = X_t.float()

                    if X_s.ndim == 2 and X_s.shape[1] == 1 and X_t.ndim == 1:
                        X_s = X_s.squeeze(-1)
                    elif X_t.ndim == 2 and X_t.shape[1] == 1 and X_s.ndim == 1:
                        X_t = X_t.squeeze(-1)
                    elif X_s.ndim == 1 and X_t.ndim == 1:
                        pass  # Both 1D, no action needed
                    elif X_s.ndim == 2 and X_t.ndim == 2:
                        pass  # Both 2D, will use SlicedWassersteinDivergence
                    else:
                        if X_s.ndim > 1:
                            X_s = X_s.squeeze()
                        if X_t.ndim > 1:
                            X_t = X_t.squeeze()

                    if X_s.ndim == 1:
                        wd = WassersteinDivergence()
                        distance, _ = wd.distance(X_s, X_t, delta=0.1)
                    else:
                        swd = SlicedWassersteinDivergence(dim=X_s.shape[1], n_proj=5000, random_state=seed)
                        distance, _ = swd.distance(X_s, X_t, delta=0.1)
                    return distance.item()

                # Calculate OT X distance 
                ot_distance_x = compute_distance(self.X.detach().cpu(), self.X_prime.detach().cpu())

                # Calculate OT Y distance 
                ot_distance_y = compute_distance(self.y.detach().cpu(), self.y_prime.detach().cpu())

                print(f"OT Distance X = {ot_distance_x:.6f}, OT Distance Y = {ot_distance_y:.6f}")

                # Check if target OT Y is reached (if specified)
                if target_ot_y is not None and ot_distance_y <= target_ot_y:
                    print(f"🎯 Target Y OT distance {target_ot_y:.6f} reached! Current: {ot_distance_y:.6f}")
                    print(f"Early stopping at iteration {i}")

                    self.best_Q = Q.item()
                    self.best_X = self.X.clone().detach()
                    self.best_y = self.y.clone().detach()
                    self.best_iter = i
                    self.found_feasible_solution = True

                    self.final_X = self.best_X.clone().detach()
                    self.final_y = self.best_y.clone().detach()
                    self.final_Q = self.best_Q

                    optimization_end_time = time.time()
                    self.optimization_time = optimization_end_time - optimization_start_time
                    print(f"⏱️  Total optimization time:{self.optimization_time:.2f} seconds")

                    if save_results and self.result_saver:
                        if save_results:
                            log_entry = {
                                'iteration': i,
                                'Q': Q.item(),
                                'term1': term1.item(),
                                'term2': term2.item(),
                                'eta': eta,
                                'Qu_upper': self.Qu_upper,
                                'Qv_upper': self.Qv_upper,
                                'U1_minus_Qu': U_1 - self.Qu_upper,
                                'U2_minus_Qv': U_2 - self.Qv_upper,
                                'is_feasible': is_feasible,
                                'interval_left': l,
                                'interval_right': r
                            }
                            if ot_distance_x is not None:
                                log_entry['OT_distance_X'] = ot_distance_x
                            if ot_distance_y is not None:
                                log_entry['OT_distance_Y'] = ot_distance_y

                            self.logger_data.append(log_entry)

                            if save_iterations:
                                X_current_denorm = self._denormalize_X(self.X, df_factual.columns)
                                self.iteration_X_list.append(X_current_denorm)
                                y_current = self.y.detach().cpu().numpy().flatten()
                                self.iteration_y_matrix.append(y_current)

                        self.result_saver.save_results(self, df_factual.columns)

                    result_X = self.best_X
                    df_result = pd.DataFrame(result_X.detach().cpu().numpy(), columns=df_factual.columns)

                    if hasattr(self.data, 'mean') and hasattr(self.data, 'std') and self.result_saver:     
                        df_result = self.result_saver.recover_data(df_result, df_factual.columns, self.data)

                    return df_result

            if save_results:
                log_entry = {
                    'iteration': i,
                    'Q': Q.item(),
                    'term1': term1.item(),
                    'term2': term2.item(),
                    'eta': eta,
                    'Qu_upper': self.Qu_upper,
                    'Qv_upper': self.Qv_upper,
                    'U1_minus_Qu': U_1 - self.Qu_upper,
                    'U2_minus_Qv': U_2 - self.Qv_upper,
                    'is_feasible': is_feasible,
                    'interval_left': l,
                    'interval_right': r
                }
                if ot_distance_x is not None:
                    log_entry['OT_distance_X'] = ot_distance_x
                if ot_distance_y is not None:
                    log_entry['OT_distance_Y'] = ot_distance_y

                self.logger_data.append(log_entry)

                if save_iterations:
                    X_current_denorm = self._denormalize_X(self.X, df_factual.columns)
                    self.iteration_X_list.append(X_current_denorm)
                    y_current = self.y.detach().cpu().numpy().flatten()
                    self.iteration_y_matrix.append(y_current)
            if is_feasible and Q.item() < getattr(self, "best_Q", float("inf")):
                self.best_Q = Q.item()
                self.best_X = self.X.clone().detach()
                self.best_y = self.y.clone().detach()
                self.best_iter = i
                self.found_feasible_solution = True
                print(f"🌟 New best Q found: {self.best_Q:.6f} at iter {i}")

                # Check coverage early stopping criterion
                if coverage is not None:
                    y_current = self.y.clone().detach()
                    y_target_tensor = self.y_prime.clone().detach()

                    if y_current.dim() == 1 or y_current.shape[1] == 1:
                        matches = (y_current.round() == y_target_tensor.round()).sum().item()
                    else:
                        matches = (y_current.argmax(dim=1) == y_target_tensor.argmax(dim=1)).sum().item()

                    total = y_current.shape[0]
                    current_coverage = matches / total if total > 0 else 0.0

                    print(f"📊 Coverage: {current_coverage:.2%} ({matches}/{total}), Target: {coverage:.2%}")

                    if current_coverage >= coverage:
                        print(f"✅ Coverage threshold {coverage:.2%} reached! Early stopping at iteration {i}")
                        self.final_X = self.best_X.clone().detach()
                        self.final_y = self.best_y.clone().detach()
                        self.final_Q = self.best_Q

                        optimization_end_time = time.time()
                        self.optimization_time = optimization_end_time - optimization_start_time
                        print(f"⏱️  Total optimization time: {self.optimization_time:.2f} seconds")

                        if save_results and self.result_saver:
                            self.result_saver.save_results(self, df_factual.columns)

                        result_X = self.best_X
                        df_result = pd.DataFrame(result_X.detach().cpu().numpy(), columns=df_factual.columns)

                        if hasattr(self.data, 'mean') and hasattr(self.data, 'std') and self.result_saver:
                            df_result = self.result_saver.recover_data(df_result, df_factual.columns, self.data)

                        return df_result

            self.final_X = self.X.clone().detach()
            self.final_y = self.y.clone().detach()
            self.final_Q = Q.item()
            print(f"best_Q = {getattr(self, 'best_Q', None)}, final_Q = {self.final_Q:.6f}")


        print("Optimization End")

        optimization_end_time = time.time()
        self.optimization_time = optimization_end_time - optimization_start_time
        print(f"⏱️  Total optimization time: {self.optimization_time:.2f} seconds")

        if save_results and self.result_saver:
            self.result_saver.save_results(self, df_factual.columns)
            
        result_X = self.best_X if self.found_feasible_solution else self.final_X
        df_result = pd.DataFrame(result_X.detach().cpu().numpy(), columns=df_factual.columns)
        
        if hasattr(self.data, 'mean') and hasattr(self.data, 'std') and self.result_saver:
            df_result = self.result_saver.recover_data(df_result, df_factual.columns, self.data)
            
        return df_result

    def evaluate_Q(self, X_candidate, y_candidate, eta, inplace=False):
        X_s = X_candidate[:, self.explain_indices] * self.costs_vector_reshaped
        X_t = self.X_prime[:, self.explain_indices] * self.costs_vector_reshaped

        self.swd.distance(X_s=X_s, X_t=X_t, delta=self.delta)
        self.wd.distance(y_s=y_candidate, y_t=self.y_prime, delta=self.delta)

        mu_list = self.swd.mu_list
        nu = self.wd.nu

        thetas = [torch.from_numpy(theta).float().to(self.device) for theta in self.swd.thetas]
        term1 = torch.tensor(0.0, dtype=torch.float, device=self.device)
        n, m = X_s.shape[0], X_t.shape[0]

        for k, theta in enumerate(thetas):
            mu = mu_list[k].to(self.device)
            for i in range(n):
                for j in range(m):
                    term1 += mu[i, j] * (torch.dot(theta, X_s[i]) - torch.dot(theta, X_t[j])) ** 2

        term1 /= float(self.swd.n_proj)

        term2 = torch.tensor(0.0, dtype=torch.float)
        for i in range(n):
            for j in range(m):
                term2 += (nu[i, j] * (y_candidate[i] - self.y_prime[j]) ** 2).item()

        Q = (1 - eta) * term1 + eta * term2

        if inplace:
            self.Q = Q
            self.term1 = term1
            self.term2 = term2

        return Q, term1, term2, mu_list, nu


    def _get_eta_interval_narrowing(self, U_1, U_2, Qu_upper, Qv_upper, l, r, kappa, eta_init=0.2):
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

        return eta, l, r

    def __choose_eta_within_interval(self, a, b, l, r):
        if (a < 0 < b) or (a > 0 > b):
            return l if a < 0 else r
        eta_proportion = b / (a + b) if a < 0 else a / (a + b)
        return l + eta_proportion * (r - l)


    def _get_topk_Q_indices(self, X_s, X_t, y, y_prime, mu_list, nu, eta, top_k=1):
        n, m = X_s.shape[0], X_t.shape[0]
        q_x_contributions = torch.zeros(n, device=self.device)
        q_y_contributions = torch.zeros(n, device=self.device)

        thetas = [torch.from_numpy(theta).float().to(self.device) for theta in self.swd.thetas]
        for k, theta in enumerate(thetas):
            mu = mu_list[k].to(self.device)
            for i in range(n):
                for j in range(m):
                    proj_diff = torch.dot(theta, X_s[i]) - torch.dot(theta, X_t[j])
                    q_x_contributions[i] += mu[i, j] * proj_diff**2
        q_x_contributions /= self.swd.n_proj

        for i in range(n):
            for j in range(m):
                q_y_contributions[i] += nu[i, j] * (y[i] - y_prime[j]) ** 2

        q_total = (1 - eta) * q_x_contributions + eta * q_y_contributions
        top_k = min(top_k, n)
        top_indices = torch.topk(q_total, k=top_k).indices.tolist()
        return top_indices

    def _update_row(self, X, idx, new_row):
        X[idx] = new_row
        return X

    def _denormalize_X(self, X_tensor, columns):
        """Denormalize standardized X back to original scale"""
        X_np = X_tensor.detach().cpu().numpy()
        df_denorm = pd.DataFrame(X_np, columns=columns)

        for col in self.explain_columns:
            if col in df_denorm.columns:
                mean_val = self.data.mean[col]
                std_val = self.data.std[col]
                df_denorm[col] = df_denorm[col] * std_val + mean_val

        # Recover data types 
        if hasattr(self.data, 'df'):
            dtype_dict = self.data.df.dtypes.to_dict()
            for col, dtype in dtype_dict.items():
                if col in df_denorm.columns and str(dtype).startswith('int'):
                    df_denorm[col] = df_denorm[col].round().astype(dtype)

        return df_denorm
        
    
        
            
    
