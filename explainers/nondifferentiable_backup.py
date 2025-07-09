import torch
import numpy as np
import pandas as pd
from typing import Optional
from abc import ABC, abstractmethod
from explainers.base_explainer import BaseExplainer
from explainers.distances import SlicedWassersteinDivergence, WassersteinDivergence
from explainers.model import Model
from explainers.visualization import CallbackVisualizer
from explainers.result_saver import ResultSaver
import math


class DCENonDifferentiable(BaseExplainer):
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
        X_init=None,
        lr=0.1,
        init_eta=0.5,
        n_proj=50,
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
        tol=1e-6,
        save_results=True,
        dataset_name=None,
        model_name=None,
        seed=None
    ):
        if not hasattr(strategy, "generate_new_X"):
            raise ValueError("The strategy must implement method `generate_new_X(eta, num_trials, top_k)`")

        # Auto-fetch if omitted
        if explain_columns is None:
            explain_columns = self.data.explain_columns
        if categorical_columns is None:
            categorical_columns = self.data.categorical_columns
        if continuous_columns is None:
            continuous_columns = self.data.continuous_columns

        X_init = self.data.get_X_init()      
        y_target = self.data.get_y_target()
        
        # Store X_init for saving
        self.X_init = X_init

        # Setup save directory FIRST if save_results is True
        if save_results:
            self.result_saver = ResultSaver()
            self.save_dir = self.result_saver.setup_save_directory(
                dataset_name, model_name, strategy, n_proj, delta, U_1, U_2, l, r, max_iter, top_k, seed
            )
        else:
            self.result_saver = None
            self.save_dir = None

        # Set up callback visualizer (mode = "off" | "final_only" | "full")
        # Now save_dir is available if needed
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

        self.X = df_factual.values
        self.explain_indices = [df_factual.columns.get_loc(col) for col in explain_columns]
        self.categorical_indices = [df_factual.columns.get_loc(col) for col in categorical_columns]
        self.continuous_indices = [df_factual.columns.get_loc(col) for col in continuous_columns]
        self.explain_columns = explain_columns

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X = torch.from_numpy(self.X).float().to(self.device)
        self.X_prime = self.X.clone()
        self.best_iter = 0

        self.X = X_init.clone().to(self.device)

        self.best_X = self.X.clone()
        self.final_X = self.X.clone()
        self.y = self.model(self.X)
        self.y_prime = y_target.clone().to(self.device)
        self.best_y = self.y.clone()
        self.final_y = self.y.clone()

        self.swd = SlicedWassersteinDivergence(
            self.X_prime[:, self.explain_indices].shape[1], n_proj=n_proj
        )
        self.wd = WassersteinDivergence()

        self.Q = torch.tensor(torch.inf, dtype=torch.float, device=self.device)
        self.best_Q = torch.tensor(torch.inf, dtype=torch.float, device=self.device)
        self.final_Q = torch.tensor(torch.inf, dtype=torch.float, device=self.device)

        self.init_eta = torch.tensor(init_eta, dtype=torch.float, device=self.device)
        self.delta = delta
        self.found_feasible_solution = False

        if costs_vector is None:
            self.costs_vector = torch.ones(len(self.explain_indices)).float()
        else:
            self.costs_vector = torch.tensor(costs_vector).float()
        self.costs_vector_reshaped = self.costs_vector.reshape(1, -1)

        self.interval_left = l
        self.interval_right = r
        
        # Initialize logger data
        self.logger_data = []
        
        # Save initial data before optimization (x_true, y_true, y_target)
        if save_results:
            self._save_initial_data(df_factual, y_target)
            
        print("Training Start")
        print("Optimization Start")

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
            
            # Log iteration data
            if save_results:
                self.logger_data.append({
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
                })
            if is_feasible and Q.item() < getattr(self, "best_Q", float("inf")):
                self.best_Q = Q.item()
                self.best_X = self.X.clone().detach()
                self.best_y = self.y.clone().detach()
                self.best_iter = i
                self.found_feasible_solution = True
                print(f"🌟 New best Q found: {self.best_Q:.6f} at iter {i}")

            self.final_X = self.X.clone().detach()
            self.final_y = self.y.clone().detach()
            self.final_Q = Q.item()
            print(f"best_Q = {getattr(self, 'best_Q', None)}, final_Q = {self.final_Q:.6f}")

            if callback:
                callback(self, i)

        print("Optimization End")
        
        # Save results if requested
        if save_results and self.result_saver:
            self.result_saver.save_results(self, df_factual.columns)
            
        # Return recovered (denormalized) data for better readability
        result_X = self.best_X if self.found_feasible_solution else self.final_X
        df_result = pd.DataFrame(result_X.detach().cpu().numpy(), columns=df_factual.columns)
        
        # Apply data recovery if possible
        if hasattr(self.data, 'mean') and hasattr(self.data, 'std'):
            df_result = self._recover_data(df_result, df_factual.columns)
            
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

    def _get_eta_interval_narrowing(self, U_1, U_2, Qu_upper, Qv_upper, l, r, kappa):
        eta = self.__choose_eta_within_interval(U_1 - Qu_upper, U_2 - Qv_upper, l, r)
        if eta > (l + r) / 2:
            l = l + kappa * (r - l)
        else:
            r = r - kappa * (r - l)
        return eta, l, r

    def __choose_eta_within_interval(self, a, b, l, r):
        if (a < 0 and b >= 0) or (a >= 0 and b < 0):
            return l if a < 0 else r
        else:
            if a < 0 and b < 0:
                eta_proportion = b / (a + b)
            else:
                eta_proportion = a / (a + b)
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
        
    def _recovering_types(self, df_to_recover, dtype_dict):
        """Setup nested directory structure for saving results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get strategy name and parameters
        strategy_name = strategy.__class__.__name__.replace('Strategy', '')
        
        # Get strategy-specific parameters with abbreviations
        strategy_params = []
        
        # GeneticStrategy parameters
        if hasattr(strategy, 'crossover_prob'):
            strategy_params.append(f"cp={strategy.crossover_prob}")
        if hasattr(strategy, 'gene_swap_prob'):
            strategy_params.append(f"gsp={strategy.gene_swap_prob}")
        if hasattr(strategy, 'mutation_prob_cat'):
            strategy_params.append(f"mpc={strategy.mutation_prob_cat}")
        if hasattr(strategy, 'mutation_prob_cont'):
            strategy_params.append(f"mpo={strategy.mutation_prob_cont}")
        if hasattr(strategy, 'mutation_noise_scale'):
            strategy_params.append(f"mns={strategy.mutation_noise_scale}")
        
        # SimulatedAnnealingStrategy parameters
        if hasattr(strategy, 'T0'):
            strategy_params.append(f"T0={strategy.T0}")
        if hasattr(strategy, 'T_final'):
            strategy_params.append(f"Tf={strategy.T_final}")
        if hasattr(strategy, 'temp_decay'):
            strategy_params.append(f"td={strategy.temp_decay}")
        
        # ParticleSwarmOptimizationStrategy parameters
        if hasattr(strategy, 'swarm_size'):
            strategy_params.append(f"swarm={strategy.swarm_size}")
        if hasattr(strategy, 'w'):
            strategy_params.append(f"w={strategy.w}")
        if hasattr(strategy, 'c1'):
            strategy_params.append(f"c1={strategy.c1}")
        if hasattr(strategy, 'c2'):
            strategy_params.append(f"c2={strategy.c2}")
        
        # DifferentialEvolutionStrategy parameters
        if hasattr(strategy, 'F'):
            strategy_params.append(f"F={strategy.F}")
        if hasattr(strategy, 'CR'):
            strategy_params.append(f"CR={strategy.CR}")
        
        # CovarianceMatrixAdaptationEvolutionStrategy parameters
        if hasattr(strategy, 'population_size'):
            strategy_params.append(f"pop={strategy.population_size}")
        if hasattr(strategy, 'sigma_decay'):
            strategy_params.append(f"sd={strategy.sigma_decay}")
        
        strategy_param_str = ",".join(strategy_params) if strategy_params else "default"
        
        # DCE parameters with abbreviations
        dce_params = f"np={n_proj},d={delta},u1={U_1},u2={U_2},l={l},r={r},mi={max_iter},tk={top_k}"
        
        # Create nested directory structure
        save_path = os.path.join(
            dataset_name or "unknown_dataset",
            model_name or "unknown_model", 
            f"DCE_{dce_params}",
            f"{strategy_name}_{strategy_param_str}",
            f"seed_{seed}_{timestamp}"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        print(f"Results will be saved to: {save_path}")
        
        return save_path
    
    def _get_strategy_params_dict(self, strategy):
        """Get full strategy parameters dictionary"""
        params = {}
        
        # GeneticStrategy parameters
        if hasattr(strategy, 'crossover_prob'):
            params['crossover_prob'] = strategy.crossover_prob
        if hasattr(strategy, 'gene_swap_prob'):
            params['gene_swap_prob'] = strategy.gene_swap_prob
        if hasattr(strategy, 'mutation_prob_cat'):
            params['mutation_prob_cat'] = strategy.mutation_prob_cat
        if hasattr(strategy, 'mutation_prob_cont'):
            params['mutation_prob_cont'] = strategy.mutation_prob_cont
        if hasattr(strategy, 'mutation_noise_scale'):
            params['mutation_noise_scale'] = strategy.mutation_noise_scale
        
        # SimulatedAnnealingStrategy parameters
        if hasattr(strategy, 'T0'):
            params['T0'] = strategy.T0
        if hasattr(strategy, 'T_final'):
            params['T_final'] = strategy.T_final
        if hasattr(strategy, 'temp_decay'):
            params['temp_decay'] = strategy.temp_decay
        
        # ParticleSwarmOptimizationStrategy parameters
        if hasattr(strategy, 'swarm_size'):
            params['swarm_size'] = strategy.swarm_size
        if hasattr(strategy, 'w'):
            params['w'] = strategy.w
        if hasattr(strategy, 'c1'):
            params['c1'] = strategy.c1
        if hasattr(strategy, 'c2'):
            params['c2'] = strategy.c2
        
        # DifferentialEvolutionStrategy parameters
        if hasattr(strategy, 'F'):
            params['F'] = strategy.F
        if hasattr(strategy, 'CR'):
            params['CR'] = strategy.CR
        
        # CovarianceMatrixAdaptationEvolutionStrategy parameters
        if hasattr(strategy, 'population_size'):
            params['population_size'] = strategy.population_size
        if hasattr(strategy, 'sigma_decay'):
            params['sigma_decay'] = strategy.sigma_decay
        
        return params
        
    def _save_results(self, columns):
        """Save all results to the designated directory"""
        if not hasattr(self, 'save_dir'):
            print("Warning: save_dir not set, skipping save")
            return
            
        try:
            # Create visualization subfolder
            viz_dir = os.path.join(self.save_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            print(f"📁 Visualization directory ensured: {viz_dir}")
            
            # Save ML model
            if hasattr(self.model, 'model'):
                model_path = os.path.join(self.save_dir, "model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(self.model.model, f)
                print(f"Model saved to: {model_path}")
            
            # X_init no longer saved - can be regenerated from seed
            
            # Save best Q and corresponding x,y as CSV (both raw and recovered)
            if hasattr(self, 'best_X') and hasattr(self, 'best_y'):
                # Raw data
                best_x_df = pd.DataFrame(self.best_X.detach().cpu().numpy(), columns=columns)
                best_y_df = pd.DataFrame(self.best_y.detach().cpu().numpy(), columns=['y'])
                best_q_df = pd.DataFrame([{'Q': self.best_Q}])
                
                best_x_df.to_csv(os.path.join(self.save_dir, "best_x_raw.csv"), index=False)
                best_y_df.to_csv(os.path.join(self.save_dir, "best_y.csv"), index=False)
                best_q_df.to_csv(os.path.join(self.save_dir, "best_q.csv"), index=False)
                
                # Recovered data
                if hasattr(self.data, 'mean') and hasattr(self.data, 'std'):
                    best_x_recovered = self._recover_data(best_x_df, columns)
                    best_x_recovered.to_csv(os.path.join(self.save_dir, "best_x.csv"), index=False)
                else:
                    best_x_df.to_csv(os.path.join(self.save_dir, "best_x.csv"), index=False)
                    
                print(f"Best results saved to: {self.save_dir}")
            
            # Save final round Q and corresponding x,y as CSV (both raw and recovered)
            if hasattr(self, 'final_X') and hasattr(self, 'final_y'):
                # Raw data
                final_x_df = pd.DataFrame(self.final_X.detach().cpu().numpy(), columns=columns)
                final_y_df = pd.DataFrame(self.final_y.detach().cpu().numpy(), columns=['y'])
                final_q_df = pd.DataFrame([{'Q': self.final_Q}])
                
                final_x_df.to_csv(os.path.join(self.save_dir, "final_x_raw.csv"), index=False)
                final_y_df.to_csv(os.path.join(self.save_dir, "final_y.csv"), index=False)
                final_q_df.to_csv(os.path.join(self.save_dir, "final_q.csv"), index=False)
                
                # Recovered data
                if hasattr(self.data, 'mean') and hasattr(self.data, 'std'):
                    final_x_recovered = self._recover_data(final_x_df, columns)
                    final_x_recovered.to_csv(os.path.join(self.save_dir, "final_x.csv"), index=False)
                else:
                    final_x_df.to_csv(os.path.join(self.save_dir, "final_x.csv"), index=False)
                    
                print(f"Final results saved to: {self.save_dir}")
            
            # Save logger info as pandas DataFrame and convert to CSV
            if self.logger_data:
                logger_df = pd.DataFrame(self.logger_data)
                logger_df.to_csv(os.path.join(self.save_dir, "optimization_log.csv"), index=False)
                print(f"Optimization log saved to: {self.save_dir}")
            
            # Save experiment metadata with full parameter details
            metadata = {
                'dataset_name': getattr(self, 'dataset_name', 'unknown'),
                'model_name': getattr(self, 'model_name', 'unknown'),
                'strategy': getattr(self, 'strategy_name', 'unknown'),
                'seed': getattr(self, 'seed', None),
                'timestamp': datetime.now().isoformat(),
                'found_feasible_solution': self.found_feasible_solution,
                'best_iter': getattr(self, 'best_iter', None),
                'total_iterations': len(self.logger_data),
                'visualization_dir': viz_dir,
                'dce_parameters': getattr(self, 'full_dce_params', {}),
                'strategy_parameters': getattr(self, 'full_strategy_params', {}),
                'parameter_abbreviations': {
                    'dce_abbrev': {
                        'np': 'n_proj', 'd': 'delta', 'u1': 'U_1', 'u2': 'U_2',
                        'l': 'l', 'r': 'r', 'mi': 'max_iter', 'tk': 'top_k'
                    },
                    'strategy_abbrev': {
                        'cp': 'crossover_prob', 'gsp': 'gene_swap_prob', 'mpc': 'mutation_prob_cat',
                        'mpo': 'mutation_prob_cont', 'mns': 'mutation_noise_scale', 'T0': 'T0',
                        'Tf': 'T_final', 'td': 'temp_decay', 'swarm': 'swarm_size', 'w': 'w',
                        'c1': 'c1', 'c2': 'c2', 'F': 'F', 'CR': 'CR', 'pop': 'population_size',
                        'sd': 'sigma_decay'
                    }
                }
            }
            
            with open(os.path.join(self.save_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"All results successfully saved to: {self.save_dir}")
            print(f"Visualizations will be saved to: {viz_dir}")
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            raise e
            
    def _recover_data(self, df, columns):
        """Recover (denormalize) data using dataset statistics"""
        try:
            if not hasattr(self.data, 'mean') or not hasattr(self.data, 'std'):
                return df
                
            # Create a copy to avoid modifying original
            df_recovered = df.copy()
            
            # Denormalize using mean and std
            if hasattr(self.data, 'explain_columns'):
                explain_cols = self.data.explain_columns
                for col in explain_cols:
                    if col in df_recovered.columns:
                        mean_val = self.data.mean[col] if hasattr(self.data.mean, '__getitem__') else getattr(self.data.mean, col, 0)
                        std_val = self.data.std[col] if hasattr(self.data.std, '__getitem__') else getattr(self.data.std, col, 1)
                        df_recovered[col] = df_recovered[col] * std_val + mean_val
            
            # Recover data types if available
            if hasattr(self.data, 'df'):
                dtype_dict = self.data.df.dtypes.apply(lambda x: x.name).to_dict()
                df_recovered = self._recovering_types(df_recovered, dtype_dict)
            
            return df_recovered
            
        except Exception as e:
            print(f"Warning: Could not recover data: {str(e)}")
            return df
            
    def _recovering_types(self, df_to_recover, dtype_dict):
        """Recover original data types"""
        df_recovered = df_to_recover.copy()
        for k, v in dtype_dict.items():
            if k in df_recovered.columns:
                if v.startswith('int'):  
                    df_recovered[k] = df_recovered[k].round().astype(v)
                else: 
                    df_recovered[k] = df_recovered[k].astype(v)
        return df_recovered
        
    # def compare_factual_counterfactual(self, df_factual, df_counterfactual, show_raw=False):
    #     """Compare factual and counterfactual data with proper denormalization"""
    #     print("🔍 Sample Comparison: Original vs Counterfactual")
        
    #     if show_raw:
    #         print("\n📊 Original (Factual) - Raw:")
    #         print(df_factual.iloc[0].to_frame().T.to_string(index=False))
    #         print("\n🎯 Counterfactual - Raw:")
    #         print(df_counterfactual.iloc[0].to_frame().T.to_string(index=False))
        
    #     # Show denormalized data
    #     try:
    #         df_factual_recovered = self._recover_data(df_factual, df_factual.columns)
    #         df_counterfactual_recovered = self._recover_data(df_counterfactual, df_counterfactual.columns)
            
    #         print("\n📊 Original (Factual):")
    #         print(df_factual_recovered.iloc[0].to_frame().T.to_string(index=False))
            
    #         print("\n🎯 Counterfactual:")
    #         print(df_counterfactual_recovered.iloc[0].to_frame().T.to_string(index=False))
            
    #         # Show key differences
    #         print("\n📋 Key Differences:")
    #         original = df_factual_recovered.iloc[0]
    #         counterfactual = df_counterfactual_recovered.iloc[0]
            
    #         differences = []
    #         for col in df_factual.columns:
    #             if original[col] != counterfactual[col]:
    #                 differences.append(f"   {col}: {original[col]} → {counterfactual[col]}")
            
    #         if differences:
    #             print("\n".join(differences))
    #         else:
    #             print("   No differences found (optimization may need more iterations)")
                
    #     except Exception as e:
    #         print(f"\n❌ Error in data recovery: {str(e)}")
    #         print("\n📊 Showing raw normalized data instead:")
    #         print("\n📊 Original (Factual):")
    #         print(df_factual.iloc[0].to_frame().T.to_string(index=False))
    #         print("\n🎯 Counterfactual:")
    #         print(df_counterfactual.iloc[0].to_frame().T.to_string(index=False))
    
    def _save_initial_data(self, df_factual, y_target):
        """Save initial data before optimization starts"""
        import os
        import pandas as pd
        
        # Create directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save x_true (factual data without noise)
        df_factual.to_csv(os.path.join(self.save_dir, "x_true.csv"), index=False)
        
        # Save y_true (actual predictions for factual data)
        y_true = self.model(self.X_prime).detach().cpu().numpy()
        pd.DataFrame(y_true, columns=['y_true']).to_csv(os.path.join(self.save_dir, "y_true.csv"), index=False)
        
        # Save y_target
        if isinstance(y_target, torch.Tensor):
            y_target_np = y_target.detach().cpu().numpy()
        else:
            y_target_np = y_target
        pd.DataFrame(y_target_np, columns=['y_target']).to_csv(os.path.join(self.save_dir, "y_target.csv"), index=False)
        
        print(f"📁 Initial data saved:")
        print(f"   - x_true.csv: {df_factual.shape}")
        print(f"   - y_true.csv: {y_true.shape}")
        print(f"   - y_target.csv: {y_target_np.shape}")