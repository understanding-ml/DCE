import os
import json
import pickle
import pandas as pd
import torch
from datetime import datetime


class ResultSaver:
    """
    A class to handle saving experiment results, metadata, and visualizations.
    Separated from the main explainer to keep the code clean and modular.
    """
    
    def __init__(self):
        self.save_dir = None
        self.dataset_name = None
        self.model_name = None
        self.strategy_name = None
        self.seed = None
        self.full_dce_params = {}
        self.full_strategy_params = {}
        
    def setup_save_directory(self, dataset_name, model_name, strategy, n_proj, delta, U_1, U_2, l, r, max_iter, top_k, seed, gradient_method="cone_sampling", num_trials=None):
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
        
        # DifferentialEvolutionStrategy parameters
        if hasattr(strategy, 'F'):
            strategy_params.append(f"F={strategy.F}")
        if hasattr(strategy, 'CR'):
            strategy_params.append(f"CR={strategy.CR}")
        
        # Add gradient guidance parameters (if enabled)
        if hasattr(strategy, 'use_gradient_guidance') and strategy.use_gradient_guidance:
            if hasattr(strategy, 'cone_angle'):
                angle_degrees = round(strategy.cone_angle * 180 / 3.14159, 1)  # Convert to degrees
                strategy_params.append(f"angle={angle_degrees}")
            elif hasattr(strategy, 'weight_alpha'):
                strategy_params.append(f"alpha={strategy.weight_alpha}")
            elif hasattr(strategy, 'noise_beta'):
                strategy_params.append(f"beta={strategy.noise_beta}")
        
        # Add new common parameters
        if hasattr(strategy, 'categorical_step'):
            strategy_params.append(f"cs={strategy.categorical_step}")
        if hasattr(strategy, 'continuous_step'):
            strategy_params.append(f"ct={strategy.continuous_step}")
        if hasattr(strategy, 'temperature'):
            strategy_params.append(f"temp={strategy.temperature}")
        
        strategy_param_str = ",".join(strategy_params) if strategy_params else "default"
        
        # DCE parameters with abbreviations - include num_trials if provided, exclude n_proj and delta
        dce_param_parts = [f"u1={U_1}", f"u2={U_2}", f"l={l}", f"r={r}", f"mi={max_iter}", f"tk={top_k}"]
        if num_trials is not None:
            dce_param_parts.append(f"nt={num_trials}")
        dce_params = ",".join(dce_param_parts)
        
        # Create nested directory structure under Results folder
        # save_path = os.path.join(
        #     "Results",
        #     dataset_name or "unknown_dataset",
        #     model_name or "unknown_model", 
        #     f"DCE_{dce_params}",
        #     f"{gradient_method}_{strategy_name}_{strategy_param_str}",
        #     f"seed_{seed}_{timestamp}"
        # )

        save_path = os.path.join(
            "Results",
            dataset_name or "unknown_dataset",
            model_name or "unknown_model", 
            f"DCE_{dce_params}",
            "_".join(filter(None, [gradient_method, strategy_name, strategy_param_str])),
            f"seed_{seed}_{timestamp}"
        )


        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        print(f"Results will be saved to: {save_path}")
        
        # Store parameters for later use
        self.save_dir = save_path
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.strategy_name = strategy_name
        self.seed = seed
        self.full_dce_params = {
            'U_1': U_1, 'U_2': U_2, 'l': l, 'r': r, 'max_iter': max_iter, 'top_k': top_k
        }
        if num_trials is not None:
            self.full_dce_params['num_trials'] = num_trials
        self.full_strategy_params = self._get_strategy_params_dict(strategy)
        
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
        
        # DifferentialEvolutionStrategy parameters
        if hasattr(strategy, 'F'):
            params['F'] = strategy.F
        if hasattr(strategy, 'CR'):
            params['CR'] = strategy.CR
        
        # Add gradient guidance parameters (if enabled) - but exclude use_gradient_guidance itself
        if hasattr(strategy, 'use_gradient_guidance') and strategy.use_gradient_guidance:
            if hasattr(strategy, 'cone_angle'):
                params['cone_angle'] = strategy.cone_angle
                params['angle_degrees'] = round(strategy.cone_angle * 180 / 3.14159, 1)
            elif hasattr(strategy, 'weight_alpha'):
                params['weight_alpha'] = strategy.weight_alpha
            elif hasattr(strategy, 'noise_beta'):
                params['noise_beta'] = strategy.noise_beta
        
        # Add new common parameters
        if hasattr(strategy, 'categorical_step'):
            params['categorical_step'] = strategy.categorical_step
        if hasattr(strategy, 'continuous_step'):
            params['continuous_step'] = strategy.continuous_step
        if hasattr(strategy, 'temperature'):
            params['temperature'] = strategy.temperature
        
        return params
    
    def save_results(self, explainer, columns):
        """Save all results to the designated directory"""
        if not self.save_dir:
            print("Warning: save_dir not set, skipping save")
            return
            
        try:
            # Create visualization subfolder
            viz_dir = os.path.join(self.save_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            print(f"📁 Visualization directory ensured: {viz_dir}")
            
            # Save ML model
            if hasattr(explainer.model, 'model'):
                model_path = os.path.join(self.save_dir, "model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(explainer.model.model, f)
                print(f"Model saved to: {model_path}")
            
            # X_init no longer saved - can be regenerated from seed
            
            # Save best Q and corresponding x,y as CSV (standardized data only)
            if hasattr(explainer, 'best_X') and hasattr(explainer, 'best_y'):
                best_x_df = pd.DataFrame(explainer.best_X.detach().cpu().numpy(), columns=columns)
                best_y_df = pd.DataFrame(explainer.best_y.detach().cpu().numpy(), columns=['y'])
                best_q_df = pd.DataFrame([{'Q': explainer.best_Q}])
                
                # Save standardized data as best_x.csv
                best_x_df.to_csv(os.path.join(self.save_dir, "best_x.csv"), index=False)
                best_y_df.to_csv(os.path.join(self.save_dir, "best_y.csv"), index=False)
                best_q_df.to_csv(os.path.join(self.save_dir, "best_q.csv"), index=False)
                    
                print(f"Best results saved to: {self.save_dir}")
            
            # Save final round Q and corresponding x,y as CSV (standardized data only)
            if hasattr(explainer, 'final_X') and hasattr(explainer, 'final_y'):
                final_x_df = pd.DataFrame(explainer.final_X.detach().cpu().numpy(), columns=columns)
                final_y_df = pd.DataFrame(explainer.final_y.detach().cpu().numpy(), columns=['y'])
                final_q_df = pd.DataFrame([{'Q': explainer.final_Q}])
                
                # Save standardized data as final_x.csv
                final_x_df.to_csv(os.path.join(self.save_dir, "final_x.csv"), index=False)
                final_y_df.to_csv(os.path.join(self.save_dir, "final_y.csv"), index=False)
                final_q_df.to_csv(os.path.join(self.save_dir, "final_q.csv"), index=False)
                    
                print(f"Final results saved to: {self.save_dir}")
            
            # Save logger info as pandas DataFrame and convert to CSV
            if hasattr(explainer, 'logger_data') and explainer.logger_data:
                logger_df = pd.DataFrame(explainer.logger_data)
                logger_df.to_csv(os.path.join(self.save_dir, "optimization_log.csv"), index=False)
                print(f"Optimization log saved to: {self.save_dir}")
            
            # Save experiment metadata with full parameter details
            metadata = {
                'dataset_name': self.dataset_name or 'unknown',
                'model_name': self.model_name or 'unknown',
                'strategy': self.strategy_name or 'unknown',
                'seed': self.seed,
                'timestamp': datetime.now().isoformat(),
                'found_feasible_solution': explainer.found_feasible_solution,
                'best_iter': getattr(explainer, 'best_iter', None),
                'total_iterations': len(explainer.logger_data) if hasattr(explainer, 'logger_data') else 0,
                'visualization_dir': viz_dir,
                'dce_parameters': self.full_dce_params,
                'strategy_parameters': self.full_strategy_params,
                'parameter_abbreviations': {
                    'dce_abbrev': {
                        'u1': 'U_1', 'u2': 'U_2', 'l': 'l', 'r': 'r', 
                        'mi': 'max_iter', 'tk': 'top_k', 'nt': 'num_trials'
                    },
                    'strategy_abbrev': {
                        'cp': 'crossover_prob', 'gsp': 'gene_swap_prob', 'mpc': 'mutation_prob_cat',
                        'mpo': 'mutation_prob_cont', 'mns': 'mutation_noise_scale', 'T0': 'T0',
                        'Tf': 'T_final', 'td': 'temp_decay', 'F': 'F', 'CR': 'CR', 'angle': 'angle_degrees', 'alpha': 'weight_alpha', 'beta': 'noise_beta',
                        'cs': 'categorical_step', 'ct': 'continuous_step', 'temp': 'temperature'
                    }
                }
            }
            
            with open(os.path.join(self.save_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"All results successfully saved to: {self.save_dir}")
            print(f"Visualizations will be saved to: {viz_dir}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
            import traceback
            traceback.print_exc()
    
    def recover_data(self, df, columns, data):
        """Recover (denormalize) data using dataset statistics"""
        try:
            if not hasattr(data, 'mean') or not hasattr(data, 'std'):
                return df
                
            # Create a copy to avoid modifying original
            df_recovered = df.copy()
            
            # Denormalize using mean and std
            if hasattr(data, 'explain_columns'):
                explain_cols = data.explain_columns
                for col in explain_cols:
                    if col in df_recovered.columns:
                        mean_val = data.mean[col] if hasattr(data.mean, '__getitem__') else getattr(data.mean, col, 0)
                        std_val = data.std[col] if hasattr(data.std, '__getitem__') else getattr(data.std, col, 1)
                        df_recovered[col] = df_recovered[col] * std_val + mean_val
            
            # Recover data types if available
            if hasattr(data, 'df'):
                dtype_dict = data.df.dtypes.apply(lambda x: x.name).to_dict()
                df_recovered = self.recovering_types(df_recovered, dtype_dict)
            
            return df_recovered
            
        except Exception as e:
            print(f"Warning: Could not recover data: {str(e)}")
            return df
    
    def recovering_types(self, df_to_recover, dtype_dict):
        """Recover original data types"""
        df_recovered = df_to_recover.copy()
        for k, v in dtype_dict.items():
            if k in df_recovered.columns:
                if v.startswith('int'):  
                    df_recovered[k] = df_recovered[k].round().astype(v)
                else: 
                    df_recovered[k] = df_recovered[k].astype(v)
        return df_recovered
    
    def save_initial_data(self, explainer, df_factual, y_target):
        """Save initial data before optimization starts"""
        if not self.save_dir:
            print("Warning: save_dir not set, skipping initial data save")
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.save_dir, exist_ok=True)
            
            # Save x_true (factual data without noise)
            df_factual.to_csv(os.path.join(self.save_dir, "x_true.csv"), index=False)
            
            # Save y_true (actual predictions for factual data)
            y_true = explainer.model(explainer.X_prime).detach().cpu().numpy()
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
            
        except Exception as e:
            print(f"Error saving initial data: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_visualization_dir(self):
        """Get the visualization directory path"""
        if self.save_dir:
            return os.path.join(self.save_dir, "visualizations")
        return None