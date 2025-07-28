#!/usr/bin/env python3
"""
Comprehensive DCE Experiments Script

This script runs all comparison experiments for DCE (Diverse Counterfactual Explanations):
- 5 strategies: MonteCarlo, Genetic, SimulatedAnnealing, Bayesian, DifferentialEvolution
- 4 gradient methods: original, cone_sampling, weighted_combination, noisy_gradient
- Total: 20 experimental combinations

Based on demo_new.ipynb with complete metrics analysis and result saving.
"""

import pandas as pd
import numpy as np
import torch
import math
import os
import json
import glob
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, entropy
from sklearn.preprocessing import MinMaxScaler

# DCE Framework imports
from explainers.model import Model
from explainers.nondifferentiable import DCENonDifferentiable
from explainers.distances import SlicedWassersteinDivergence, WassersteinDivergence

# Dataset imports
from dataset.cardio import CardioData
from dataset.german_credit import GermanCreditData
from dataset.hotel_booking import HotelBookingData
from dataset.heloc import HelocData
from dataset.compas import CompasData

# Model imports
from models.svm import LinearSVM
from models.mlp import BlackBoxModel

import torch.nn as nn
import torch.optim as optim

class ComprehensiveDCEExperiments:
    """
    Comprehensive DCE experiments runner
    
    Runs all strategy-gradient method combinations and collects metrics
    """
    
    def __init__(self, 
                 dataset_name: str = "german_credit",
                 model_name: str = "SVM", 
                 seed: int = 42,
                 sample_num: int = 50):
        """
        Initialize the experiment runner
        
        Args:
            dataset_name: Dataset to use ('german_credit', 'cardio', 'heloc', 'compas', 'hotel_booking')
            model_name: Model to use ('SVM', 'MLP', 'RandomForest', 'LightGBM', 'XGBoost')
            seed: Random seed for reproducibility
            sample_num: Number of samples to explain
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.seed = seed
        self.sample_num = sample_num
        
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Common DCE parameters (exactly from demo_new.ipynb)
        self.dce_params = {
            'X_init': True,
            'n_proj': 5,
            'delta': 0.1,
            'U_1': 0.5,
            'U_2': 0.3,
            'l': 0.2,
            'r': 1.0,
            'max_iter': 50,
            'top_k': 1,
            'callback': 'full',  # Set to 'off' for batch experiments (no display)
            'save_results': True
        }
        
        # Strategy configurations (exclude PSO and CMA-ES as requested)
        self.strategy_configs = {
            'MonteCarlo': {},
            'Genetic': {
                'crossover_prob': 0.8,
                'gene_swap_prob': 0.5,
                'mutation_prob_cat': 0.3,
                'mutation_prob_cont': 0.8,
                'mutation_noise_scale': 0.1
            },
            'SimulatedAnnealing': {
                'T0': 1.5,
                'T_final': 0.01
            },
            'Bayesian': {},
            'DifferentialEvolution': {
                'F': 0.5,
                'CR': 0.9
            }
        }
        
        # Gradient method configurations
        self.gradient_configs = {
            'original': {},  # Pure random sampling
            'cone_sampling': {
                'use_gradient_guidance': True,
                'cone_angle': math.pi/4  # 45 degree cone
            },
            'weighted_combination': {
                'use_gradient_guidance': True,
                'weight_alpha': 0.7  # 70% gradient, 30% random
            },
            'noisy_gradient': {
                'use_gradient_guidance': True,
                'noise_beta': 0.1  # β=0.1 noise scaling
            }
        }
        
        # Results storage
        self.all_results = {}
        self.all_metrics = {}
        
        print(f"🚀 Comprehensive DCE Experiments Initialized")
        print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Dataset: {dataset_name}")
        print(f"🤖 Model: {model_name}")
        print(f"🎲 Seed: {seed}")
        print(f"📋 Sample size: {sample_num}")
        print(f"🔧 DCE parameters: {self.dce_params}")
        print(f"📈 Strategy count: {len(self.strategy_configs)}")
        print(f"🧭 Gradient methods: {len(self.gradient_configs)}")
        print(f"🎯 Total experiments: {len(self.strategy_configs) * len(self.gradient_configs)}")
    
    def load_dataset(self):
        """Load the specified dataset"""
        print(f"\n📊 Loading dataset: {self.dataset_name}")
        
        if self.dataset_name == "german_credit":
            self.data = GermanCreditData(seed=self.seed)
        elif self.dataset_name == "cardio":
            self.data = CardioData(seed=self.seed)
        elif self.dataset_name == "heloc":
            self.data = HelocData(seed=self.seed)
        elif self.dataset_name == "compas":
            self.data = CompasData(seed=self.seed)
        elif self.dataset_name == "hotel_booking":
            self.data = HotelBookingData(seed=self.seed)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        # Get explanation data
        self.df_explain = self.data.get_df_explain(sample_num=self.sample_num)
        self.X_train, self.X_test, self.y_train, self.y_test = self.data.get_train_test()
        
        print(f"✅ Dataset loaded: {len(self.df_explain)} samples for explanation")
        print(f"📈 Training data: {len(self.X_train)} samples")
        print(f"📉 Test data: {len(self.X_test)} samples")
    
    def train_model(self):
        """Train the specified model"""
        print(f"\n🤖 Training model: {self.model_name}")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(self.X_train.values)
        y_train_tensor = torch.FloatTensor(self.y_train.values).view(-1, 1)
        X_test_tensor = torch.FloatTensor(self.X_test.values)
        y_test_tensor = torch.FloatTensor(self.y_test.values).view(-1, 1)
        
        if self.model_name == "SVM":
            model_raw = LinearSVM(input_dim=self.X_train.shape[1])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model_raw.parameters(), lr=0.01)
            backend = "pytorch"
            
        elif self.model_name == "MLP":
            model_raw = BlackBoxModel(input_dim=self.X_train.shape[1])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model_raw.parameters(), lr=0.01)
            backend = "pytorch"
            
        elif self.model_name == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier
            model_raw = RandomForestClassifier(n_estimators=100, random_state=self.seed)
            model_raw.fit(self.X_train, self.y_train)
            self.model = Model(model=model_raw, backend="sklearn", data=self.data)
            
            # Calculate accuracy
            accuracy = model_raw.score(self.X_test, self.y_test)
            print(f"✅ Model trained with accuracy: {accuracy:.4f}")
            return
            
        elif self.model_name == "LightGBM":
            from lightgbm import LGBMClassifier
            model_raw = LGBMClassifier(n_estimators=100, random_state=self.seed)
            model_raw.fit(self.X_train, self.y_train)
            self.model = Model(model=model_raw, backend="lightgbm", data=self.data)
            
            # Calculate accuracy
            accuracy = model_raw.score(self.X_test, self.y_test)
            print(f"✅ Model trained with accuracy: {accuracy:.4f}")
            return
            
        elif self.model_name == "XGBoost":
            from xgboost import XGBClassifier
            model_raw = XGBClassifier(n_estimators=100, random_state=self.seed, eval_metric="logloss")
            model_raw.fit(self.X_train, self.y_train)
            self.model = Model(model=model_raw, backend="xgboost", data=self.data)
            
            # Calculate accuracy
            accuracy = model_raw.score(self.X_test, self.y_test)
            print(f"✅ Model trained with accuracy: {accuracy:.4f}")
            return
            
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # For PyTorch models (SVM, MLP)
        num_epochs = 300
        for epoch in range(num_epochs):
            outputs = model_raw(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Evaluate on test set
        model_raw.eval()
        with torch.no_grad():
            test_outputs = model_raw(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
            
            # Convert outputs to binary using 0.5 as threshold
            y_pred_tensor = (test_outputs > 0.5).float()
            correct_predictions = (y_pred_tensor == y_test_tensor).float().sum()
            accuracy = correct_predictions / y_test_tensor.shape[0]
        
        self.model = Model(model=model_raw, backend=backend, data=self.data)
        print(f"✅ Model trained with accuracy: {accuracy.item():.4f}")
    
    def import_strategy_classes(self, gradient_method: str):
        """Dynamically import strategy classes based on gradient method"""
        if gradient_method == "original":
            from explainers.original.monte_carlo import MonteCarloStrategy
            from explainers.original.genetic import GeneticStrategy
            from explainers.original.simulated_annealing import SimulatedAnnealingStrategy
            from explainers.original.bayesian import BayesianStrategy
            from explainers.original.differential_evolution import DifferentialEvolutionStrategy
        elif gradient_method == "cone_sampling":
            from explainers.cone_sampling.monte_carlo import MonteCarloStrategy
            from explainers.cone_sampling.genetic import GeneticStrategy
            from explainers.cone_sampling.simulated_annealing import SimulatedAnnealingStrategy
            from explainers.cone_sampling.bayesian import BayesianStrategy
            from explainers.cone_sampling.differential_evolution import DifferentialEvolutionStrategy
        elif gradient_method == "weighted_combination":
            from explainers.weighted_combination.monte_carlo import MonteCarloStrategy
            from explainers.weighted_combination.genetic import GeneticStrategy
            from explainers.weighted_combination.simulated_annealing import SimulatedAnnealingStrategy
            from explainers.weighted_combination.bayesian import BayesianStrategy
            from explainers.weighted_combination.differential_evolution import DifferentialEvolutionStrategy
        elif gradient_method == "noisy_gradient":
            from explainers.noisy_gradient.monte_carlo import MonteCarloStrategy
            from explainers.noisy_gradient.genetic import GeneticStrategy
            from explainers.noisy_gradient.simulated_annealing import SimulatedAnnealingStrategy
            from explainers.noisy_gradient.bayesian import BayesianStrategy
            from explainers.noisy_gradient.differential_evolution import DifferentialEvolutionStrategy
        else:
            raise ValueError(f"Unknown gradient method: {gradient_method}")
        
        return {
            'MonteCarlo': MonteCarloStrategy,
            'Genetic': GeneticStrategy,
            'SimulatedAnnealing': SimulatedAnnealingStrategy,
            'Bayesian': BayesianStrategy,
            'DifferentialEvolution': DifferentialEvolutionStrategy
        }
    
    def create_strategy(self, strategy_name: str, gradient_method: str, explainer):
        """Create a strategy instance with appropriate parameters"""
        strategy_classes = self.import_strategy_classes(gradient_method)
        strategy_class = strategy_classes[strategy_name]
        
        # Get base parameters for this strategy
        base_params = self.strategy_configs[strategy_name].copy()
        base_params['random_state'] = self.seed
        
        # Add gradient-specific parameters
        gradient_params = self.gradient_configs[gradient_method].copy()
        base_params.update(gradient_params)
        
        # Create strategy instance
        strategy = strategy_class(explainer, **base_params)
        
        return strategy
    
    def run_single_experiment(self, strategy_name: str, gradient_method: str) -> Dict[str, Any]:
        """Run a single DCE experiment with demo_new.ipynb style metrics calculation and saving"""
        exp_name = f"{strategy_name}_{gradient_method}"
        print(f"\n🔬 Running experiment: {exp_name}")
        
        # Create explainer
        explainer = DCENonDifferentiable(self.model, self.data)
        
        # Create strategy
        strategy = self.create_strategy(strategy_name, gradient_method, explainer)
        
        # Generate strategy name for saving
        if hasattr(strategy, 'cone_angle'):
            strategy_suffix = f"cone_{math.degrees(strategy.cone_angle):.0f}deg"
        elif hasattr(strategy, 'weight_alpha'):
            strategy_suffix = f"weighted_{strategy.weight_alpha}"
        elif hasattr(strategy, 'noise_beta'):
            strategy_suffix = f"noisy_{strategy.noise_beta}"
        else:
            strategy_suffix = "original" if gradient_method == "original" else "default"
        
        strategy_display_name = f"{strategy_name}_{strategy_suffix}"
        
        print(f"🎯 Strategy: {strategy_display_name}")
        print(f"🧭 Gradient method: {gradient_method}")
        
        # Print gradient method info like demo_new.ipynb
        if gradient_method == "original":
            print("🎲 Using pure random sampling (original method, no gradient information)")
        elif hasattr(strategy, 'use_gradient_guidance') and strategy.use_gradient_guidance:
            if hasattr(strategy, 'cone_angle'):
                print(f"🧭 Cone sampling gradient guidance enabled with cone angle: {strategy.cone_angle:.3f} rad ({math.degrees(strategy.cone_angle):.1f}°)")
            elif hasattr(strategy, 'weight_alpha'):
                print(f"🧭 Weighted combination gradient guidance enabled with alpha={strategy.weight_alpha} ({strategy.weight_alpha*100:.1f}% gradient + {(1-strategy.weight_alpha)*100:.1f}% random)")
            elif hasattr(strategy, 'noise_beta'):
                print(f"🧭 Noisy gradient guidance enabled with β={strategy.noise_beta} (final_dir = guide_dir + β*noise)")
        else:
            print("🎲 Using traditional random sampling")
        
        # Run DCE explanation
        start_time = time.time()
        
        try:
            df_cf = explainer.explain(
                df_factual=self.df_explain,
                strategy=strategy,
                dataset_name=self.dataset_name,
                model_name=self.model_name,
                seed=self.seed,
                gradient_method=gradient_method,
                **self.dce_params
            )
            
            elapsed_time = time.time() - start_time
            
            print(f"✅ Experiment completed in {elapsed_time:.2f} seconds")
            
            # Calculate and save metrics exactly like demo_new.ipynb
            metrics = self.calculate_and_save_metrics_like_demo(explainer)
            
            return {
                'strategy_name': strategy_name,
                'gradient_method': gradient_method,
                'strategy_display_name': strategy_display_name,
                'experiment_name': exp_name,
                'elapsed_time': elapsed_time,
                'explainer': explainer,
                'df_cf': df_cf,
                'metrics': metrics,  # Add metrics to result
                'success': True,
                'error': None
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ Experiment failed after {elapsed_time:.2f} seconds: {str(e)}")
            
            return {
                'strategy_name': strategy_name,
                'gradient_method': gradient_method,
                'strategy_display_name': strategy_display_name,
                'experiment_name': exp_name,
                'elapsed_time': elapsed_time,
                'explainer': None,
                'df_cf': None,
                'metrics': {},  # Empty metrics on failure
                'success': False,
                'error': str(e)
            }
    
    def calculate_and_save_metrics_like_demo(self, explainer):
        """
        Calculate and save metrics exactly like demo_new.ipynb
        This replicates the entire metrics calculation and saving process from the demo
        Returns: Dictionary of calculated metrics
        """
        print("\n📊 Results Analysis and Metrics Calculation")
        
        # Universal model prediction function (from demo_new.ipynb)
        def get_model_predictions(model_obj, data_array):
            """
            Universal function to get predictions from any model type
            Returns probabilities for class 1 (positive class)
            """
            if self.model.backend == "pytorch":
                # PyTorch models
                import torch
                with torch.no_grad():
                    data_tensor = torch.FloatTensor(data_array)
                    predictions = model_obj.model(data_tensor).cpu().numpy()
                    return predictions.flatten()
            elif self.model.backend in ["sklearn", "lightgbm", "xgboost"]:
                # Sklearn-like models (RandomForest, LightGBM, XGBoost)
                try:
                    # Try predict_proba first (for probability estimates)
                    prob_predictions = model_obj.model.predict_proba(data_array)
                    return prob_predictions[:, 1]  # Return probability for class 1
                except AttributeError:
                    # Fall back to predict if predict_proba is not available
                    predictions = model_obj.model.predict(data_array)
                    return predictions
            else:
                raise ValueError(f"Unknown model backend: {self.model.backend}")
        
        # Get the results directory from the explainer
        if hasattr(explainer, 'save_dir') and explainer.save_dir:
            results_directory = explainer.save_dir
            print(f"📁 Using results directory: {results_directory}")
        else:
            # Fallback: search for the most recent results directory
            pattern = f"{self.dataset_name}/*/*/seed_{self.seed}_*"
            result_dirs = glob.glob(pattern)
            
            if result_dirs:
                results_directory = max(result_dirs, key=os.path.getctime)
                print(f"📁 Found results directory: {results_directory}")
            else:
                results_directory = None
                print("❌ No results directory found")
                return
        
        # Global variable to store all metrics (like demo_new.ipynb)
        all_metrics = {}
        
        # Process metrics if results directory is available
        if results_directory and os.path.exists(results_directory):
            latest_dir = results_directory
            
            try:
                # Load required data files
                best_x_path = os.path.join(latest_dir, "best_x.csv")
                best_y_path = os.path.join(latest_dir, "best_y.csv")
                y_target_path = os.path.join(latest_dir, "y_target.csv")
                x_true_path = os.path.join(latest_dir, "x_true.csv")  # Load factual X from saved file
                
                if all(os.path.exists(path) for path in [best_x_path, best_y_path, y_target_path, x_true_path]):
                    # Load standardized counterfactual data
                    best_x_standardized = pd.read_csv(best_x_path)  # Standardized counterfactual X
                    best_y_df = pd.read_csv(best_y_path)  # Counterfactual Y (may need recalculation)
                    y_target_df = pd.read_csv(y_target_path)  # Target Y
                    x_true_standardized = pd.read_csv(x_true_path)  # Standardized factual X (from saved file)
                    
                    # ⚠️ CRITICAL FIX: Get original data statistics from the data object
                    # data.mean and data.std are the original statistics before standardization
                    try:
                        mean_vals = self.data.mean  # Original mean before standardization  
                        std_vals = self.data.std    # Original std before standardization
                    except AttributeError:
                        # Fallback: calculate from training data if attributes don't exist
                        print("⚠️ Data object missing mean/std attributes, calculating from training data...")
                        mean_vals = self.X_train.mean()
                        std_vals = self.X_train.std()
                    
                    print("🔄 Processing data for metrics calculation...")
                    
                    # === 1. Prepare Counterfactual Data (Inverse Standardization + Type Recovery) ===
                    # Inverse standardization: x_original = x_standardized * std + mean
                    counterfactual_X_original = best_x_standardized * std_vals + mean_vals
                    
                    # Get original data types for type recovery
                    # Need to get original data before standardization for dtype
                    X_original_sample = self.data.X  # This is the original data before split and standardization
                    dtype_dict = X_original_sample.dtypes.apply(lambda x: x.name).to_dict()
                    
                    # Recover data types (important for integer features)
                    for k, v in dtype_dict.items():
                        if k in counterfactual_X_original.columns:
                            if v[:3] == 'int':
                                counterfactual_X_original[k] = counterfactual_X_original[k].round().astype(v)
                            else:
                                counterfactual_X_original[k] = counterfactual_X_original[k].astype(v)
                    
                    # === 2. Prepare Factual Data (from saved x_true.csv) ===
                    # Get original factual data (same inverse standardization)
                    factual_X_standardized = x_true_standardized  # Use saved standardized factual data
                    factual_X_original = factual_X_standardized * std_vals + mean_vals
                    
                    # Recover data types for factual data
                    for k, v in dtype_dict.items():
                        if k in factual_X_original.columns:
                            if v[:3] == 'int':
                                factual_X_original[k] = factual_X_original[k].round().astype(v)
                            else:
                                factual_X_original[k] = factual_X_original[k].astype(v)
                    
                    # === 3. Recalculate Counterfactual Y with Proper Data ===
                    # We need to re-standardize the original data for model prediction
                    # because the model was trained on standardized data
                    counterfactual_X_for_model = (counterfactual_X_original - mean_vals) / std_vals
                    
                    # Use unified prediction function instead of PyTorch-specific code
                    counterfactual_y_prob = best_y_df
                    counterfactual_y = np.array((counterfactual_y_prob > 0.5).astype(int))
                    y_target = y_target_df['y_target'].values
                    
                    # === 4. Prepare data for different metric calculations ===
                    # For distance-based metrics: use standardized data (same scale)
                    counterfactual_X_df_standardized = best_x_standardized
                    factual_X_df_standardized = factual_X_standardized
                    counterfactual_X_np_standardized = counterfactual_X_df_standardized.values
                    factual_X_np_standardized = factual_X_df_standardized.values
                    
                    # For interpretable metrics: use original data
                    counterfactual_X_df_original = counterfactual_X_original
                    factual_X_df_original = factual_X_original
                    counterfactual_X_np_original = counterfactual_X_df_original.values
                    factual_X_np_original = factual_X_df_original.values
                    
                    print("✅ Data processed for metrics calculation")
                    print(f"   Factual X (original) range: [{factual_X_df_original.min().min():.2f}, {factual_X_df_original.max().max():.2f}]")
                    print(f"   Counterfactual X (original) range: [{counterfactual_X_df_original.min().min():.2f}, {counterfactual_X_df_original.max().max():.2f}]")
                    print(f"   Factual X (standardized) range: [{factual_X_df_standardized.min().min():.2f}, {factual_X_df_standardized.max().max():.2f}]")
                    print(f"   Counterfactual X (standardized) range: [{counterfactual_X_df_standardized.min().min():.2f}, {counterfactual_X_df_standardized.max().max():.2f}]")
                    print(f"   Counterfactual Y range: [{counterfactual_y.min():.4f}, {counterfactual_y.max():.4f}]")
                    print(f"   Loaded factual X from: x_true.csv")
                    print(f"   Using model backend: {self.model.backend}")
                    
                    # === METRICS CALCULATION (exactly like demo_new.ipynb) ===
                    
                    # Metric 1: Coverage Rate - using simple sum-based calculation like baseline_experiments_heloc.ipynb
                    try:
                        # ⚠️ CRITICAL FIX: Ensure shape alignment to prevent broadcasting errors
                        # Flatten for shape alignment
                        y_cf = counterfactual_y.flatten() if hasattr(counterfactual_y, 'flatten') else counterfactual_y
                        y_tgt = y_target.flatten() if len(y_target.shape) > 1 else y_target
                        
                        # Fixed coverage rate calculation with shape-aligned arrays
                        match_mask = ((y_cf > 0.5) & (y_tgt > 0.5)) | ((y_cf <= 0.5) & (y_tgt <= 0.5))
                        coverage_rate = np.mean(match_mask)
                        
                        all_metrics['Coverage Rate'] = coverage_rate
                        
                        print(f"📊 Coverage Rate (fixed shape alignment): {coverage_rate:.3f} ({coverage_rate*100:.2f}%)")
                        print(f"   Match count: {np.sum(match_mask)}/{len(match_mask)}")
                        print(f"   Positive predictions: {y_cf.sum()}/{len(y_cf)}")
                        print(f"   counterfactual_y shape: {y_cf.shape}, range: [{y_cf.min():.4f}, {y_cf.max():.4f}]")
                        print(f"   y_target shape: {y_tgt.shape}, range: [{y_tgt.min():.4f}, {y_tgt.max():.4f}]")
                        
                    except Exception as e:
                        print(f"❌ Error calculating Coverage Rate: {e}")
                        all_metrics['Coverage Rate'] = float('nan')
                    
                    # Sigma for MMD
                    sigma = 1.0
                    
                    # Metric 2: MMD (Maximum Mean Discrepancy) Calculation
                    try:
                        # MMD calculation using standardized data (same scale)
                        def gaussian_kernel(x, y, sigma=sigma):
                            return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))
                        
                        def mmd(X_s, X_t, kernel=gaussian_kernel):
                            n = X_s.shape[0]
                            m = X_t.shape[0]
                            
                            # Calculate kernel values
                            XX = np.sum([kernel(X_s[i], X_s[j]) for i in range(n) for j in range(n)])
                            YY = np.sum([kernel(X_t[i], X_t[j]) for i in range(m) for j in range(m)])
                            XY = np.sum([kernel(X_s[i], X_t[j]) for i in range(n) for j in range(m)])
                            
                            return XX / (n ** 2) + YY / (m ** 2) - 2 * XY / (n * m)
                        
                        mmd_value = mmd(counterfactual_X_np_standardized, factual_X_np_standardized)
                        all_metrics['MMD'] = mmd_value
                        
                        print(f"📊 MMD: {mmd_value:.3f}")
                        print(f"   Method: Gaussian kernel with sigma=1.0 on standardized data")
                        print(f"   Comparing counterfactual_X vs factual_X (both standardized)")
                        print(f"   Data shapes: {counterfactual_X_np_standardized.shape} vs {factual_X_np_standardized.shape}")
                        print(f"   Standardized data range: [{counterfactual_X_np_standardized.min():.2f}, {counterfactual_X_np_standardized.max():.2f}]")
                        
                    except Exception as e:
                        print(f"❌ Error calculating MMD: {e}")
                        all_metrics['MMD'] = float('nan')
                    
                    # Metric 3: OT Distance (Optimal Transport) Calculation
                    try:
                        # OT Distance calculation using standardized data (OT is sensitive to scale)
                        def compute_distance(X_s, X_t):
                            if type(X_s) == pd.DataFrame:
                                X_s = torch.FloatTensor(X_s.values)
                            if type(X_t) == pd.DataFrame:
                                X_t = torch.FloatTensor(X_t.values)
                            if type(X_s) == np.ndarray:
                                X_s = torch.FloatTensor(X_s)
                            if type(X_t) == np.ndarray:
                                X_t = torch.FloatTensor(X_t)
                            
                            if X_s.ndim == 1:
                                wd = WassersteinDivergence()
                                distance, _ = wd.distance(X_s, X_t, delta=0.1)
                            else:
                                swd = SlicedWassersteinDivergence(dim=X_s.shape[1], n_proj=5, random_state=self.seed)
                                distance, _ = swd.distance(X_s, X_t, delta=0.1)
                            return distance.item()
                        
                        # Use standardized data for OT distance (scale-sensitive metric)
                        ot_distance = compute_distance(counterfactual_X_df_standardized, factual_X_df_standardized)
                        all_metrics['OT Distance'] = ot_distance
                        
                        print(f"📊 OT Distance: {ot_distance:.4f}")
                        print(f"   Method: SlicedWassersteinDivergence with delta=0.1, n_proj=5")
                        print(f"   Using standardized data (OT is scale-sensitive)")
                        print(f"   Comparing counterfactual_X vs factual_X (both standardized)")
                        print(f"   Input dimensions: {counterfactual_X_df_standardized.shape[1]} features")
                        print(f"   Standardized data range: [{counterfactual_X_df_standardized.min().min():.2f}, {counterfactual_X_df_standardized.max().max():.2f}]")
                        
                        # Optional: Also calculate OT distance on original data with normalization
                        # Normalize original data to [0,1] range for OT calculation
                        scaler = MinMaxScaler()
                        
                        # Fit scaler on combined data
                        combined_data = pd.concat([factual_X_df_original, counterfactual_X_df_original], axis=0)
                        scaler.fit(combined_data)
                        
                        factual_normalized = scaler.transform(factual_X_df_original)
                        counterfactual_normalized = scaler.transform(counterfactual_X_df_original)
                        
                        ot_distance_normalized = compute_distance(counterfactual_normalized, factual_normalized)
                        all_metrics['OT Distance (normalized original)'] = ot_distance_normalized
                        print(f"   OT Distance on normalized original data: {ot_distance_normalized:.4f} (for comparison)")
                        
                    except Exception as e:
                        print(f"❌ Error calculating OT Distance: {e}")
                        all_metrics['OT Distance'] = float('nan')
                        import traceback
                        traceback.print_exc()
                    
                    # Metric 4: Percentile Difference Calculation
                    try:
                        # Percentile Difference calculation using standardized data (consistent scale)
                        def compute_percentile_difference(counterfactual_X, factual_X, percentiles):
                            columns = counterfactual_X.columns  # Use DataFrame columns like in baseline
                            diff_list = []
                            for percentile in percentiles:
                                for column in columns:
                                    perc_cf = np.percentile(counterfactual_X[column].values, percentile)
                                    perc_f = np.percentile(factual_X[column].values, percentile)
                                    
                                    # Use baseline method (no zero-division check)
                                    if abs(perc_f) > 1e-8:  # Avoid division by very small numbers
                                        diff_list.append(abs(perc_cf - perc_f)/abs(perc_f) * 100)
                                    else:
                                        diff_list.append(0.0 if abs(perc_cf) < 1e-8 else 100.0)
                            
                            return np.mean(diff_list)
                        
                        # Calculate for different percentile ranges as in baseline
                        percentile_ranges = [
                            (0, 15, "0-15%"),
                            (15, 30, "15-30%"),
                            (30, 70, "30-70%"),
                            (70, 85, "70-85%"),
                            (85, 100, "85-100%")
                        ]
                        
                        print("📊 Percentile Differences (using standardized data):")
                        for low, high, label in percentile_ranges:
                            percentiles = np.arange(low, high, 0.1 if high - low <= 15 else 1)
                            try:
                                diff_pct = compute_percentile_difference(counterfactual_X_df_standardized, factual_X_df_standardized, percentiles)
                                all_metrics[f"Percentile Difference {label}"] = diff_pct
                                print(f"   {label}: {diff_pct:.3f}%")
                            except Exception as e:
                                print(f"   {label}: Error - {e}")
                                all_metrics[f"Percentile Difference {label}"] = float('nan')
                        
                        print(f"   Features compared: {list(counterfactual_X_df_standardized.columns)}")
                        print(f"   Standardized data range: [{counterfactual_X_df_standardized.min().min():.2f}, {counterfactual_X_df_standardized.max().max():.2f}]")
                        
                    except Exception as e:
                        print(f"❌ Error calculating Percentile Differences: {e}")
                        for low, high, label in [(0, 15, "0-15%"), (15, 30, "15-30%"), (30, 70, "30-70%"), (70, 85, "70-85%"), (85, 100, "85-100%")]:
                            all_metrics[f"Percentile Difference {label}"] = float('nan')
                    
                    # Metric 5: AReS Cost Calculation
                    try:
                        # AReS Cost calculation using original data (more interpretable costs)
                        def compute_cost(delta, costs_vector):
                            """
                            Compute AReS cost using the baseline method
                            Args:
                                delta: Feature change matrix (counterfactual - factual) - shape (n_instances, n_features)
                                costs_vector: Feature cost vector - shape (n_features,)
                            Returns:
                                L2 norm of weighted feature change matrix (single scalar value)
                            """
                            return np.linalg.norm(delta @ np.diag(costs_vector))
                        
                        # Create costs vector based on feature characteristics
                        n_features = counterfactual_X_np_original.shape[1]
                        costs_vector = np.ones(n_features)
                        
                        # Get feature names for better cost assignment
                        feature_names = list(counterfactual_X_df_original.columns)
                        
                        # Assign costs based on feature type and range (similar to Globe CE approach)
                        print("📊 AReS Cost - Feature Cost Assignment:")
                        for i, feature_name in enumerate(feature_names):
                            feature_values = factual_X_np_original[:, i]
                            
                            # Calculate feature range for normalization
                            feature_range = np.max(feature_values) - np.min(feature_values)
                            
                            # Check if feature appears to be categorical (small number of unique values)
                            unique_values = np.unique(feature_values)
                            n_unique = len(unique_values)
                            
                            if n_unique <= 10 and np.all(unique_values == unique_values.astype(int)):
                                # Categorical feature - use moderate fixed cost
                                costs_vector[i] = 0.5
                                print(f"   {feature_name}: Categorical (cost=0.5, unique values={n_unique})")
                            else:
                                # Continuous feature - use inverse of range for normalization
                                if feature_range > 0:
                                    costs_vector[i] = 1.0 / feature_range
                                else:
                                    costs_vector[i] = 1.0
                                print(f"   {feature_name}: Continuous (cost={costs_vector[i]:.3f}, range={feature_range:.2f})")
                        
                        print(f"\n📊 AReS Cost Calculation:")
                        print(f"   Using original data for interpretable costs")
                        print(f"   Cost vector: {costs_vector}")
                        
                        # Calculate delta matrix (ALL instances at once - this is the key difference)
                        deltas = counterfactual_X_np_original - factual_X_np_original
                        print(f"   Delta matrix shape: {deltas.shape}")
                        
                        # Compute AReS cost using baseline method (L2 norm of entire weighted matrix)
                        ares_cost = compute_cost(deltas, costs_vector)
                        
                        all_metrics['AReS Cost'] = ares_cost
                        
                        print(f"📊 AReS Cost Results:")
                        print(f"   Cost: {ares_cost:.3f}")
                        print(f"   Method: L2 norm of weighted feature change matrix")
                        print(f"   Matrix dimensions: {deltas.shape[0]} instances × {deltas.shape[1]} features")
                        
                    except Exception as e:
                        print(f"❌ Error calculating AReS Cost: {e}")
                        import traceback
                        traceback.print_exc()
                        all_metrics['AReS Cost'] = float('nan')
                    
                    # Visualization: Q value optimization progress (like demo_new.ipynb)
                    try:
                        # Load optimization log
                        optimization_log_path = os.path.join(results_directory, "optimization_log.csv")
                        
                        if os.path.exists(optimization_log_path):
                            # Read the optimization log
                            log_df = pd.read_csv(optimization_log_path)
                            
                            print("📊 Creating Q optimization progress visualization...")
                            
                            # Create the plot
                            plt.figure(figsize=(12, 8))
                            
                            # Plot Q values over iterations
                            plt.plot(log_df['iteration'], log_df['Q'], 'b-', linewidth=2, label='Q value', alpha=0.7)
                            
                            # Find and highlight the best Q value
                            best_q_idx = log_df['Q'].idxmin()
                            best_q_value = log_df.loc[best_q_idx, 'Q']
                            best_iteration = log_df.loc[best_q_idx, 'iteration']
                            
                            plt.scatter(best_iteration, best_q_value, color='red', s=100, zorder=5, 
                                       label=f'Best Q: {best_q_value:.4f} (iter {best_iteration})')
                            
                            # Highlight feasible solutions if available
                            if 'is_feasible' in log_df.columns:
                                feasible_points = log_df[log_df['is_feasible'] == True]
                                if not feasible_points.empty:
                                    plt.scatter(feasible_points['iteration'], feasible_points['Q'], 
                                              color='green', s=50, alpha=0.6, label='Feasible solutions')
                            
                            # Formatting
                            plt.xlabel('Iteration', fontsize=12)
                            plt.ylabel('Q Value', fontsize=12)
                            plt.title(f'DCE Optimization Progress\nDataset: {self.dataset_name}, Model: {self.model_name}', 
                                     fontsize=14, pad=20)
                            plt.grid(True, alpha=0.3)
                            plt.legend()
                            
                            # Add annotation for best point
                            plt.annotate(f'Best Q: {best_q_value:.4f}', 
                                        xy=(best_iteration, best_q_value), 
                                        xytext=(best_iteration + len(log_df) * 0.1, best_q_value + (log_df['Q'].max() - log_df['Q'].min()) * 0.1),
                                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                                        fontsize=10, color='red')
                            
                            # Add text box with key statistics
                            stats_text = f"""Statistics:
Total iterations: {len(log_df)}
Best Q: {best_q_value:.4f}
Best iteration: {best_iteration}
Final Q: {log_df['Q'].iloc[-1]:.4f}
Q range: [{log_df['Q'].min():.4f}, {log_df['Q'].max():.4f}]"""
                            
                            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                                    verticalalignment='top', fontsize=9, 
                                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                            
                            plt.tight_layout()
                            
                            # Prepare save paths
                            max_iter = len(log_df)
                            plot_filename = f"best_Q_{max_iter}.png"
                            
                            # Save to results directory
                            results_plot_path = os.path.join(results_directory, plot_filename)
                            
                            try:
                                plt.savefig(results_plot_path, dpi=300, bbox_inches='tight')
                                print(f"✅ Q optimization plot saved: {results_plot_path}")
                                
                                # Verify file was created
                                if os.path.exists(results_plot_path):
                                    file_size = os.path.getsize(results_plot_path)
                                    print(f"📄 File verified: {plot_filename} ({file_size} bytes)")
                                
                            except Exception as save_error:
                                print(f"❌ Error saving Q optimization plot: {save_error}")
                            
                            # Close the plot to free memory (no plt.show() to avoid display)
                            plt.close()
                            
                        else:
                            print(f"❌ Optimization log not found: {optimization_log_path}")
                            
                    except Exception as e:
                        print(f"❌ Error creating Q optimization plot: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Final Metrics Summary (like demo_new.ipynb)
                    print("\n📈 Final Metrics Summary")
                    print("=" * 60)
                    
                    if all_metrics:
                        # Core metrics
                        core_metrics = ['Coverage Rate', 'MMD', 'OT Distance', 'AReS Cost']
                        
                        print("🎯 Core Metrics:")
                        for metric in core_metrics:
                            if metric in all_metrics:
                                value = all_metrics[metric]
                                if isinstance(value, float) and not np.isnan(value):
                                    print(f"   {metric}: {value:.3f}")
                                else:
                                    print(f"   {metric}: {value}")
                        
                        # Percentile differences
                        percentile_metrics = [k for k in all_metrics.keys() if k.startswith('Percentile Difference')]
                        if percentile_metrics:
                            print(f"\n📊 Percentile Differences:")
                            for metric in sorted(percentile_metrics):
                                value = all_metrics[metric]
                                if isinstance(value, float) and not np.isnan(value):
                                    print(f"   {metric}: {value:.3f}%")
                                else:
                                    print(f"   {metric}: {value}")
                        
                        # Save metrics to file (like demo_new.ipynb)
                        metrics_file = os.path.join(results_directory, "metrics_summary.json")
                        try:
                            # Convert any numpy types to Python types for JSON serialization
                            metrics_for_json = {}
                            for k, v in all_metrics.items():
                                if isinstance(v, np.ndarray):
                                    metrics_for_json[k] = v.tolist()
                                elif isinstance(v, (np.integer, np.floating)):
                                    metrics_for_json[k] = v.item()
                                else:
                                    metrics_for_json[k] = v
                            
                            with open(metrics_file, 'w') as f:
                                json.dump(metrics_for_json, f, indent=2)
                            print(f"\n💾 Metrics saved to: {metrics_file}")
                        except Exception as e:
                            print(f"\n❌ Error saving metrics: {e}")
                        
                    else:
                        print("❌ No metrics calculated")
                    
                    print("=" * 60)
                    
                    # Return metrics dictionary
                    return all_metrics
                    
                else:
                    missing_files = []
                    for path_name, path in [("best_x.csv", best_x_path), ("best_y.csv", best_y_path), 
                                          ("y_target.csv", y_target_path), ("x_true.csv", x_true_path)]:
                        if not os.path.exists(path):
                            missing_files.append(path_name)
                    print(f"❌ Required data files not found: {missing_files}")
                    return {}
                    
            except Exception as e:
                print(f"❌ Error loading data: {e}")
                import traceback
                traceback.print_exc()
                return {}
                
        else:
            print("❌ Results directory not available")
            return {}
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all strategy-gradient method combinations"""
        print(f"\n🚀 Starting comprehensive experiments...")
        print(f"🔬 Total experiments: {len(self.strategy_configs) * len(self.gradient_configs)}")
        
        all_results = []
        experiment_count = 0
        total_experiments = len(self.strategy_configs) * len(self.gradient_configs)
        
        # Run experiments
        for strategy_name in self.strategy_configs.keys():
            for gradient_method in self.gradient_configs.keys():
                experiment_count += 1
                print(f"\n📊 Progress: {experiment_count}/{total_experiments}")
                
                result = self.run_single_experiment(strategy_name, gradient_method)
                all_results.append(result)
                
                # Store metrics
                exp_key = f"{strategy_name}_{gradient_method}"
                if result['success'] and 'metrics' in result:
                    self.all_metrics[exp_key] = result['metrics']
                    # Print quick summary for successful experiments
                    metrics = result['metrics']
                    if metrics:
                        coverage = metrics.get('Coverage Rate', float('nan'))
                        mmd = metrics.get('MMD', float('nan'))
                        ot_dist = metrics.get('OT Distance', float('nan'))
                        ares_cost = metrics.get('AReS Cost', float('nan'))
                        print(f"  📈 Quick metrics - Coverage: {coverage:.3f}, MMD: {mmd:.3f}, OT: {ot_dist:.4f}, AReS: {ares_cost:.3f}")
                else:
                    print(f"⚠️  Missing metrics for: {exp_key}")
                    if result.get("error"):
                        print(f"  ↪️  Error: {result.get('error')}")
                    self.all_metrics[exp_key] = {}

        self.all_results = all_results
        return all_results
    
    def save_comprehensive_results(self):
        """Save comprehensive results and analysis"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = f"ComprehensiveResults_{self.dataset_name}_{self.model_name}_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"\n💾 Saving comprehensive results to: {results_dir}")
        
        # 1. Save summary table
        summary_data = []
        for result in self.all_results:
            if result['success']:
                row = {
                    'Strategy': result['strategy_name'],
                    'Gradient_Method': result['gradient_method'],
                    'Experiment_Name': result['experiment_name'],
                    'Elapsed_Time': result['elapsed_time'],
                    'Success': result['success']
                }
                # Add metrics if available
                if 'metrics' in result and result['metrics']:
                    row.update(result['metrics'])
                summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(results_dir, "experiment_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"📊 Summary table saved: {summary_path}")
        
        # 2. Save detailed metrics
        metrics_path = os.path.join(results_dir, "all_metrics.json")
        with open(metrics_path, 'w') as f:
            # Convert numpy types for JSON serialization
            metrics_for_json = {}
            for exp_key, metrics in self.all_metrics.items():
                metrics_clean = {}
                for k, v in metrics.items():
                    if isinstance(v, (np.integer, np.floating)):
                        metrics_clean[k] = v.item()
                    elif isinstance(v, np.ndarray):
                        metrics_clean[k] = v.tolist()
                    else:
                        metrics_clean[k] = v
                metrics_for_json[exp_key] = metrics_clean
            json.dump(metrics_for_json, f, indent=2)
        print(f"📈 Detailed metrics saved: {metrics_path}")
        
        # 3. Create comparison plots
        self.create_comparison_plots(summary_df, results_dir)
        
        # 4. Save configuration
        config_path = os.path.join(results_dir, "experiment_config.json")
        config_data = {
            'dataset_name': self.dataset_name,
            'model_name': self.model_name,
            'seed': self.seed,
            'sample_num': self.sample_num,
            'dce_params': self.dce_params,
            'strategy_configs': self.strategy_configs,
            'gradient_configs': self.gradient_configs,
            'timestamp': timestamp
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"⚙️ Configuration saved: {config_path}")
        
        print(f"✅ All results saved to: {results_dir}")
        return results_dir
    
    def create_comparison_plots(self, summary_df: pd.DataFrame, results_dir: str):
        """Create comparison plots for all experiments"""
        print(f"📊 Creating comparison plots...")
        
        # Core metrics to plot
        core_metrics = ['Coverage Rate', 'MMD', 'OT Distance', 'AReS Cost']
        
        # Create subplot for core metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(core_metrics):
            if metric in summary_df.columns:
                # Create pivot table for heatmap
                pivot_data = summary_df.pivot(index='Strategy', columns='Gradient_Method', values=metric)
                
                # Plot heatmap
                im = axes[i].imshow(pivot_data.values, cmap='viridis', aspect='auto')
                
                # Set labels
                axes[i].set_title(f'{metric}', fontsize=12, fontweight='bold')
                axes[i].set_xticks(range(len(pivot_data.columns)))
                axes[i].set_xticklabels(pivot_data.columns, rotation=45)
                axes[i].set_yticks(range(len(pivot_data.index)))
                axes[i].set_yticklabels(pivot_data.index)
                
                # Add colorbar
                plt.colorbar(im, ax=axes[i], shrink=0.8)
                
                # Add value annotations
                for row in range(len(pivot_data.index)):
                    for col in range(len(pivot_data.columns)):
                        value = pivot_data.iloc[row, col]
                        if not np.isnan(value):
                            axes[i].text(col, row, f'{value:.3f}', 
                                       ha='center', va='center', 
                                       color='white' if value < pivot_data.values.max()/2 else 'black',
                                       fontsize=8)
        
        plt.tight_layout()
        
        # Save core metrics plot
        core_plot_path = os.path.join(results_dir, "core_metrics_comparison.png")
        plt.savefig(core_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Core metrics plot saved: {core_plot_path}")
        
        # Create bar plots for strategy comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(core_metrics):
            if metric in summary_df.columns:
                # Group by strategy and calculate mean
                strategy_means = summary_df.groupby('Strategy')[metric].mean()
                
                bars = axes[i].bar(strategy_means.index, strategy_means.values, 
                                 color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
                axes[i].set_title(f'{metric} by Strategy (Average)', fontsize=12, fontweight='bold')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, strategy_means.values):
                    if not np.isnan(value):
                        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save strategy comparison plot
        strategy_plot_path = os.path.join(results_dir, "strategy_comparison.png")
        plt.savefig(strategy_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Strategy comparison plot saved: {strategy_plot_path}")
        
        # Create gradient method comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(core_metrics):
            if metric in summary_df.columns:
                # Group by gradient method and calculate mean
                gradient_means = summary_df.groupby('Gradient_Method')[metric].mean()
                
                bars = axes[i].bar(gradient_means.index, gradient_means.values, 
                                 color=['#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
                axes[i].set_title(f'{metric} by Gradient Method (Average)', fontsize=12, fontweight='bold')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, gradient_means.values):
                    if not np.isnan(value):
                        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save gradient method comparison plot
        gradient_plot_path = os.path.join(results_dir, "gradient_method_comparison.png")
        plt.savefig(gradient_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🧭 Gradient method comparison plot saved: {gradient_plot_path}")
    
    def run_comprehensive_experiments(self):
        """Main method to run all experiments"""
        print(f"🚀 Starting comprehensive DCE experiments...")
        
        # Load dataset
        self.load_dataset()
        
        # Train model
        self.train_model()
        
        # Run all experiments
        results = self.run_all_experiments()
        
        # Save results
        results_dir = self.save_comprehensive_results()
        
        # Create final comprehensive table
        table_path = self.create_final_comprehensive_table(results_dir)
        
        # Print final summary
        successful_experiments = sum(1 for r in results if r['success'])
        total_experiments = len(results)
        
        print(f"\n🎉 Comprehensive experiments completed!")
        print(f"✅ Successful experiments: {successful_experiments}/{total_experiments}")
        print(f"📊 Results saved to: {results_dir}")
        print(f"📋 Final comprehensive table: {table_path}")
        
        if successful_experiments > 0:
            # Show best performers
            print(f"\n🏆 Best performers:")
            summary_data = []
            for result in results:
                if result['success'] and 'metrics' in result and result['metrics']:
                    summary_data.append({
                        'Experiment': result['experiment_name'],
                        'Coverage Rate': result['metrics'].get('Coverage Rate', float('nan')),
                        'MMD': result['metrics'].get('MMD', float('nan')),
                        'OT Distance': result['metrics'].get('OT Distance', float('nan')),
                        'AReS Cost': result['metrics'].get('AReS Cost', float('nan'))
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                
                # Best coverage rate
                best_coverage = summary_df.loc[summary_df['Coverage Rate'].idxmax()]
                print(f"🎯 Best Coverage Rate: {best_coverage['Experiment']} ({best_coverage['Coverage Rate']:.3f})")
                
                # Lowest MMD (most similar distributions)
                best_mmd = summary_df.loc[summary_df['MMD'].idxmin()]
                print(f"📊 Lowest MMD: {best_mmd['Experiment']} ({best_mmd['MMD']:.3f})")
                
                # Lowest OT Distance
                best_ot = summary_df.loc[summary_df['OT Distance'].idxmin()]
                print(f"🔄 Lowest OT Distance: {best_ot['Experiment']} ({best_ot['OT Distance']:.4f})")
                
                # Lowest AReS Cost
                best_ares = summary_df.loc[summary_df['AReS Cost'].idxmin()]
                print(f"💰 Lowest AReS Cost: {best_ares['Experiment']} ({best_ares['AReS Cost']:.3f})")
        
        return results_dir
    
    def create_final_comprehensive_table(self, results_dir: str):
        """
        Create final comprehensive table with all metrics as requested
        - Rows: Experiments (Strategy + Gradient Method)
        - Columns: All specified metrics
        - Save with timestamp in same directory as script
        """
        print(f"\n📊 Creating final comprehensive metrics table...")
        
        # Define metrics in the exact order requested
        metric_columns = [
            'Coverage Rate',
            'MMD', 
            'OT Distance',
            'OT Distance (normalized original)',
            'AReS Cost',
            'Percentile Difference 0-15%',
            'Percentile Difference 15-30%',
            'Percentile Difference 30-70%',
            'Percentile Difference 70-85%',
            'Percentile Difference 85-100%'
        ]
        
        # Create experiment names in order (strategies × gradient methods)
        experiment_rows = []
        for strategy_name in self.strategy_configs.keys():
            for gradient_method in self.gradient_configs.keys():
                # Create descriptive experiment name
                if gradient_method == "original":
                    exp_display = f"{strategy_name}_original"
                elif gradient_method == "cone_sampling":
                    exp_display = f"{strategy_name}_cone_45deg"
                elif gradient_method == "weighted_combination":
                    exp_display = f"{strategy_name}_weighted_0.7"
                elif gradient_method == "noisy_gradient":
                    exp_display = f"{strategy_name}_noisy_0.1"
                else:
                    exp_display = f"{strategy_name}_{gradient_method}"
                
                experiment_rows.append({
                    'Experiment': exp_display,
                    'Strategy': strategy_name,
                    'Gradient_Method': gradient_method,
                    'Key': f"{strategy_name}_{gradient_method}"
                })
        
        # Create final table with experiments as rows and metrics as columns
        table_data = []
        
        for exp_info in experiment_rows:
            row = {'Experiment': exp_info['Experiment']}
            
            # Get metrics for this experiment
            exp_key = exp_info['Key']
            metrics = self.all_metrics.get(exp_key, {})
            
            # Add each metric column
            for metric in metric_columns:
                if metric in metrics:
                    value = metrics[metric]
                    if isinstance(value, (np.integer, np.floating)):
                        row[metric] = value.item()
                    elif isinstance(value, float) and not np.isnan(value):
                        row[metric] = value
                    else:
                        row[metric] = 'N/A'
                else:
                    row[metric] = 'N/A'
            
            table_data.append(row)
        
        # Create DataFrame
        final_df = pd.DataFrame(table_data)
        
        # Generate timestamp and filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        table_filename = f"comprehensive_dce_results_{self.dataset_name}_{self.model_name}_{timestamp}.csv"
        
        # Save in same directory as the script (current working directory)
        script_dir = os.getcwd()
        table_path = os.path.join(script_dir, table_filename)
        
        # Save the table
        final_df.to_csv(table_path, index=False)
        
        print(f"✅ Final comprehensive table saved: {table_path}")
        print(f"📊 Table dimensions: {final_df.shape[0]} experiments × {final_df.shape[1]} columns")
        print(f"🔧 Dataset: {self.dataset_name}, Model: {self.model_name}")
        print(f"📅 Timestamp: {timestamp}")
        
        # Display table preview
        print(f"\n📋 Table Preview (first 5 rows):")
        print("=" * 120)
        
        # Format and display first few rows
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 15)
        
        preview_df = final_df.head(5).round(6)
        print(preview_df.to_string(index=False))
        
        print("=" * 120)
        
        # Summary statistics
        print(f"\n📈 Summary Statistics:")
        numeric_columns = [col for col in final_df.columns if col != 'Experiment' and final_df[col].dtype in ['float64', 'int64']]
        
        if numeric_columns:
            for col in numeric_columns[:5]:  # Show first 5 metrics
                valid_values = pd.to_numeric(final_df[col], errors='coerce').dropna()
                if len(valid_values) > 0:
                    print(f"   {col}: mean={valid_values.mean():.6f}, std={valid_values.std():.6f}")
        
        return table_path


def main():
    """Main function to run comprehensive experiments"""
    
    # Configuration
    DATASET_NAME = "german_credit"  # Options: "german_credit", "cardio", "heloc", "compas", "hotel_booking"
    MODEL_NAME = "SVM"              # Options: "SVM", "MLP", "RandomForest", "LightGBM", "XGBoost"
    SEED = 42
    SAMPLE_NUM = 50
    
    print("=" * 80)
    print("🚀 COMPREHENSIVE DCE EXPERIMENTS")
    print("=" * 80)
    print(f"📊 Dataset: {DATASET_NAME}")
    print(f"🤖 Model: {MODEL_NAME}")
    print(f"🎲 Seed: {SEED}")
    print(f"📋 Sample size: {SAMPLE_NUM}")
    print(f"🔬 Strategy count: 5 (MonteCarlo, Genetic, SimulatedAnnealing, Bayesian, DifferentialEvolution)")
    print(f"🧭 Gradient methods: 4 (original, cone_sampling, weighted_combination, noisy_gradient)")
    print(f"🎯 Total experiments: 20")
    print("=" * 80)
    
    # Create and run experiments
    experiment_runner = ComprehensiveDCEExperiments(
        dataset_name=DATASET_NAME,
        model_name=MODEL_NAME,
        seed=SEED,
        sample_num=SAMPLE_NUM
    )
    
    results_dir = experiment_runner.run_comprehensive_experiments()
    
    print("=" * 80)
    print(f"🎉 ALL EXPERIMENTS COMPLETED!")
    print(f"📁 Results directory: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()