#!/usr/bin/env python3
"""
DCE Server Experiment Script with Configuration File Support
Automated DCE experiments for server deployment with JSON configuration files.
"""

import pandas as pd
import numpy as np
import torch
import math
import os
import json
import glob
import argparse
from datetime import datetime
from typing import List, Dict, Any

# DCE imports
from explainers.model import Model
from explainers.DCEHeuristic import DCENonDifferentiable

# Data loader imports
from data_loader.cardio import CardioData
from data_loader.german_credit import GermanCreditData
from data_loader.hotel_booking import HotelBookingData
from data_loader.heloc import HelocData
from data_loader.compas import CompasData

# Strategy imports
from explainers.cone_sampling.genetic import GeneticStrategy
from explainers.cone_sampling.monte_carlo import MonteCarloStrategy
from explainers.cone_sampling.simulated_annealing import SimulatedAnnealingStrategy
from explainers.cone_sampling.bayesian import BayesianStrategy
from explainers.cone_sampling.differential_evolution import DifferentialEvolutionStrategy

# Model imports
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim

# Metrics imports
from explainers.distances import SlicedWassersteinDivergence, WassersteinDivergence
from scipy.stats import gaussian_kde, entropy
from numpy.linalg import LinAlgError
from sklearn.preprocessing import MinMaxScaler


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DCE Server Experiment Runner with Configuration File Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration File Format:
{
  "experiment_name": "my_experiment",
  "global": {
    "dataset": "german_credit",
    "model": "RandomForest",
    "seed": 42,  // Single seed OR [42, 123, 456] for multiple seeds
    "save_results": true,
    "callback": "final_only",
    "verbose": true
  },
  "dce_params": {
    "X_init": false,
    "U_1": 0.5,
    "U_2": 0.3,
    "l": 0.2,
    "r": 1.0,
    "max_iter": 50,
    "num_trials": 10,
    "top_k": 1
  },
  "cone_sampling": {
    "cone_cat": true,
    "cone_cont": true,
    "cone_angle": 0.7854
  },
  "explain_data": {
    "method": "standard",
    "sample_num": 50,
    "risk_filter": null,
    "fixed_indices": null
  },
  "strategies": [
    {
      "name": "monte_carlo",
      "alias": "mc_default",
      "params": {
        "categorical_step": 1.2,
        "continuous_step": 0.1,
        "temperature": 2.0
      }
    },
    {
      "name": "genetic",
      "alias": "ga_conservative",
      "params": {
        "crossover_prob": 0.8,
        "gene_swap_prob": 0.5,
        "mutation_prob_cat": 0.3,
        "mutation_prob_cont": 0.8,
        "mutation_noise_scale": 0.1,
        "categorical_step": 1.2,
        "continuous_step": 0.1,
        "temperature": 2.0
      }
    }
  ]
}

Examples:
  # Single seed experiment
  python server_experiment_config.py --config experiment1.json
  
  # Override seed from command line
  python server_experiment_config.py --config experiment1.json --seed 123
  
  # Multi-seed experiment (configured in JSON)
  python server_experiment_config.py --config multi_seed_experiment.json
  
  # Batch experiments with wildcards
  python server_experiment_config.py --config batch_experiments/*.json
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to JSON configuration file(s). Supports wildcards.')
    parser.add_argument('--seed', type=int, default=None,
                       help='Override seed from config file (useful for batch runs)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Override output directory')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print configuration and exit without running experiments')
    
    return parser.parse_args()


def default_config():
    """Return default configuration structure."""
    return {
        "experiment_name": "dce_experiment",
        "global": {
            "dataset": "german_credit",
            "model": "RandomForest",
            "seed": 42,
            "save_results": True,
            "callback": "final_only",
            "results_dir": None,
            "verbose": False
        },
        "dce_params": {
            "X_init": False,
            "U_1": 0.5,
            "U_2": 0.3,
            "l": 0.2,
            "r": 1.0,
            "max_iter": 50,
            "num_trials": 10,
            "top_k": 1
        },
        "cone_sampling": {
            "cone_cat": True,
            "cone_cont": True,
            "cone_angle": math.pi/4
        },
        "explain_data": {
            "method": "standard",
            "sample_num": 50,
            "risk_filter": None,
            "fixed_indices": None
        },
        "strategies": [
            {
                "name": "monte_carlo",
                "alias": "mc_default",
                "params": {
                    "categorical_step": 1.2,
                    "continuous_step": 0.1,
                    "temperature": 2.0
                }
            }
        ]
    }


def load_and_validate_config(config_path: str, seed_override: int = None) -> Dict:
    """Load and validate configuration file."""
    print(f"📄 Loading configuration: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    # Merge with defaults
    default = default_config()
    
    # Deep merge function
    def deep_merge(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value
    
    deep_merge(default, config)
    config = default
    
    # Override seed if provided
    if seed_override is not None:
        config["global"]["seed"] = seed_override
        print(f"🎲 Seed overridden to: {seed_override}")
    
    # Normalize seed format (convert single int to list)
    seed_value = config["global"]["seed"]
    if isinstance(seed_value, int):
        config["global"]["seeds"] = [seed_value]
    elif isinstance(seed_value, list):
        config["global"]["seeds"] = seed_value
        # Validate all seeds are integers
        if not all(isinstance(s, int) for s in seed_value):
            raise ValueError("All seeds must be integers")
    else:
        raise ValueError("Seed must be an integer or list of integers")
    
    # Keep original seed field for backward compatibility
    config["global"]["seed"] = config["global"]["seeds"][0]
    
    # Validate required fields
    required_fields = {
        "global": ["dataset", "model"],
        "strategies": []
    }
    
    for section, fields in required_fields.items():
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
        for field in fields:
            if field not in config[section]:
                raise ValueError(f"Missing required field: {section}.{field}")
    
    # Validate strategy configurations
    if not config["strategies"]:
        raise ValueError("At least one strategy must be specified")
    
    valid_strategies = ['monte_carlo', 'genetic', 'simulated_annealing', 'bayesian', 'differential_evolution']
    valid_datasets = ['cardio', 'german_credit', 'hotel_booking', 'heloc', 'compas']
    valid_models = ['RandomForest', 'LightGBM', 'XGBoost', 'SVM', 'MLP']
    
    if config["global"]["dataset"] not in valid_datasets:
        raise ValueError(f"Invalid dataset: {config['global']['dataset']}")
    
    if config["global"]["model"] not in valid_models:
        raise ValueError(f"Invalid model: {config['global']['model']}")
    
    # Validate explain_data configuration
    explain_config = config.get("explain_data", {})
    method = explain_config.get("method", "standard")
    valid_methods = ['standard', 'risk_filter', 'fixed_indices']
    
    if method not in valid_methods:
        raise ValueError(f"Invalid explain_data method: {method}. Valid methods: {valid_methods}")
    
    if method == "risk_filter":
        risk_filter = explain_config.get("risk_filter")
        if risk_filter is None:
            raise ValueError("risk_filter must be specified when method='risk_filter'")
        if not isinstance(risk_filter, int) or risk_filter not in [0, 1]:
            raise ValueError("risk_filter must be 0 (low risk) or 1 (high risk)")
    
    elif method == "fixed_indices":
        fixed_indices = explain_config.get("fixed_indices")
        if not fixed_indices or len(fixed_indices) == 0:
            raise ValueError("fixed_indices must be non-empty when method='fixed_indices'")
        if not all(isinstance(idx, int) and idx >= 0 for idx in fixed_indices):
            raise ValueError("All fixed_indices must be non-negative integers")
    
    sample_num = explain_config.get("sample_num", 50)
    if not isinstance(sample_num, int) or sample_num <= 0:
        raise ValueError("sample_num must be a positive integer")
    
    for i, strategy in enumerate(config["strategies"]):
        if "name" not in strategy:
            raise ValueError(f"Strategy {i} missing required field: name")
        if strategy["name"] not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy['name']}")
        if "params" not in strategy:
            config["strategies"][i]["params"] = {}
        if "alias" not in strategy:
            config["strategies"][i]["alias"] = f"{strategy['name']}_{i}"
    
    print(f"✅ Configuration validated successfully")
    return config


def print_config_summary(config: Dict):
    """Print comprehensive configuration summary."""
    print(f"\n🚀 DCE Experiment Configuration")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📝 Experiment: {config.get('experiment_name', 'Unnamed')}")
    
    global_config = config["global"]
    print(f"\n🎯 Global Settings:")
    print(f"   Dataset: {global_config['dataset']}")
    print(f"   Model: {global_config['model']}")
    seeds = global_config.get('seeds', [global_config['seed']])
    if len(seeds) == 1:
        print(f"   Seed: {seeds[0]}")
    else:
        print(f"   Seeds: {seeds} ({len(seeds)} seeds)")
    print(f"   Sample size: {config.get('explain_data', {}).get('sample_num', 'N/A')}")
    print(f"   Save results: {global_config['save_results']}")
    print(f"   Callback mode: {global_config['callback']}")
    print(f"   Verbose: {global_config['verbose']}")
    
    dce_params = config["dce_params"]
    print(f"\n📊 DCE Parameters:")
    print(f"   X_init: {dce_params['X_init']}")
    print(f"   U_1: {dce_params['U_1']}, U_2: {dce_params['U_2']}")
    print(f"   l: {dce_params['l']}, r: {dce_params['r']}")
    print(f"   max_iter: {dce_params['max_iter']}")
    print(f"   num_trials: {dce_params['num_trials']}")
    print(f"   top_k: {dce_params['top_k']}")
    print(f"   Using function defaults for: n_proj, delta, alpha, kappa, bootstrap")
    
    explain_config = config["explain_data"]
    print(f"\n📊 Explain Data:")
    print(f"   Method: {explain_config['method']}")
    print(f"   Sample num: {explain_config['sample_num']}")
    if explain_config['method'] == 'risk_filter' and explain_config['risk_filter'] is not None:
        print(f"   Risk filter: {explain_config['risk_filter']}")
    elif explain_config['method'] == 'fixed_indices' and explain_config['fixed_indices'] is not None:
        indices_preview = explain_config['fixed_indices'][:5] if len(explain_config['fixed_indices']) > 5 else explain_config['fixed_indices']
        print(f"   Fixed indices: {indices_preview}{'...' if len(explain_config['fixed_indices']) > 5 else ''} (total: {len(explain_config['fixed_indices'])})")
    
    cone_config = config["cone_sampling"]
    print(f"\n🧭 Cone Sampling:")
    print(f"   Categorical: {cone_config['cone_cat']}")
    print(f"   Continuous: {cone_config['cone_cont']}")
    print(f"   Angle: {cone_config['cone_angle']:.3f} rad ({math.degrees(cone_config['cone_angle']):.1f}°)")
    
    print(f"\n🎯 Strategies ({len(config['strategies'])}):")
    for i, strategy in enumerate(config["strategies"]):
        print(f"   {i+1}. {strategy['name']} (alias: {strategy['alias']})")
        if global_config.get('verbose', False):
            for param, value in strategy['params'].items():
                print(f"      {param}: {value}")


def get_explain_data(data, explain_config: Dict, seed: int):
    """Generate df_explain based on configuration."""
    method = explain_config["method"]
    sample_num = explain_config["sample_num"]
    
    print(f"📊 Generating explain data using method: {method}")
    
    if method == "standard":
        # Standard random sampling
        df_explain = data.get_df_explain(sample_num=sample_num)
        print(f"   Standard sampling: {len(df_explain)} samples")
        
    elif method == "risk_filter":
        # Filter by risk level
        risk_filter = explain_config.get("risk_filter")
        if risk_filter is None:
            raise ValueError("risk_filter must be specified when method='risk_filter'")
        
        # Get raw data and filter by risk
        df_raw = data.df[data.df["Risk"] == risk_filter].head(sample_num)
        if len(df_raw) < sample_num:
            print(f"   Warning: Only {len(df_raw)} samples available with Risk={risk_filter}, requested {sample_num}")
        
        # Standardize the filtered data
        df_explain = (df_raw[data.features] - data.mean) / data.std
        print(f"   Risk filter (Risk={risk_filter}): {len(df_explain)} samples")
        
    elif method == "fixed_indices":
        # Use fixed indices
        fixed_indices = explain_config.get("fixed_indices")
        if fixed_indices is None or len(fixed_indices) == 0:
            raise ValueError("fixed_indices must be specified and non-empty when method='fixed_indices'")
        
        # Get train/test split to use test data for indexing
        X_train, X_test, y_train, y_test = data.get_train_test()
        
        # Check if indices are valid
        max_available_index = len(X_test) - 1
        valid_indices = [idx for idx in fixed_indices if idx <= max_available_index]
        
        if len(valid_indices) < len(fixed_indices):
            print(f"   Warning: Some indices exceed X_test size ({max_available_index})")
            print(f"   Using {len(valid_indices)} valid indices out of {len(fixed_indices)}")
        
        if len(valid_indices) == 0:
            raise ValueError("No valid indices found in fixed_indices")
        
        # Select samples using valid indices
        df_explain = X_test.iloc[valid_indices]
        print(f"   Fixed indices: {len(df_explain)} samples from indices {valid_indices[:5]}{'...' if len(valid_indices) > 5 else ''}")
        
    else:
        raise ValueError(f"Unknown explain_data method: {method}")
    
    return df_explain


def load_dataset(dataset_name: str, seed: int):
    """Load the specified dataset."""
    print(f"📊 Loading dataset: {dataset_name}")
    
    if dataset_name == 'cardio':
        return CardioData(seed=seed), dataset_name
    elif dataset_name == 'german_credit':
        return GermanCreditData(seed=seed), dataset_name
    elif dataset_name == 'hotel_booking':
        return HotelBookingData(seed=seed), dataset_name
    elif dataset_name == 'heloc':
        return HelocData(seed=seed), dataset_name
    elif dataset_name == 'compas':
        return CompasData(seed=seed), dataset_name
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_model(model_name: str, X_train, y_train, seed: int):
    """Create and train the specified model."""
    print(f"🤖 Creating model: {model_name}")
    
    if model_name == 'RandomForest':
        model_raw = RandomForestClassifier(n_estimators=100, random_state=seed)
        backend = "sklearn"
        # Train the model
        model_raw.fit(X_train, y_train)
        
    elif model_name == 'LightGBM':
        model_raw = LGBMClassifier(n_estimators=100, random_state=seed, verbose=-1)
        backend = "lightgbm"
        # Train the model
        model_raw.fit(X_train, y_train)
        
    elif model_name == 'XGBoost':
        model_raw = XGBClassifier(n_estimators=100, random_state=seed, eval_metric="logloss")
        backend = "xgboost"
        # Train the model
        model_raw.fit(X_train, y_train)
        
    elif model_name == 'SVM':
        from models.svm import LinearSVM
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
        
        # Initialize the model
        model_raw = LinearSVM(input_dim=X_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model_raw.parameters(), lr=0.01)
        
        # Training loop
        num_epochs = 300
        for epoch in range(num_epochs):
            outputs = model_raw(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model_raw.eval()
        backend = "pytorch"
        
    elif model_name == 'MLP':
        from models.mlp import BlackBoxModel
        # Initialize the model
        input_dim = X_train.shape[1]
        model_raw = BlackBoxModel(input_dim=input_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model_raw.parameters(), lr=0.01)
        
        X_tensor = torch.FloatTensor(X_train.values)
        y_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
        
        # Training loop
        for epoch in range(300):
            pred = model_raw(X_tensor)
            loss = criterion(pred, y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model_raw.eval()
        backend = "pytorch"
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return Model(model=model_raw, backend=backend, data=None), model_name


def create_strategy(strategy_config: Dict, explainer, cone_config: Dict, verbose: bool = False):
    """Create strategy from configuration."""
    strategy_name = strategy_config["name"]
    alias = strategy_config["alias"]
    params = strategy_config["params"]
    
    print(f"🎯 Creating strategy: {strategy_name} (alias: {alias})")
    
    # Common parameters for all strategies
    common_params = {
        'random_state': explainer.device.index if hasattr(explainer.device, 'index') else 42,  # Will be set properly
        'cone_angle': cone_config['cone_angle'],
        'use_cone_sampling_categorical': cone_config['cone_cat'],
        'use_cone_sampling_continuous': cone_config['cone_cont'],
        'categorical_step': params.get('categorical_step', 1.2),
        'continuous_step': params.get('continuous_step', 0.1),
        'temperature': params.get('temperature', 2.0)
    }
    
    if verbose:
        print(f"   Common parameters:")
        for key, value in common_params.items():
            if key != 'random_state':  # Skip random_state as it's set later
                print(f"     {key}: {value}")
    
    if strategy_name == 'monte_carlo':
        strategy = MonteCarloStrategy(explainer, **common_params)
        if verbose:
            print(f"   Monte Carlo strategy created")
        
    elif strategy_name == 'genetic':
        genetic_params = {
            'crossover_prob': params.get('crossover_prob', 0.8),
            'gene_swap_prob': params.get('gene_swap_prob', 0.5),
            'mutation_prob_cat': params.get('mutation_prob_cat', 0.3),
            'mutation_prob_cont': params.get('mutation_prob_cont', 0.8),
            'mutation_noise_scale': params.get('mutation_noise_scale', 0.1),
        }
        strategy = GeneticStrategy(explainer, **genetic_params, **common_params)
        if verbose:
            print(f"   Genetic algorithm parameters:")
            for key, value in genetic_params.items():
                print(f"     {key}: {value}")
        
    elif strategy_name == 'simulated_annealing':
        sa_params = {
            'T0': params.get('T0', 1.5),
            'T_final': params.get('T_final', 0.01),
            'temp_decay': params.get('temp_decay', None),
        }
        strategy = SimulatedAnnealingStrategy(explainer, **sa_params, **common_params)
        if verbose:
            print(f"   Simulated annealing parameters:")
            for key, value in sa_params.items():
                print(f"     {key}: {value}")
        
    elif strategy_name == 'bayesian':
        strategy = BayesianStrategy(explainer, **common_params)
        if verbose:
            print(f"   Bayesian optimization strategy created")
        
    elif strategy_name == 'differential_evolution':
        de_params = {
            'F': params.get('F', 0.5),
            'CR': params.get('CR', 0.9),
        }
        strategy = DifferentialEvolutionStrategy(explainer, **de_params, **common_params)
        if verbose:
            print(f"   Differential evolution parameters:")
            for key, value in de_params.items():
                print(f"     {key}: {value}")
        
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return strategy, alias


def get_model_predictions(model_obj, data_array):
    """Universal function to get predictions from any model type."""
    if model_obj.backend == "pytorch":
        import torch
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data_array)
            predictions = model_obj.model(data_tensor).cpu().numpy()
            return predictions.flatten()
    elif model_obj.backend in ["sklearn", "lightgbm", "xgboost"]:
        try:
            prob_predictions = model_obj.model.predict_proba(data_array)
            return prob_predictions[:, 1]
        except AttributeError:
            predictions = model_obj.model.predict(data_array)
            return predictions
    else:
        raise ValueError(f"Unknown model backend: {model_obj.backend}")


def calculate_metrics(results_directory: str, model, data):
    """Calculate comprehensive metrics for the experiment results."""
    print("📊 Calculating metrics...")
    
    if not results_directory or not os.path.exists(results_directory):
        print("❌ Results directory not available")
        return {}
    
    all_metrics = {}
    
    try:
        # Load required data files
        best_x_path = os.path.join(results_directory, "best_x.csv")
        best_y_path = os.path.join(results_directory, "best_y.csv")
        y_target_path = os.path.join(results_directory, "y_target.csv")
        x_true_path = os.path.join(results_directory, "x_true.csv")
        
        if not all(os.path.exists(path) for path in [best_x_path, best_y_path, y_target_path, x_true_path]):
            print("❌ Required data files not found")
            return {}
        
        # Load data
        best_x_standardized = pd.read_csv(best_x_path)
        best_y_df = pd.read_csv(best_y_path)
        y_target_df = pd.read_csv(y_target_path)
        x_true_standardized = pd.read_csv(x_true_path)
        
        # Get original data statistics
        mean_vals = data.mean
        std_vals = data.std
        
        # Prepare data for metrics
        counterfactual_X_original = best_x_standardized * std_vals + mean_vals
        factual_X_original = x_true_standardized * std_vals + mean_vals
        
        # Recover data types
        X_original_sample = data.X
        dtype_dict = X_original_sample.dtypes.apply(lambda x: x.name).to_dict()
        
        for k, v in dtype_dict.items():
            if k in counterfactual_X_original.columns:
                if v[:3] == 'int':
                    counterfactual_X_original[k] = counterfactual_X_original[k].round().astype(v)
                    factual_X_original[k] = factual_X_original[k].round().astype(v)
                else:
                    counterfactual_X_original[k] = counterfactual_X_original[k].astype(v)
                    factual_X_original[k] = factual_X_original[k].astype(v)
        
        # Prepare arrays for metrics
        counterfactual_y = np.array(best_y_df.values)
        y_target = y_target_df['y_target'].values
        
        counterfactual_X_np_standardized = best_x_standardized.values
        factual_X_np_standardized = x_true_standardized.values
        counterfactual_X_np_original = counterfactual_X_original.values
        factual_X_np_original = factual_X_original.values
        
        print("✅ Data processed for metrics calculation")
        
        # Metric 1: Coverage Rate
        try:
            y_cf = counterfactual_y.flatten()
            y_tgt = y_target.flatten() if len(y_target.shape) > 1 else y_target
            match_mask = ((y_cf > 0.5) & (y_tgt > 0.5)) | ((y_cf <= 0.5) & (y_tgt <= 0.5))
            coverage_rate = np.mean(match_mask)
            all_metrics['Coverage Rate'] = coverage_rate
            print(f"   Coverage Rate: {coverage_rate:.3f}")
        except Exception as e:
            print(f"❌ Error calculating Coverage Rate: {e}")
            all_metrics['Coverage Rate'] = float('nan')
        
        # Metric 2: MMD
        try:
            sigma = 1.0
            def gaussian_kernel(x, y, sigma=sigma):
                return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

            def mmd(X_s, X_t, kernel=gaussian_kernel):
                n = X_s.shape[0]
                m = X_t.shape[0]
                XX = np.sum([kernel(X_s[i], X_s[j]) for i in range(n) for j in range(n)])
                YY = np.sum([kernel(X_t[i], X_t[j]) for i in range(m) for j in range(m)])
                XY = np.sum([kernel(X_s[i], X_t[j]) for i in range(n) for j in range(m)])
                return XX / (n ** 2) + YY / (m ** 2) - 2 * XY / (n * m)
            
            mmd_value = mmd(counterfactual_X_np_standardized, factual_X_np_standardized)
            all_metrics['MMD'] = mmd_value
            print(f"   MMD: {mmd_value:.3f}")
        except Exception as e:
            print(f"❌ Error calculating MMD: {e}")
            all_metrics['MMD'] = float('nan')
        
        # Metric 3: OT Distance
        try:
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
                    swd = SlicedWassersteinDivergence(dim=X_s.shape[1], n_proj=5000, random_state=42)
                    distance, _ = swd.distance(X_s, X_t, delta=0.1)
                return distance.item()
            
            ot_distance = compute_distance(best_x_standardized, x_true_standardized)
            all_metrics['OT Distance'] = ot_distance
            print(f"   OT Distance: {ot_distance:.4f}")
        except Exception as e:
            print(f"❌ Error calculating OT Distance: {e}")
            all_metrics['OT Distance'] = float('nan')
        
        # Metric 4: Percentile Differences
        try:
            def compute_percentile_difference(counterfactual_X, factual_X, percentiles):
                columns = counterfactual_X.columns
                diff_list = []
                for percentile in percentiles:
                    for column in columns:
                        perc_cf = np.percentile(counterfactual_X[column].values, percentile)
                        perc_f = np.percentile(factual_X[column].values, percentile)
                        if abs(perc_f) > 1e-8:
                            diff_list.append(abs(perc_cf - perc_f)/abs(perc_f) * 100)
                        else:
                            diff_list.append(0.0 if abs(perc_cf) < 1e-8 else 100.0)
                return np.mean(diff_list)
            
            percentile_ranges = [
                (0, 15, "0-15%"), (15, 30, "15-30%"), (30, 70, "30-70%"), 
                (70, 85, "70-85%"), (85, 100, "85-100%")
            ]
            
            print("   Percentile Differences:")
            for low, high, label in percentile_ranges:
                percentiles = np.arange(low, high, 0.1 if high - low <= 15 else 1)
                try:
                    diff_pct = compute_percentile_difference(best_x_standardized, x_true_standardized, percentiles)
                    all_metrics[f"Percentile Difference {label}"] = diff_pct
                    print(f"     {label}: {diff_pct:.3f}%")
                except Exception as e:
                    print(f"     {label}: Error - {e}")
                    all_metrics[f"Percentile Difference {label}"] = float('nan')
        except Exception as e:
            print(f"❌ Error calculating Percentile Differences: {e}")
        
        # Metric 5: AReS Cost
        try:
            def compute_cost(delta, costs_vector):
                return np.linalg.norm(delta @ np.diag(costs_vector))
            
            n_features = counterfactual_X_np_original.shape[1]
            costs_vector = np.ones(n_features)
            feature_names = list(counterfactual_X_original.columns)
            
            for i, feature_name in enumerate(feature_names):
                feature_values = factual_X_np_original[:, i]
                feature_range = np.max(feature_values) - np.min(feature_values)
                unique_values = np.unique(feature_values)
                n_unique = len(unique_values)
                
                if n_unique <= 10 and np.all(unique_values == unique_values.astype(int)):
                    costs_vector[i] = 0.5
                else:
                    if feature_range > 0:
                        costs_vector[i] = 1.0 / feature_range
                    else:
                        costs_vector[i] = 1.0
            
            deltas = counterfactual_X_np_original - factual_X_np_original
            ares_cost = compute_cost(deltas, costs_vector)
            all_metrics['AReS Cost'] = ares_cost
            print(f"   AReS Cost: {ares_cost:.3f}")
        except Exception as e:
            print(f"❌ Error calculating AReS Cost: {e}")
            all_metrics['AReS Cost'] = float('nan')
        
    except Exception as e:
        print(f"❌ Error in metrics calculation: {e}")
        import traceback
        traceback.print_exc()
    
    return all_metrics


def run_single_seed_experiment(config: Dict, seed: int):
    """Run a complete DCE experiment for a single seed."""
    global_config = config["global"]
    dce_params = config["dce_params"]
    cone_config = config["cone_sampling"]
    
    print(f"\n🎲 Running experiment with seed: {seed}")
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset
    data, dataset_name = load_dataset(global_config["dataset"], seed)
    
    # Get train/test split
    X_train, X_test, y_train, y_test = data.get_train_test()
    print(f"✅ Training data: {len(X_train)} samples")
    print(f"✅ Test data: {len(X_test)} samples")
    
    # Get explanation data using new explain_data configuration
    explain_config = config["explain_data"]
    df_explain = get_explain_data(data, explain_config, seed)
    print(f"✅ Explanation data: {len(df_explain)} samples")
    
    # Create and train model
    model, model_name = create_model(global_config["model"], X_train, y_train, seed)
    
    # Store results for all strategies
    all_results = {}
    
    # Run experiments for each strategy
    for strategy_config in config["strategies"]:
        strategy_name = strategy_config["name"]
        alias = strategy_config["alias"]
        
        print(f"\n{'='*60}")
        print(f"🎲 Running experiment: {strategy_name} (alias: {alias})")
        print(f"{'='*60}")
        
        # Initialize explainer
        explainer = DCENonDifferentiable(model, data)
        
        # Create strategy
        strategy, strategy_alias = create_strategy(
            strategy_config, explainer, cone_config, global_config.get('verbose', False)
        )
        
        # Determine gradient method name
        if cone_config['cone_cat'] and cone_config['cone_cont']:
            gradient_method = "cone_all"
            print("🧭 Full cone sampling mode (both categorical and continuous)")
        elif not cone_config['cone_cat'] and not cone_config['cone_cont']:
            gradient_method = "original_all"
            print("🎲 Full original mode (pure random sampling for both)")
        elif cone_config['cone_cat'] and not cone_config['cone_cont']:
            gradient_method = "cone_cat"
            print("🔀 Mixed mode: cone sampling for categorical, original for continuous")
        elif not cone_config['cone_cat'] and cone_config['cone_cont']:
            gradient_method = "cone_cont"
            print("🔀 Mixed mode: original for categorical, cone sampling for continuous")
        
        if cone_config['cone_cat'] or cone_config['cone_cont']:
            print(f"📐 Cone angle: {math.degrees(cone_config['cone_angle']):.1f}° ({cone_config['cone_angle']:.3f} rad)")
        
        # Run DCE explanation
        print("🚀 Starting DCE explanation...")
        if global_config.get('verbose', False):
            print(f"   DCE parameters: {dce_params}")
            
        try:
            df_cf = explainer.explain(
                df_factual=df_explain,
                X_init=dce_params['X_init'],
                U_1=dce_params['U_1'],
                U_2=dce_params['U_2'],
                l=dce_params['l'],
                r=dce_params['r'],
                strategy=strategy,
                max_iter=dce_params['max_iter'],
                num_trials=dce_params['num_trials'],
                top_k=dce_params['top_k'],
                callback=global_config['callback'],
                save_results=global_config['save_results'],
                dataset_name=dataset_name,
                model_name=model_name,
                seed=seed,
                gradient_method=gradient_method,
                results_dir=global_config.get('results_dir')
            )
            
            print("✅ DCE explanation completed!")
            print(f"🎯 Best Q: {explainer.best_Q:.6f}")
            print(f"📊 Final Q: {explainer.final_Q:.6f}")
            print(f"🎉 Found feasible solution: {explainer.found_feasible_solution}")
            
            # Calculate metrics
            results_directory = getattr(explainer, 'save_dir', None)
            if results_directory:
                print(f"💾 Results saved to: {results_directory}")
                
                # Calculate comprehensive metrics
                metrics = calculate_metrics(results_directory, model, data)
                
                # Store results
                all_results[alias] = {
                    'strategy_name': strategy_name,
                    'alias': alias,
                    'best_Q': explainer.best_Q,
                    'final_Q': explainer.final_Q,
                    'found_feasible_solution': explainer.found_feasible_solution,
                    'best_iter': getattr(explainer, 'best_iter', None),
                    'results_directory': results_directory,
                    'metrics': metrics,
                    'config': strategy_config
                }
                
                # Save metrics to results directory
                if metrics:
                    metrics_file = os.path.join(results_directory, "metrics_summary.json")
                    try:
                        # Convert numpy types for JSON serialization
                        metrics_for_json = {}
                        for k, v in metrics.items():
                            if isinstance(v, np.ndarray):
                                metrics_for_json[k] = v.tolist()
                            elif isinstance(v, (np.integer, np.floating)):
                                metrics_for_json[k] = v.item()
                            else:
                                metrics_for_json[k] = v
                        
                        with open(metrics_file, 'w') as f:
                            json.dump(metrics_for_json, f, indent=2)
                        print(f"💾 Metrics saved to: {metrics_file}")
                    except Exception as e:
                        print(f"❌ Error saving metrics: {e}")
            else:
                print("❌ No results directory available")
                all_results[alias] = {
                    'strategy_name': strategy_name,
                    'alias': alias,
                    'best_Q': explainer.best_Q,
                    'final_Q': explainer.final_Q,
                    'found_feasible_solution': explainer.found_feasible_solution,
                    'error': 'No results directory',
                    'config': strategy_config
                }
        
        except Exception as e:
            print(f"❌ Error in {strategy_name} ({alias}) experiment: {e}")
            import traceback
            traceback.print_exc()
            all_results[alias] = {
                'strategy_name': strategy_name,
                'alias': alias,
                'error': str(e),
                'config': strategy_config
            }
    
    # Print final summary
    print(f"\n{'='*80}")
    print("📈 EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"🎯 Experiment: {config.get('experiment_name', 'Unnamed')}")
    print(f"📊 Dataset: {global_config['dataset']}, Model: {global_config['model']}")
    print(f"🎲 Seed: {seed}")
    
    for alias, results in all_results.items():
        print(f"\n🎯 {results['strategy_name'].upper()} ({alias}):")
        if 'error' in results:
            print(f"   ❌ Error: {results['error']}")
        else:
            print(f"   Best Q: {results.get('best_Q', 'N/A'):.6f}")
            print(f"   Final Q: {results.get('final_Q', 'N/A'):.6f}")
            print(f"   Feasible: {results.get('found_feasible_solution', 'N/A')}")
            
            if 'metrics' in results and results['metrics']:
                metrics = results['metrics']
                core_metrics = ['Coverage Rate', 'MMD', 'OT Distance', 'AReS Cost']
                print("   Core Metrics:")
                for metric in core_metrics:
                    if metric in metrics:
                        value = metrics[metric]
                        if isinstance(value, float) and not np.isnan(value):
                            print(f"     {metric}: {value:.3f}")
                        else:
                            print(f"     {metric}: {value}")
    
    print(f"\n✅ All experiments completed!")
    return all_results


def run_experiment(config: Dict):
    """Run complete DCE experiments for all seeds in configuration."""
    print_config_summary(config)
    
    global_config = config["global"] 
    seeds = global_config["seeds"]
    
    # Results storage for all seeds
    all_seeds_results = {}
    
    print(f"\n{'='*80}")
    print(f"🎯 Multi-Seed Experiment Setup")
    print(f"{'='*80}")
    print(f"📊 Total seeds to process: {len(seeds)}")
    print(f"🎲 Seeds: {seeds}")
    print(f"📈 Strategies per seed: {len(config['strategies'])}")
    print(f"📊 Total experiments: {len(seeds) * len(config['strategies'])}")
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*80}")
        print(f"🎲 SEED {i+1}/{len(seeds)}: {seed}")
        print(f"{'='*80}")
        
        try:
            # Run experiment for this seed
            seed_results = run_single_seed_experiment(config, seed)
            all_seeds_results[f"seed_{seed}"] = {
                'seed': seed,
                'results': seed_results
            }
            print(f"✅ Seed {seed} completed successfully!")
            
        except Exception as e:
            print(f"❌ Error in seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            all_seeds_results[f"seed_{seed}"] = {
                'seed': seed,
                'error': str(e)
            }
    
    # Print multi-seed summary
    print(f"\n{'='*80}")
    print("📈 MULTI-SEED EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"🎯 Experiment: {config.get('experiment_name', 'Unnamed')}")
    print(f"📊 Dataset: {global_config['dataset']}, Model: {global_config['model']}")
    print(f"🎲 Seeds processed: {len(seeds)}")
    
    # Summary statistics across seeds
    successful_seeds = sum(1 for result in all_seeds_results.values() if 'error' not in result)
    failed_seeds = len(seeds) - successful_seeds
    
    print(f"✅ Successful seeds: {successful_seeds}/{len(seeds)}")
    if failed_seeds > 0:
        print(f"❌ Failed seeds: {failed_seeds}/{len(seeds)}")
    
    # Strategy performance across seeds
    if successful_seeds > 0:
        print(f"\n📊 Strategy Performance Summary:")
        for strategy_config in config["strategies"]:
            alias = strategy_config["alias"]
            strategy_name = strategy_config["name"]
            
            # Collect metrics across successful seeds
            best_qs = []
            feasible_counts = 0
            
            for seed_data in all_seeds_results.values():
                if 'error' not in seed_data and alias in seed_data['results']:
                    result = seed_data['results'][alias]
                    if 'error' not in result:
                        if 'best_Q' in result:
                            best_qs.append(result['best_Q'])
                        if result.get('found_feasible_solution', False):
                            feasible_counts += 1
            
            if best_qs:
                print(f"   🎯 {strategy_name.upper()} ({alias}):")
                print(f"     Best Q - Mean: {np.mean(best_qs):.6f}, Std: {np.std(best_qs):.6f}")
                print(f"     Feasible solutions: {feasible_counts}/{successful_seeds}")
    
    return all_seeds_results


def create_example_config():
    """Create example configuration files."""
    # Example 1: Basic comparison
    basic_config = {
        "experiment_name": "basic_comparison",
        "global": {
            "dataset": "german_credit",
            "model": "RandomForest",
            "seed": 42,
            "save_results": True,
            "callback": "final_only",
            "verbose": False
        },
        "dce_params": {
            "X_init": False,
            "U_1": 0.5,
            "U_2": 0.3,
            "l": 0.2,
            "r": 1.0,
            "max_iter": 50,
            "num_trials": 10,
            "top_k": 1
        },
        "cone_sampling": {
            "cone_cat": True,
            "cone_cont": True,
            "cone_angle": 0.7854
        },
        "strategies": [
            {
                "name": "monte_carlo",
                "alias": "mc_default",
                "params": {
                    "categorical_step": 1.2,
                    "continuous_step": 0.1,
                    "temperature": 2.0
                }
            },
            {
                "name": "genetic",
                "alias": "ga_default",
                "params": {
                    "crossover_prob": 0.8,
                    "gene_swap_prob": 0.5,
                    "mutation_prob_cat": 0.3,
                    "mutation_prob_cont": 0.8,
                    "mutation_noise_scale": 0.1,
                    "categorical_step": 1.2,
                    "continuous_step": 0.1,
                    "temperature": 2.0
                }
            }
        ]
    }
    
    # Example 2: Parameter tuning
    tuning_config = {
        "experiment_name": "genetic_parameter_tuning",
        "global": {
            "dataset": "hotel_booking",
            "model": "LightGBM",
            "seed": 123,
            "save_results": True,
            "callback": "final_only",
            "verbose": True
        },
        "dce_params": {
            "X_init": False,
            "U_1": 0.6,
            "U_2": 0.4,
            "l": 0.2,
            "r": 1.0,
            "max_iter": 100,
            "num_trials": 20,
            "top_k": 1
        },
        "explain_data": {
            "method": "standard",
            "sample_num": 100,
            "risk_filter": None,
            "fixed_indices": None
        },
        "strategies": [
            {
                "name": "genetic",
                "alias": "ga_conservative",
                "params": {
                    "crossover_prob": 0.7,
                    "mutation_prob_cat": 0.2,
                    "categorical_step": 1.0
                }
            },
            {
                "name": "genetic",
                "alias": "ga_aggressive",
                "params": {
                    "crossover_prob": 0.9,
                    "mutation_prob_cat": 0.5,
                    "categorical_step": 1.8
                }
            }
        ]
    }
    
    # Example 3: Multi-seed experiment
    multi_seed_config = {
        "experiment_name": "multi_seed_comparison",
        "global": {
            "dataset": "compas",
            "model": "XGBoost",
            "seed": [42, 123, 456, 789, 999],  # Multiple seeds
            "save_results": True,
            "callback": "final_only",
            "verbose": False
        },
        "dce_params": {
            "X_init": False,
            "U_1": 0.5,
            "U_2": 0.3,
            "l": 0.2,
            "r": 1.0,
            "max_iter": 50,
            "num_trials": 10,
            "top_k": 1
        },
        "explain_data": {
            "method": "standard",
            "sample_num": 60,
            "risk_filter": None,
            "fixed_indices": None
        },
        "cone_sampling": {
            "cone_cat": True,
            "cone_cont": True,
            "cone_angle": 0.7854
        },
        "strategies": [
            {
                "name": "monte_carlo",
                "alias": "mc_robust",
                "params": {
                    "categorical_step": 1.2,
                    "continuous_step": 0.1,
                    "temperature": 2.0
                }
            },
            {
                "name": "genetic",
                "alias": "ga_robust",
                "params": {
                    "crossover_prob": 0.8,
                    "gene_swap_prob": 0.5,
                    "mutation_prob_cat": 0.3,
                    "mutation_prob_cont": 0.8,
                    "mutation_noise_scale": 0.1,
                    "categorical_step": 1.2,
                    "continuous_step": 0.1,
                    "temperature": 2.0
                }
            }
        ]
    }
    
    # Save example configs
    os.makedirs("example_configs", exist_ok=True)
    
    with open("example_configs/basic_comparison.json", 'w') as f:
        json.dump(basic_config, f, indent=2)
    
    with open("example_configs/genetic_tuning.json", 'w') as f:
        json.dump(tuning_config, f, indent=2)
    
    with open("example_configs/multi_seed_comparison.json", 'w') as f:
        json.dump(multi_seed_config, f, indent=2)
    
    print("📁 Example configuration files created:")
    print("   - example_configs/basic_comparison.json")
    print("   - example_configs/genetic_tuning.json")
    print("   - example_configs/multi_seed_comparison.json")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Handle wildcards in config path
    config_paths = []
    if '*' in args.config or '?' in args.config:
        config_paths = glob.glob(args.config)
        if not config_paths:
            print(f"❌ No configuration files found matching: {args.config}")
            return 1
    else:
        config_paths = [args.config]
    
    if args.dry_run:
        print("🔍 Dry run mode - loading and validating configurations only")
    
    total_experiments = 0
    successful_experiments = 0
    
    for config_path in config_paths:
        try:
            print(f"\n{'='*80}")
            print(f"📄 Processing configuration: {config_path}")
            print(f"{'='*80}")
            
            config = load_and_validate_config(config_path, args.seed)
            
            if args.dry_run:
                print_config_summary(config)
                print(f"✅ Configuration is valid")
                continue
            
            results = run_experiment(config)
            total_experiments += 1
            successful_experiments += 1
            
        except Exception as e:
            total_experiments += 1
            print(f"❌ Error processing {config_path}: {e}")
            import traceback
            traceback.print_exc()
    
    if args.dry_run:
        print(f"\n✅ Dry run completed. Validated {len(config_paths)} configuration files.")
    else:
        print(f"\n📊 Batch processing completed:")
        print(f"   Total experiments: {total_experiments}")
        print(f"   Successful: {successful_experiments}")
        print(f"   Failed: {total_experiments - successful_experiments}")
    
    return 0 if successful_experiments == total_experiments else 1


if __name__ == "__main__":
    # Create example configs if they don't exist
    if not os.path.exists("example_configs"):
        print("📁 Creating example configuration files...")
        create_example_config()
        print()
    
    exit(main())