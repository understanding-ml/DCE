#!/usr/bin/env python3
"""
DCE Server Experiment Script with Multi-Dataset/Model Support
Extended version supporting multiple datasets and models in a single configuration.
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
from explainers.DCESolver import DCESolver

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

# Visualization imports
import matplotlib.pyplot as plt


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
    "dataset": "german_credit",  // Single dataset OR ["german_credit", "heloc"] for multiple
    "model": "RandomForest",     // Single model OR ["RandomForest", "LightGBM"] for multiple
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
    "top_k": 1,
    "use_global_ranges": true
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
        "temperature": 2.0,
        "h": 2
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
        "temperature": 2.0,
        "h": 2
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
            "callback": None,
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
            "top_k": 1,
            "use_global_ranges": True,
            "save_iterations": False
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
                    "temperature": 2.0,
                    "h": 2
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
    
    # Normalize dataset format (convert single string to list)
    dataset_value = config["global"]["dataset"]
    if isinstance(dataset_value, str):
        config["global"]["datasets"] = [dataset_value]
    elif isinstance(dataset_value, list):
        config["global"]["datasets"] = dataset_value
        # Validate all datasets are strings
        if not all(isinstance(d, str) for d in dataset_value):
            raise ValueError("All datasets must be strings")
    else:
        raise ValueError("Dataset must be a string or list of strings")
    
    # Keep original dataset field for backward compatibility
    config["global"]["dataset"] = config["global"]["datasets"][0]
    
    # Normalize model format (convert single string to list)
    model_value = config["global"]["model"]
    if isinstance(model_value, str):
        config["global"]["models"] = [model_value]
    elif isinstance(model_value, list):
        config["global"]["models"] = model_value
        # Validate all models are strings
        if not all(isinstance(m, str) for m in model_value):
            raise ValueError("All models must be strings")
    else:
        raise ValueError("Model must be a string or list of strings")
    
    # Keep original model field for backward compatibility
    config["global"]["model"] = config["global"]["models"][0]
    
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
    
    # Validate all datasets
    for dataset in config["global"]["datasets"]:
        if dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset: {dataset}. Valid datasets: {valid_datasets}")
    
    # Validate all models
    for model in config["global"]["models"]:
        if model not in valid_models:
            raise ValueError(f"Invalid model: {model}. Valid models: {valid_models}")
    
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
    
    # Display datasets
    datasets = global_config.get('datasets', [global_config['dataset']])
    if len(datasets) == 1:
        print(f"   Dataset: {datasets[0]}")
    else:
        print(f"   Datasets: {datasets} ({len(datasets)} datasets)")
    
    # Display models
    models = global_config.get('models', [global_config['model']])
    if len(models) == 1:
        print(f"   Model: {models[0]}")
    else:
        print(f"   Models: {models} ({len(models)} models)")
    
    # Display seeds
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
        'random_state': explainer.device.index if hasattr(explainer.device, 'index') else 42,  
        'cone_angle': cone_config['cone_angle'],
        'use_cone_sampling_categorical': cone_config['cone_cat'],
        'use_cone_sampling_continuous': cone_config['cone_cont'],
        'categorical_step': params.get('categorical_step', 1.2),
        'continuous_step': params.get('continuous_step', 0.1),
        'temperature': params.get('temperature', 2.0),
        'h': params.get('h', 2)  
    }
    
    if verbose:
        print(f"   Common parameters:")
        for key, value in common_params.items():
            if key != 'random_state':  
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
        
        # Check is_feasible status from optimization log
        is_feasible = False
        log_df = None
        optimization_log_path = os.path.join(results_directory, "optimization_log.csv")
        if os.path.exists(optimization_log_path):
            try:
                log_df = pd.read_csv(optimization_log_path)
                if 'is_feasible' in log_df.columns:
                    is_feasible = log_df['is_feasible'].any() 
                print(f"✅ Feasible solution found: {is_feasible}")
            except Exception as e:
                print(f"⚠️  Warning: Could not read feasible status: {e}")
                is_feasible = False
        else:
            print("⚠️  Warning: optimization_log.csv not found, is_feasible set to False")
        
        # Load data - NOTE: both best_x and x_true are now in ORIGINAL scale
        best_x_original = pd.read_csv(best_x_path)
        best_y_df = pd.read_csv(best_y_path)
        y_target_df = pd.read_csv(y_target_path)
        x_true_original = pd.read_csv(x_true_path)

        # Get original data statistics for standardization when needed
        mean_vals = data.mean
        std_vals = data.std

        # Since both files are in original scale, use them directly
        counterfactual_X_original = best_x_original
        factual_X_original = x_true_original

        # For metrics that need standardized data, standardize from original
        counterfactual_X_standardized = (best_x_original - mean_vals) / std_vals
        factual_X_standardized = (x_true_original - mean_vals) / std_vals
        
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

        # Arrays in both scales
        counterfactual_X_np_standardized = counterfactual_X_standardized.values
        factual_X_np_standardized = factual_X_standardized.values
        counterfactual_X_np_original = counterfactual_X_original.values
        factual_X_np_original = factual_X_original.values
        
        print("✅ Data processed for metrics calculation")

        # Metric 1: MMD
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
        
        # Metric 3: OT Distance X (using standardized data - following baseline methodology)
        try:
            # Common OT distance helper
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

            # If optimization log already has OT_distance_X, reuse the best-iteration value to stay consistent
            ot_distance_x = None
            if log_df is not None and 'OT_distance_X' in log_df.columns:
                log_df_valid = log_df.dropna(subset=['OT_distance_X'])
                if 'is_feasible' in log_df_valid.columns and log_df_valid['is_feasible'].any():
                    log_df_valid = log_df_valid[log_df_valid['is_feasible'] == True]
                if not log_df_valid.empty:
                    if 'Q' in log_df_valid.columns:
                        best_idx = log_df_valid['Q'].idxmin()
                        ot_distance_x = log_df_valid.loc[best_idx, 'OT_distance_X']
                    else:
                        ot_distance_x = log_df_valid['OT_distance_X'].iloc[-1]
            if ot_distance_x is not None:
                all_metrics['OT_distance_X'] = float(ot_distance_x)
                print(f"   OT Distance X (from log, best iter): {ot_distance_x:.6f}")
            else:
                # Use standardized data for OT X calculation (baseline methodology)
                ot_distance_x = compute_distance(counterfactual_X_standardized, factual_X_standardized)
                all_metrics['OT_distance_X'] = ot_distance_x
                print(f"   OT Distance X (standardized): {ot_distance_x:.6f}")
        except Exception as e:
            print(f"❌ Error calculating OT Distance X: {e}")
            all_metrics['OT_distance_X'] = float('nan')

        # Metric 4: OT Distance Y (between predicted y and target y - following baseline methodology)
        try:
            # Prefer using optimization log for consistency with OT X
            ot_distance_y = None
            if log_df is not None and 'OT_distance_Y' in log_df.columns:
                log_df_valid = log_df.dropna(subset=['OT_distance_Y'])
                if 'is_feasible' in log_df_valid.columns and log_df_valid['is_feasible'].any():
                    log_df_valid = log_df_valid[log_df_valid['is_feasible'] == True]
                if not log_df_valid.empty:
                    if 'Q' in log_df_valid.columns:
                        best_idx = log_df_valid['Q'].idxmin()
                        ot_distance_y = log_df_valid.loc[best_idx, 'OT_distance_Y']
                    else:
                        ot_distance_y = log_df_valid['OT_distance_Y'].iloc[-1]

            if ot_distance_y is None:
                # Calculate OT Y distance
                counterfactual_y_tensor = torch.FloatTensor(counterfactual_y.flatten())
                y_target_tensor = torch.FloatTensor(y_target.flatten())
                ot_distance_y = compute_distance(counterfactual_y_tensor, y_target_tensor)

            all_metrics['OT_distance_Y'] = float(ot_distance_y)
            print(f"   OT Distance Y: {ot_distance_y:.6f}")
        except Exception as e:
            print(f"❌ Error calculating OT Distance Y: {e}")
            all_metrics['OT_distance_Y'] = float('nan')

        # Metric 5: AReS Cost (matching demo_new.ipynb implementation)
        try:
            # 1) Build the baseline-heloc style feature cost vector
            feature_names = list(x_true_original.columns)
            # Use original scale data directly (no need to convert)
            cost_list = []
            for fname in feature_names:
                values = x_true_original[fname].values
                unique_count = np.unique(values).size
                ratio = unique_count / len(values)
                if unique_count <= 10 and ratio < 0.5:
                    # categorical feature → fixed cost
                    cost_list.append(0.5)
                else:
                    # continuous feature → inverse of the range
                    val_range = values.max() - values.min()
                    cost_list.append(1.0 / val_range if val_range > 0 else 1.0)
            costs_vector = np.array(cost_list)

            # 2) Compute the delta matrix using STANDARDIZED data for AReS
            # (AReS cost should be scale-invariant, so use standardized delta)
            delta = (counterfactual_X_standardized - factual_X_standardized).values

            # 3) Compute the weighted L2 norm as AReS cost
            ares_cost = np.linalg.norm(delta @ np.diag(costs_vector))

            all_metrics['AReS Cost'] = ares_cost
            print(f"   AReS Cost: {ares_cost:.6f}")
        except Exception as e:
            print(f"❌ Error calculating AReS Cost: {e}")
            all_metrics['AReS Cost'] = float('nan')

        all_metrics['is_feasible'] = is_feasible

        metadata_path = os.path.join(results_directory, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                optimization_time = metadata.get('optimization_time_seconds')
                if optimization_time is not None:
                    all_metrics['optimization_time_seconds'] = optimization_time
                    print(f"   Optimization time: {optimization_time:.2f} seconds")

                if is_feasible:
                    best_iter = metadata.get('best_iter')
                    if best_iter is not None:
                        all_metrics['best_iteration'] = best_iter
                        print(f"   Best iteration: {best_iter}")
                    else:
                        all_metrics['best_iteration'] = None
                else:
                    all_metrics['best_iteration'] = None

            except Exception as meta_error:
                print(f"⚠️  Warning: Could not read metadata: {meta_error}")
                all_metrics['optimization_time_seconds'] = None
                all_metrics['best_iteration'] = None
        else:
            print("⚠️  Warning: metadata.json not found")
            all_metrics['optimization_time_seconds'] = None
            all_metrics['best_iteration'] = None

    except Exception as e:
        print(f"❌ Error in metrics calculation: {e}")
        import traceback
        traceback.print_exc()
        all_metrics['is_feasible'] = False

    required_metrics = {
        'MMD': all_metrics.get('MMD', float('nan')),
        'OT_distance_X': all_metrics.get('OT_distance_X', float('nan')),
        'OT_distance_Y': all_metrics.get('OT_distance_Y', float('nan')),
        'AReS Cost': all_metrics.get('AReS Cost', float('nan')),
        'is_feasible': all_metrics.get('is_feasible', False),
        'optimization_time_seconds': all_metrics.get('optimization_time_seconds'),
        'best_iteration': all_metrics.get('best_iteration')
    }

    return required_metrics


def generate_ot_distance_plot(results_directory: str, dataset_name: str, model_name: str, strategy_name: str):
    """Generate OT Distance evolution visualization (ported from demo_new.ipynb)."""
    if not results_directory or not os.path.exists(results_directory):
        print("❌ Results directory not available for OT Distance plotting")
        return
    
    try:
        # Load optimization log
        optimization_log_path = os.path.join(results_directory, "optimization_log.csv")
        
        if os.path.exists(optimization_log_path):
            # Read the optimization log
            log_df = pd.read_csv(optimization_log_path)
            
            # Check if OT distance data is available
            if 'OT_distance_X' in log_df.columns:
                print("📊 Creating OT Distance X optimization progress visualization...")

                # Create the plot
                plt.figure(figsize=(12, 8))

                # Plot OT Distance X values over iterations
                plt.plot(log_df['iteration'], log_df['OT_distance_X'], 'g-', linewidth=2, label='OT Distance X', alpha=0.7)
                
                # Highlight feasible solutions if available
                if 'is_feasible' in log_df.columns:
                    feasible_points = log_df[log_df['is_feasible'] == True]
                    if not feasible_points.empty:
                        plt.scatter(feasible_points['iteration'], feasible_points['OT_distance_X'],
                                  color='orange', s=50, alpha=0.6, label='Feasible solutions')

                # Formatting
                plt.xlabel('Iteration', fontsize=12)
                plt.ylabel('OT Distance X', fontsize=12)
                plt.title(f'OT Distance X Evolution\nDataset: {dataset_name}, Model: {model_name}, Strategy: {strategy_name}',
                         fontsize=14, pad=20)
                plt.grid(True, alpha=0.3)
                plt.legend()

                # Add text box with key statistics
                stats_text = f"OT Distance X Statistics:\nTotal iterations: {len(log_df)}\nFinal OT Distance X: {log_df['OT_distance_X'].iloc[-1]:.6f}\nOT X range: [{log_df['OT_distance_X'].min():.6f}, {log_df['OT_distance_X'].max():.6f}]"
                
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                        verticalalignment='top', fontsize=9, 
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                
                plt.tight_layout()
                
                # Prepare save paths
                max_iter = len(log_df)
                plot_filename = f"ot_distance_{max_iter}.png"
                
                # Save to current working directory
                current_dir = os.getcwd()
                plot_path = os.path.join(current_dir, plot_filename)
                print(f"🗂️ Current working directory: {current_dir}")
                print(f"📁 Attempting to save OT Distance plot to: {plot_path}")
                
                try:
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    print(f"✅ Successfully saved OT Distance plot to current directory: {plot_path}")
                    
                    # Verify file was created
                    if os.path.exists(plot_path):
                        file_size = os.path.getsize(plot_path)
                        print(f"📄 File verified: {plot_filename} ({file_size} bytes)")
                    else:
                        print(f"❌ File not found after saving: {plot_path}")
                        
                except Exception as save_error:
                    print(f"❌ Error saving OT Distance plot to current directory: {save_error}")
                
                # Save to results directory
                if results_directory and os.path.exists(results_directory):
                    results_plot_path = os.path.join(results_directory, plot_filename)
                    print(f"📁 Attempting to save to results directory: {results_plot_path}")
                    
                    try:
                        plt.savefig(results_plot_path, dpi=300, bbox_inches='tight')
                        print(f"✅ Successfully saved OT Distance plot to results directory: {results_plot_path}")
                        
                        # Verify file was created
                        if os.path.exists(results_plot_path):
                            file_size = os.path.getsize(results_plot_path)
                            print(f"📄 File verified in results: {plot_filename} ({file_size} bytes)")
                        else:
                            print(f"❌ File not found in results directory: {results_plot_path}")
                            
                    except Exception as save_error:
                        print(f"❌ Error saving OT Distance plot to results directory: {save_error}")
                else:
                    print(f"❌ Results directory not accessible: {results_directory}")
                

                print(f"📈 OT Distance X plot details:")
                print(f"   Total iterations: {len(log_df)}")
                print(f"   Final OT Distance X: {log_df['OT_distance_X'].iloc[-1]:.6f}")
                if len(log_df) > 1:
                    print(f"   OT Distance X change: {log_df['OT_distance_X'].iloc[-1] - log_df['OT_distance_X'].iloc[0]:.6f}")

                try:
                    current_files = [f for f in os.listdir(current_dir) if f.startswith('ot_distance_') and f.endswith('.png')]
                    if current_files:
                        print(f"🔍 Found OT Distance plot files in current directory: {current_files}")
                    else:
                        print(f"🔍 No OT Distance plot files found in current directory")
                except Exception as list_error:
                    print(f"❌ Error listing current directory: {list_error}")
                
                plt.close()
                
            else:
                print("📊 OT Distance X data not available in optimization log")
                print("   Note: OT Distance tracking is enabled by default in save_results mode")
                
        else:
            print(f"❌ Optimization log not found: {optimization_log_path}")
            
    except Exception as e:
        print(f"❌ Error creating OT Distance optimization plot: {e}")
        import traceback
        traceback.print_exc()


def generate_q_optimization_plot(results_directory: str, dataset_name: str, model_name: str, strategy_name: str):
    """Generate Q optimization progress visualization (ported from demo_new.ipynb)."""
    if not results_directory or not os.path.exists(results_directory):
        print("❌ Results directory not available for plotting")
        return
    
    try:
        # Load optimization log
        optimization_log_path = os.path.join(results_directory, "optimization_log.csv")
        
        if os.path.exists(optimization_log_path):

            log_df = pd.read_csv(optimization_log_path)
            
            print("📊 Creating Q optimization progress visualization...")
            
            plt.figure(figsize=(12, 8))
            
            plt.plot(log_df['iteration'], log_df['Q'], 'b-', linewidth=2, label='Q value', alpha=0.7)
            
            # Find and highlight the best Q value
            best_q_idx = log_df['Q'].idxmin()
            best_q_value = log_df.loc[best_q_idx, 'Q']
            best_iteration = log_df.loc[best_q_idx, 'iteration']
            
            plt.scatter(best_iteration, best_q_value, color='red', s=100, zorder=5, 
                       label=f'Best Q: {best_q_value:.6f} (iter {best_iteration})')
            
            # Highlight feasible solutions if available
            if 'is_feasible' in log_df.columns:
                feasible_points = log_df[log_df['is_feasible'] == True]
                if not feasible_points.empty:
                    plt.scatter(feasible_points['iteration'], feasible_points['Q'], 
                              color='green', s=50, alpha=0.6, label='Feasible solutions')
            
            # Formatting
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Q Value', fontsize=12)
            plt.title(f'DCE Optimization Progress\nDataset: {dataset_name}, Model: {model_name}, Strategy: {strategy_name}', 
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
Best Q: {best_q_value:.6f}
Best iteration: {best_iteration}
Final Q: {log_df['Q'].iloc[-1]:.6f}
Q range: [{log_df['Q'].min():.6f}, {log_df['Q'].max():.6f}]"""
            
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Prepare save paths
            max_iter = len(log_df)
            plot_filename = f"best_Q_{max_iter}.png"
            
            # Save to current working directory
            current_dir = os.getcwd()
            plot_path = os.path.join(current_dir, plot_filename)
            print(f"🗂️ Current working directory: {current_dir}")
            print(f"📁 Attempting to save plot to: {plot_path}")
            
            try:
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"✅ Successfully saved to current directory: {plot_path}")
                
                # Verify file was created
                if os.path.exists(plot_path):
                    file_size = os.path.getsize(plot_path)
                    print(f"📄 File verified: {plot_filename} ({file_size} bytes)")
                else:
                    print(f"❌ File not found after saving: {plot_path}")
                    
            except Exception as save_error:
                print(f"❌ Error saving to current directory: {save_error}")
            
            # Save to results directory
            if results_directory and os.path.exists(results_directory):
                results_plot_path = os.path.join(results_directory, plot_filename)
                print(f"📁 Attempting to save to results directory: {results_plot_path}")
                
                try:
                    plt.savefig(results_plot_path, dpi=300, bbox_inches='tight')
                    print(f"✅ Successfully saved to results directory: {results_plot_path}")
                    
                    # Verify file was created
                    if os.path.exists(results_plot_path):
                        file_size = os.path.getsize(results_plot_path)
                        print(f"📄 File verified in results: {plot_filename} ({file_size} bytes)")
                    else:
                        print(f"❌ File not found in results directory: {results_plot_path}")
                        
                except Exception as save_error:
                    print(f"❌ Error saving to results directory: {save_error}")
            else:
                print(f"❌ Results directory not accessible: {results_directory}")
            
            # Show the plot (commented out for server environment)
            # plt.show()
            
            print(f"📈 Plot details:")
            print(f"   Total iterations: {len(log_df)}")
            print(f"   Best Q value: {best_q_value:.6f} at iteration {best_iteration}")
            print(f"   Final Q value: {log_df['Q'].iloc[-1]:.6f}")
            print(f"   Q improvement: {log_df['Q'].iloc[0] - best_q_value:.6f}")
            
            # List files in current directory to verify
            try:
                current_files = [f for f in os.listdir(current_dir) if f.startswith('best_Q_') and f.endswith('.png')]
                if current_files:
                    print(f"🔍 Found Q plot files in current directory: {current_files}")
                else:
                    print(f"🔍 No Q plot files found in current directory")
            except Exception as list_error:
                print(f"❌ Error listing current directory: {list_error}")
            
            # Close the figure to free memory
            plt.close()
            
        else:
            print(f"❌ Optimization log not found: {optimization_log_path}")
            
    except Exception as e:
        print(f"❌ Error creating Q optimization plot: {e}")
        import traceback
        traceback.print_exc()


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
        explainer = DCESolver(model, data)
        
        # Create strategy
        strategy, strategy_alias = create_strategy(
            strategy_config, explainer, cone_config, global_config.get('verbose', False)
        )
        
        # Determine method name
        if cone_config['cone_cat'] and cone_config['cone_cont']:
            g_method = "cone_all"
            print("🧭 Full cone sampling mode (both categorical and continuous)")
        elif not cone_config['cone_cat'] and not cone_config['cone_cont']:
            g_method = "original_all"
            print("🎲 Full original mode (pure random sampling for both)")
        elif cone_config['cone_cat'] and not cone_config['cone_cont']:
            g_method = "cone_cat"
            print("🔀 Mixed mode: cone sampling for categorical, original for continuous")
        elif not cone_config['cone_cat'] and cone_config['cone_cont']:
            g_method = "cone_cont"
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
                use_global_ranges=dce_params.get('use_global_ranges', True),
                callback=global_config['callback'],
                save_results=global_config['save_results'],
                dataset_name=dataset_name,
                model_name=model_name,
                seed=seed,
                g_method=g_method,
                results_dir=global_config.get('results_dir'),
                save_iterations=dce_params.get('save_iterations', True)
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
                
                # Generate Q optimization plot
                generate_q_optimization_plot(results_directory, dataset_name, model_name, strategy_name)
                
                # Generate OT Distance optimization plot
                generate_ot_distance_plot(results_directory, dataset_name, model_name, strategy_name)
                
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
                            elif isinstance(v, (np.integer, np.floating, np.bool_)):
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
                core_metrics = ['MMD', 'OT_distance_X', 'OT_distance_Y', 'AReS Cost']
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
    """Run complete DCE experiments for all dataset-model-seed combinations in configuration."""
    print_config_summary(config)
    
    global_config = config["global"] 
    datasets = global_config["datasets"]
    models = global_config["models"]
    seeds = global_config["seeds"]
    
    # Results storage for all combinations
    all_combinations_results = {}
    
    print(f"\n{'='*80}")
    print(f"🎯 Multi-Dataset/Model/Seed Experiment Setup")
    print(f"{'='*80}")
    print(f"📊 Datasets to process: {len(datasets)} - {datasets}")
    print(f"🤖 Models to process: {len(models)} - {models}")
    print(f"🎲 Seeds to process: {len(seeds)} - {seeds}")
    print(f"📈 Strategies per combination: {len(config['strategies'])}")
    total_combinations = len(datasets) * len(models) * len(seeds)
    total_experiments = total_combinations * len(config['strategies'])
    print(f"📊 Total combinations: {total_combinations}")
    print(f"📊 Total experiments: {total_experiments}")
    
    combination_count = 0
    
    for dataset in datasets:
        for model in models:
            for i, seed in enumerate(seeds):
                combination_count += 1
                
                print(f"\n{'='*80}")
                print(f"🎯 COMBINATION {combination_count}/{total_combinations}")
                print(f"📊 Dataset: {dataset}")
                print(f"🤖 Model: {model}")
                print(f"🎲 Seed: {seed}")
                print(f"{'='*80}")
                
                # Create a temporary config for this specific combination
                temp_config = config.copy()
                temp_config["global"] = global_config.copy()
                temp_config["global"]["dataset"] = dataset
                temp_config["global"]["model"] = model
                temp_config["global"]["seed"] = seed
                temp_config["global"]["seeds"] = [seed]
                temp_config["global"]["datasets"] = [dataset]
                temp_config["global"]["models"] = [model]
                
                # Update experiment name to include combination info
                original_name = config.get('experiment_name', 'dce_experiment')
                temp_config["experiment_name"] = f"{original_name}_{dataset}_{model}_seed{seed}"
                
                combination_key = f"{dataset}_{model}_seed_{seed}"
                
                try:
                    # Run experiment for this combination
                    combination_results = run_single_seed_experiment(temp_config, seed)
                    all_combinations_results[combination_key] = {
                        'dataset': dataset,
                        'model': model,
                        'seed': seed,
                        'results': combination_results
                    }
                    print(f"✅ Combination {dataset}-{model}-{seed} completed successfully!")
                    
                except Exception as e:
                    print(f"❌ Error in combination {dataset}-{model}-{seed}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_combinations_results[combination_key] = {
                        'dataset': dataset,
                        'model': model,
                        'seed': seed,
                        'error': str(e)
                    }
    
    # Print multi-combination summary
    print(f"\n{'='*80}")
    print("📈 MULTI-DATASET/MODEL/SEED EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"🎯 Experiment: {config.get('experiment_name', 'Unnamed')}")
    print(f"📊 Datasets: {datasets}")
    print(f"🤖 Models: {models}")
    print(f"🎲 Seeds: {seeds}")
    print(f"📊 Total combinations processed: {combination_count}")
    
    # Summary statistics across all combinations
    successful_combinations = sum(1 for result in all_combinations_results.values() if 'error' not in result)
    failed_combinations = combination_count - successful_combinations
    
    print(f"✅ Successful combinations: {successful_combinations}/{combination_count}")
    if failed_combinations > 0:
        print(f"❌ Failed combinations: {failed_combinations}/{combination_count}")
    
    # Performance summary by dataset and model
    if successful_combinations > 0:
        print(f"\n📊 Performance Summary by Dataset:")
        for dataset in datasets:
            dataset_results = [r for k, r in all_combinations_results.items() if r.get('dataset') == dataset and 'error' not in r]
            print(f"   📊 {dataset}: {len(dataset_results)}/{len(models) * len(seeds)} combinations successful")
        
        print(f"\n🤖 Performance Summary by Model:")
        for model in models:
            model_results = [r for k, r in all_combinations_results.items() if r.get('model') == model and 'error' not in r]
            print(f"   🤖 {model}: {len(model_results)}/{len(datasets) * len(seeds)} combinations successful")
        
        print(f"\n📈 Strategy Performance Across All Combinations:")
        for strategy_config in config["strategies"]:
            alias = strategy_config["alias"]
            strategy_name = strategy_config["name"]
            
            # Collect metrics across all successful combinations
            best_qs = []
            feasible_counts = 0
            
            for combination_data in all_combinations_results.values():
                if 'error' not in combination_data and 'results' in combination_data:
                    if alias in combination_data['results']:
                        result = combination_data['results'][alias]
                        if 'error' not in result:
                            if 'best_Q' in result:
                                best_qs.append(result['best_Q'])
                            if result.get('found_feasible_solution', False):
                                feasible_counts += 1
            
            if best_qs:
                print(f"   🎯 {strategy_name.upper()} ({alias}):")
                print(f"     Best Q - Mean: {np.mean(best_qs):.6f}, Std: {np.std(best_qs):.6f}")
                print(f"     Feasible solutions: {feasible_counts}/{successful_combinations}")
    
    return all_combinations_results


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
            "top_k": 1,
            "use_global_ranges": True,
            "save_iterations": True
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
                    "temperature": 2.0,
                    "h": 2
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
                    "temperature": 2.0,
                    "h": 2
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
            "top_k": 1,
            "use_global_ranges": True,
            "save_iterations": True
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
                    "categorical_step": 1.0,
                    "h": 2
                }
            },
            {
                "name": "genetic",
                "alias": "ga_aggressive",
                "params": {
                    "crossover_prob": 0.9,
                    "mutation_prob_cat": 0.5,
                    "categorical_step": 1.8,
                    "h": 3
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
            "top_k": 1,
            "use_global_ranges": True,
            "save_iterations": True
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
                    "temperature": 2.0,
                    "h": 2
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
                    "temperature": 2.0,
                    "h": 2
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

    if not os.path.exists("example_configs"):
        print("📁 Creating example configuration files...")
        create_example_config()
        print()
    
    exit(main())
