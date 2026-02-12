"""
Helper functions to load DCE experiment results for baseline comparison.
This ensures baseline experiments use exactly the same model and samples as DCE.
"""

import os
import glob
import pickle
import pandas as pd
import torch
from pathlib import Path


def find_dce_result_dir(dataset_name, model_name, seed, strategy_name="MonteCarlo"):
    """
    Find the DCE result directory for a given dataset, model, and seed.

    Parameters:
    -----------
    dataset_name : str
        Dataset name (e.g., 'compas', 'heloc')
    model_name : str
        Model name (e.g., 'SVM', 'RandomForest')
    seed : int
        Random seed used in DCE experiment
    strategy_name : str, optional
        Strategy name to look for (default: 'MonteCarlo')

    Returns:
    --------
    str : Path to the result directory

    Raises:
    -------
    FileNotFoundError : If no matching directory is found
    """
    # Pattern to search for result directories
    pattern = f"DCE_Results/{dataset_name}/{model_name}/*/*/seed_{seed}_*/"

    print(f"🔍 Searching for DCE results:")
    print(f"   Pattern: {pattern}")

    # Find matching directories
    dirs = glob.glob(pattern)

    if not dirs:
        raise FileNotFoundError(
            f"No DCE results found for {dataset_name}/{model_name}/seed_{seed}\n"
            f"Please run DCE experiment first with:\n"
            f"  python run_dce_experiments.py --config config_seed42_for_baseline.json"
        )

    # Filter by strategy if multiple results exist
    if len(dirs) > 1 and strategy_name:
        strategy_dirs = [d for d in dirs if strategy_name in d]
        if strategy_dirs:
            dirs = strategy_dirs

    # Use the first matching directory
    result_dir = dirs[0]
    print(f"✅ Found DCE result directory:")
    print(f"   {result_dir}")

    return result_dir


def load_dce_model(result_dir):
    """
    Load the trained model from DCE experiment.

    Parameters:
    -----------
    result_dir : str
        Path to DCE result directory

    Returns:
    --------
    model : The loaded model object
    """
    model_path = os.path.join(result_dir, "model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"📦 Loading model from: {os.path.basename(result_dir)}/model.pkl")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"✅ Model loaded successfully")

    return model


def load_dce_sample_indices(result_dir):
    """
    Load the sample indices used in DCE experiment.

    Parameters:
    -----------
    result_dir : str
        Path to DCE result directory

    Returns:
    --------
    pd.Index : The indices of samples used in DCE experiment
    """
    x_true_path = os.path.join(result_dir, "x_true.csv")

    if not os.path.exists(x_true_path):
        raise FileNotFoundError(f"x_true.csv not found: {x_true_path}")

    print(f"📊 Loading sample indices from: x_true.csv")

    # Load x_true to get the indices
    x_true = pd.read_csv(x_true_path)
    indice = x_true.index

    print(f"✅ Loaded {len(indice)} sample indices")

    return indice


def load_dce_experiment_data(dataset_name, model_name, seed, strategy_name="MonteCarlo"):
    """
    Load both model and sample indices from DCE experiment.
    This is the main function to use in baseline experiments.

    Parameters:
    -----------
    dataset_name : str
        Dataset name (e.g., 'compas', 'heloc')
    model_name : str
        Model name (e.g., 'SVM', 'RandomForest')
    seed : int
        Random seed used in DCE experiment
    strategy_name : str, optional
        Strategy name to look for (default: 'MonteCarlo')

    Returns:
    --------
    tuple : (model, indice, result_dir)
        - model: The trained model
        - indice: pd.Index of sample indices
        - result_dir: Path to result directory (for reference)

    Example:
    --------
    >>> model, indice, result_dir = load_dce_experiment_data('compas', 'SVM', 42)
    >>> print(f"Using {len(indice)} samples from {result_dir}")
    """
    print("\n" + "="*80)
    print(f"🔄 Loading DCE experiment data for baseline comparison")
    print("="*80)
    print(f"📋 Dataset: {dataset_name}")
    print(f"🤖 Model: {model_name}")
    print(f"🎲 Seed: {seed}")
    print(f"🎯 Strategy: {strategy_name}")
    print("="*80 + "\n")

    # Find result directory
    result_dir = find_dce_result_dir(dataset_name, model_name, seed, strategy_name)

    # Load model
    model = load_dce_model(result_dir)

    # Load sample indices
    indice = load_dce_sample_indices(result_dir)

    print("\n" + "="*80)
    print("✅ DCE experiment data loaded successfully!")
    print("="*80)
    print(f"📦 Model: Loaded from DCE experiment")
    print(f"📊 Samples: {len(indice)} indices")
    print(f"📁 Source: {result_dir}")
    print("="*80 + "\n")

    return model, indice, result_dir


def get_dce_metadata(result_dir):
    """
    Load metadata from DCE experiment (optional, for verification).

    Parameters:
    -----------
    result_dir : str
        Path to DCE result directory

    Returns:
    --------
    dict : Metadata dictionary
    """
    metadata_path = os.path.join(result_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        return {}

    import json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata


if __name__ == "__main__":
    # Test the functions
    print("Testing load_dce_experiment_data...")

    try:
        model, indice, result_dir = load_dce_experiment_data('compas', 'SVM', 42)
        print(f"\n✅ Test passed!")
        print(f"   Model type: {type(model)}")
        print(f"   Number of samples: {len(indice)}")
        print(f"   Result directory: {result_dir}")

        # Try loading metadata
        metadata = get_dce_metadata(result_dir)
        if metadata:
            print(f"\n📋 Metadata:")
            for key, value in metadata.items():
                print(f"   {key}: {value}")

    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("\nThis is expected if DCE experiment hasn't been run yet.")
