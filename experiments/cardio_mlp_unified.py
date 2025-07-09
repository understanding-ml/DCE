#!/usr/bin/env python3
"""
Unified experiment script for Cardio MLP with different U values.
Usage: python cardio_mlp_unified.py --u_value 0.05 --output_dir data/cardio/mlp/U_005/
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from models.mlp import BlackBoxModel
from models.svm import LinearSVM
from models.lr import LogisticRegression
from models.rbf import RBFNet
import pickle
import os
from explainers.dce import DistributionalCounterfactualExplainer
from explainers.distances import bootstrap_1d, bootstrap_sw
from utils.logger_config import setup_logger
from utils.data_processing import *
from experiments.input_index import y_target


def main():
    parser = argparse.ArgumentParser(description='Run Cardio MLP experiments with different U values')
    parser.add_argument('--u_value', type=float, default=0.05, help='U value for the experiment')
    parser.add_argument('--output_dir', type=str, default='data/cardio/mlp/U_005/', help='Output directory')
    parser.add_argument('--data_path', type=str, default='data/cardio', help='Data path')
    parser.add_argument('--csv_file', type=str, default='cardio.csv', help='CSV filename')
    
    args = parser.parse_args()
    
    logger = setup_logger()
    
    # Use current working directory as base path
    base_path = os.getcwd()
    read_data_path = os.path.join(base_path, args.data_path)
    dump_data_path = os.path.join(base_path, args.output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(dump_data_path, exist_ok=True)
    
    logger.info(f"Starting Cardio MLP experiment with U={args.u_value}")
    logger.info(f"Reading data from: {read_data_path}")
    logger.info(f"Output directory: {dump_data_path}")
    
    # Load and process data
    df = pd.read_csv(os.path.join(read_data_path, args.csv_file))
    
    # Add your experiment logic here
    # This is a template - you'll need to add the actual experiment code from the original files
    
    logger.info("Experiment completed successfully")


if __name__ == "__main__":
    main()