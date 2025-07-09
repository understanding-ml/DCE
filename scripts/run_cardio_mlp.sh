#!/bin/bash
# Unified script to run Cardio MLP experiments with different U values

echo "Running Cardio MLP experiments..."

# Run experiments with different U values
python experiments/cardio_mlp_unified.py --u_value 0.05 --output_dir data/cardio/mlp/U_005/
python experiments/cardio_mlp_unified.py --u_value 0.10 --output_dir data/cardio/mlp/U_010/
python experiments/cardio_mlp_unified.py --u_value 0.20 --output_dir data/cardio/mlp/U_020/
python experiments/cardio_mlp_unified.py --u_value 0.40 --output_dir data/cardio/mlp/U_040/

echo "All Cardio MLP experiments completed."