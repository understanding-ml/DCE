
# Distributional Counterfactual Explanation (DCE)

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Paper](https://img.shields.io/badge/arXiv-2401.13112-B31B1B.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)

This project implements **Distributional Counterfactual Explanation** (DCE) using **Optimal Transport**, with extended support for **non-differentiable models** (e.g., Random Forest, XGBoost, LightGBM).

> 📄 Based on: [You et al., 2024. Distributional Counterfactual Explanations With Optimal Transport](https://arxiv.org/pdf/2401.13112)

## 🧠 Motivation

Traditional counterfactual explanations find one alternative instance per sample. DCE generalizes this by optimizing over **counterfactual distributions**, aiming to match a target output distribution while maintaining similarity in feature space using **Wasserstein distances**.

## ✨ Features

- 🤖 **Non-differentiable model support**:
  - PyTorch models (MLP, RBF, SVM)
  - Scikit-learn models (Random Forest, etc.)
  - XGBoost / LightGBM
- 🎯 **Multiple optimization strategies**:
  - Monte Carlo
  - Simulated Annealing
  - Genetic Algorithm
  - Differential Evolution
  - Covariance Matrix Adaptation Evolution Strategy
  - Particle Swarm Optimization
  - Bayesian Optimization
- 📊 **Benchmark datasets**:
  - German Credit
  - Cardiovascular Disease
  - Hotel Booking

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from dataset.german_credit import GermanCreditData
from explainers.model import Model
from explainers.nondifferentiable import DCENonDifferentiable
from explainers.cone_sampling.monte_carlo import MonteCarloStrategy

# Load dataset
data = GermanCreditData()
df_explain = data.get_df_explain(sample_num=50)

# Initialize model
model = Model(model_name='random_forest', data=data)

# Initialize explainer and strategy
explainer = DCENonDifferentiable(model=model, data=data)
strategy = MonteCarloStrategy(explainer)

# Run explanation
df_cf = explainer.explain(
    df_factual=df_explain,
    strategy=strategy,
    max_iter=10
)
```

### Running Experiments

```bash
# Run unified experiments
python experiments/cardio_mlp_unified.py --u_value 0.05 --output_dir results/cardio/

# Or use the convenience script
bash scripts/run_cardio_mlp.sh
```

## 📁 Project Structure

```
DCE/
├── explainers/          # Core explanation algorithms
│   ├── dce.py          # Main DCE implementation
│   ├── nondifferentiable.py  # Non-differentiable model support
│   ├── model.py        # Unified model interface
│   └── strategies/     # Optimization strategies
├── models/             # Machine learning models
├── dataset/            # Data preprocessing
├── experiments/        # Experiment scripts
├── utils/             # Utility functions
├── demo.ipynb         # Main demonstration
└── results/           # Experiment results
```

## 📊 Experiments & Results

The project includes comprehensive experiments on three datasets:

- **Cardiovascular Disease**: `analysis_cardio.ipynb`
- **German Credit**: `analysis_german_credit.ipynb`
- **Hotel Booking**: `analysis_hotel_booking.ipynb`

Results are saved in the `results/` directory with visualization plots.

## 🛠️ Development

### Running Tests
```bash
pytest
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 Citation

```bibtex
@article{you2024distributional,
  title={Distributional Counterfactual Explanations With Optimal Transport},
  author={You, Lei and Bian, Yijun and Cao, Lele},
  journal={arXiv preprint arXiv:2401.13112},
  year={2024}
}
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📬 Contact

- Lei You: leiyo@dtu.dk

For questions or suggestions, please open an issue on GitHub.