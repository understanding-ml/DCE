# Extending Distributional Counterfactual Explanations to Non-Differentiable Models  
@(layout=centered)

**Yikai Gu**  
s232263  
Technical University of Denmark  

---

## What is XAI?

- **Explainable AI (XAI)** aims to make machine learning decisions interpretable.
- Critical in domains like:
  - 🏥 Healthcare
  - 💰 Finance
  - ⚖️ Policy & Law
- Two major categories:
  - **Local explanations** (e.g., LIME, SHAP): instance-level insight
  - **Global explanations**: explain overall model behavior

===

> 🤔 XAI answers:  
> *“Why did the model make this decision?”*

---

## Counterfactual Explanations and DCE

- **Counterfactual Explanation (CE)**:
  - Finds minimal changes to input to flip the model’s output
  - Usually focused on **one instance**
  - Typically assumes **gradient-based optimization**

===

- **Distributional Counterfactual Explanation (DCE)**:
  - Generates a **distribution** of counterfactuals
  - Uses **Optimal Transport** to:
    - Match target output distribution
    - Maintain feature similarity (Wasserstein distance)

---

## Limitations of Original DCE

- ❌ Only works with **differentiable models**
- ⚙️ Optimization based solely on **gradient descent**
- 🔗 Tightly coupled to **PyTorch**

===

- 🚫 Cannot handle:
  - Random Forest
  - XGBoost / LightGBM
- ⚠️ Not suitable for many real-world ML pipelines

