import numpy as np
import torch
from typing import Union
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from models.svm import LinearSVM, svm_loss
from models.lr import LogisticRegression
from models.mlp import BlackBoxModel
from models.rbf import RBFNet


class Model:
    def __init__(self, model=None, backend: str = "auto", model_name=None, X_train=None, y_train=None, data=None):
        self.model = model
        self.backend = self._infer_backend(backend)
        self.data = data

    def _infer_backend(self, backend):
        if backend != "auto":
            return backend.lower()
        cls = str(type(self.model)).lower()
        if "torch" in cls:
            return "pytorch"
        elif "keras" in cls or "tensorflow" in cls:
            return "keras"
        elif "xgboost" in cls:
            return "xgboost"
        elif "lightgbm" in cls:
            return "lightgbm"
        else:
            return "sklearn"

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.backend == "pytorch":
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                y = self.model(X_tensor)
                return y.cpu().numpy()

        elif self.backend == "keras":
            return self.model.predict(X, verbose=0)

        elif self.backend in ["xgboost", "lightgbm", "sklearn"]:
            X = self._ensure_dataframe(X)
            return self.model.predict_proba(X)

        else:
            raise NotImplementedError(f"Unsupported backend: {self.backend}")

    def __call__(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(X, np.ndarray):
            X_input = torch.FloatTensor(X)
        else:
            X_input = X

        if self.backend == "pytorch" or self.backend == "PYT":
            if hasattr(self.model, "eval"):
                self.model.eval()
            if not X_input.requires_grad:
                with torch.no_grad():
                    output = self.model(X_input).float()
                    # Ensure output is 1D for consistency with sklearn models
                    if output.ndim == 2 and output.shape[1] == 1:
                        output = output.squeeze(-1)
                    return output
            else:
                output = self.model(X_input).float()
                # Ensure output is 1D for consistency with sklearn models
                if output.ndim == 2 and output.shape[1] == 1:
                    output = output.squeeze(-1)
                return output

        elif self.backend == "keras":
            X_np = X_input.detach().cpu().numpy() if torch.is_tensor(X_input) else X_input
            preds = self.model.predict(X_np, verbose=0)
            return torch.from_numpy(preds).float()

        elif self.backend in ["xgboost", "lightgbm", "sklearn"]:
            X_np = X_input.detach().cpu().numpy() if torch.is_tensor(X_input) else X_input
            X_df = self._ensure_dataframe(X_np)
            preds = self.model.predict_proba(X_df)
            # Extract positive class probability and ensure 1D output
            if preds.ndim == 2:
                if preds.shape[1] > 1:
                    preds = preds[:, 1]  # Binary classification: take positive class
                else:
                    preds = preds.squeeze(-1)  # Single output: squeeze to 1D
            return torch.from_numpy(preds).float()

        else:
            raise NotImplementedError(f"Unsupported backend: {self.backend}")

    def _ensure_dataframe(self, X: np.ndarray) -> pd.DataFrame:
        if hasattr(self.model, "feature_names_in_"):
            return pd.DataFrame(X, columns=self.model.feature_names_in_)
        return pd.DataFrame(X)

    def to(self, device):
        if self.backend == "pytorch" and hasattr(self.model, "to"):
            self.model.to(device)
        return self
