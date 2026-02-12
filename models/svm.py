import torch.nn as nn
import torch
import numpy as np

class LinearSVM(nn.Module):
    """Linear SVM Classifier"""

    def __init__(self, input_dim):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

        self.name = "svm"

    def forward(self, x):
        out = self.fc(x)
        # Check for NaN in the output and replace with a default value (e.g., 0)
        if torch.isnan(out).any():
            # Handling NaN values - can choose to set to a specific value or handle differently
            out = torch.where(torch.isnan(out), torch.zeros_like(out), out)
        return out

    def predict(self, x):
        x = torch.FloatTensor(x)
        return (self(x).reshape(-1) > 0.5).float().detach().numpy()
    
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x)
            probs = self(x_t).reshape(-1, 1).cpu().numpy()
            return np.concatenate([1 - probs, probs], axis=1)


def svm_loss(outputs, labels):
    """Hinge loss for SVM"""
    return torch.mean(torch.clamp(1 - outputs.t() * labels, min=0))
