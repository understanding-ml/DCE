import pandas as pd
import numpy as np

from .base_model import BaseModel

class SklearnModel(BaseModel):
    """Classwrapper for pre-trained models."""
    def __init__(self, model, backend):
        super().__init__(model, backend)
        """Initialize the Model class
            
        :param model: Sklearn model
        :param backend: 'sklearn'
        """
        # ✅ 保存列名用于后续DataFrame恢复
        if hasattr(model, "feature_names_in_"):
            self.feature_names = list(model.feature_names_in_)
        else:
            self.feature_names = None  # fallback: 使用外部传入
            
    def _ensure_dataframe(self, X):
        """
        如果传入的是 ndarray，则转成 DataFrame 并补充列名（避免 warning）
        """
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, np.ndarray):
            if self.feature_names:
                return pd.DataFrame(X, columns=self.feature_names)
            else:
                return pd.DataFrame(X)  # fallback
        raise ValueError("Unsupported input type")

    def predict(self, x_factual):
        """
        Generate predictions using the pre-trained model
        
        Returns:
        Numpy type prediction results
        """
        # if isinstance(x_factual, pd.DataFrame):
        #     x_factual = x_factual.values
        if isinstance(x_factual, np.ndarray):
            x_factual = pd.DataFrame(x_factual, columns=self.model.feature_names_in_)  
        return self.model.predict(x_factual)
    
    def predict_proba(self, X):
        """
        Predict probability function that returns the probability distribution for each class.

        Args:
        X (numpy.ndarray or pandas.DataFrame): Input data for which to predict probabilities.

        Returns:
        numpy.ndarray: Array of shape (n_samples, n_classes) containing the predicted probabilities
                    for each class. Each row corresponds to a sample, and each column corresponds
                    to a class.
        """
        # if X is DataFrame，transfer to np.array
        # if isinstance(X, pd.DataFrame):
        #     X = X.values
        # # get the probability of each class for each sample
        # probabilities = self.model.predict_proba(X)

        # # check if it is a 1D array (usually occurs in binary classification, only return the probability of the positive class)        
        # if probabilities.ndim == 1:
        #     # expand to a 2D array, column 0 is the probability of class 0, column 1 is the probability of class 1
        #     probabilities = np.vstack([1 - probabilities, probabilities]).T
        # return probabilities
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.model.feature_names_in_)
        """
        classwrapper.py use this return (But I dont know why)
        """
        return self.model.predict_proba(X)[:, 1]
