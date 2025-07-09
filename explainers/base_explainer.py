from abc import ABC, abstractmethod
import pandas as pd

class BaseExplainer(ABC):
    def __init__(self, data, model):
        self.data = data  
        self.model = model  

    @abstractmethod
    def explain(self, df_factual: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters:
            df_factual: DataFrame, factual data

        Returns
            DataFrame, counterfactual results
        """
        pass
