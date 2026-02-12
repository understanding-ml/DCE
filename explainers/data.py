import torch
import pandas as pd
from typing import List


class Data:
    def __init__(self, df: pd.DataFrame, explain_columns: List[str], 
                 categorical_columns: List[str], continuous_columns: List[str],
                 costs_vector: List[float] = None,
                 continuous_ranges: dict = None, categorical_ranges: dict = None):
        self.df = df.reset_index(drop=True)
        self.explain_columns = explain_columns
        self.categorical_columns = categorical_columns
        self.continuous_columns = continuous_columns

        self.explain_indices = [self.df.columns.get_loc(c) for c in explain_columns]
        self.categorical_indices = [self.df.columns.get_loc(c) for c in categorical_columns]
        self.continuous_indices = [self.df.columns.get_loc(c) for c in continuous_columns]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if costs_vector is None:
            self.costs_vector = torch.ones(len(self.explain_indices)).float()
        else:
            self.costs_vector = torch.tensor(costs_vector).float()
        self.costs_vector_reshaped = self.costs_vector.reshape(1, -1).to(self.device)
        
        # Global range information for sampling
        self.continuous_ranges = continuous_ranges or {}
        self.categorical_ranges = categorical_ranges or {}

        self.continuous_ranges_by_index = {}
        self.categorical_ranges_by_index = {}
        
        for col in continuous_columns:
            if col in self.continuous_ranges:
                col_idx = self.df.columns.get_loc(col)
                self.continuous_ranges_by_index[col_idx] = self.continuous_ranges[col]
        
        for col in categorical_columns:
            if col in self.categorical_ranges:
                col_idx = self.df.columns.get_loc(col)
                self.categorical_ranges_by_index[col_idx] = self.categorical_ranges[col]

    def sample(self, n, random_state=None):
        if random_state is None:
            random_state = getattr(self, 'random_state', 42)

        cache_key = (n, random_state)
        
        if hasattr(self, "_df_explain_cache") and self._cache_key == cache_key:
            return self._df_explain_cache.copy()
        
        df_explain = self.df.sample(n, random_state=random_state).reset_index(drop=True)
        self._df_explain_cache = df_explain
        self._cache_key = cache_key
        return df_explain.copy()

    def to_tensor(self, df):

        return torch.tensor(df.values, dtype=torch.float32).to(self.device)