import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class CompasData:
    def __init__(self, test_size=0.2, seed=42, sample_num=50):
        self.name = "compas"
        self.seed = seed 
        self.test_size = test_size
        self.random_state = seed
        self.sample_num = sample_num

        self.target_name = 'two_year_recid'
        self.features = [
            'age', 'sex', 'race', 'priors_count', 'c_charge_degree', 'decile_score'
        ]
        self.categorical_columns = [
            'sex', 'race', 'c_charge_degree'
        ]
        self.continuous_columns = [
            'age', 'priors_count', 'decile_score'
        ]
        self.explain_columns = self.features.copy()

        self._load_data()
        self._preprocess()
        self._split()
        self._standardize()

    def _load_data(self):
        self.df_raw = pd.read_csv("data/compas/compas.csv")
        self.df = self.df_raw.copy()

    def _preprocess(self):
        # Select relevant columns
        relevant_cols = self.features + [self.target_name]
        self.df = self.df[relevant_cols].copy()
        
        # Handle missing values
        self.df = self.df.dropna()
        
        # Encode categorical variables
        label_encoder = LabelEncoder()
        self.label_mappings = {}

        for column in self.categorical_columns:
            if column in self.df.columns:
                self.df[column] = self.df[column].astype(str)
                self.df[column] = label_encoder.fit_transform(self.df[column])
                self.label_mappings[column] = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

        # Fill any remaining missing values with median for continuous columns
        for column in self.continuous_columns:
            if column in self.df.columns and self.df[column].isna().any():
                self.df[column].fillna(self.df[column].median(), inplace=True)

        # Ensure target is binary (0 or 1)
        self.df[self.target_name] = self.df[self.target_name].astype(int)

        self.X = self.df[self.features].copy()
        self.y = self.df[self.target_name]

    def _split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.seed
        )

    def _standardize(self):
        self.mean = self.X_train.mean()
        self.std = self.X_train.std()
        self.X_train = (self.X_train - self.mean) / self.std
        self.X_test = (self.X_test - self.mean) / self.std

    def get_train_test(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_features(self):
        return self.features

    def get_explain_columns(self):
        return self.explain_columns, self.categorical_columns, self.continuous_columns

    def get_df_explain(self, sample_num=None):
        if sample_num is not None:
            self.sample_num = sample_num
        
        self.df_explain = self.X_test.sample(self.sample_num, random_state=self.seed)
        return self.df_explain.copy()

    def get_y_target(self):
        torch.manual_seed(self.seed)
        y_target = torch.distributions.Beta(0.1, 0.9).sample((self.sample_num,)).to(torch.float32)
        return y_target

    def get_X_init(self):
        X = torch.tensor(self.df_explain.values, dtype=torch.float32)
        explain_indices = [self.df_explain.columns.get_loc(col) for col in self.explain_columns]
        
        torch.manual_seed(self.seed)
        noise = torch.randn_like(X[:, explain_indices]) * 0.01
        X_init = X.clone()
        X_init[:, explain_indices] += noise
        return X_init

    def get_y_true(self):
        return self.y_test.loc[self.df_explain.index]