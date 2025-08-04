import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

class CardioData:
    def __init__(self, test_size=0.2, seed=42, sample_num=50):
        self.name = "cardio"
        self.test_size = test_size
        self.random_state = seed
        self.sample_num = sample_num

        self.target_name = 'cardio'
        self.features = [
            'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active'
        ]
        self.categorical_columns = [
            'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'
        ]
        self.continuous_columns = [
            'age', 'height', 'weight', 'ap_hi', 'ap_lo'
        ]
        self.explain_columns = self.features.copy()
        self.feature_names = self.features.copy()
        self._load_data()
        self._preprocess()
        self._split()
        self._standardize()

    def _load_data(self):
        self.df_raw = pd.read_csv("data/cardio/cardio.csv", sep=';')
        self.df = self.df_raw.drop(columns=['id'], axis=1).copy()

    def _preprocess(self):
        self.df['cardio'] = self.df['cardio'].astype(int)
        for column in self.df.columns:
            if self.df[column].isna().any():
                self.df[column].fillna(self.df[column].median(), inplace=True)
        self.X = self.df[self.features].copy()
        self.y = self.df[self.target_name]

    def _split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
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
        
        self.df_explain = self.X_test.sample(self.sample_num, random_state=self.random_state)
        return self.df_explain.copy()

    def get_y_target(self):
        torch.manual_seed(self.random_state)
        y_target = torch.distributions.Beta(0.2, 0.3).sample((self.sample_num,)).to(torch.float32)
        return y_target

    def get_X_init(self):
        X = torch.tensor(self.df_explain.values, dtype=torch.float32)
        explain_indices = [self.df_explain.columns.get_loc(col) for col in self.explain_columns]
        
        torch.manual_seed(self.random_state)
        noise = torch.randn_like(X[:, explain_indices]) * 0.01
        X_init = X.clone()
        X_init[:, explain_indices] += noise
        return X_init

    def get_y_true(self):
        return self.y_test.loc[self.df_explain.index]
