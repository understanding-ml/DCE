import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class HelocData:
    def __init__(self, test_size=0.2, seed=42, sample_num=50):
        self.name = "heloc"
        self.seed = seed
        self.test_size = test_size
        self.random_state = seed
        self.sample_num = sample_num

        self.target_name = 'RiskPerformance'
        self.features = [
            'ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen', 
            'AverageMInFile', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 
            'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq', 
            'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M', 
            'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M', 
            'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden', 
            'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization', 
            'PercentTradesWBalance'
        ]
        
        self.categorical_columns = [
            'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTrades60Ever2DerogPubRec', 
            'NumTrades90Ever2DerogPubRec', 'NumInqLast6M', 'NumInqLast6Mexcl7days',
            'NumBank2NatlTradesWHighUtilization'
        ]
        
        self.continuous_columns = [
            'ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen', 
            'AverageMInFile', 'NumSatisfactoryTrades', 'PercentTradesNeverDelq', 
            'MSinceMostRecentDelq', 'NumTotalTrades', 'NumTradesOpeninLast12M', 
            'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 'NetFractionRevolvingBurden', 
            'NetFractionInstallBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 
            'PercentTradesWBalance'
        ]
        
        self.explain_columns = self.features.copy()
        self.feature_names = self.features.copy()
        self._load_data()
        self._preprocess()
        self._split()
        self._standardize()
        self._compute_global_ranges()

    def _load_data(self):
        self.df_raw = pd.read_csv("data/heloc/heloc.csv")
        self.df = self.df_raw.copy()

    def _preprocess(self):

        self.df['RiskPerformance'] = self.df['RiskPerformance'].map({'Bad': 0, 'Good': 1}).astype(int)
        
        # Handle missing values represented as -7, -8, -9
        # Replace with NaN first, then handle
        self.df = self.df.replace([-7, -8, -9], np.nan)
        
        for column in self.continuous_columns:
            if column in self.df.columns and self.df[column].isna().any():
                self.df[column] = self.df[column].fillna(self.df[column].median())
        
        for column in self.categorical_columns:
            if column in self.df.columns and self.df[column].isna().any():
                self.df[column] = self.df[column].fillna(self.df[column].mode()[0])

        self.X = self.df[self.features].copy()
        self.y = self.df[self.target_name]

    def _split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.seed
        )

    def _standardize(self):
        # Save original unstandardized data for computing global ranges
        self.X_raw = self.X.copy()
        
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

        # y_target = torch.distributions.Beta(0.1, 0.9).sample((self.sample_num,)).to(torch.float32)
        y_target = torch.full((self.sample_num,), 1.0).to(torch.float32)  
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

    def _compute_global_ranges(self):
        """Compute global feature range information for sampling"""
        raw_data = self.X_raw

        # Global ranges for continuous features 
        self.continuous_ranges = {}
        for col in self.continuous_columns:
            if col in raw_data.columns:
                raw_min = float(raw_data[col].min())
                raw_max = float(raw_data[col].max())

                if col in self.mean.index and col in self.std.index:
                    mean_val = float(self.mean[col])
                    std_val = float(self.std[col])
                    if std_val > 0:  
                        standardized_min = (raw_min - mean_val) / std_val
                        standardized_max = (raw_max - mean_val) / std_val
                    else:
                        standardized_min = 0.0
                        standardized_max = 0.0
                else:
                    standardized_min = raw_min
                    standardized_max = raw_max

                self.continuous_ranges[col] = {
                    'min': standardized_min,
                    'max': standardized_max
                }

        # Global possible values for categorical features 
        self.categorical_ranges = {}
        for col in self.categorical_columns:
            if col in raw_data.columns:
                unique_vals = raw_data[col].unique()
                standardized_vals = []
                for val in unique_vals:
                    if col in self.mean.index and col in self.std.index:
                        mean_val = float(self.mean[col])
                        std_val = float(self.std[col])
                        if std_val > 0:
                            standardized_val = (float(val) - mean_val) / std_val
                        else:
                            standardized_val = float(val)
                    else:
                        standardized_val = float(val)
                    standardized_vals.append(standardized_val)
                self.categorical_ranges[col] = standardized_vals