import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class CompasData:
    def __init__(self, test_size=0.2, seed=42, sample_num=50):
        self.name = "compas"
        self.seed = seed 
        self.test_size = test_size
        self.random_state = seed
        self.sample_num = sample_num

        self.target_name = 'Status'
        self.features = [
            'Sex = Female', 'Sex = Male', 'Age_Cat = Less than 25', 'Age_Cat = 25 - 45', 
            'Age_Cat = Greater than 45', 'Race = African-American', 'Race = Asian', 
            'Race = Caucasian', 'Race = Hispanic', 'Race = Native American', 'Race = Other',
            'C_Charge_Degree = F', 'C_Charge_Degree = M', 'Priors_Count', 'Time_Served'
        ]
        self.categorical_columns = [
            'Sex = Female', 'Sex = Male', 'Age_Cat = Less than 25', 'Age_Cat = 25 - 45', 
            'Age_Cat = Greater than 45', 'Race = African-American', 'Race = Asian', 
            'Race = Caucasian', 'Race = Hispanic', 'Race = Native American', 'Race = Other',
            'C_Charge_Degree = F', 'C_Charge_Degree = M'
        ]
        self.continuous_columns = ['Priors_Count', 'Time_Served']
        self.explain_columns = self.features.copy()
        self.feature_names = self.features.copy()
        self._load_data()
        self._preprocess()
        self._split()
        self._standardize()
        self._compute_global_ranges()

    def _load_data(self):
        self.df_raw = pd.read_csv("data/compas/compas.csv")
        self.df = self.df_raw.copy()

    def _preprocess(self):

        self.df = self.df.dropna(subset=['days_b_screening_arrest'])
        
        dates_in = pd.to_datetime(self.df['c_jail_in'], errors='coerce')
        dates_out = pd.to_datetime(self.df['c_jail_out'], errors='coerce')
        time_served = (dates_out - dates_in).dt.days
        time_served = time_served.fillna(0)
        time_served[time_served < 0] = 0
        self.df['time_served'] = time_served
        
        idx = (self.df['days_b_screening_arrest'] <= 30) & (self.df['days_b_screening_arrest'] >= -30)
        idx = idx & (self.df['is_recid'] != -1)
        idx = idx & (self.df['c_charge_degree'] != 'O')
        self.df = self.df[idx].copy()

        processed_data = []
        

        for _, row in self.df.iterrows():
            processed_row = {}
            
            # Sex (one-hot encoded)
            processed_row['Sex = Female'] = 1 if row['sex'] == 'Female' else 0
            processed_row['Sex = Male'] = 1 if row['sex'] == 'Male' else 0
            
            # Age Category (one-hot encoded)
            processed_row['Age_Cat = Less than 25'] = 1 if row['age_cat'] == 'Less than 25' else 0
            processed_row['Age_Cat = 25 - 45'] = 1 if row['age_cat'] == '25 - 45' else 0
            processed_row['Age_Cat = Greater than 45'] = 1 if row['age_cat'] == 'Greater than 45' else 0
            
            # Race (one-hot encoded)
            processed_row['Race = African-American'] = 1 if row['race'] == 'African-American' else 0
            processed_row['Race = Asian'] = 1 if row['race'] == 'Asian' else 0
            processed_row['Race = Caucasian'] = 1 if row['race'] == 'Caucasian' else 0
            processed_row['Race = Hispanic'] = 1 if row['race'] == 'Hispanic' else 0
            processed_row['Race = Native American'] = 1 if row['race'] == 'Native American' else 0
            processed_row['Race = Other'] = 1 if row['race'] == 'Other' else 0
            
            # Charge Degree (one-hot encoded)
            processed_row['C_Charge_Degree = F'] = 1 if row['c_charge_degree'] == 'F' else 0
            processed_row['C_Charge_Degree = M'] = 1 if row['c_charge_degree'] == 'M' else 0
            
            # Continuous variables (ensure numeric)
            processed_row['Priors_Count'] = float(row['priors_count']) if pd.notna(row['priors_count']) else 0.0
            processed_row['Time_Served'] = float(time_served.loc[row.name]) if pd.notna(time_served.loc[row.name]) else 0.0
            
            # Target (flip to match baseline: 1 - original, ensure numeric)
            recid_val = float(row['two_year_recid']) if pd.notna(row['two_year_recid']) else 0.0
            processed_row['Status'] = 1.0 - recid_val
            
            processed_data.append(processed_row)
        
        self.df = pd.DataFrame(processed_data)
        
        for col in self.features:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype(float)
        
        self.df[self.target_name] = pd.to_numeric(self.df[self.target_name], errors='coerce').astype(float)
        
        self.df = self.df.dropna()
        
        self.X = self.df[self.features].copy()
        self.y = self.df[self.target_name]
        
        self.features_tree = {}
        
        self.features_tree['Sex'] = ['Sex = Female', 'Sex = Male']
        self.features_tree['Age_Cat'] = ['Age_Cat = Less than 25', 'Age_Cat = 25 - 45', 'Age_Cat = Greater than 45']
        self.features_tree['Race'] = ['Race = African-American', 'Race = Asian', 'Race = Caucasian', 
                                     'Race = Hispanic', 'Race = Native American', 'Race = Other']
        self.features_tree['C_Charge_Degree'] = ['C_Charge_Degree = F', 'C_Charge_Degree = M']
        self.features_tree['Priors_Count'] = [] 
        self.features_tree['Time_Served'] = []   
        
        self.categorical_features = {
            self.name: ['Sex', 'Age_Cat', 'Race', 'C_Charge_Degree']
        }
        
        self.continuous_features = {
            self.name: ['Priors_Count', 'Time_Served']
        }
        
        self.features = self.features + [self.target_name]

    def _split(self):
        X_reset = self.X.reset_index(drop=True)
        y_reset = self.y.reset_index(drop=True)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_reset, y_reset, test_size=self.test_size, random_state=self.seed
        )
        
        self.train_indices = self.X_train.index
        self.test_indices = self.X_test.index

    def _standardize(self):
        self.mean = self.X_train.mean()
        self.std = self.X_train.std()
        self.X_train = (self.X_train - self.mean) / self.std
        self.X_test = (self.X_test - self.mean) / self.std
        

        self.X_raw = self.X.reset_index(drop=True)  
        self.data_raw = self.df.reset_index(drop=True) 
        
        self.data = self.data_raw

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
        if TORCH_AVAILABLE:
            torch.manual_seed(self.seed)
            y_target = torch.distributions.Beta(0.1, 0.9).sample((self.sample_num,)).to(torch.float32)
            return y_target
        else:
            np.random.seed(self.seed)
            y_target = np.random.beta(0.1, 0.9, self.sample_num).astype(np.float32)
            return y_target

    def get_X_init(self):
        if TORCH_AVAILABLE:
            X = torch.tensor(self.df_explain.values, dtype=torch.float32)
            explain_indices = [self.df_explain.columns.get_loc(col) for col in self.explain_columns]
            
            torch.manual_seed(self.seed)
            noise = torch.randn_like(X[:, explain_indices]) * 0.01
            X_init = X.clone()
            X_init[:, explain_indices] += noise
            return X_init
        else:
            X = self.df_explain.values.astype(np.float32)
            explain_indices = [self.df_explain.columns.get_loc(col) for col in self.explain_columns]
            
            np.random.seed(self.seed)
            noise = np.random.randn(X.shape[0], len(explain_indices)) * 0.01
            X_init = X.copy()
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