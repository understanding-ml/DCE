import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class HotelBookingData:
    def __init__(self, test_size=0.2, seed=42, sample_num=50):
        self.name = "hotel"
        self.test_size = test_size
        self.random_state = seed
        self.sample_num = sample_num

        self.target_name = 'is_canceled'
        self.features = [
            'hotel', 'lead_time', 'arrival_date_year', 'arrival_date_month',
            'arrival_date_week_number', 'arrival_date_day_of_month',
            'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 
            'babies', 'meal', 'country', 'market_segment', 'distribution_channel',
            'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
            'reserved_room_type', 'assigned_room_type', 'booking_changes',
            'deposit_type', 'agent', 'company', 'days_in_waiting_list',
            'customer_type', 'adr', 'required_car_parking_spaces',
            'total_of_special_requests'
        ]
        self.categorical_columns = [
            'hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
            'distribution_channel', 'is_repeated_guest', 'reserved_room_type',
            'assigned_room_type', 'deposit_type', 'agent', 'company', 'customer_type'
        ]
        self.continuous_columns = list(set(self.features) - set(self.categorical_columns))
        self.explain_columns = self.features.copy()
        self.feature_names = self.features.copy()
        self._load_data()
        self._preprocess()
        self._split()
        self._standardize()
        self._compute_global_ranges()

    def _load_data(self):
        self.df_raw = pd.read_csv("data/hotel_booking/hotel_bookings.csv")
        self.df = self.df_raw.drop(columns=["reservation_status", "reservation_status_date"], errors="ignore").copy()

    def _preprocess(self):
        self.df[self.target_name] = self.df[self.target_name].astype(int)
        label_encoder = LabelEncoder()
        self.label_mappings = {}

        for col in self.df.columns:
            if col != self.target_name and self.df[col].dtype == 'object':
                self.df[col] = self.df[col].fillna('Unknown')
                self.df[col] = label_encoder.fit_transform(self.df[col])
                self.label_mappings[col] = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

        for col in self.df.columns:
            if self.df[col].isna().any():
                self.df[col] = self.df[col].fillna(self.df[col].median())

        self.X = self.df[self.features].copy()
        self.y = self.df[self.target_name]

    def _split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

    def _standardize(self):
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
        
        self.df_explain = self.X_test.sample(self.sample_num, random_state=self.random_state)
        return self.df_explain.copy()

    def get_y_target(self):
        torch.manual_seed(self.random_state)
        y_target = torch.distributions.Beta(0.1, 0.9).sample((self.sample_num,)).to(torch.float32)
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

    def _compute_global_ranges(self):
        """Compute global feature range information for sampling"""
        raw_data = self.X_raw

        # Global ranges for continuous features
        self.continuous_ranges = {}
        for col in self.continuous_columns:
            if col in raw_data.columns:
                # Get original range
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
