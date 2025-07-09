# main.py

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from models.mlp import BlackBoxModel
import pickle
import os
from explainers.dce import DistributionalCounterfactualExplainer
from explainers.distances import bootstrap_1d, bootstrap_sw
from utils.logger_config import setup_logger
from utils.data_processing import *


logger = setup_logger()

absolute_path = "/zhome/b2/8/197929/GitHub/distributional-counterfactual-explanation"

read_data_path = os.path.join(absolute_path, "data/hotel_booking")
dump_data_path = os.path.join(absolute_path, "data/hotel_booking/")
csv_file = "hotel_bookings.csv"
target_name = "is_canceled"

# Parameters
sample_num = 100
delta = 0.15
alpha = 0.05
U_1 = 0.5
U_2 = 0.3
n_proj = 10
interval_left = 0
interval_right = 1.0
kappa = 0.001

max_iter = 100
tau = 1e3

# Define features for model training
features = [
    "hotel",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "assigned_room_type",
    "booking_changes",
    "deposit_type",
    "agent",
    "company",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

target_name = "is_canceled"

explain_columns = [
    "lead_time",
    "booking_changes",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "days_in_waiting_list",
]

y_target = torch.distributions.beta.Beta(0.1, 0.9).sample((sample_num,))


def main():
    # Load dataset and create a copy for manipulation
    df_ = pd.read_csv(os.path.join(read_data_path, csv_file))
    df = df_.copy()

    logger.info("Dataset loaded.")

    df, label_mappings = feature_encoding(
        df=df, target_name=target_name, target_encode_dict={}
    )

    logger.info("Data preprocessing done.")

    df_X = df[features].copy()
    df_y = df[target_name].copy()

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)

    # Normalize training and test datasets
    std = X_train.std()
    mean = X_train.mean()
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)

    # Model initialization, defining loss and optimizer
    model = BlackBoxModel(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Model training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Model evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        y_pred_tensor = (test_outputs > 0.5).float()
        correct_predictions = (y_pred_tensor == y_test_tensor).float().sum()
        accuracy = correct_predictions / y_test_tensor.shape[0]

    logger.info(f"Accuracy = {accuracy.item()}")

    indice = (X_test.sample(sample_num)).index
    df_explain = X_test.loc[indice]
    y = model(torch.FloatTensor(df_explain.values))
    y_true = y_test.loc[indice]

    # Counterfactual explanation
    logger.info("Counterfactual explanation optimization started.")
    explainer = DistributionalCounterfactualExplainer(
        model=model,
        df_X=df_explain,
        explain_columns=explain_columns,
        y_target=y_target,
        lr=1e-1,
        n_proj=n_proj,
        delta=delta,
    )

    logger.info(
        f"WD: {np.sqrt(explainer.wd.distance(y, y_target, delta=delta)[0].item())}"
    )
    logger.info(
        f"WD Exact Interval: {explainer.wd.distance_interval(y, y_target, delta=delta, alpha=alpha)}",
    )
    logger.info(
        f"WD Bootstrap Interval: {bootstrap_1d(y, y_target, delta=delta, alpha=alpha)}"
    )

    explainer.optimize(
        U_1=U_1,
        U_2=U_2,
        l=interval_left,
        r=interval_right,
        kappa=kappa,
        max_iter=max_iter,
        tau=tau,
    )
    logger.info(f"Feasible solution found: {explainer.found_feasible_solution}")

    if explainer.best_X is not None:
        which_X = explainer.best_X
    else:
        which_X = explainer.X

    logger.info(
        f"SWD: {np.sqrt(explainer.swd.distance(which_X[:,explainer.explain_indices], explainer.X_prime[:,explainer.explain_indices], delta)[0].item())}",
    )
    logger.info(
        f"SWD Exact Interval: {explainer.swd.distance_interval(which_X[:,explainer.explain_indices], explainer.X_prime[:,explainer.explain_indices], delta, alpha=alpha)}",
    )
    logger.info(
        f"SWD Bootstrap Interval: {bootstrap_sw(which_X[:,explainer.explain_indices], explainer.X_prime[:,explainer.explain_indices], delta=delta, alpha=alpha, swd=explainer.swd)}",
    )

    factual_X = df[df_X.columns].loc[indice].copy()
    counterfactual_X = pd.DataFrame(
        which_X.detach().numpy() * std[df_X.columns].values + mean[df_X.columns].values,
        columns=df_X.columns,
    )

    dtype_dict = df.dtypes.apply(lambda x: x.name).to_dict()
    for k, v in dtype_dict.items():
        if k in counterfactual_X.columns:
            if v[:3] == "int":
                counterfactual_X[k] = counterfactual_X[k].round().astype(v)
            else:
                counterfactual_X[k] = counterfactual_X[k].astype(v)

    factual_y = pd.DataFrame(
        y.detach().numpy(), columns=[target_name], index=factual_X.index
    )
    counterfactual_y = pd.DataFrame(
        explainer.y.detach().numpy(), columns=[target_name], index=factual_X.index
    )

    # Recover the type of counterfactual_X
    dtype_dict = df.dtypes.apply(lambda x: x.name).to_dict()
    for k, v in dtype_dict.items():
        if k in counterfactual_X.columns:
            if v[:3] == "int":
                counterfactual_X[k] = counterfactual_X[k].round().astype(v)
            else:
                counterfactual_X[k] = counterfactual_X[k].astype(v)

    counterfactual_X.index = factual_X.index
    counterfactual_X[target_name] = counterfactual_y
    factual_X[target_name] = factual_y

    # Now, reverse the label encoding using the label_mappings
    for dft in [factual_X, counterfactual_X]:
        for column, mapping in label_mappings.items():
            if column in dft.columns:
                # Invert the label mapping dictionary
                inv_mapping = {v: k for k, v in mapping.items()}
                # Map the encoded labels back to the original strings
                dft[column] = dft[column].map(inv_mapping)

    factual_X.to_csv(os.path.join(dump_data_path, "factual.csv"), index=False)
    counterfactual_X.to_csv(
        os.path.join(dump_data_path, "counterfactual.csv"), index=False
    )
    with open(os.path.join(dump_data_path, "explainer.pkl"), "wb") as file:
        pickle.dump(explainer, file)
    logger.info("Files dumped.")


if __name__ == "__main__":
    logger.info("Hotel booking analysis started")
    main()
