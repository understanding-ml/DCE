from sklearn.preprocessing import LabelEncoder


def feature_encoding(df, target_name, target_encode_dict):
    target = df[target_name].replace(target_encode_dict)
    df[target_name] = target

    # Initialize a label encoder and a dictionary to store label mappings
    label_encoder = LabelEncoder()
    label_mappings = {}

    # Convert categorical columns to numerical representations using label encoding
    for column in df.columns:
        if column is not target_name and df[column].dtype == "object":
            df[column] = df[column].fillna("Unknown")  # Handle missing values
            df[column] = label_encoder.fit_transform(df[column])
            label_mappings[column] = dict(
                zip(label_encoder.classes_, range(len(label_encoder.classes_)))
            )

    # Impute missing values in numerical columns with their median
    for column in df.columns:
        if df[column].isna().any():
            median_val = df[column].median()
            df[column].fillna(median_val, inplace=True)

    return df, label_mappings
