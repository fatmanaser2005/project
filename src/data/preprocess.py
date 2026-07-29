import pandas as pd
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from data.merge_data import merge_data
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os


def preprocess():

    # Load merged data
    df = merge_data()

    print("Before preprocessing:", df.shape)

    # Remove duplicates
    df.drop_duplicates(inplace=True)


    # Handle missing values
    for col in df.columns:

        if df[col].dtype == "object" or df[col].dtype == "string":
            df[col] = df[col].fillna(df[col].mode()[0])

        else:
            df[col] = df[col].fillna(df[col].median())


    # =========================
    # Feature Engineering
    # =========================

    # Price difference
    df["discount"] = (
        df["base_price"] -
        df["checkout_price"]
    )

    # Discount percentage
    df["discount_percentage"] = (
        df["discount"] /
        df["base_price"]
    ) * 100


    # Avoid infinity values
    df["discount_percentage"] = (
        df["discount_percentage"]
        .replace([float("inf"), -float("inf")], 0)
    )


    # =========================
    # Encoding
    # =========================

    categorical_cols = [
        "category",
        "cuisine",
        "center_type"
    ]

    encoders = {}

    for col in categorical_cols:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(
            df[col]
        )

        encoders[col] = encoder


    # Save encoders
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        encoders,
        "models/encoders.pkl"
    )


    # =========================
    # Split Features and Target
    # =========================

    X = df.drop(
        "num_orders",
        axis=1
    )

    y = df["num_orders"]


    # =========================
    # Scaling
    # =========================

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)


    # Save scaler
    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )


    return X_scaled, y



if __name__ == "__main__":

    X, y = preprocess()

    print("\nAfter preprocessing:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)