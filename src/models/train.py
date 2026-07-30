import os
import sys
import joblib
import numpy as np

# Add src folder to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from data.preprocess import preprocess
from models.evaluate import evaluate_model, save_metrics

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def train_model():

    # Load preprocessed data
    X, y = preprocess()


    # Split data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    # Create model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )


    # Train model
    print("Training started...")

    model.fit(
        X_train,
        y_train
    )


    # Prediction
    y_pred = model.predict(
        X_val
    )


    # Evaluation
    metrics = evaluate_model(
        y_val,
        y_pred
    )

    print("\nModel Results:")
    print("----------------")
    print("MAE:", metrics["mae"])
    print("RMSE:", metrics["rmse"])
    print("MAPE:", metrics["mape"])
    print("R2 Score:", metrics["r2"])

    metrics_path = save_metrics(metrics)
    print("\nEvaluation metrics saved to:", metrics_path)


    # Save model
    os.makedirs(
        "models",
        exist_ok=True
    )


    joblib.dump(
        model,
        "models/best_model.pkl"
    )


    print("\nModel saved successfully!")


if __name__ == "__main__":

    train_model()