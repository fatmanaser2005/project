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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    mae = mean_absolute_error(
        y_val,
        y_pred
    )

    import numpy as np

    rmse = np.sqrt(
    mean_squared_error(
        y_val,
        y_pred
    )
    )

    r2 = r2_score(
        y_val,
        y_pred
    )


    print("\nModel Results:")
    print("----------------")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)


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