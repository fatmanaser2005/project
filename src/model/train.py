import os
import sys
import joblib

# Add src folder to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from data.preprocess import preprocess



from random_forest import random_forest
from linear_regression import linear_regression
from decision_tree import decision_tree
from gradient_boosting import gradient_boosting
from xgboost_model import xgboost_model



from evaluate import evaluate_model

from sklearn.model_selection import train_test_split


def train_model():

    # Load data
    X, y = preprocess()


    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    # Create Models
    models = {

        "Random Forest": random_forest(),

        "Linear Regression": linear_regression(),
        
        "Decision Tree": decision_tree(),
        
        "Gradient Boosting": gradient_boosting(),
        
        "XGBoost": xgboost_model()


    }


    # Create models folder
    os.makedirs(
        "models",
        exist_ok=True
    )


    # Train all models
    for name, model in models.items():

        print("\n====================")
        print("Training", name)
        print("====================")


        # Training
        model.fit(
            X_train,
            y_train
        )


        # Evaluation
        mae, rmse, mape, r2 = evaluate_model(
            model,
            X_val,
            y_val
        )


        print("\nModel Results")
        print("---------------------")
        print("MAE :", mae)
        print("RMSE :", rmse)
        print("MAPE :", mape)
        print("R2 Score :", r2)


        # Save model
        filename = name.replace(
            " ",
            "_"
        )


        joblib.dump(
            model,
            f"models/{filename}.pkl"
        )


        print(
            "\n",
            name,
            "saved successfully!"
        )


if __name__ == "__main__":
    train_model()
    


