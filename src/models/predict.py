import os
import sys
import joblib
import numpy as np


# Add src path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)


from data.preprocess import preprocess


def predict():

    # Load model
    model = joblib.load(
        "models/best_model.pkl"
    )

    # Load processed data
    X, y = preprocess()


    # Predict first 10 samples
    predictions = model.predict(
        X[:10]
    )


    print("Predictions:")
    print(predictions)


    print("\nActual values:")
    print(y[:10].values)



if __name__ == "__main__":

    predict()