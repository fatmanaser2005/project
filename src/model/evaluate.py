import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_val, y_val):

    y_pred = model.predict(X_val)

    mae = mean_absolute_error(
        y_val,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val,
            y_pred
        )
    )

    mape = np.mean(
        np.abs(
            (y_val - y_pred) / y_val
        )
    ) * 100

    r2 = r2_score(
        y_val,
        y_pred
    )

    return mae, rmse, mape, r2