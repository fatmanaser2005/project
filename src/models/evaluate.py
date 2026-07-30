import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def evaluate_model(y_true, y_pred):
    return {
        "mae": float(mae(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "mape": float(mape(y_true, y_pred)),
        "r2": float(r2(y_true, y_pred)),
    }


def save_metrics(metrics, output_path="models/evaluation_metrics.json"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return output_path