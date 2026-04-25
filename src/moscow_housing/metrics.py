from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred_safe = np.maximum(y_pred, 0)

    mae = mean_absolute_error(y_true, y_pred_safe)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_safe)))
    rmsle = float(
        np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred_safe)))
    )
    r2 = r2_score(y_true, y_pred_safe)

    return {
        "mae": float(mae),
        "rmse": rmse,
        "rmsle": rmsle,
        "r2": float(r2),
    }
