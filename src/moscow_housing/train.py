from __future__ import annotations

import json
import warnings

import joblib
import pandas as pd

from moscow_housing.config import PATHS
from moscow_housing.constants import TARGET_COLUMN
from moscow_housing.data import load_split
from moscow_housing.metrics import regression_metrics
from moscow_housing.modeling import get_experiments


def predict_without_known_sklearn_warnings(pipeline, x: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Found unknown categories.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul",
            category=RuntimeWarning,
        )
        return pipeline.predict(x)


def run_experiments() -> None:
    PATHS.ensure_dirs()

    train_df, val_df, test_df = load_split()

    rows: list[dict[str, float | str | int]] = []
    fitted_models = {}

    for name, (pipeline, feature_columns) in get_experiments(train_df).items():
        x_train = train_df[feature_columns]
        y_train = train_df[TARGET_COLUMN]

        x_val = val_df[feature_columns]
        y_val = val_df[TARGET_COLUMN]

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*encountered in matmul",
                category=RuntimeWarning,
            )
            pipeline.fit(x_train, y_train)
        val_pred = predict_without_known_sklearn_warnings(pipeline, x_val)
        val_metrics = regression_metrics(y_val.to_numpy(), val_pred)

        row = {
            "model": name,
            "n_features_before_encoding": len(feature_columns),
            **{f"val_{metric}": value for metric, value in val_metrics.items()},
        }
        rows.append(row)
        fitted_models[name] = (pipeline, feature_columns)

        print(f"{name}: {json.dumps(row, ensure_ascii=False)}")

    results = pd.DataFrame(rows).sort_values("val_rmsle", ascending=True)
    results.to_csv(PATHS.experiments_path, index=False)

    best_model_name = str(results.iloc[0]["model"])
    best_pipeline, best_features = fitted_models[best_model_name]

    test_pred = predict_without_known_sklearn_warnings(best_pipeline, test_df[best_features])
    test_metrics = regression_metrics(test_df[TARGET_COLUMN].to_numpy(), test_pred)

    model_payload = {
        "model_name": best_model_name,
        "feature_columns": best_features,
        "pipeline": best_pipeline,
        "validation_results": results.to_dict(orient="records"),
        "test_metrics": test_metrics,
    }
    joblib.dump(model_payload, PATHS.best_model_path)

    test_metrics_path = PATHS.metrics / "test_metrics.json"
    test_metrics_path.write_text(
        json.dumps(
            {
                "best_model": best_model_name,
                **test_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Experiments saved to {PATHS.experiments_path}")
    print(f"Best model: {best_model_name}")
    print(f"Test metrics saved to {test_metrics_path}")
    print(f"Best model saved to {PATHS.best_model_path}")


if __name__ == "__main__":
    run_experiments()
