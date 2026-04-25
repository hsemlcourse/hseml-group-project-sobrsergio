from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from moscow_housing.config import PATHS
from moscow_housing.constants import RANDOM_STATE, TARGET_COLUMN
from moscow_housing.data import load_split
from moscow_housing.metrics import regression_metrics
from moscow_housing.modeling import get_cp2_experiments


def _predict_without_known_sklearn_warnings(pipeline, x: pd.DataFrame) -> np.ndarray:
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


def _fit_without_known_sklearn_warnings(pipeline, x: pd.DataFrame, y: pd.Series) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul",
            category=RuntimeWarning,
        )
        pipeline.fit(x, y)


def _estimator_summary(pipeline) -> str:
    model = pipeline.named_steps["model"]
    estimator = getattr(model, "regressor", model)
    params = estimator.get_params()
    important_params = {
        key: value
        for key, value in params.items()
        if key
        in {
            "alpha",
            "n_neighbors",
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "l2_regularization",
        }
    }
    return json.dumps(important_params, ensure_ascii=False, sort_keys=True)


def _save_feature_importance(
    pipeline,
    feature_columns: list[str],
    val_df: pd.DataFrame,
) -> None:
    sample = val_df.sample(min(len(val_df), 1200), random_state=RANDOM_STATE)
    x_val = sample[feature_columns]
    y_val = sample[TARGET_COLUMN]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Found unknown categories.*",
            category=UserWarning,
        )
        result = permutation_importance(
            pipeline,
            x_val,
            y_val,
            scoring="neg_mean_absolute_error",
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    importance = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(PATHS.feature_importance_path, index=False)

    top = importance.head(12).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance_mean"])
    plt.title("CP2 permutation feature importance, validation sample")
    plt.xlabel("increase in MAE when permuted")
    plt.tight_layout()
    output_path = PATHS.figures / "07_cp2_feature_importance.png"
    plt.savefig(output_path, dpi=160)
    plt.close()


def _save_pca_artifacts(pca_pipeline, train_df: pd.DataFrame, feature_columns: list[str]) -> None:
    pca = pca_pipeline.named_steps["pca"]
    explained = pca.explained_variance_ratio_
    payload = {
        "n_components": int(pca.n_components_),
        "explained_variance_sum": float(np.sum(explained)),
        "explained_variance_ratio": [float(value) for value in explained],
    }
    PATHS.pca_variance_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cumulative = np.cumsum(explained)
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", markersize=3)
    plt.axhline(0.95, color="red", linestyle="--", linewidth=1)
    plt.title("CP2 PCA cumulative explained variance")
    plt.xlabel("number of components")
    plt.ylabel("cumulative explained variance")
    plt.tight_layout()
    plt.savefig(PATHS.figures / "08_cp2_pca_explained_variance.png", dpi=160)
    plt.close()

    sample = train_df.sample(min(len(train_df), 1500), random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul",
            category=RuntimeWarning,
        )
        transformed = pca_pipeline.named_steps["preprocessor"].transform(sample[feature_columns])
        projected = pca.transform(transformed)
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        projected[:, 0],
        projected[:, 1],
        c=np.log1p(sample[TARGET_COLUMN]),
        alpha=0.4,
        s=12,
    )
    plt.colorbar(scatter, label="log1p(price)")
    plt.title("CP2 PCA projection of encoded features")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(PATHS.figures / "09_cp2_pca_projection.png", dpi=160)
    plt.close()


def run_cp2_experiments() -> None:
    PATHS.ensure_dirs()
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

    train_df, val_df, test_df = load_split()
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    rows: list[dict[str, float | str | int]] = []
    fitted_models = {}

    for name, (pipeline, feature_columns, comment) in get_cp2_experiments(train_df).items():
        x_train = train_df[feature_columns]
        y_train = train_df[TARGET_COLUMN]
        x_val = val_df[feature_columns]
        y_val = val_df[TARGET_COLUMN]

        started_at = time.perf_counter()
        _fit_without_known_sklearn_warnings(pipeline, x_train, y_train)
        fit_time_sec = time.perf_counter() - started_at

        val_pred = _predict_without_known_sklearn_warnings(pipeline, x_val)
        val_metrics = regression_metrics(y_val.to_numpy(), val_pred)

        row = {
            "model": name,
            "feature_set": "raw" if len(feature_columns) == 11 else "engineered",
            "n_features_before_encoding": len(feature_columns),
            "params": _estimator_summary(pipeline),
            "fit_time_sec": round(fit_time_sec, 3),
            "comment": comment,
            **{f"val_{metric}": value for metric, value in val_metrics.items()},
        }
        rows.append(row)
        fitted_models[name] = (pipeline, feature_columns)
        print(f"{name}: {json.dumps(row, ensure_ascii=False)}")

    results = pd.DataFrame(rows).sort_values("val_rmsle", ascending=True)
    results.to_csv(PATHS.cp2_experiments_path, index=False)

    best_model_name = str(results.iloc[0]["model"])
    best_pipeline_train_only, best_features = fitted_models[best_model_name]

    final_pipeline = clone(best_pipeline_train_only)
    _fit_without_known_sklearn_warnings(
        final_pipeline,
        train_val_df[best_features],
        train_val_df[TARGET_COLUMN],
    )
    test_pred = _predict_without_known_sklearn_warnings(final_pipeline, test_df[best_features])
    test_metrics = regression_metrics(test_df[TARGET_COLUMN].to_numpy(), test_pred)

    model_payload = {
        "checkpoint": "cp2",
        "model_name": best_model_name,
        "feature_columns": best_features,
        "pipeline": final_pipeline,
        "validation_results": results.to_dict(orient="records"),
        "test_metrics": test_metrics,
    }
    joblib.dump(model_payload, PATHS.final_model_path)

    PATHS.cp2_test_metrics_path.write_text(
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

    _save_feature_importance(best_pipeline_train_only, best_features, val_df)

    pca_pipeline, pca_features = fitted_models["ridge_pca_95"]
    _save_pca_artifacts(pca_pipeline, train_df, pca_features)

    print(f"CP2 experiments saved to {PATHS.cp2_experiments_path}")
    print(f"CP2 best model: {best_model_name}")
    print(f"CP2 test metrics saved to {PATHS.cp2_test_metrics_path}")
    print(f"CP2 final model saved to {PATHS.final_model_path}")
    print(f"CP2 feature importance saved to {PATHS.feature_importance_path}")
    print(f"CP2 PCA variance saved to {PATHS.pca_variance_path}")


if __name__ == "__main__":
    run_cp2_experiments()
