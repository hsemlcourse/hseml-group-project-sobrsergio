from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = Path(__file__).resolve().parents[2]
    data_raw: Path = root / "data" / "raw"
    data_processed: Path = root / "data" / "processed"
    report: Path = root / "report"
    figures: Path = report / "images"
    metrics: Path = report / "metrics"
    models: Path = root / "models"

    @property
    def cleaned_data_path(self) -> Path:
        return self.data_processed / "moscow_housing_cleaned.csv"

    @property
    def train_path(self) -> Path:
        return self.data_processed / "train.csv"

    @property
    def val_path(self) -> Path:
        return self.data_processed / "val.csv"

    @property
    def test_path(self) -> Path:
        return self.data_processed / "test.csv"

    @property
    def dataset_stats_path(self) -> Path:
        return self.metrics / "dataset_stats.json"

    @property
    def experiments_path(self) -> Path:
        return self.metrics / "experiments.csv"

    @property
    def best_model_path(self) -> Path:
        return self.models / "best_model.joblib"

    @property
    def final_model_path(self) -> Path:
        return self.models / "final_model_cp2.joblib"

    @property
    def cp1_report_path(self) -> Path:
        return self.report / "report.md"

    @property
    def cp2_experiments_path(self) -> Path:
        return self.metrics / "cp2_experiments.csv"

    @property
    def cp2_test_metrics_path(self) -> Path:
        return self.metrics / "cp2_test_metrics.json"

    @property
    def feature_importance_path(self) -> Path:
        return self.metrics / "cp2_feature_importance.csv"

    @property
    def pca_variance_path(self) -> Path:
        return self.metrics / "cp2_pca_explained_variance.json"

    def ensure_dirs(self) -> None:
        for path in [
            self.data_raw,
            self.data_processed,
            self.report,
            self.figures,
            self.metrics,
            self.models,
        ]:
            path.mkdir(parents=True, exist_ok=True)


PATHS = ProjectPaths()
