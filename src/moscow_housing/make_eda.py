from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from moscow_housing.config import PATHS
from moscow_housing.constants import TARGET_COLUMN


def _save_current_figure(filename: str) -> None:
    output_path = PATHS.figures / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Saved {output_path}")


def plot_price_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(np.log1p(df[TARGET_COLUMN]), bins=50)
    plt.title("Distribution of log1p(price)")
    plt.xlabel("log1p(price)")
    plt.ylabel("count")
    _save_current_figure("01_price_distribution_log.png")


def plot_price_vs_area(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sample = df.sample(min(len(df), 5000), random_state=42)
    plt.scatter(sample["area"], sample[TARGET_COLUMN], alpha=0.25, s=10)
    plt.title("Price vs total area")
    plt.xlabel("area, m²")
    plt.ylabel("price")
    _save_current_figure("02_price_vs_area.png")


def plot_region_boxplot(df: pd.DataFrame) -> None:
    regions = sorted(df["region"].dropna().astype(str).unique())
    values = [np.log1p(df.loc[df["region"].astype(str) == region, TARGET_COLUMN]) for region in regions]

    plt.figure(figsize=(8, 5))
    plt.boxplot(values, tick_labels=regions, showfliers=False)
    plt.title("log1p(price) by region")
    plt.xlabel("region")
    plt.ylabel("log1p(price)")
    _save_current_figure("03_price_by_region_boxplot.png")


def plot_renovation_median_price(df: pd.DataFrame) -> None:
    grouped = (
        df.groupby("renovation", dropna=False)[TARGET_COLUMN]
        .median()
        .sort_values(ascending=False)
        .head(15)
    )

    plt.figure(figsize=(10, 5))
    plt.bar(grouped.index.astype(str), grouped.values)
    plt.title("Median price by renovation type")
    plt.xlabel("renovation")
    plt.ylabel("median price")
    plt.xticks(rotation=30, ha="right")
    _save_current_figure("04_median_price_by_renovation.png")


def plot_numeric_correlation(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include=["number"]).copy()
    if len(numeric.columns) < 2:
        return

    corr = numeric.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    image = plt.imshow(corr, aspect="auto")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Numeric feature correlation")
    _save_current_figure("05_numeric_correlation.png")


def plot_price_per_m2_distribution(df: pd.DataFrame) -> None:
    price_per_m2 = df[TARGET_COLUMN] / df["area"].replace(0, np.nan)
    price_per_m2 = price_per_m2.replace([np.inf, -np.inf], np.nan).dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(price_per_m2.clip(upper=price_per_m2.quantile(0.99)), bins=50)
    plt.title("Price per m² distribution, clipped at 99% for EDA")
    plt.xlabel("price per m²")
    plt.ylabel("count")
    _save_current_figure("06_price_per_m2_distribution.png")


def plot_split_target_distribution() -> None:
    if not (PATHS.train_path.exists() and PATHS.val_path.exists() and PATHS.test_path.exists()):
        return

    train_df = pd.read_csv(PATHS.train_path)
    val_df = pd.read_csv(PATHS.val_path)
    test_df = pd.read_csv(PATHS.test_path)

    plt.figure(figsize=(8, 5))
    plt.hist(np.log1p(train_df[TARGET_COLUMN]), bins=45, alpha=0.45, label="train")
    plt.hist(np.log1p(val_df[TARGET_COLUMN]), bins=45, alpha=0.45, label="validation")
    plt.hist(np.log1p(test_df[TARGET_COLUMN]), bins=45, alpha=0.45, label="test")
    plt.title("Train/validation/test target distribution")
    plt.xlabel("log1p(price)")
    plt.ylabel("count")
    plt.legend()
    _save_current_figure("10_cp2_split_target_distribution.png")


def make_eda() -> None:
    PATHS.ensure_dirs()

    if not PATHS.cleaned_data_path.exists():
        raise FileNotFoundError(
            f"{PATHS.cleaned_data_path} not found. Run python -m moscow_housing.prepare_data first."
        )

    df = pd.read_csv(PATHS.cleaned_data_path)

    plot_price_distribution(df)
    plot_price_vs_area(df)
    plot_region_boxplot(df)
    plot_renovation_median_price(df)
    plot_numeric_correlation(df)
    plot_price_per_m2_distribution(df)
    plot_split_target_distribution()


if __name__ == "__main__":
    make_eda()
