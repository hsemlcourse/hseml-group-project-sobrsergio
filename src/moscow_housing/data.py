from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from moscow_housing.config import PATHS
from moscow_housing.constants import (
    RANDOM_STATE,
    RAW_TO_CANONICAL_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)
from moscow_housing.features import add_features


def find_raw_csv(raw_dir: Path = PATHS.data_raw) -> Path:
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"CSV не найден в {raw_dir}. "
            "Скачайте датасет с Kaggle и положите CSV-файл в data/raw."
        )
    return csv_files[0]


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or find_raw_csv()
    return pd.read_csv(csv_path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result = result.rename(columns=RAW_TO_CANONICAL_COLUMNS)
    result.columns = (
        result.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return result


def validate_columns(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(
            "В датасете не хватает обязательных колонок: "
            f"{missing}. Текущие колонки: {list(df.columns)}"
        )


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | float]]:
    result = normalize_columns(df)
    validate_columns(result)

    stats: dict[str, int | float] = {
        "rows_raw": int(len(result)),
        "columns_raw": int(result.shape[1]),
    }

    result = result[REQUIRED_COLUMNS].copy()
    stats["missing_values_raw_total"] = int(result.isna().sum().sum())
    for column, missing_count in result.isna().sum().items():
        stats[f"missing_raw_{column}"] = int(missing_count)

    numeric_columns = [
        "price",
        "minutes_to_metro",
        "number_of_rooms",
        "area",
        "living_area",
        "kitchen_area",
        "floor",
        "number_of_floors",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column, missing_count in result[numeric_columns].isna().sum().items():
        stats[f"missing_after_type_cast_{column}"] = int(missing_count)

    categorical_columns = ["apartment_type", "metro_station", "region", "renovation"]
    for column in categorical_columns:
        result[column] = (
            result[column]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        )
    for column, missing_count in result[categorical_columns].isna().sum().items():
        stats[f"missing_after_string_cleanup_{column}"] = int(missing_count)

    stats["duplicate_rows"] = int(result.duplicated().sum())
    result = result.drop_duplicates()
    stats["rows_after_duplicates"] = int(len(result))

    before_business_rules = len(result)

    business_mask = (
        result["price"].gt(0)
        & result["area"].gt(0)
        & result["living_area"].gt(0)
        & result["kitchen_area"].gt(0)
        & result["number_of_rooms"].gt(0)
        & result["floor"].gt(0)
        & result["number_of_floors"].gt(0)
        & result["minutes_to_metro"].ge(0)
        & result["living_area"].le(result["area"])
        & result["kitchen_area"].le(result["area"])
        & result["floor"].le(result["number_of_floors"])
    )
    result = result[business_mask].copy()
    stats["rows_removed_by_business_rules"] = int(before_business_rules - len(result))

    result = add_features(result)

    stats["rows_cleaned"] = int(len(result))
    stats["columns_cleaned"] = int(result.shape[1])
    stats["target_min"] = float(result[TARGET_COLUMN].min())
    stats["target_median"] = float(result[TARGET_COLUMN].median())
    stats["target_mean"] = float(result[TARGET_COLUMN].mean())
    stats["target_max"] = float(result[TARGET_COLUMN].max())
    stats["target_q01"] = float(result[TARGET_COLUMN].quantile(0.01))
    stats["target_q99"] = float(result[TARGET_COLUMN].quantile(0.99))
    stats["area_q01"] = float(result["area"].quantile(0.01))
    stats["area_q99"] = float(result["area"].quantile(0.99))

    return result, stats


def _make_price_bins(y: pd.Series, q: int = 10) -> pd.Series | None:
    try:
        bins = pd.qcut(y, q=q, labels=False, duplicates="drop")
        if bins.nunique(dropna=True) < 2:
            return None
        return bins
    except ValueError:
        return None


def split_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test = 70/15/15.

    Для регрессии используем стратификацию по бинам цены, чтобы во всех частях
    выборки были похожие ценовые диапазоны.
    """

    y = df[target_column]
    stratify_bins = _make_price_bins(y)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=stratify_bins,
    )

    temp_bins = _make_price_bins(temp_df[target_column])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_bins,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def prepare_and_save() -> None:
    PATHS.ensure_dirs()

    raw_df = load_raw_data()
    cleaned_df, stats = clean_data(raw_df)
    train_df, val_df, test_df = split_data(cleaned_df)

    cleaned_df.to_csv(PATHS.cleaned_data_path, index=False)
    train_df.to_csv(PATHS.train_path, index=False)
    val_df.to_csv(PATHS.val_path, index=False)
    test_df.to_csv(PATHS.test_path, index=False)

    stats.update(
        {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_price_mean": float(train_df[TARGET_COLUMN].mean()),
            "val_price_mean": float(val_df[TARGET_COLUMN].mean()),
            "test_price_mean": float(test_df[TARGET_COLUMN].mean()),
        }
    )

    PATHS.dataset_stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Data preparation finished.")
    print(f"Cleaned data: {PATHS.cleaned_data_path}")
    print(f"Train: {PATHS.train_path}")
    print(f"Val: {PATHS.val_path}")
    print(f"Test: {PATHS.test_path}")
    print(f"Stats: {PATHS.dataset_stats_path}")


def load_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(PATHS.train_path),
        pd.read_csv(PATHS.val_path),
        pd.read_csv(PATHS.test_path),
    )
