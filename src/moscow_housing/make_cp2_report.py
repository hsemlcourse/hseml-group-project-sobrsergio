from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from moscow_housing.config import PATHS


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_table(path: Path, columns: list[str] | None = None) -> str:
    if not path.exists():
        return "Таблица пока не создана."
    table = pd.read_csv(path)
    if columns is not None:
        table = table[columns]
    return table.to_markdown(index=False)


def make_report() -> None:
    PATHS.ensure_dirs()

    stats = _read_json(PATHS.dataset_stats_path)
    preliminary_test_metrics = _read_json(PATHS.metrics / "test_metrics.json")
    cp2_test_metrics = _read_json(PATHS.cp2_test_metrics_path)
    pca_stats = _read_json(PATHS.pca_variance_path)

    cp2_columns = [
        "model",
        "feature_set",
        "n_features_before_encoding",
        "val_rmsle",
        "val_mae",
        "val_r2",
        "fit_time_sec",
        "comment",
    ]
    importance_columns = ["feature", "importance_mean", "importance_std"]

    cp2_experiments_md = _read_table(PATHS.cp2_experiments_path, cp2_columns)
    importance_md = _read_table(PATHS.feature_importance_path, importance_columns)

    report = f"""# CP2: Предсказание стоимости квартир в Москве

## 1. Постановка задачи

Решается задача регрессии: по характеристикам объявления и локации нужно предсказать стоимость квартиры в Москве.

Основная метрика — RMSLE. Для недвижимости она удобнее MAE/RMSE как основная метрика, потому что цены имеют большой разброс и относительная ошибка важна при сравнении дешёвых и дорогих квартир. Дополнительно считаются MAE, RMSE и R2.

## 2. Данные

Источник: Kaggle Moscow Housing Price Dataset. Датасет выбран, потому что он напрямую относится к monetary regression, содержит реальные признаки объявлений о недвижимости и подходит по объёму для сравнения нескольких ML-моделей.

- строк до очистки: {stats.get("rows_raw", "нет данных")}
- колонок до очистки: {stats.get("columns_raw", "нет данных")}
- строк после удаления дублей: {stats.get("rows_after_duplicates", "нет данных")}
- строк после очистки: {stats.get("rows_cleaned", "нет данных")}
- колонок после feature engineering: {stats.get("columns_cleaned", "нет данных")}
- дублей удалено: {stats.get("duplicate_rows", "нет данных")}
- строк удалено бизнес-правилами: {stats.get("rows_removed_by_business_rules", "нет данных")}

Датасет удовлетворяет требованиям курса: больше 10 000 строк, больше 10 колонок, задача monetary regression.

## 3. Обработка и подготовка данных

Что сделано:

- нормализованы названия колонок;
- проверено наличие обязательных колонок;
- числовые признаки приведены к числовому типу;
- категориальные признаки очищены от пустых строк;
- удалены дубли;
- удалены физически некорректные строки: цена/площади/этажность <= 0, этаж выше этажности дома, жилая или кухонная площадь больше общей, отрицательное расстояние до метро;
- выбросы в числовых признаках ограничиваются по train-квантилям внутри `sklearn Pipeline`, чтобы не использовать validation/test при расчёте порогов.

Feature engineering:

- CP1-признаки: `area_per_room`, `living_area_share`, `kitchen_area_share`, `floor_ratio`, `is_first_floor`, `is_last_floor`, `metro_distance_bucket`, `is_moscow`;
- CP2-признаки: `area_log`, `minutes_to_metro_log`, `room_area_interaction`, `kitchen_to_living_ratio`, `is_studio`, `floor_category`.

Сплит: train/validation/test = 70/15/15 со стратификацией по ценовым бинам.

Защита от leakage:

- препроцессинг обучается только на train внутри `Pipeline`;
- target-derived признаки не используются;
- `price_per_m2` используется только для EDA;
- дубли удаляются до split.

## 4. Визуализации

Графики сохраняются в `report/images`:

- распределение `log1p(price)`;
- зависимость цены от площади;
- boxplot цены по региону;
- медианная цена по ремонту;
- корреляция числовых признаков;
- распределение цены за м² только для анализа;
- сравнение распределений target на train/validation/test;
- feature importance финальной модели;
- PCA cumulative explained variance и 2D PCA projection.

## 5. Baseline

Baseline-модели:

- `DummyRegressor` — sanity check, предсказание медианы;
- `KNeighborsRegressor` — простая модель из коробки без feature engineering.

Test RMSLE предварительного пайплайна до CP2 tuning: {preliminary_test_metrics.get("rmsle", "нет данных")}.

## 6. CP2-эксперименты

В CP2 проверялись гипотезы:

- регуляризация линейной модели влияет на устойчивость Ridge;
- feature engineering должен улучшить качество относительно raw baseline;
- ансамбли деревьев лучше ловят нелинейные зависимости цены от площади, локации и этажа;
- разные ограничения глубины и размера листа меняют переобучение RandomForest/ExtraTrees;
- HistGradientBoosting должен быть сильным кандидатом на финальную модель;
- PCA может ухудшить качество, если важные one-hot признаки плохо сохраняются в главных компонентах.

Таблица экспериментов:

{cp2_experiments_md}

## 7. Уменьшение размерности

Проведён эксперимент `ridge_pca_95`: после encoding признаков PCA сохраняет 95% дисперсии и затем обучается Ridge.

- число PCA-компонент: {pca_stats.get("n_components", "нет данных")}
- сохранённая доля дисперсии: {pca_stats.get("explained_variance_sum", "нет данных")}

Вывод: PCA добавлен как контрольный эксперимент. Для табличных данных с категориальными one-hot признаками он не обязан улучшать качество, но помогает проверить, можно ли сжать пространство признаков без сильной потери качества.

## 8. Финальная модель и интерпретируемость

Финальная модель выбрана по validation RMSLE: `{cp2_test_metrics.get("best_model", "нет данных")}`.

Метрики финальной модели на test:

- RMSLE: {cp2_test_metrics.get("rmsle", "нет данных")}
- MAE: {cp2_test_metrics.get("mae", "нет данных")}
- RMSE: {cp2_test_metrics.get("rmse", "нет данных")}
- R2: {cp2_test_metrics.get("r2", "нет данных")}

Permutation importance на validation sample:

{importance_md}

## 9. Воспроизводимость

Для воспроизводимости:

- fixed seed = 42;
- зависимости закреплены в `requirements.txt`;
- настройки `ruff` и `pytest` вынесены в `pyproject.toml`;
- есть `Dockerfile` и `docker-compose.yml`;
- есть `Makefile`;
- есть тесты для feature engineering, split и метрик.

Запуск CP2:

```bash
make cp2
```

## 10. Что остаётся на CP3

На CP3 нужно добавить FastAPI-деплой, финальный отчёт/демонстрацию деплоя и материалы для защиты.
"""

    PATHS.cp1_report_path.write_text(report, encoding="utf-8")
    print(f"CP2 report saved to {PATHS.cp1_report_path}")


if __name__ == "__main__":
    make_report()
