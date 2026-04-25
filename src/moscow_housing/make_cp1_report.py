from __future__ import annotations

import json

import pandas as pd

from moscow_housing.config import PATHS


def _read_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def make_report() -> None:
    PATHS.ensure_dirs()

    stats = _read_json(PATHS.dataset_stats_path)
    test_metrics = _read_json(PATHS.metrics / "test_metrics.json")

    if PATHS.experiments_path.exists():
        experiments = pd.read_csv(PATHS.experiments_path)
        experiments_md = experiments.to_markdown(index=False)
    else:
        experiments_md = "Таблица экспериментов пока не создана."

    report = f"""# CP1: Предсказание стоимости квартир в Москве

## 1. Постановка задачи

Решается задача регрессии: по характеристикам объявления и локации нужно предсказать стоимость квартиры.

Основная метрика — RMSLE. Она удобна для цен недвижимости, потому что цена имеет большой разброс, а ошибка в 1 млн рублей для дешёвой и дорогой квартиры воспринимается по-разному. Дополнительно считаются MAE, RMSE и R2.

## 2. Данные

Источник: Kaggle Moscow Housing Price Dataset.

Размер исходного датасета после загрузки:

- строк до очистки: {stats.get("rows_raw", "заполнится после запуска")}
- колонок до очистки: {stats.get("columns_raw", "заполнится после запуска")}
- строк после очистки: {stats.get("rows_cleaned", "заполнится после запуска")}
- колонок после feature engineering: {stats.get("columns_cleaned", "заполнится после запуска")}

Таргет: `price`.

Признаки: тип квартиры, ближайшее метро, минуты до метро, регион, количество комнат, общая/жилая/кухонная площадь, этаж, этажность дома, тип ремонта.

## 3. Обработка и подготовка данных

Что сделано:

- нормализованы названия колонок;
- проверено наличие обязательных колонок;
- числовые признаки приведены к числовому типу;
- категориальные признаки очищены от пустых строк;
- удалены дубли;
- удалены физически некорректные строки:
  - цена <= 0;
  - площадь <= 0;
  - этаж <= 0;
  - этаж выше количества этажей;
  - жилая или кухонная площадь больше общей;
  - отрицательное расстояние до метро;
- добавлены новые признаки:
  - `area_per_room`;
  - `living_area_share`;
  - `kitchen_area_share`;
  - `floor_ratio`;
  - `is_first_floor`;
  - `is_last_floor`;
  - `metro_distance_bucket`;
  - `is_moscow`.

Важно: `price_per_m2` используется только для EDA и не передаётся в модель, потому что он напрямую зависит от таргета и вызвал бы data leakage.

## 4. Сплит

Использован split 70/15/15:

- train: {stats.get("train_rows", "заполнится после запуска")}
- validation: {stats.get("val_rows", "заполнится после запуска")}
- test: {stats.get("test_rows", "заполнится после запуска")}

Для регрессии использована стратификация по ценовым бинам, чтобы train/val/test имели похожее распределение цен.

Препроцессинг обучается только на train внутри sklearn Pipeline.

## 5. Визуализации

Графики сохраняются в `report/images`:

- распределение `log1p(price)`;
- зависимость цены от площади;
- boxplot цены по региону;
- медианная цена по типу ремонта;
- корреляция числовых признаков;
- распределение цены за м² только для анализа.

## 6. Baseline и первые эксперименты

| Тип | Что проверяется |
| --- | --- |
| DummyRegressor | sanity check, предсказание медианы |
| KNeighborsRegressor | простой baseline без feature engineering |
| Ridge | линейная модель с регуляризацией |
| RandomForest | нелинейная ансамблевая модель |
| ExtraTrees | ансамбль деревьев с большей рандомизацией |
| HistGradientBoosting | градиентный бустинг из sklearn |

## 7. Таблица экспериментов

{experiments_md}

## 8. Лучшая модель на CP1

Лучшая модель по validation RMSLE: `{test_metrics.get("best_model", "заполнится после запуска")}`.

Метрики на test:

- RMSLE: {test_metrics.get("rmsle", "заполнится после запуска")}
- MAE: {test_metrics.get("mae", "заполнится после запуска")}
- RMSE: {test_metrics.get("rmse", "заполнится после запуска")}
- R2: {test_metrics.get("r2", "заполнится после запуска")}

## 9. Воспроизводимость

Для воспроизводимости добавлены:

- fixed seed = 42;
- `requirements.txt`;
- `pyproject.toml` с ruff;
- `Dockerfile`;
- `docker-compose.yml`;
- `Makefile`;
- базовые тесты для feature engineering.

Запуск всего CP1:

```bash
make cp1
```
"""

    PATHS.cp1_report_path.write_text(report, encoding="utf-8")
    print(f"Report saved to {PATHS.cp1_report_path}")


if __name__ == "__main__":
    make_report()
