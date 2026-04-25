[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)

# Предсказание стоимости квартир в Москве

Проект решает задачу регрессии: предсказание стоимости квартиры в Москве по характеристикам объявления и локации.

Источник данных: [Kaggle Moscow Housing Price Dataset](https://www.kaggle.com/datasets/egorkainov/moscow-housing-price-dataset).

## Данные

Исходный датасет содержит 22 676 строк и 12 колонок. После удаления дублей, приведения типов и очистки некорректных строк осталось 16 280 строк и 20 колонок после feature engineering.

Таргет: `price`.

Исходные признаки:

- тип квартиры;
- ближайшая станция метро;
- минуты до метро;
- регион;
- количество комнат;
- общая площадь;
- жилая площадь;
- площадь кухни;
- этаж;
- количество этажей в доме;
- тип ремонта.

## Подготовка данных

В пайплайне подготовки данных выполняются:

- загрузка CSV из `data/raw`;
- нормализация названий колонок;
- проверка обязательных колонок;
- приведение числовых признаков к числовому типу;
- очистка категориальных признаков от пустых значений;
- удаление дублей;
- удаление физически некорректных строк по бизнес-правилам:
  - цена больше 0;
  - площади больше 0;
  - жилая и кухонная площадь не больше общей;
  - этаж больше 0;
  - этаж не выше количества этажей;
  - расстояние до метро неотрицательное.

Добавленные признаки:

- `area_per_room`;
- `living_area_share`;
- `kitchen_area_share`;
- `floor_ratio`;
- `is_first_floor`;
- `is_last_floor`;
- `metro_distance_bucket`;
- `is_moscow`.

Данные делятся на train/validation/test в пропорции 70/15/15 с фиксированным seed. Для регрессии используется стратификация по ценовым бинам, чтобы сохранить похожее распределение цен во всех частях выборки.

Защита от data leakage:

- препроцессинг обучается только на train внутри `sklearn Pipeline`;
- признаки, напрямую использующие target, не передаются в модель;
- `price_per_m2` используется только для EDA;
- дубли удаляются до split.

## EDA

Графики сохраняются в `report/images`:

- распределение `log1p(price)`;
- зависимость цены от площади;
- boxplot цены по региону;
- медианная цена по типу ремонта;
- корреляция числовых признаков;
- распределение цены за квадратный метр для анализа.

## Моделирование

Основная метрика: RMSLE. Она выбрана, потому что цена недвижимости имеет большой разброс, а относительная ошибка важнее абсолютной для сравнения дешёвых и дорогих квартир.

Дополнительные метрики:

- MAE — интерпретируемая ошибка в рублях;
- RMSE — метрика с большим штрафом за крупные ошибки;
- R2 — общая доля объяснённой дисперсии.

В CP1 сравниваются модели:

- `DummyRegressor`;
- `KNeighborsRegressor` без feature engineering;
- `Ridge`;
- `RandomForestRegressor`;
- `ExtraTreesRegressor`;
- `HistGradientBoostingRegressor`.

Результаты экспериментов сохраняются в `report/metrics/experiments.csv`.

Лучшая модель по validation RMSLE: `hist_gradient_boosting_with_features`.

Метрики лучшей модели на test:

| Метрика | Значение |
| --- | ---: |
| RMSLE | 0.2223 |
| MAE | 9 787 013.72 |
| RMSE | 33 028 252.87 |
| R2 | 0.8175 |

## Воспроизводимость

В проекте есть:

- `Makefile` для запуска всего CP1;
- `requirements.txt` с зафиксированными версиями зависимостей;
- `pyproject.toml` с настройками `ruff`;
- `Dockerfile` и `docker-compose.yml`;
- fixed seed в split и моделях;
- тесты для feature engineering.

## Запуск

Если `data/raw/data.csv` уже лежит в проекте, можно сразу установить зависимости и запустить пайплайн.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
make cp1
```

Если исходного CSV нет, его можно скачать через Kaggle CLI:

```bash
mkdir -p data/raw
kaggle datasets download -d egorkainov/moscow-housing-price-dataset -p data/raw --unzip
```

Запуск отдельных шагов:

```bash
make prepare
make eda
make train
make report
make lint
make test
```

Запуск через Docker:

```bash
docker compose up --build
```

## Структура проекта

```text
.
├── data/
│   ├── raw/                 # исходный CSV
│   └── processed/           # очищенные данные и train/val/test
├── models/                  # сохранённые модели
├── notebooks/               # дополнительные материалы
├── presentation/            # материалы для защиты
├── report/
│   ├── images/              # EDA-графики
│   ├── metrics/             # метрики и таблица экспериментов
│   └── report.md            # отчёт по CP1
├── src/moscow_housing/
│   ├── config.py
│   ├── constants.py
│   ├── data.py
│   ├── features.py
│   ├── make_eda.py
│   ├── metrics.py
│   ├── modeling.py
│   ├── prepare_data.py
│   ├── train.py
│   └── make_cp1_report.py
├── tests/
├── Makefile
├── requirements.txt
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```
