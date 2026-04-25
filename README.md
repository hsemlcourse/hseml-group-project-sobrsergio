[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)

# Предсказание стоимости квартир в Москве

Проект решает задачу регрессии: предсказание стоимости квартиры в Москве по характеристикам объявления и локации.

Источник данных: [Kaggle Moscow Housing Price Dataset](https://www.kaggle.com/datasets/egorkainov/moscow-housing-price-dataset). Датасет выбран, потому что он напрямую относится к monetary regression, содержит реальные признаки объявлений о недвижимости и подходит по объёму для сравнения нескольких ML-моделей.

## Данные

Исходный датасет содержит 22 676 строк и 12 колонок. После удаления дублей, приведения типов и очистки некорректных строк остаётся 16 280 строк и 26 колонок, включая target и engineered features.

Датасет соответствует требованиям курса: это monetary regression, больше 10 000 строк и больше 10 колонок.

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

Feature engineering:

- `area_per_room`;
- `living_area_share`;
- `kitchen_area_share`;
- `floor_ratio`;
- `is_first_floor`;
- `is_last_floor`;
- `metro_distance_bucket`;
- `is_moscow`;
- `area_log`;
- `minutes_to_metro_log`;
- `room_area_interaction`;
- `kitchen_to_living_ratio`;
- `is_studio`;
- `floor_category`.

Данные делятся на train/validation/test в пропорции 70/15/15 с фиксированным seed. Для регрессии используется стратификация по ценовым бинам, чтобы сохранить похожее распределение цен во всех частях выборки.

Защита от data leakage:

- препроцессинг обучается только на train внутри `sklearn Pipeline`;
- признаки, напрямую использующие target, не передаются в модель;
- `price_per_m2` используется только для EDA;
- дубли удаляются до split;
- clipping выбросов считается только по train-квантилям.

## EDA

Графики сохраняются в `report/images`:

- распределение `log1p(price)`;
- зависимость цены от площади;
- boxplot цены по региону;
- медианная цена по типу ремонта;
- корреляция числовых признаков;
- распределение цены за квадратный метр для анализа;
- сравнение распределений target на train/validation/test;
- feature importance;
- PCA explained variance и PCA projection.

## Моделирование

Основная метрика: RMSLE. Она выбрана, потому что цена недвижимости имеет большой разброс, а относительная ошибка важнее абсолютной для сравнения дешёвых и дорогих квартир.

Дополнительные метрики:

- MAE — интерпретируемая ошибка в рублях;
- RMSE — метрика с большим штрафом за крупные ошибки;
- R2 — общая доля объяснённой дисперсии.

В CP2 сравниваются:

- `DummyRegressor`;
- `KNeighborsRegressor` без feature engineering;
- `Ridge` с разными `alpha`;
- `Ridge + PCA`;
- `RandomForestRegressor` с разными ограничениями глубины и листа;
- `ExtraTreesRegressor` с разными ограничениями глубины и листа;
- `HistGradientBoostingRegressor` с разными `learning_rate`, `max_iter`, `max_leaf_nodes` и `l2_regularization`.

Результаты CP2 сохраняются в `report/metrics/cp2_experiments.csv`.

Лучшая модель по validation RMSLE: `hist_gradient_lr_007_leaf_31`.

Метрики лучшей CP2-модели на test:

| Метрика | Значение |
| --- | ---: |
| RMSLE | 0.2135 |
| MAE | 9 430 619.48 |
| RMSE | 32 346 528.12 |
| R2 | 0.8249 |

Дополнительные артефакты CP2:

- `report/metrics/cp2_test_metrics.json`;
- `report/metrics/cp2_feature_importance.csv`;
- `report/metrics/cp2_pca_explained_variance.json`;
- `models/final_model_cp2.joblib`, генерируется локально и не коммитится.

## Деплой

Для CP3 добавлен FastAPI-сервис.

Эндпоинты:

- `GET /health` — проверка статуса сервиса;
- `GET /model-info` — информация о модели, признаках и test-метриках;
- `POST /predict` — предсказание цены квартиры по признакам объявления.

Пример входа для `POST /predict`:

```json
{
  "apartment_type": "Secondary",
  "metro_station": "Каширская",
  "minutes_to_metro": 7,
  "region": "Moscow",
  "number_of_rooms": 2,
  "area": 50,
  "living_area": 30,
  "kitchen_area": 10,
  "floor": 5,
  "number_of_floors": 10,
  "renovation": "cosmetic"
}
```

Swagger UI доступен после запуска API:

```text
http://127.0.0.1:8000/docs
```

Финальные CP3-артефакты:

- `report/report.md`;
- `report/report.pdf`;
- `report/demo_cp3.mp4`;
- `report/images/11_cp3_api_docs.png`;
- `report/images/12_cp3_predict_request.png`;
- `report/images/13_cp3_predict_response.png`.

## Воспроизводимость

В проекте есть:

- `Makefile` для запуска CP1/CP2;
- `requirements.txt` с зафиксированными версиями зависимостей;
- `pyproject.toml` с настройками `ruff` и `pytest`;
- `Dockerfile` и `docker-compose.yml`;
- fixed seed в split и моделях;
- тесты для feature engineering, split и метрик;
- GitHub Actions для `ruff`.

## Запуск

Если `data/raw/data.csv` уже лежит в проекте, можно сразу установить зависимости и запустить пайплайн.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
make cp3
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
make train-cp2
make report-cp2
make report-cp3
make lint
make test
```

Запуск API:

```bash
make api
```

Проверка API без браузера:

```bash
make api-smoke
```

Запуск через Docker:

```bash
docker compose up --build
```

При контейнерном запуске модель обучается перед стартом API, поэтому отдельный локальный `make train-cp2` не нужен.

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
│   ├── images/              # EDA-графики и CP2-графики
│   ├── metrics/             # метрики и таблицы экспериментов
│   ├── report.md            # финальный markdown-отчёт
│   ├── report.pdf           # финальный PDF-отчёт
│   └── demo_cp3.mp4         # видео демонстрации API
├── src/moscow_housing/
│   ├── api.py
│   ├── config.py
│   ├── constants.py
│   ├── data.py
│   ├── features.py
│   ├── make_cp1_report.py
│   ├── make_cp2_report.py
│   ├── make_cp3_report.py
│   ├── make_eda.py
│   ├── metrics.py
│   ├── modeling.py
│   ├── prepare_data.py
│   ├── smoke_api.py
│   ├── train.py
│   └── train_cp2.py
├── tests/
├── Makefile
├── requirements.txt
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```
