# CP3: Предсказание стоимости квартир в Москве

## 1. Введение и постановка задачи

Решается задача регрессии: предсказание стоимости квартиры в Москве по характеристикам объявления и локации.

Практический смысл задачи — получить быстрый ориентир рыночной цены по параметрам объявления. Такой прогноз можно использовать для первичной проверки адекватности цены, сравнения похожих объектов и поиска объявлений, которые сильно отличаются от ожидаемой стоимости.

Основная метрика — RMSLE. Она выбрана потому, что цены недвижимости имеют большой разброс, и относительная ошибка важнее абсолютной при сравнении дешёвых и дорогих квартир. Дополнительно считаются MAE, RMSE и R2: MAE интерпретируется в рублях, RMSE сильнее штрафует крупные ошибки, R2 показывает общую объяснённую долю дисперсии.

## 2. Поиск и описание данных

Источник: [Kaggle Moscow Housing Price Dataset](https://www.kaggle.com/datasets/egorkainov/moscow-housing-price-dataset). Датасет выбран, потому что он напрямую относится к monetary regression, содержит реальные признаки объявлений и подходит по объёму.

- строк до очистки: 22676
- колонок до очистки: 12
- пропусков в исходных обязательных колонках: 0
- дублей удалено: 1851
- строк удалено бизнес-правилами: 4545
- строк после очистки: 16280
- колонок после feature engineering: 26
- медианная цена: 13500000.0
- 99% квантиль цены: 400803486.24999666

## 3. Обработка и подготовка данных

Выполнены нормализация колонок, проверка обязательных полей, приведение типов, очистка категориальных признаков, удаление дублей и физически некорректных строк.

Удалялись строки, где цена или площади были неположительными, этаж был выше этажности дома, жилая или кухонная площадь превышала общую, а расстояние до метро было отрицательным. Выбросы числовых признаков обрабатываются через `QuantileClipper` внутри `sklearn Pipeline`: пороги считаются только на train, чтобы не подглядывать в validation/test.

Добавлены доменные признаки:

- признаки площади: `area_per_room`, `living_area_share`, `kitchen_area_share`, `area_log`;
- признаки этажа: `floor_ratio`, `is_first_floor`, `is_last_floor`, `floor_category`;
- признаки комнат и планировки: `room_area_interaction`, `kitchen_to_living_ratio`, `is_studio`;
- признаки локации: `minutes_to_metro_log`, `metro_distance_bucket`, `is_moscow`.

Сплит: train/validation/test = 70/15/15 со стратификацией по ценовым бинам. Препроцессинг обучается только на train внутри `sklearn Pipeline`; target-derived признаки не используются. `price_per_m2` используется только в EDA и не передаётся в модель.

## 4. Baseline-модель

Baseline:

- `DummyRegressor` — предсказание медианы;
- `KNeighborsRegressor` — простая модель без feature engineering.

Baseline нужен как нижняя граница качества: если сложная модель не выигрывает у медианы или простой KNN-модели, значит pipeline построен плохо или признаки не дают полезного сигнала.

## 5. Эксперименты

Формат экспериментов: гипотеза → настройка модели → результат на validation.

Основные гипотезы:

- регуляризация Ridge может стабилизировать линейную модель на one-hot признаках;
- feature engineering должен улучшить качество относительно raw baseline;
- ансамбли деревьев должны лучше ловить нелинейность между ценой, площадью, этажом и локацией;
- HistGradientBoosting должен быть сильным кандидатом для финальной модели;
- PCA может уменьшить размерность encoded-признаков, но для one-hot категорий возможна потеря полезного сигнала.

| model                         | feature_set   |   val_rmsle |     val_mae |     val_r2 | comment                                                                  |
|:------------------------------|:--------------|------------:|------------:|-----------:|:-------------------------------------------------------------------------|
| hist_gradient_lr_007_leaf_31  | engineered    |    0.210527 | 1.03423e+07 |  0.736412  | Gradient boosting tuning: CP1-like learning rate with more iterations.   |
| hist_gradient_lr_004_leaf_63  | engineered    |    0.211063 | 1.04702e+07 |  0.725613  | Gradient boosting tuning: larger trees with stronger L2 regularization.  |
| hist_gradient_lr_005_leaf_31  | engineered    |    0.213051 | 1.0565e+07  |  0.728068  | Gradient boosting tuning: lower learning rate and more iterations.       |
| ridge_alpha_1                 | engineered    |    0.226706 | 1.19345e+07 |  0.631919  | Regularized linear model with moderate penalty.                          |
| extra_trees_depth_18_leaf_2   | engineered    |    0.233923 | 1.0958e+07  |  0.725959  | ExtraTrees tuning: randomized tree ensemble.                             |
| extra_trees_depth_none_leaf_3 | engineered    |    0.236668 | 1.13543e+07 |  0.714104  | ExtraTrees tuning: unrestricted depth with stronger leaf regularization. |
| random_forest_depth_20_leaf_2 | engineered    |    0.241755 | 1.17173e+07 |  0.702513  | RandomForest tuning: deeper trees with moderate regularization.          |
| random_forest_depth_14_leaf_3 | engineered    |    0.24685  | 1.19923e+07 |  0.694201  | RandomForest tuning: shallower trees and stronger leaf regularization.   |
| ridge_alpha_30                | engineered    |    0.250961 | 1.30156e+07 |  0.621958  | Regularized linear model with stronger penalty.                          |
| knn_raw_k10                   | raw           |    0.27444  | 1.37061e+07 |  0.631662  | Simple out-of-the-box KNN baseline without feature engineering.          |
| ridge_pca_95                  | engineered    |    0.313512 | 1.56852e+07 |  0.610487  | Dimensionality reduction: PCA keeps 95% of encoded-feature variance.     |
| dummy_median_raw              | raw           |    1.09622  | 3.17239e+07 | -0.0811494 | Baseline: constant median prediction on raw features.                    |

## 6. Финальная модель и интерпретируемость результатов

Финальная модель выбрана по validation RMSLE: `hist_gradient_lr_007_leaf_31`.

Метрики на test:

- RMSLE: 0.21354964294317733
- MAE: 9430619.479579482
- RMSE: 32346528.115745854
- R2: 0.8249125266833573

Permutation importance:

| feature                 |   importance_mean |   importance_std |
|:------------------------|------------------:|-----------------:|
| area                    |       2.11001e+07 |         697145   |
| metro_station           |       3.91341e+06 |         200505   |
| number_of_floors        |       2.17194e+06 |         580750   |
| apartment_type          |       2.13808e+06 |         206713   |
| renovation              |       1.98812e+06 |         243220   |
| area_per_room           |       1.68019e+06 |         131402   |
| is_moscow               |       1.63267e+06 |         121879   |
| kitchen_area            |  742414           |         226134   |
| minutes_to_metro        |  673225           |         105265   |
| living_area             |  650793           |          78280.7 |
| room_area_interaction   |  553573           |          58813.6 |
| floor                   |  451712           |         105193   |
| floor_ratio             |  420252           |          47202.4 |
| kitchen_to_living_ratio |  408550           |         144221   |
| living_area_share       |  322389           |         160204   |

PCA:

- число компонент для 95% дисперсии: 20
- сохранённая доля дисперсии: 0.9518396543729467

По permutation importance наиболее важными оказались площадь, станция метро, этажность дома, тип квартиры и ремонт. Это выглядит ожидаемо для рынка недвижимости: базовая площадь и локация дают основной вклад, а ремонт и тип объекта уточняют цену.

## 7. Деплой

Добавлен FastAPI-сервис.

Запуск:

```bash
make api
```

Эндпоинты:

- `GET /health`
- `GET /model-info`
- `POST /predict`

Сервис загружает сохранённую финальную модель `models/final_model_cp2.joblib`, валидирует входные параметры через Pydantic и возвращает прогноз цены в рублях. Если модель ещё не создана, API возвращает понятную ошибку и просит запустить `make train-cp2`.

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Пример запроса:

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

Пример ответа:

```json
{
  "predicted_price": 16840677.554136697,
  "model_name": "hist_gradient_lr_007_leaf_31",
  "currency": "RUB"
}
```

Скриншоты:

- [API docs](images/11_cp3_api_docs.png)
- [Predict request](images/12_cp3_predict_request.png)
- [Predict response](images/13_cp3_predict_response.png)

Видео демонстрации: [Google Drive](https://drive.google.com/file/d/1WevJNReWjvA7DrlCg0kZgCtoC6JF3y3M/view?usp=sharing)

## 8. Заключение и выводы

Финальная модель показывает заметное улучшение относительно baseline и упакована в воспроизводимый FastAPI-сервис. Проект можно запустить локально через `make cp3`, проверить API через `make api-smoke` и открыть документацию в Swagger UI.

Ограничения: модель обучена на одном датасете с Kaggle, поэтому при использовании на новых объявлениях нужно следить за изменением рынка и переобучать модель на более свежих данных. Следующие улучшения: добавить мониторинг качества, расширить признаки района и подключить регулярное обновление данных.
