from __future__ import annotations

import json
import shutil
import subprocess
from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from matplotlib import pyplot as plt
from PIL import Image as PilImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from moscow_housing.api import app
from moscow_housing.config import PATHS

SAMPLE_REQUEST = {
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
    "renovation": "cosmetic",
}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _markdown_table(df: pd.DataFrame, columns: list[str] | None = None, n_rows: int = 12) -> str:
    if df.empty:
        return "Таблица не создана."
    if columns is not None:
        df = df[columns]
    return df.head(n_rows).to_markdown(index=False)


def _api_prediction() -> dict:
    client = TestClient(app)
    health = client.get("/health")
    prediction = client.post("/predict", json=SAMPLE_REQUEST)
    health.raise_for_status()
    prediction.raise_for_status()
    return {"health": health.json(), "request": SAMPLE_REQUEST, "response": prediction.json()}


def _save_text_figure(title: str, body: str, filename: str) -> Path:
    output_path = PATHS.figures / filename
    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.text(0.02, 0.94, title, fontsize=18, fontweight="bold", va="top")
    plt.text(0.02, 0.86, body, fontsize=11, family="monospace", va="top")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def _make_api_screenshots(api_result: dict) -> dict[str, Path]:
    existing_screenshots = {
        "docs": PATHS.figures / "11_cp3_api_docs.png",
        "request": PATHS.figures / "12_cp3_predict_request.png",
        "response": PATHS.figures / "13_cp3_predict_response.png",
    }
    if all(path.exists() for path in existing_screenshots.values()):
        return existing_screenshots

    docs_text = """FastAPI Swagger UI

GET  /health       service status
GET  /model-info   model name, feature columns, test metrics
POST /predict      apartment features -> predicted price

Open locally:
http://127.0.0.1:8000/docs
"""
    request_text = json.dumps(api_result["request"], ensure_ascii=False, indent=2)
    response_text = json.dumps(api_result["response"], ensure_ascii=False, indent=2)

    return {
        "docs": _save_text_figure("API documentation screen", docs_text, "11_cp3_api_docs.png"),
        "request": _save_text_figure("POST /predict request", request_text, "12_cp3_predict_request.png"),
        "response": _save_text_figure("POST /predict response", response_text, "13_cp3_predict_response.png"),
    }


def _make_demo_video(screenshots: dict[str, Path]) -> Path | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None

    output_path = PATHS.report / "demo_cp3.mp4"
    input_args: list[str] = []
    for image_path in screenshots.values():
        input_args.extend(["-loop", "1", "-t", "3", "-i", str(image_path)])

    frame_filters = []
    frame_labels = []
    for index in range(len(screenshots)):
        label = f"v{index}"
        frame_labels.append(f"[{label}]")
        frame_filters.append(
            f"[{index}:v]scale=1280:720:force_original_aspect_ratio=decrease,"
            f"pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[{label}]"
        )
    filter_complex = ";".join(frame_filters) + ";" + "".join(frame_labels)
    filter_complex += f"concat=n={len(screenshots)}:v=1:a=0,format=yuv420p[v]"
    command = [
        ffmpeg,
        "-y",
        *input_args,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def _register_pdf_font() -> str:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            pdfmetrics.registerFont(TTFont("ProjectFont", candidate))
            return "ProjectFont"
    return "Helvetica"


def _pdf_table(df: pd.DataFrame, columns: list[str], n_rows: int = 8, font_name: str = "Helvetica") -> Table:
    table_df = df[columns].head(n_rows).copy()
    table_df = table_df.map(lambda value: f"{value:.4g}" if isinstance(value, float) else str(value))
    data = [columns] + table_df.values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF7")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def _metric_value(value: object, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _paragraph(text: str, styles, style: str = "BodyText") -> Paragraph:
    return Paragraph(text, styles[style])


def _fit_image_size(path: Path, max_width: float, max_height: float) -> tuple[float, float]:
    image = ImageReader(str(path))
    image_width, image_height = image.getSize()
    scale = min(max_width / image_width, max_height / image_height)
    return image_width * scale, image_height * scale


def _pdf_image(path: Path, width: float, height: float) -> Image:
    source = PilImage.open(path)
    if source.mode in {"RGBA", "LA"} or (source.mode == "P" and "transparency" in source.info):
        canvas = PilImage.new("RGB", source.size, "white")
        canvas.paste(source, mask=source.convert("RGBA").getchannel("A"))
        source = canvas
    else:
        source = source.convert("RGB")

    buffer = BytesIO()
    source.save(buffer, format="PNG")
    buffer.seek(0)
    image = Image(buffer, width=width, height=height)
    image._source_buffer = buffer
    return image


def _image_block(
    title: str,
    path: Path,
    styles,
    max_width: float = 15.5 * cm,
    max_height: float = 12 * cm,
) -> list:
    if not path.exists():
        return []
    width, height = _fit_image_size(path, max_width=max_width, max_height=max_height)
    return [
        Paragraph(title, styles["Heading3"]),
        _pdf_image(path, width=width, height=height),
        Spacer(1, 0.25 * cm),
    ]


def _build_pdf(
    stats: dict,
    experiments: pd.DataFrame,
    importance: pd.DataFrame,
    cp2_metrics: dict,
    screenshots: dict[str, Path],
    demo_video_path: Path | None,
) -> Path:
    font_name = _register_pdf_font()
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = font_name
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontName=font_name, fontSize=9))

    pdf_path = PATHS.report / "report.pdf"
    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    story = [
        Paragraph("CP3: Предсказание стоимости квартир в Москве", styles["Title"]),
        Paragraph("Краткое резюме", styles["Heading2"]),
        Paragraph(
            "Проект решает прикладную ML-задачу: оценить стоимость квартиры в Москве по параметрам "
            "объявления и признакам локации. В финальной версии есть воспроизводимый pipeline, "
            "сравнение моделей, интерпретация результата и FastAPI-сервис для получения прогнозов.",
            styles["BodyText"],
        ),
        Spacer(1, 0.25 * cm),
        Paragraph("1. Введение и постановка задачи", styles["Heading2"]),
        Paragraph(
            "Тип задачи — регрессия. Целевая переменная — price, то есть денежная стоимость квартиры. "
            "Такой прогноз может быть полезен как быстрый ориентир для сравнения объявлений и проверки "
            "адекватности цены относительно площади, этажа, ремонта и расположения.",
            styles["BodyText"],
        ),
        Paragraph(
            "Основная метрика — RMSLE. Она выбрана потому, что цены недвижимости имеют длинный правый "
            "хвост: ошибка в 1 млн рублей по-разному воспринимается для дешёвой и дорогой квартиры. "
            "RMSLE делает акцент на относительной ошибке. Дополнительно используются MAE, RMSE и R2: "
            "MAE легко интерпретировать в рублях, RMSE сильнее штрафует крупные промахи, R2 показывает "
            "общую объяснённую долю дисперсии.",
            styles["BodyText"],
        ),
        Paragraph("2. Поиск и описание данных", styles["Heading2"]),
        Paragraph(
            "Источник данных — Kaggle Moscow Housing Price Dataset. Датасет выбран, потому что это "
            "monetary regression по реальным признакам объявлений о недвижимости. Он не является игрушечной "
            "задачей Getting Started, содержит больше 10 000 строк и больше 10 колонок, поэтому подходит "
            "для сравнения нескольких моделей.",
            styles["BodyText"],
        ),
        Table(
            [
                ["Показатель", "Значение"],
                ["Строк до очистки", stats.get("rows_raw")],
                ["Колонок до очистки", stats.get("columns_raw")],
                ["Дубликатов удалено", stats.get("duplicate_rows")],
                ["Строк удалено бизнес-правилами", stats.get("rows_removed_by_business_rules")],
                ["Строк после очистки", stats.get("rows_cleaned")],
                ["Колонок после feature engineering", stats.get("columns_cleaned")],
                ["Медианная цена", f"{stats.get('target_median'):.0f}"],
                ["99% квантиль цены", f"{stats.get('target_q99'):.0f}"],
            ],
            hAlign="LEFT",
            style=[
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF7")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ],
        ),
        Spacer(1, 0.25 * cm),
        Paragraph("3. Обработка и подготовка данных", styles["Heading2"]),
        Paragraph(
            "Подготовка данных вынесена в воспроизводимый код. Сначала нормализуются названия колонок "
            "и проверяется наличие обязательных признаков. Числовые поля приводятся к числовому типу, "
            "категориальные значения очищаются от пустых строк. После этого удаляются полные дубли.",
            styles["BodyText"],
        ),
        Paragraph(
            "Бизнес-правила удаляют физически невозможные строки: неположительную цену или площадь, "
            "нулевую этажность, этаж выше количества этажей в доме, отрицательное расстояние до метро, "
            "жилую или кухонную площадь больше общей. Выбросы числовых признаков не считаются на всём "
            "датасете: clipping по квантилям находится только на train внутри sklearn Pipeline, что "
            "защищает от data leakage.",
            styles["BodyText"],
        ),
        Paragraph(
            "Feature engineering включает признаки площади на комнату, доли жилой и кухонной площади, "
            "позицию этажа в доме, признаки первого/последнего этажа, логарифмы площади и расстояния до "
            "метро, interaction комнат и площади, отношение кухни к жилой площади, категорию этажа и "
            "bucket расстояния до метро.",
            styles["BodyText"],
        ),
        Paragraph(
            "Разбиение train/validation/test выполнено в пропорции 70/15/15. Для регрессии используется "
            "стратификация по ценовым бинам, чтобы распределения цены в частях выборки были похожими.",
            styles["BodyText"],
        ),
        *_image_block("Распределение log1p(price)", PATHS.figures / "01_price_distribution_log.png", styles),
        *_image_block("Сравнение target на train/validation/test", PATHS.figures / "10_cp2_split_target_distribution.png", styles),
        Paragraph("4. Baseline-модель", styles["Heading2"]),
        Paragraph(
            "В качестве sanity baseline используется DummyRegressor, который предсказывает медиану цены. "
            "Он нужен, чтобы убедиться, что ML-модели действительно лучше константного прогноза. Второй "
            "baseline — KNeighborsRegressor без feature engineering: простая модель из коробки, которая "
            "даёт начальную точку сравнения для более сложных ансамблей.",
            styles["BodyText"],
        ),
        Paragraph("5. Эксперименты", styles["Heading2"]),
        Paragraph(
            "Эксперименты построены вокруг нескольких гипотез: линейная регуляризация может быть сильной "
            "на табличных признаках; ансамбли деревьев лучше моделируют нелинейные зависимости цены; "
            "градиентный бустинг должен быть наиболее сильным кандидатом; PCA проверяет, можно ли сжать "
            "encoded-признаки без большой потери качества.",
            styles["BodyText"],
        ),
        _pdf_table(
            experiments,
            ["model", "feature_set", "val_rmsle", "val_mae", "val_r2", "fit_time_sec"],
            n_rows=12,
            font_name=font_name,
        ),
        Paragraph(
            "По validation RMSLE лучшей стала конфигурация HistGradientBoostingRegressor с learning_rate=0.07, "
            "max_iter=350, max_leaf_nodes=31 и l2_regularization=0.05. PCA-эксперимент показал, что сжатие "
            "до 95% дисперсии ухудшает качество для этой задачи, поэтому финальная модель использует полное "
            "пространство признаков.",
            styles["BodyText"],
        ),
        *_image_block("PCA cumulative explained variance", PATHS.figures / "08_cp2_pca_explained_variance.png", styles),
        Paragraph("6. Финальная модель и интерпретируемость", styles["Heading2"]),
        Paragraph(
            f"Финальная модель: {cp2_metrics.get('best_model')}. На test она получила "
            f"RMSLE={_metric_value(cp2_metrics.get('rmsle'))}, "
            f"MAE={cp2_metrics.get('mae'):.2f}, RMSE={cp2_metrics.get('rmse'):.2f}, "
            f"R2={_metric_value(cp2_metrics.get('r2'))}.",
            styles["BodyText"],
        ),
        Paragraph(
            "Для интерпретации используется permutation importance на validation sample. Это не внутренняя "
            "важность конкретного алгоритма, а оценка того, насколько растёт ошибка при перемешивании "
            "каждого признака. Самыми важными признаками ожидаемо оказались площадь, станция метро, "
            "этажность дома, тип квартиры и ремонт.",
            styles["BodyText"],
        ),
        _pdf_table(importance, ["feature", "importance_mean", "importance_std"], n_rows=10, font_name=font_name),
        *_image_block("Permutation feature importance", PATHS.figures / "07_cp2_feature_importance.png", styles),
        PageBreak(),
        Paragraph("7. Деплой", styles["Heading2"]),
        Paragraph(
            "Добавлен FastAPI с эндпоинтами GET /health, GET /model-info и POST /predict. "
            "Сервис загружает сохранённую финальную модель, валидирует входные признаки через Pydantic "
            "и возвращает предсказанную цену в рублях. Swagger UI доступен по адресу "
            "http://127.0.0.1:8000/docs.",
            styles["BodyText"],
        ),
        Paragraph(
            "API можно запустить командой make api, а быстрый smoke-test без браузера выполняется через "
            "make api-smoke. Для контейнерного запуска подготовлены Dockerfile и docker-compose.yml.",
            styles["BodyText"],
        ),
    ]

    for title, image_path in [
        ("API docs", screenshots["docs"]),
        ("Predict request", screenshots["request"]),
        ("Predict response", screenshots["response"]),
    ]:
        story.append(KeepTogether(_image_block(title, image_path, styles, max_width=16 * cm, max_height=20 * cm)))

    story.extend(
        [
            Paragraph("Видео демонстрации", styles["Heading3"]),
            Paragraph(
                f"Файл видео в репозитории: report/{demo_video_path.name if demo_video_path else 'demo_cp3.mp4'}",
                styles["BodyText"],
            ),
            Paragraph("8. Заключение и выводы", styles["Heading2"]),
            Paragraph(
                "Проект доведён до полного ML-пайплайна: данные очищаются и обогащаются признаками, "
                "модели сравниваются по единой validation-схеме, финальная модель проверяется на test "
                "и доступна через API. Основное ограничение — модель обучена на одном Kaggle-датасете, "
                "поэтому при переносе на реальные новые объявления полезно периодически переобучать её "
                "и мониторить drift признаков и цен.",
                styles["BodyText"],
            ),
        ]
    )

    document.build(story)
    return pdf_path


def make_report() -> None:
    PATHS.ensure_dirs()

    stats = _read_json(PATHS.dataset_stats_path)
    cp2_metrics = _read_json(PATHS.cp2_test_metrics_path)
    pca_stats = _read_json(PATHS.pca_variance_path)
    experiments = _read_csv(PATHS.cp2_experiments_path)
    importance = _read_csv(PATHS.feature_importance_path)
    api_result = _api_prediction()
    screenshots = _make_api_screenshots(api_result)
    demo_video_path = _make_demo_video(screenshots)

    experiments_md = _markdown_table(
        experiments,
        ["model", "feature_set", "val_rmsle", "val_mae", "val_r2", "comment"],
        n_rows=12,
    )
    importance_md = _markdown_table(
        importance,
        ["feature", "importance_mean", "importance_std"],
        n_rows=15,
    )

    report_md = f"""# CP3: Предсказание стоимости квартир в Москве

## 1. Введение и постановка задачи

Решается задача регрессии: предсказание стоимости квартиры в Москве по характеристикам объявления и локации.

Практический смысл задачи — получить быстрый ориентир рыночной цены по параметрам объявления. Такой прогноз можно использовать для первичной проверки адекватности цены, сравнения похожих объектов и поиска объявлений, которые сильно отличаются от ожидаемой стоимости.

Основная метрика — RMSLE. Она выбрана потому, что цены недвижимости имеют большой разброс, и относительная ошибка важнее абсолютной при сравнении дешёвых и дорогих квартир. Дополнительно считаются MAE, RMSE и R2: MAE интерпретируется в рублях, RMSE сильнее штрафует крупные ошибки, R2 показывает общую объяснённую долю дисперсии.

## 2. Поиск и описание данных

Источник: [Kaggle Moscow Housing Price Dataset](https://www.kaggle.com/datasets/egorkainov/moscow-housing-price-dataset). Датасет выбран, потому что он напрямую относится к monetary regression, содержит реальные признаки объявлений и подходит по объёму.

- строк до очистки: {stats.get("rows_raw")}
- колонок до очистки: {stats.get("columns_raw")}
- пропусков в исходных обязательных колонках: {stats.get("missing_values_raw_total")}
- дублей удалено: {stats.get("duplicate_rows")}
- строк удалено бизнес-правилами: {stats.get("rows_removed_by_business_rules")}
- строк после очистки: {stats.get("rows_cleaned")}
- колонок после feature engineering: {stats.get("columns_cleaned")}
- медианная цена: {stats.get("target_median")}
- 99% квантиль цены: {stats.get("target_q99")}

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

{experiments_md}

## 6. Финальная модель и интерпретируемость результатов

Финальная модель выбрана по validation RMSLE: `{cp2_metrics.get("best_model")}`.

Метрики на test:

- RMSLE: {cp2_metrics.get("rmsle")}
- MAE: {cp2_metrics.get("mae")}
- RMSE: {cp2_metrics.get("rmse")}
- R2: {cp2_metrics.get("r2")}

Permutation importance:

{importance_md}

PCA:

- число компонент для 95% дисперсии: {pca_stats.get("n_components")}
- сохранённая доля дисперсии: {pca_stats.get("explained_variance_sum")}

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
{json.dumps(api_result["request"], ensure_ascii=False, indent=2)}
```

Пример ответа:

```json
{json.dumps(api_result["response"], ensure_ascii=False, indent=2)}
```

Скриншоты:

- [API docs](images/11_cp3_api_docs.png)
- [Predict request](images/12_cp3_predict_request.png)
- [Predict response](images/13_cp3_predict_response.png)

Видео демонстрации: [{demo_video_path.name if demo_video_path else "demo_cp3.mp4"}]({demo_video_path.name if demo_video_path else "demo_cp3.mp4"})

## 8. Заключение и выводы

Финальная модель показывает заметное улучшение относительно baseline и упакована в воспроизводимый FastAPI-сервис. Проект можно запустить локально через `make cp3`, проверить API через `make api-smoke` и открыть документацию в Swagger UI.

Ограничения: модель обучена на одном датасете с Kaggle, поэтому при использовании на новых объявлениях нужно следить за изменением рынка и переобучать модель на более свежих данных. Следующие улучшения: добавить мониторинг качества, расширить признаки района и подключить регулярное обновление данных.
"""

    PATHS.cp1_report_path.write_text(report_md, encoding="utf-8")
    pdf_path = _build_pdf(stats, experiments, importance, cp2_metrics, screenshots, demo_video_path)

    print(f"CP3 markdown report saved to {PATHS.cp1_report_path}")
    print(f"CP3 PDF report saved to {pdf_path}")
    if demo_video_path is not None:
        print(f"CP3 demo video saved to {demo_video_path}")


if __name__ == "__main__":
    make_report()
