FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

COPY . .

CMD ["sh", "-c", "python -m moscow_housing.prepare_data && python -m moscow_housing.train_cp2 && uvicorn moscow_housing.api:app --host 0.0.0.0 --port 8000"]
