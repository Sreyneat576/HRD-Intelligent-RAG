FROM python:3.12-slim

ARG WITH_PADDLE=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir ".[ocr-tesseract,eureka,ui]" \
    && if [ "$WITH_PADDLE" = "true" ]; then pip install --no-cache-dir ".[ocr-paddle]"; fi

COPY alembic.ini ./
COPY migrations ./migrations
COPY ui ./ui

EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
