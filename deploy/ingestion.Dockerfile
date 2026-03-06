FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip

# DB
RUN pip install --no-cache-dir psycopg2-binary

# CPU torch
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# ColQwen stack
RUN pip install --no-cache-dir \
    pillow \
    accelerate \
    "transformers>=4.45.0" \
    "colpali-engine==0.3.7"

COPY src/ /app/src/
ENV PYTHONPATH=/app

CMD ["python", "-m", "src.ingestion.run"]