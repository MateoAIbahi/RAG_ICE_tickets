FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY certs/firebox.crt /usr/local/share/ca-certificates/firebox.crt
RUN update-ca-certificates

RUN pip install --no-cache-dir -U pip

RUN pip install --no-cache-dir psycopg2-binary

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    pillow \
    accelerate \
    "transformers>=4.45.0" \
    "colpali-engine==0.3.7"

COPY src/ /app/src/
ENV PYTHONPATH=/app

CMD ["python", "-m", "src.ingestion.run"]