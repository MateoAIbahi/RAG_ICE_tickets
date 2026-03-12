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

# PyTorch + TorchVision compatibles, même index CPU
RUN pip install --no-cache-dir \
    torch==2.9.1 \
    torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Dépendances ColQwen
RUN pip install --no-cache-dir pillow accelerate pymupdf
RUN pip install --no-cache-dir git+https://github.com/huggingface/transformers
RUN pip install --no-cache-dir git+https://github.com/illuin-tech/colpali@main

COPY src/ /app/src/
ENV PYTHONPATH=/app

CMD ["python", "-m", "src.ingestion.run"]