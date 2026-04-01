FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY certs/firebox.crt /usr/local/share/ca-certificates/firebox.crt
RUN update-ca-certificates

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    torch==2.9.1 \
    torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    transformers==4.40.2 \
    peft==0.10.0 \
    accelerate

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    psycopg2-binary \
    python-multipart

RUN pip install --no-cache-dir git+https://github.com/illuin-tech/colpali@main

RUN pip install --no-cache-dir requests

COPY src/ /app/src/
ENV PYTHONPATH=/app

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]