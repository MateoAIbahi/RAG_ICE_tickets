FROM python:3.11-slim

WORKDIR /app

# libreoffice : conversion doc/docx/odt/ppt -> pdf
# tesseract    : OCR des pages scannées (léger en RAM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libreoffice-writer-nogui \
    libreoffice-impress-nogui \
    tesseract-ocr \
    tesseract-ocr-fra \
    && rm -rf /var/lib/apt/lists/*

COPY certs/firebox.crt /usr/local/share/ca-certificates/firebox.crt
RUN update-ca-certificates

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV HF_HUB_DISABLE_TELEMETRY=1

RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-ingestion.txt .
RUN pip install --no-cache-dir -r requirements-ingestion.txt

COPY src/ /app/src/
ENV PYTHONPATH=/app

CMD ["python", "-m", "src.ingestion.sync"]
