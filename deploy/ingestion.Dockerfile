FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir psycopg2-binary

COPY src/ /app/src/
ENV PYTHONPATH=/app

CMD ["python", "-m", "src.ingestion.run"]