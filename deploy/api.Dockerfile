FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.9.1 \
    torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    transformers==4.40.2 \
    peft==0.10.0 \
    accelerate

RUN pip install --no-cache-dir git+https://github.com/illuin-tech/colpali@main

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -r /app/requirements.txt

COPY src/ /app/src/
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]