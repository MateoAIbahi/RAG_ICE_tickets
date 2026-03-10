import os

os.environ["SSL_CERT_FILE"] = os.getenv("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")
os.environ["REQUESTS_CA_BUNDLE"] = os.getenv("REQUESTS_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")
os.environ["CURL_CA_BUNDLE"] = os.getenv("CURL_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")

import time
import torch

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


def run_probe():
    model_name = os.getenv("EMBED_MODEL", "Metric-AI/ColQwen2.5-3b-multilingual-v1.0")
    device = os.getenv("DEVICE", "cpu")

    print(f"[PROBE] Loading model: {model_name}")
    print(f"[PROBE] Device: {device}")

    start = time.time()

    model = ColQwen2_5.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
    ).eval()

    processor = ColQwen2_5_Processor.from_pretrained(model_name)

    load_time = time.time() - start
    print(f"[PROBE] Model loaded in {load_time:.2f}s")

    query = "Comment réinitialiser un mot de passe dans l'ERP ?"

    batch = processor.process_queries([query]).to(device)

    with torch.no_grad():
        embeddings = model(**batch)

    print(f"[PROBE] Query embedding shape: {tuple(embeddings.shape)}")

    if len(embeddings.shape) == 3:
        print(f"[PROBE] batch={embeddings.shape[0]}, n_vectors={embeddings.shape[1]}, dim={embeddings.shape[2]}")
    elif len(embeddings.shape) == 2:
        print(f"[PROBE] n_vectors={embeddings.shape[0]}, dim={embeddings.shape[1]}")

    print("[PROBE] OK")