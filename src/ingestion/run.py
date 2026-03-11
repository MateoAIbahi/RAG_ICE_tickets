import os
from pathlib import Path

from src.ingestion.pdf_ingest import ingest_pdf_file


UPLOAD_DIR = Path("/data/uploads")


def main():
    print("[INGESTION] Starting PDF ingestion")

    if not UPLOAD_DIR.exists():
        print("[INGESTION] Upload directory does not exist:", UPLOAD_DIR)
        return

    pdf_files = list(UPLOAD_DIR.rglob("*.pdf"))

    if not pdf_files:
        print("[INGESTION] No PDF files found")
        return

    print(f"[INGESTION] {len(pdf_files)} PDF(s) found")

    success = 0
    failed = 0

    for pdf in pdf_files:
        print(f"[INGESTION] Processing {pdf}")

        try:
            ingest_pdf_file(str(pdf))
            success += 1

        except Exception as e:
            print(f"[INGESTION] ERROR on {pdf}: {e}")
            failed += 1

    print("\n[INGESTION] Summary")
    print("Success:", success)
    print("Failed :", failed)


if __name__ == "__main__":
    main()