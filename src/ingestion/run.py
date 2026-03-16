import shutil
import subprocess
from pathlib import Path

from src.ingestion.pdf_ingest import ingest_pdf_file


UPLOAD_DIR = Path("/data/uploads")
SUPPORTED_DIRECT_PDF = {".pdf"}
SUPPORTED_CONVERTIBLE = {".doc", ".docx", ".odt"}


def convert_to_pdf(input_file: Path) -> Path:
    """
    Convertit un fichier bureautique en PDF avec LibreOffice headless.
    Retourne le chemin du PDF généré.
    """
    output_dir = input_file.parent

    cmd = [
        "libreoffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(input_file),
    ]

    print(f"[INGESTION] Converting to PDF: {input_file}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Conversion failed for {input_file}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    pdf_path = input_file.with_suffix(".pdf")

    if not pdf_path.exists():
        raise FileNotFoundError(f"Converted PDF not found: {pdf_path}")

    print(f"[INGESTION] Conversion OK: {pdf_path}")
    return pdf_path


def move_metadata_to_pdf_name(source_file: Path, pdf_file: Path):
    """
    Si on convertit mon_doc.docx -> mon_doc.pdf,
    on déplace aussi :
      mon_doc.docx.meta.json -> mon_doc.pdf.meta.json
    comme ça il ne reste pas de .doc.meta.json orphelin.
    """
    source_meta = source_file.parent / f"{source_file.name}.meta.json"
    pdf_meta = pdf_file.parent / f"{pdf_file.name}.meta.json"

    if source_meta.exists():
        shutil.move(str(source_meta), str(pdf_meta))
        print(f"[INGESTION] Metadata moved: {source_meta} -> {pdf_meta}")
    else:
        print(f"[INGESTION] No metadata file found for {source_file}")


def main():
    print("[INGESTION] Starting document ingestion")

    if not UPLOAD_DIR.exists():
        print("[INGESTION] Upload directory does not exist:", UPLOAD_DIR)
        return

    all_files = [
        p for p in UPLOAD_DIR.rglob("*")
        if p.is_file() and not p.name.endswith(".meta.json")
    ]

    if not all_files:
        print("[INGESTION] No files found")
        return

    print(f"[INGESTION] {len(all_files)} file(s) found")

    success = 0
    failed = 0

    for file_path in all_files:
        print(f"[INGESTION] Processing {file_path}")

        try:
            suffix = file_path.suffix.lower()

            if suffix in SUPPORTED_DIRECT_PDF:
                pdf_path = file_path

            elif suffix in SUPPORTED_CONVERTIBLE:
                pdf_path = convert_to_pdf(file_path)

                # Copie les metadata vers le nom du PDF
                move_metadata_to_pdf_name(file_path, pdf_path)

                # Supprime le fichier bureautique source après conversion réussie
                file_path.unlink(missing_ok=True)
                print(f"[INGESTION] Source file removed after conversion: {file_path}")

            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

            ingest_pdf_file(str(pdf_path))
            success += 1

        except Exception as e:
            print(f"[INGESTION] ERROR on {file_path}: {e}")
            failed += 1

    print("\n[INGESTION] Summary")
    print("Success:", success)
    print("Failed :", failed)


if __name__ == "__main__":
    main()