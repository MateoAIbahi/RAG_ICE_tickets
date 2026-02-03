import os
import time
from src.ingestion.sync import sync_sharepoint

def main():
    # Lit l’intervalle depuis .env (en secondes)
    interval = int(os.getenv("SYNC_INTERVAL_SECONDS", "300"))

    print(f"[INGESTION] Starting SharePoint sync loop (every {interval}s)")

    while True:
        try:
            sync_sharepoint()
        except Exception as e:
            print(f"[INGESTION] Error during sync: {e}")

        time.sleep(interval)

if __name__ == "__main__":
    main()
