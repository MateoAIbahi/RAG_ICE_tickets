"""
Compatibilité : l'ancien CMD du conteneur (et peut-être le cron actuel) lance
`python -m src.ingestion.run`, qui n'ingérait QUE les documents. Ce point
d'entrée lance désormais la synchronisation complète.
"""
import sys

from src.ingestion.sync import main

if __name__ == "__main__":
    sys.exit(main())
