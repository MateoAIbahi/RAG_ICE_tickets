# RAG SharePoint Entreprise

## Architecture
- Ingestion : job (toutes les 5 min) → SharePoint → embeddings → DB
- API : FastAPI → retrieval → réponse + sources
- DB : Postgres (ou Vespa)
- Déploiement : Podman (Linux)

## Lancer
1. Copier .env.example → .env
2. podman-compose up -d
