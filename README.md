# RAG ICE

Assistant de recherche sur la documentation PCCN (V1/V2/V3) et les tickets
clients (Sylob, Mantis), avec un LLM Devstral hébergé sur un serveur dédié.

## Architecture

| Service     | Rôle                                                            | En continu ? |
|-------------|-----------------------------------------------------------------|--------------|
| `db`        | Postgres + pgvector (chunks, embeddings, état de synchro)       | oui          |
| `api`       | FastAPI : recherche hybride (vecteurs + mots-clés), upload, admin | oui        |
| `web`       | nginx : interface + proxy `/api`                                 | oui          |
| `ingestion` | job ponctuel : documents en attente + tickets Sylob/Mantis       | vendredi soir|

Embedding : `BAAI/bge-m3` (~2,5 Go de RAM par processus, CPU). Configurable via
`.env` (voir `.env.example`), mais **changer de modèle impose une réindexation complète**.

## Dossiers sur l'hôte

```
/data/uploads   boîte d'entrée : fichiers reçus, pas encore ingérés (+ _echecs/)
/data/archive   originaux ingérés (permet une réindexation future)
/data/state     drapeau du mode maintenance
/data/models    cache des modèles Hugging Face
```

## Exploitation courante

```sh
# Ingestion manuelle (complète, ou documents seuls)
deploy/run_ingestion.sh
deploy/run_ingestion.sh --docs-only

# Importer un lot livré par ICE (archive .7z extraite au préalable)
python3 src/ingestion/import_tree.py "/chemin/Fichiers pour IA" --dry-run   # vérifier le classement
python3 src/ingestion/import_tree.py "/chemin/Fichiers pour IA"             # copier dans la boîte d'entrée

# Mode maintenance (écran "Service en maintenance" pour les utilisateurs)
deploy/maintenance.sh on ["message personnalisé"]
deploy/maintenance.sh off
```

Planification hebdomadaire (vendredi 20h) avec systemd :

```sh
sudo cp deploy/systemd/rag-ice-ingestion.* /etc/systemd/system/
# adapter WorkingDirectory / ExecStart si le dépôt n'est pas dans /opt/RAG_ICE_tickets
sudo systemctl daemon-reload && sudo systemctl enable --now rag-ice-ingestion.timer
systemctl list-timers rag-ice-ingestion.timer
journalctl -u rag-ice-ingestion.service      # logs de la dernière exécution
```

Ou en cron (une seule des deux méthodes !) :
`0 20 * * 5 /opt/RAG_ICE_tickets/deploy/run_ingestion.sh >> /var/log/rag-ice-ingestion.log 2>&1`

## Passage à la v2 (migration depuis ColQwen) : procédure

À faire hors heures de travail. La première synchronisation réencode TOUS les
tickets et documents : compter plusieurs heures sur CPU.

1. **Sauvegarde** (la suppression des anciens vecteurs ColQwen est irréversible) :
   `podman exec rag_db pg_dump -U rag -Fc ragdb > ragdb_avant_v2.dump`
2. Optionnel, pour savoir quelles pages n'étaient trouvables que par l'image :
   ```sql
   SELECT source_path, pccn_version, count(*) FROM documents
   WHERE source_type='pdf' AND content LIKE '[PAGE_IMAGE_ONLY]%' GROUP BY 1,2 ORDER BY 3 DESC;
   ```
3. Désactiver l'ancienne planification (`crontab -e` / ancien timer).
4. Activer la maintenance **avant** de toucher aux conteneurs :
   `deploy/maintenance.sh on`
5. Créer `.env` depuis `.env.example` (au minimum `ADMIN_PASSWORD`,
   `ERP_DATABASE_URL`, `MANTIS_DB_PASSWORD`), puis :
   `mkdir -p /data/uploads /data/archive /data/state /data/models`
6. Reconstruire et redémarrer : `podman-compose build && podman-compose up -d`
   (au 1er démarrage, l'API télécharge bge-m3 et applique la migration).
7. Lancer la réindexation complète. Elle retire la maintenance seule si tout réussit :
   `deploy/run_ingestion.sh --maintenance`
8. Installer le timer du vendredi (voir plus haut).

Retour arrière : restaurer le dump (`pg_restore -U rag -d ragdb --clean`) et
redéployer l'ancienne version.

## Sécurité

- Le dépôt ne doit contenir **aucun secret** : tout passe par `.env` (non versionné).
- Le token Hugging Face présent dans l'historique doit être révoqué (il n'est plus utilisé).
- Postgres n'écoute plus que sur `127.0.0.1`.
