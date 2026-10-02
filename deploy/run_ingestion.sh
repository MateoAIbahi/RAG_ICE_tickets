#!/bin/sh
# Ingestion RAG ICE (appelé par le timer systemd ou le cron du vendredi soir).
#   deploy/run_ingestion.sh                    -> synchronisation complète
#   deploy/run_ingestion.sh --docs-only        -> uniquement les documents en attente
#   deploy/run_ingestion.sh --maintenance      -> active l'écran de maintenance pendant
#                                                 la synchro, le retire si elle réussit
#                                                 (à utiliser pour la 1re réindexation)
set -eu
cd "$(dirname "$0")/.."
COMPOSE="${COMPOSE_CMD:-podman-compose}"

MAINT=0
ARGS=""
for a in "$@"; do
  case "$a" in
    --maintenance) MAINT=1 ;;
    *) ARGS="$ARGS $a" ;;
  esac
done

mkdir -p /data/uploads /data/archive /data/state /data/models
[ "$MAINT" = 1 ] && deploy/maintenance.sh on

set +e
# flock : jamais deux ingestions en parallèle (RAM)
# shellcheck disable=SC2086
flock -n /tmp/rag-ice-ingestion.lock \
    $COMPOSE --profile jobs run --rm ingestion python -m src.ingestion.sync $ARGS
RC=$?
set -e

if [ "$MAINT" = 1 ]; then
  if [ "$RC" = 0 ]; then
    deploy/maintenance.sh off
  else
    echo "La synchronisation a échoué (code $RC) : la maintenance reste ACTIVE." >&2
    echo "Corriger puis relancer, ou : deploy/maintenance.sh off" >&2
  fi
fi
exit "$RC"
