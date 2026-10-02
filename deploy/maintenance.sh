#!/bin/sh
# Mode maintenance de l'interface RAG ICE.
#   deploy/maintenance.sh on                 -> message par défaut
#   deploy/maintenance.sh on "Texte perso"   -> message personnalisé
#   deploy/maintenance.sh off
#   deploy/maintenance.sh status
set -eu
STATE_DIR="${MAINTENANCE_DIR:-/data/state}"
FLAG="$STATE_DIR/maintenance"

case "${1:-status}" in
  on)
    mkdir -p "$STATE_DIR"
    printf '%s\n' "${2:-}" > "$FLAG"
    echo "Maintenance ACTIVÉE (l'interface affiche l'écran de maintenance)."
    ;;
  off)
    rm -f "$FLAG"
    echo "Maintenance désactivée (service de nouveau accessible)."
    ;;
  status)
    if [ -f "$FLAG" ]; then echo "Maintenance : ACTIVE"; else echo "Maintenance : inactive"; fi
    ;;
  *)
    echo "Usage : $0 on [message] | off | status" >&2
    exit 2
    ;;
esac
