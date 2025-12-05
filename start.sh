#!/bin/bash
set -euo pipefail

ROLE=${NODE_ROLE:-worker}

if [ "$ROLE" = "head" ]; then
  echo "[start.sh] Starting Ray head..."
  ray start --head --port=6379 --dashboard-port=8265
  echo "[start.sh] Ray head started. Running training (main.py) on head..."
  python /app/main.py
  echo "[start.sh] Training finished. Stopping Ray on head..."
  ray stop
  echo "[start.sh] Head shutdown complete."
else
  if [ -z "${RAY_HEAD_ADDRESS:-}" ]; then
    echo "[start.sh] ERROR: RAY_HEAD_ADDRESS not set (expected HOST:PORT). Exiting."
    exit 1
  fi
  echo "[start.sh] Starting Ray worker and connecting to $RAY_HEAD_ADDRESS..."
  ray start --address="$RAY_HEAD_ADDRESS"
  echo "[start.sh] Worker started. Keeping container alive..."
  tail -f /dev/null
fi
