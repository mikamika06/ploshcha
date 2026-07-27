#!/usr/bin/env bash
set -euo pipefail

MODELS="$HOME/models"
PORT="${2:-8080}"
case "${1:-}" in
  lapa)    GGUF="$MODELS/lapa/lapa-v0.1.2-instruct-Q8_0.gguf" ;;
  gemma12) GGUF="$MODELS/gemma-3-12b-it/gemma-3-12b-it-Q8_0.gguf" ;;
  gemma27) GGUF="$MODELS/gemma-3-27b-it/gemma-3-27b-it-Q4_K_M.gguf" ;;
  mamay27) GGUF="$MODELS/mamay-3-27b-it/MamayLM-Gemma-3-27B-IT-v2.0-Q4_K_M.gguf" ;;
  bge)     GGUF="$MODELS/bge-m3/bge-m3-q8_0.gguf" ;;
  *) echo "usage: serve_local.sh lapa|gemma12|gemma27|mamay27|bge [port]"; exit 1 ;;
esac

[ -f "$GGUF" ] || { echo "no file: $GGUF"; exit 1; }
if [ "$1" = "bge" ]; then
  exec llama-server -m "$GGUF" --host 127.0.0.1 --port "$PORT" --embedding --pooling cls -ngl 999 --log-prefix
fi
exec llama-server -m "$GGUF" --host 127.0.0.1 --port "$PORT" -c 8192 -ngl 999 --log-prefix
