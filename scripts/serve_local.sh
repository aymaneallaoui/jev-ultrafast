#!/usr/bin/env bash
# Serve a fine-tuned Kev checkpoint on this machine with the /v1/systemone API Jev's client speaks.
# usage: scripts/serve_local.sh [08b|08b-6144|4b|<run dir in ~/kev>] [port]
set -euo pipefail
KEV_DIR=${KEV_DIR:-$HOME/kev}
case "${1:-08b}" in
  08b) RUN=runs/jev-08b ;;
  08b-6144) RUN=runs/jev-08b-6144 ;;
  4b) RUN=runs/jev-4b ;;
  *) RUN=$1 ;;
esac
PORT=${2:-8009}
cd "$KEV_DIR"
[ -e "$RUN" ] || { echo "no checkpoint at $KEV_DIR/$RUN" >&2; exit 1; }
if [ "${1:-08b}" = 4b ]; then
  # A 16 GB GPU: CUDA graph buffers alone take ~5.5 GB on Kev-4B. See docs/kev-serve-memory.md.
  export KEV_CUDA_GRAPHS=${KEV_CUDA_GRAPHS:-0} KEV_MAX_BATCH=${KEV_MAX_BATCH:-1}
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
fi
exec uv run --extra serve python -m kev.serve --run "$RUN" --port "$PORT"
