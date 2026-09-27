#!/usr/bin/env bash
# Serve a fine-tuned Kev checkpoint on this machine with the /v1/systemone API Jev's client speaks.
# usage: scripts/serve_local.sh [08b|08b-d1a|08b-d1b|08b-6144|4b|4b-int8|4b-nf4|<run dir in ~/kev>] [port]
# The 4b-int8 and 4b-nf4 modes need ~/kev on the jev-serve-memory branch (KEV_LOAD_IN_8BIT / KEV_LOAD_IN_4BIT).
set -euo pipefail
KEV_DIR=${KEV_DIR:-$HOME/kev}
case "${1:-08b}" in
  08b) RUN=runs/jev-08b ;;
  08b-d1a) RUN=runs/jev-08b-d1a ;;
  08b-d1b) RUN=runs/jev-08b-d1b ;;
  08b-6144) RUN=runs/jev-08b-6144 ;;
  4b) RUN=runs/jev-4b ;;
  4b-int8) RUN=runs/jev-4b; export KEV_LOAD_IN_8BIT=1 ;;
  4b-nf4) RUN=runs/jev-4b; export KEV_LOAD_IN_4BIT=1 ;;
  *) RUN=$1 ;;
esac
PORT=${2:-8009}
cd "$KEV_DIR"
[ -e "$RUN" ] || { echo "no checkpoint at $KEV_DIR/$RUN" >&2; exit 1; }
if [[ "${1:-08b}" == 4b* ]]; then
  # A 16 GB GPU: CUDA graph buffers alone take ~5.5 GB on Kev-4B. See docs/kev-serve-memory.md.
  export KEV_CUDA_GRAPHS=${KEV_CUDA_GRAPHS:-0} KEV_MAX_BATCH=${KEV_MAX_BATCH:-1}
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
fi
exec uv run --extra serve python -m kev.serve --run "$RUN" --port "$PORT"
