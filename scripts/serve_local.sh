#!/usr/bin/env bash
# Serve a fine-tuned Kev checkpoint on this machine with the /v1/systemone API Jev's client speaks.
# usage: scripts/serve_local.sh [08b|08b-d1a|08b-d1b|08b-6144|4b|4b-int8|4b-nf4|<run dir in ~/kev>] [port]
#        scripts/serve_local.sh accurate   jev-4b nf4 (8009) + text helper on the GPU (8080); 8 GB GPUs
#        scripts/serve_local.sh fast       08b-d1a (8009) + 4b-nf4 verifier (8010) + text helper on the CPU; 16 GB GPUs
# A preset starts its servers largest first, prints the exports to use, and stops everything on exit. Logs: $LOG_DIR/serve-<preset>-*.log
# The 4b-int8 and 4b-nf4 modes need ~/kev on the jev-serve-memory branch (KEV_LOAD_IN_8BIT / KEV_LOAD_IN_4BIT).
set -euo pipefail
SCRIPTS_DIR=$(cd "$(dirname "$0")" && pwd)
case "${1:-}" in
  accurate | fast)
    PRESET=$1
    source "$SCRIPTS_DIR/serve_common.sh"
    if [ "$PRESET" = fast ]; then
      start_child "$LOG_DIR/serve-fast-verifier.log" "$SCRIPTS_DIR/serve_local.sh" 4b-nf4 8010
      wait_ready http://127.0.0.1:8010/v1/models 120
      start_child "$LOG_DIR/serve-fast-primary.log" "$SCRIPTS_DIR/serve_local.sh" 08b-d1a 8009
    else
      start_child "$LOG_DIR/serve-accurate-decision.log" "$SCRIPTS_DIR/serve_local.sh" 4b-nf4 8009
    fi
    wait_ready http://127.0.0.1:8009/v1/models 120
    if [ "$PRESET" = fast ]; then
      start_child "$LOG_DIR/serve-fast-text.log" env TEXT_NGL=0 "$SCRIPTS_DIR/serve_text_local.sh" 8080
    else
      start_child "$LOG_DIR/serve-accurate-text.log" "$SCRIPTS_DIR/serve_text_local.sh" 8080
    fi
    echo "waiting for the text helper (the first run may download the model, up to 15 minutes)" >&2
    wait_ready http://127.0.0.1:8080/v1/models 180
    echo "export TYPESAFE_BASE_URL=http://127.0.0.1:8009"
    echo "export TYPESAFE_API_KEY=local"
    if [ "$PRESET" = fast ]; then
      echo "export JEV_VERIFIER_BASE_URL=http://127.0.0.1:8010"
      echo "export JEV_VERIFIER_API_KEY=local"
    fi
    echo "export TEXT_MODEL_BASE_URL=http://127.0.0.1:8080/v1"
    echo "export TEXT_MODEL=qwen3-1.7b"
    echo "export TEXT_MODEL_API_KEY=local"
    [ "$PRESET" = fast ] && echo "export TEXT_TIMEOUT_S=30"
    wait
    exit 0
    ;;
esac
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
