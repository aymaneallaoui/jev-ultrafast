#!/usr/bin/env bash
# Run the 28-task smoke set with decisions from a locally served Kev checkpoint (start scripts/serve_local.sh first).
# Traces go to ~/jev-traces/kev-smoke-jev-<model> so they never mix with TypeSafe Jev labels.
# usage: scripts/kev_smoke.sh [08b|08b-6144|4b] [port]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-08b}
PORT=${2:-8009}
TAG=jev-$MODEL
curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null || { echo "no kev.serve on port $PORT; run scripts/serve_local.sh $MODEL $PORT" >&2; exit 1; }
curl -sf http://127.0.0.1:9222/json/version > /dev/null || { echo "jev-chrome is not running on port 9222" >&2; exit 1; }
export TYPESAFE_BASE_URL="http://127.0.0.1:$PORT" TYPESAFE_API_KEY=local TYPESAFE_MODEL="$TAG"
export BU_CDP_URL=http://127.0.0.1:9222 TRACE_DIR="$HOME/jev-traces/kev-smoke-$TAG"
exec uv run --env-file .env python scripts/collect.py --ids @scripts/smoke_ids.txt --repeat 1 --max-runtime 45m --model-tag "$TAG"
