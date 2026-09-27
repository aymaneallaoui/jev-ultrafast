#!/usr/bin/env bash
# Serve a primary and a verifier Kev checkpoint for cascade mode (primary on 8009, verifier on 8010).
# usage: scripts/serve_cascade.sh [primary-mode] [verifier-mode]
set -euo pipefail
cd "$(dirname "$0")/.."
PRIMARY=${1:-08b-d1a}
VERIFIER=${2:-4b-nf4}
source scripts/serve_common.sh
start_child "$LOG_DIR/serve-cascade-$VERIFIER.log" scripts/serve_local.sh "$VERIFIER" 8010
wait_ready http://127.0.0.1:8010/v1/models 120
start_child "$LOG_DIR/serve-cascade-$PRIMARY.log" scripts/serve_local.sh "$PRIMARY" 8009
wait_ready http://127.0.0.1:8009/v1/models 120
echo "export TYPESAFE_BASE_URL=http://127.0.0.1:8009"
echo "export JEV_VERIFIER_BASE_URL=http://127.0.0.1:8010"
wait
