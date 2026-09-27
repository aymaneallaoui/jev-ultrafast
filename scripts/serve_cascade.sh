#!/usr/bin/env bash
# Serve a primary and a verifier Kev checkpoint for cascade mode (primary on 8009, verifier on 8010).
# usage: scripts/serve_cascade.sh [primary-mode] [verifier-mode]
set -euo pipefail
cd "$(dirname "$0")/.."
PRIMARY=${1:-08b-d1a}
VERIFIER=${2:-4b-nf4}
LOG_DIR=${LOG_DIR:-$HOME/jev-traces/logs}
mkdir -p "$LOG_DIR"
PIDS=()
stop() {
  trap - EXIT INT TERM
  [ ${#PIDS[@]} -eq 0 ] || kill "${PIDS[@]}" 2> /dev/null || true
  wait 2> /dev/null || true
}
trap stop EXIT INT TERM
serve() {
  scripts/serve_local.sh "$1" "$2" > "$LOG_DIR/serve-cascade-$1.log" 2>&1 &
  PIDS+=($!)
}
wait_ready() {
  for _ in $(seq 120); do
    curl -sf "http://127.0.0.1:$1/v1/models" > /dev/null && return 0
    kill -0 "${PIDS[-1]}" 2> /dev/null || { echo "server on port $1 exited; see $LOG_DIR" >&2; exit 1; }
    sleep 5
  done
  echo "server on port $1 not ready after 10 minutes" >&2
  exit 1
}
serve "$VERIFIER" 8010
wait_ready 8010
serve "$PRIMARY" 8009
wait_ready 8009
echo "export TYPESAFE_BASE_URL=http://127.0.0.1:8009"
echo "export JEV_VERIFIER_BASE_URL=http://127.0.0.1:8010"
wait
