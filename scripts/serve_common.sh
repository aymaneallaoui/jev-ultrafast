# Shared helpers for serve_cascade.sh and the serve_local.sh presets. Source it; do not run it.
LOG_DIR=${LOG_DIR:-$HOME/jev-traces/logs}
mkdir -p "$LOG_DIR"
PIDS=()
LAST_LOG=
stop_children() {
  trap - EXIT INT TERM
  local pid
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do kill -TERM -- "-$pid" 2> /dev/null || true; done
  sleep 2
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do kill -KILL -- "-$pid" 2> /dev/null || true; done
  wait 2> /dev/null || true
}
trap stop_children EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# setsid makes the child a session and process-group leader (pgid == pid), so stop_children reaches uv and python too.
start_child() {
  LAST_LOG=$1
  shift
  setsid "$@" > "$LAST_LOG" 2>&1 &
  PIDS+=($!)
}
wait_ready() {
  local url=$1 tries=$2
  for _ in $(seq "$tries"); do
    curl -sf "$url" > /dev/null && return 0
    kill -0 "${PIDS[-1]}" 2> /dev/null || { echo "server for $url exited; see $LAST_LOG" >&2; exit 1; }
    sleep 5
  done
  echo "$url not ready after $((tries * 5 / 60)) minutes; see $LAST_LOG" >&2
  exit 1
}
