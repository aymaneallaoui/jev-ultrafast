#!/usr/bin/env bash
# Serve Qwen3-1.7B (Q4_K_M, thinking disabled) as the OpenAI-compatible text helper.
# usage: scripts/serve_text_local.sh [port]   (TEXT_NGL=0 keeps it off the GPU)
# env: TEXT_MODEL_BASE_URL=http://127.0.0.1:8080/v1 TEXT_MODEL=qwen3-1.7b TEXT_MODEL_API_KEY=local
set -euo pipefail
PORT=${1:-8080}
exec llama-server -hfr unsloth/Qwen3-1.7B-GGUF -hff Qwen3-1.7B-Q4_K_M.gguf --alias qwen3-1.7b \
  --host 127.0.0.1 --port "$PORT" -ngl "${TEXT_NGL:-99}" -c 4096 -np 1 --reasoning off --jinja
