#!/usr/bin/env bash
# Start Ollama, wait for it, choose the backends, then run the app.
set -euo pipefail

ollama serve >/tmp/ollama.log 2>&1 &
until curl -sf http://127.0.0.1:11434/api/version >/dev/null; do sleep 1; done
# Load the model now so the first visitor does not wait for it.
curl -sf http://127.0.0.1:11434/api/generate -d "{\"model\":\"$SPACE_MODEL\"}" >/dev/null || true

cpu="{\"name\":\"space-cpu\",\"base_url\":\"http://127.0.0.1:11434/v1\",\"model\":\"$SPACE_MODEL\",\"timeout_s\":300}"
# Optional faster first layer: HF Inference Providers, when both Space secrets are set.
if [[ -n "${HF_TOKEN:-}" && -n "${HF_ROUTER_MODEL:-}" ]]; then
  router="{\"name\":\"hf-router\",\"base_url\":\"https://router.huggingface.co/v1\",\"model\":\"$HF_ROUTER_MODEL\",\"api_key\":\"$HF_TOKEN\",\"timeout_s\":60}"
  export PLANNER_LLM_ENDPOINTS="[$router,$cpu]"
else
  export PLANNER_LLM_ENDPOINTS="[$cpu]"
fi
exec planner
