#!/usr/bin/env bash
# Run the local-LLM NER arms against a local llama.cpp OpenAI-compatible server.
#
# 1) Start the server (from ~/projects/harness), ONE model at a time (shares port 8080):
#      scripts/serve-local.sh                                                   # Qwen3-Coder-30B-A3B (default)
#      HARNESS_LOCAL_MODEL=unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:IQ4_XS scripts/serve-local.sh
#    Confirm the model id it expects:  curl -s localhost:8080/v1/models
#
# 2) Run the local arms (this script). The bakeoff's hermes LLM client is pointed at
#    the local server; the arms pass BAKEOFF_LOCAL_MODEL as the OpenAI `model` field.
#
# 3) Compare to the cloud baselines already scored in results.json:
#      closed_vocab_clean (gpt-4o-mini)   big_model_clean (gpt-4o)
#    Hypothesis: closed-vocab is the lever (not model size), so local_clean should
#    land close to closed_vocab_clean at $0.
set -euo pipefail

export HERMES_LLM_BASE_URL="${HERMES_LLM_BASE_URL:-http://localhost:8080/v1}"
export HERMES_LLM_API_KEY="${HERMES_LLM_API_KEY:-local}"   # llama.cpp ignores the value
export BAKEOFF_LOCAL_MODEL="${BAKEOFF_LOCAL_MODEL:-qwen3-instruct}"

echo "[run-local] base_url=$HERMES_LLM_BASE_URL  model=$BAKEOFF_LOCAL_MODEL"
for arm in local_clean local_open; do
    echo "[run-local] === $arm ==="
    poetry run python run_bakeoff.py "$arm"
done
