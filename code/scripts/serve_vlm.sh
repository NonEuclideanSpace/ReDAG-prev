#!/bin/bash
# Serve the evaluator with vLLM's OpenAI-compatible API. Logprobs are required (the client reads the first-token
# top logprobs for Yes/No); thinking is disabled per request by chat_template_kwargs.
# Usage: scripts/serve_vlm.sh <model_dir> [port] [tensor_parallel] [gpu_mem_fraction]
set -e
MODEL="${1:?model dir}"
PORT="${2:-18080}"
TP="${3:-1}"
MEM="${4:-0.90}"
NAME="$(basename "$MODEL")"
exec vllm serve "$MODEL" --served-model-name "$NAME" --port "$PORT" --tensor-parallel-size "$TP" --gpu-memory-utilization "$MEM" \
  --max-model-len 16384 --limit-mm-per-prompt '{"image": 2}' --max-num-seqs 32 --trust-remote-code --dtype auto
