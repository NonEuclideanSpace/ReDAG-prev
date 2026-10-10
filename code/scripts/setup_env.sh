#!/bin/bash
# One-time environment setup on a GPU machine. Two environments under $P0/envs:
#   p0-gen  : generation, labels, loop, training (torch + diffusers>=0.41 with QwenImage21Pipeline + peft)
#   p0-eval : the evaluator server only (vllm pins its own torch; keep it apart)
# Usage: bash code/scripts/setup_env.sh [cu128|cu124]   (cu128 for RTX 5090 / Blackwell, cu124 for A6000 / 4090)
# Re-runnable; prints SETUP OK at the end or stops at the first failed check. Not yet verified on a GPU machine:
# the first run on each new machine type is part of the preflight (必读 第九条).
set -e
CUDA="${1:-cu128}"
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export P0="${P0:-$(dirname "$(dirname "$_here")")}"
mkdir -p "$P0/envs" "$P0/logs" "$P0/cache/hf" "$P0/models" "$P0/queue" "$P0/configs"
cp -n "$P0/code/configs/"*.json "$P0/configs/" 2>/dev/null || true   # loop/train defaults and machine.example live under code/configs in the package
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_INDEX="https://download.pytorch.org/whl/$CUDA"
PY="${PY:-python3}"

mkenv() {  # name
  if [ ! -x "$P0/envs/$1/bin/python" ]; then
    if command -v conda >/dev/null 2>&1; then
      conda create -y -p "$P0/envs/$1" python=3.11 >/dev/null
    else
      "$PY" -m venv "$P0/envs/$1"
    fi
  fi
  "$P0/envs/$1/bin/python" -m pip install -q -U pip -i "$PIP_INDEX"
}

echo "== p0-gen"
mkenv p0-gen
GP="$P0/envs/p0-gen/bin/python"
"$GP" -m pip install -q torch torchvision --index-url "$TORCH_INDEX"
"$GP" -m pip install -q -i "$PIP_INDEX" "diffusers>=0.41" "transformers>=4.57" accelerate peft safetensors sentencepiece pillow numpy scipy requests modelscope huggingface_hub
"$GP" - <<'EOF'
import torch, diffusers, transformers, peft
from diffusers import QwenImage21Pipeline  # the executor's pipeline class; missing -> diffusers too old
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(), "| diffusers", diffusers.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
assert torch.cuda.is_available(), "no CUDA"
EOF

echo "== p0-eval"
mkenv p0-eval
EP="$P0/envs/p0-eval/bin/python"
"$EP" -m pip install -q -i "$PIP_INDEX" "vllm>=0.11" requests pillow
"$EP" - <<'EOF'
import vllm, torch
print("vllm", vllm.__version__, "torch", torch.__version__, "gpus", torch.cuda.device_count())
EOF

echo "== tests (no GPU needed)"
cd "$P0/code" && PYTHONPATH="$P0/code" "$GP" -c "import draftdag" && bash tests/run_all.sh | tail -1

# Qwen-Image-2.1 is 33 GB in bf16 (text encoder 17.5 + transformer 14.2 + VAE 1.4): on cards under 40 GB it only runs with CPU offload
MEM_GB=$("$GP" -c "import torch; print(int(min(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count()))/2**30) if torch.cuda.device_count() else 0)")
OFFLOAD=0; [ "$MEM_GB" -lt 40 ] && OFFLOAD=1
cat > "$P0/env.local.sh.new" <<EOF
# machine-local settings (sourced by code/env.sh)
export P0_QI_OFFLOAD=\${P0_QI_OFFLOAD:-$OFFLOAD}   # 1 on 32 GB cards (5090), 0 on 48/80 GB
export P0_API_TOKEN_CAP=\${P0_API_TOKEN_CAP:-70000000}
export VLM_URL=\${VLM_URL:-http://127.0.0.1:18080/v1}
export VLM_MODEL=\${VLM_MODEL:-Qwen3.8-27B-FP8}
export P0_GEN_PY=$GP
export P0_EVAL_PY=$EP
EOF
[ -f "$P0/env.local.sh" ] || mv "$P0/env.local.sh.new" "$P0/env.local.sh"
echo "SETUP OK  (gen: $GP, eval: $EP; env.local.sh $( [ -f "$P0/env.local.sh.new" ] && echo 'left as env.local.sh.new, merge by hand' || echo written ))"
