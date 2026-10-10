_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export P0="${P0:-$(dirname "$_here")}"
export P0_CODE="$P0/code"
[ -f "$P0/env.local.sh" ] && source "$P0/env.local.sh"
export ENV_EVAL="${ENV_EVAL:-$P0/envs/p0-eval}"
export PY_EVAL="${PY_EVAL:-$ENV_EVAL/bin/python}"
export QI_SITE="${QI_SITE:-$P0/envs/qi21_site}"
export PYTHONPATH="$P0_CODE:${QI_SITE}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$P0/cache/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export VLM_URL="${VLM_URL:-http://127.0.0.1:18080/v1}"
export VLM_MODEL="${VLM_MODEL:-Qwen3.8-27B}"
mkdir -p "$P0/logs" "$P0/runs" "$P0/gates" "$P0/tmp" 2>/dev/null
noproxy() { env -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY "$@"; }
now() { date '+%F %T'; }
