#!/usr/bin/env bash
set -euo pipefail
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L "127.0.0.1:${QWEN_IMAGE_LOCAL_PORT:-${FLUX_LOCAL_PORT:-8188}}:127.0.0.1:${QWEN_IMAGE_REMOTE_PORT:-${FLUX_REMOTE_PORT:-8188}}" \
  "${QWEN_IMAGE_SSH_TARGET:-${FLUX_SSH_TARGET:-}}"
