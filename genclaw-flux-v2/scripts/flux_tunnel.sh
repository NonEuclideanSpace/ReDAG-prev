#!/usr/bin/env bash
set -euo pipefail
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L "127.0.0.1:${FLUX_LOCAL_PORT:-8188}:127.0.0.1:${FLUX_REMOTE_PORT:-8188}" \
  "${FLUX_SSH_TARGET}"
