#!/bin/bash
# Run a command's Python in environment a or b on one GPU inside the kit sandbox, in the foreground.
# Usage (cap006a host): mrun.sh {a|b} GPU SCRIPT_OR_-m [args...]
#   Runs: venv-X/bin/python scripts/runner.py SCRIPT args   with cwd = src-X
# Mirrors the kit's run.sh environment; TMPDIR and NUMBA_CACHE_DIR are per environment.
set -euo pipefail
ROOT=${DISCO_ROOT:-$(cat ~/disco-eb-p100/ROOT)}
MC=$ROOT/modax-check
e=$1; gpu=$2; shift 2
V=$MC/venv-$e
th=${THREADS:-8}
exec nice -n 10 "$ROOT/bin/u2404" env \
  LD_PRELOAD="$V/lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12" \
  TMPDIR="$MC/tmp-$e" \
  CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$MC/scripts:$ROOT/scripts" \
  PYTHONDONTWRITEBYTECODE=1 \
  NUMBA_CACHE_DIR="$MC/cache-$e" \
  NUMBA_NUM_THREADS="$th" OMP_NUM_THREADS="$th" OPENBLAS_NUM_THREADS="$th" MKL_NUM_THREADS="$th" \
  MC="$MC" MC_ENV="$e" ${EXTRA_ENV:-} \
  bash -c 'cd "$MC/src-$MC_ENV" || exit 1
    echo "[mrun] $(date -Is) host $(uname -n) env $MC_ENV GPU $CUDA_VISIBLE_DEVICES modax $(git rev-parse --short HEAD 2>/dev/null)"
    echo "[mrun] $*"
    start=$(date +%s)
    timeout "${RUN_TIMEOUT:-3600}" "$MC/venv-$MC_ENV/bin/python" "$MC/scripts/runner.py" "$@"
    echo "exit $? after $(( $(date +%s) - start )) s"' run "$@"
