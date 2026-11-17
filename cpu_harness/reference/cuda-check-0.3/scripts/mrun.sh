#!/bin/bash
# Run a script's Python for side a or b on one GPU inside the kit sandbox, in the foreground.
# Usage (cap006a host): mrun.sh {a|b} GPU SCRIPT_OR_pytest [args...]
#   Runs: modax-check/venv-a/bin/python scripts/runner.py SCRIPT args   with cwd = src-X
# Both sides use the same interpreter and site-packages; PYTHONPATH puts src-X first, so its
# modax shadows the editable modax 0.4 of venv-a (a meta-path finder appended after PathFinder).
# Mirrors the kit's run.sh environment; TMPDIR and NUMBA_CACHE_DIR are per side.
set -euo pipefail
ROOT=${DISCO_ROOT:-$(cat ~/disco-eb-p100/ROOT)}
MC=$ROOT/modax-check-03
V=$ROOT/modax-check/venv-a
e=$1; gpu=$2; shift 2
th=${THREADS:-8}
exec nice -n 10 "$ROOT/bin/u2404" env \
  LD_PRELOAD="$V/lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12" \
  TMPDIR="$MC/tmp-$e" \
  CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$MC/src-$e:$MC/scripts:$ROOT/scripts" \
  PYTHONDONTWRITEBYTECODE=1 \
  NUMBA_CACHE_DIR="$MC/cache-$e" \
  NUMBA_NUM_THREADS="$th" OMP_NUM_THREADS="$th" OPENBLAS_NUM_THREADS="$th" MKL_NUM_THREADS="$th" \
  MC="$MC" MC_ENV="$e" MC_SRC="$MC/src-$e" MC_PY="$V/bin/python" ${EXTRA_ENV:-} \
  bash -c 'cd "$MC_SRC" || exit 1
    echo "[mrun] $(date -Is) host $(uname -n) side $MC_ENV GPU $CUDA_VISIBLE_DEVICES modax $(cat .git/HEAD)"
    echo "[mrun] $*"
    start=$(date +%s)
    timeout "${RUN_TIMEOUT:-3600}" "$MC_PY" "$MC/scripts/runner.py" "$@"
    echo "exit $? after $(( $(date +%s) - start )) s"' run "$@"
