#!/bin/bash
# DISCO-EB 2.0.1's test suite on GPU $1 with side b's modax first on PYTHONPATH, otherwise the
# kit's run.sh environment (venv310, pascal_run.py, pytest_main.py), its own numba cache and TMPDIR.
# Foreground; launch it detached.
set -euo pipefail
ROOT=$(cat ~/disco-eb-p100/ROOT)
MC=$ROOT/modax-check-03
G=${1:-0}
mkdir -p "$MC/cache-disco-b" "$MC/tmp-disco-b"
th=${THREADS:-8}
exec nice -n 10 "$ROOT/bin/u2404" env \
  LD_PRELOAD="$ROOT/venv310/lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12" \
  TMPDIR="$MC/tmp-disco-b" CUDA_VISIBLE_DEVICES="$G" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$MC/src-b:$ROOT/scripts" PYTHONDONTWRITEBYTECODE=1 \
  NUMBA_CACHE_DIR="$MC/cache-disco-b" \
  NUMBA_NUM_THREADS="$th" OMP_NUM_THREADS="$th" OPENBLAS_NUM_THREADS="$th" MKL_NUM_THREADS="$th" \
  RUN_TIMEOUT="${RUN_TIMEOUT:-5400}" DISCO_ROOT="$ROOT" MC_SRC="$MC/src-b" \
  bash -c 'cd "$DISCO_ROOT/DISCO-EB" || exit 1
    echo "[disco_b] $(date -Is) host $(uname -n) GPU $CUDA_VISIBLE_DEVICES DISCO-EB $(cat .git/HEAD 2>/dev/null) modax $MC_SRC $(cat $MC_SRC/.git/HEAD)"
    "$DISCO_ROOT/venv310/bin/python" -c "import modax, sys; print(\"[disco_b] modax from\", modax.__file__); import modax._target as t; print(\"[disco_b] TARGET\", t.TARGET, t.CPU_DERIVATIVES, [k for k in sys.modules if k == \"modax._enzyme_cpu\"])"
    start=$(date +%s)
    timeout "$RUN_TIMEOUT" "$DISCO_ROOT/venv310/bin/python" "$DISCO_ROOT/scripts/pascal_run.py" "$DISCO_ROOT/scripts/pytest_main.py" -v -p no:cacheprovider -rA --benchmark-disable tests
    echo "exit $? after $(( $(date +%s) - start )) s"'
