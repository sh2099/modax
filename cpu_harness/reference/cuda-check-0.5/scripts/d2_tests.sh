#!/bin/bash
# D2 step 4: each side's own test suite (its tests/ against its modax), then the outcomes test by test.
# Usage: d2_tests.sh PY A B OUT [extra pytest args]
source "$(dirname "$0")/common.sh"
shift 4
for side in upstream branch; do
  RUN_TIMEOUT=${RUN_TIMEOUT:-5400} run_side $side tests pytest tests -v -rA -p no:cacheprovider \
    --benchmark-disable --durations=25 "$@" > "$OUT/logs/pytest_$side.txt" 2>&1
  tail -3 "$OUT/logs/pytest_$side.txt"
done
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PY" "$HERE/cmp_tests.py" "$OUT/logs/pytest_upstream.txt" "$OUT/logs/pytest_branch.txt" \
  | tee "$OUT/logs/test_comparison.txt"
