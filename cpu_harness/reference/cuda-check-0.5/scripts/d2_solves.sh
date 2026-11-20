#!/bin/bash
# D2 step 2: solves.py on each side twice (upstream, branch, upstream, branch), each with an empty
# cache of its own, then bitwise comparisons: across sides (a1-b1, a2-b2) and run to run (a1-a2, b1-b2).
# Usage: d2_solves.sh PY A B OUT
source "$(dirname "$0")/common.sh"
for run in upstream:1 branch:1 upstream:2 branch:2; do
  side=${run%%:*}; i=${run##*:}
  run_side $side solves$i "$HERE/solves.py" "$OUT/solves_${side}$i.npz" > "$OUT/logs/solves_${side}$i.txt" 2>&1
  tail -2 "$OUT/logs/solves_${side}$i.txt"
done
c() { env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PY" "$HERE/cmp_npz.py" "$@" || true; }
{
  c "$OUT/solves_upstream1.npz" "$OUT/solves_branch1.npz" "upstream 1 vs branch 1"
  c "$OUT/solves_upstream2.npz" "$OUT/solves_branch2.npz" "upstream 2 vs branch 2"
  c "$OUT/solves_upstream1.npz" "$OUT/solves_upstream2.npz" "upstream run to run"
  c "$OUT/solves_branch1.npz" "$OUT/solves_branch2.npz" "branch run to run"
} > "$OUT/logs/solves_comparison.md"
grep -E "^## |bitwise identical|DIFFERENT|MISSING|DIFFER " "$OUT/logs/solves_comparison.md"
for side in upstream branch; do for i in 1 2; do
  echo "errors $side$i: $(grep -c '^===' "$OUT/solves_${side}$i.npz.errors.txt" 2>/dev/null || true)"
done; done
