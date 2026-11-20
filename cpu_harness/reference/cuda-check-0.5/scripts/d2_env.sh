#!/bin/bash
# D2 step 1: which modax each side imports, the versions, and the installed wheel against the upstream tree.
# Usage: d2_env.sh PY A B OUT     (seconds, no GPU work)
source "$(dirname "$0")/common.sh"
for side in upstream branch; do
  run_side $side env "$HERE/env_report.py" > "$OUT/logs/env_$side.txt" 2>&1
done
( cd "$OUT" && env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PY" "$HERE/check_installed.py" "$A" ) > "$OUT/logs/installed_vs_upstream_tree.txt" 2>&1 || true
diff "$OUT/logs/env_upstream.txt" "$OUT/logs/env_branch.txt" > "$OUT/logs/env_diff.txt" || true
cat "$OUT/logs/env_upstream.txt" "$OUT/logs/env_branch.txt" "$OUT/logs/installed_vs_upstream_tree.txt"
