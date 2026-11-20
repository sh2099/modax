#!/bin/bash
# D2 step 3: generated kernel source (and device-code digests) on each side, then a recursive diff.
# Usage: d2_kernels.sh PY A B OUT
source "$(dirname "$0")/common.sh"
for side in upstream branch; do
  run_side $side dump "$HERE/dump_kernels.py" "$OUT/dump_$side" > "$OUT/logs/dump_$side.txt" 2>&1
  tail -2 "$OUT/logs/dump_$side.txt"
done
if diff -r "$OUT/dump_upstream" "$OUT/dump_branch" > "$OUT/logs/dump_diff.txt"; then
  echo "kernel dumps identical: $(ls "$OUT/dump_branch" | wc -l) files"
else
  echo "kernel dumps DIFFER: see $OUT/logs/dump_diff.txt"; head -40 "$OUT/logs/dump_diff.txt"
fi
