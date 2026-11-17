#!/bin/bash
# Set up the modax 0.3 CUDA check beside the DISCO-EB kit. Run on the cap006a host.
# No new environment: both sides run the existing modax-check/venv-a interpreter (DISCO-EB
# 2.0.1's dependency set, the kit's pins, the modax test dependencies) and differ only in the
# modax source put first on PYTHONPATH: src-a = upstream v0.3.0, src-b = the fork's cpu-target-0.3.
set -euo pipefail
umask 077
ROOT=${DISCO_ROOT:-$(cat ~/disco-eb-p100/ROOT)}
MC=$ROOT/modax-check-03
V=$ROOT/modax-check/venv-a
B_COMMIT=${B_COMMIT:-30a2be3}
mkdir -p "$MC"/{logs,out,scripts,tmp-a,tmp-b,cache-a,cache-b}
df -h /tmp | tail -1

[ -d "$MC/src-a/.git" ] || git clone -q https://github.com/lawrenceberry/modax "$MC/src-a"
git -C "$MC/src-a" checkout -q -f --detach v0.3.0
[ -d "$MC/src-b/.git" ] || git clone -q -b cpu-target-0.3 https://github.com/sh2099/modax "$MC/src-b"
git -C "$MC/src-b" fetch -q origin cpu-target-0.3
git -C "$MC/src-b" checkout -q -f --detach "$B_COMMIT"
echo "a: $(git -C "$MC/src-a" rev-parse HEAD) $(git -C "$MC/src-a" describe --tags --always)"
echo "b: $(git -C "$MC/src-b" rev-parse HEAD) $(git -C "$MC/src-b" describe --tags --always) (origin/cpu-target-0.3 is $(git -C "$MC/src-b" rev-parse origin/cpu-target-0.3))"
git -C "$MC/src-a" rev-parse HEAD > "$MC/logs/commit_a.txt"
git -C "$MC/src-b" rev-parse HEAD > "$MC/logs/commit_b.txt"
git -C "$MC/src-a" status --short; git -C "$MC/src-b" status --short

# The installed distributions: the same interpreter on both sides, so the same list; the modax
# each side actually imports is recorded by env_report.py inside each run (it is the PYTHONPATH one).
"$ROOT/bin/u2404" bash -c "uv pip list --python '$V' > '$MC/logs/uv_pip_list_venv-a.txt'"
for e in a b; do
  bash "$MC/scripts/mrun.sh" $e "" "$MC/scripts/env_report.py" > "$MC/logs/pip_list_$e.txt" 2>&1
done
diff "$MC/logs/pip_list_a.txt" "$MC/logs/pip_list_b.txt" > "$MC/logs/pip_list_diff.txt" || true
echo "--- env report diff (a vs b):"; cat "$MC/logs/pip_list_diff.txt"
du -sh "$MC"; df -h /tmp | tail -1
