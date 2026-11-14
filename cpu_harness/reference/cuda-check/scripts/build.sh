#!/bin/bash
# Build two modax environments beside the DISCO-EB kit env, differing only in the modax source.
# Run on the cap006a host. Reuses the kit's sandbox (ROOT/bin/u2404), uv, uv cache and pins.
set -euo pipefail
umask 077
ROOT=${DISCO_ROOT:-$(cat ~/disco-eb-p100/ROOT)}
MC=$ROOT/modax-check
U=$ROOT/bin/u2404
PINS=(numba==0.61.2 llvmlite==0.44.0 numba-enzyme-cuda==0.3.0 "numba-cuda-mlir[cu12]==0.5.4")
TESTDEPS=(pytest pytest-benchmark diffrax==0.7.0 blackjax==1.3 matplotlib anesthetic==2.16.0)
mkdir -p "$MC"/{logs,out,scripts,tmp-a,tmp-b,cache-a,cache-b}
df -h /tmp | tail -1

[ -d "$MC/src-a/.git" ] || git clone -q https://github.com/lawrenceberry/modax "$MC/src-a"
git -C "$MC/src-a" checkout -q -f --detach v0.4.0
[ -d "$MC/src-b/.git" ] || git clone -q -b cpu-target-0.4 https://github.com/sh2099/modax "$MC/src-b"
git -C "$MC/src-b" fetch -q origin cpu-target-0.4
git -C "$MC/src-b" checkout -q -f --detach origin/cpu-target-0.4
echo "a: $(git -C "$MC/src-a" rev-parse HEAD) $(git -C "$MC/src-a" describe --tags --always)"
echo "b: $(git -C "$MC/src-b" rev-parse HEAD) $(git -C "$MC/src-b" describe --tags --always)"
git -C "$MC/src-a" rev-parse HEAD > "$MC/logs/commit_a.txt"
git -C "$MC/src-b" rev-parse HEAD > "$MC/logs/commit_b.txt"

q=$(printf '%q ' "${PINS[@]}" "${TESTDEPS[@]}")
nice -n 10 "$U" bash -c "set -e
  cd '$MC/src-a'
  UV_PROJECT_ENVIRONMENT='$MC/venv-a' uv sync --python 3.10 --extra cuda12 --no-group dev
  cd '$MC'
  uv pip install --no-config --python '$MC/venv-a' $q
  uv pip freeze --python '$MC/venv-a' --exclude-editable > '$MC/logs/requirements_frozen.txt'
  [ -x '$MC/venv-b/bin/python' ] || uv venv --no-config --python 3.10 '$MC/venv-b'
  uv pip install --no-config --python '$MC/venv-b' --no-deps -r '$MC/logs/requirements_frozen.txt'
  uv pip install --no-config --python '$MC/venv-b' --no-deps -e '$MC/src-b'
  uv pip check --python '$MC/venv-a' || true
  uv pip check --python '$MC/venv-b' || true
  uv pip list --python '$MC/venv-a' > '$MC/logs/pip_list_a.txt'
  uv pip list --python '$MC/venv-b' > '$MC/logs/pip_list_b.txt'
"
git -C "$MC/src-a" status --short; git -C "$MC/src-b" status --short
diff "$MC/logs/pip_list_a.txt" "$MC/logs/pip_list_b.txt" > "$MC/logs/pip_list_diff.txt" || true
echo "--- pip list diff (a vs b):"; cat "$MC/logs/pip_list_diff.txt"
du -sh "$MC"; df -h /tmp | tail -1
