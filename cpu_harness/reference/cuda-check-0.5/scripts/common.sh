# Sourced by the d2_*.sh drivers. Arguments of every driver: PY A B OUT
#   PY   the environment's python (DISCO-EB 2.1's venv, modax-solvers 0.5.0 installed)
#   A    the upstream v0.5.0 tree (a/ of the tarball), B the cpu-target-0.5 tree (b/)
#   OUT  where every log, npz, dump and cache goes; nothing is written elsewhere
# run_side SIDE TAG SCRIPT|pytest ARGS... runs one process on one side with a cache of its own:
# NUMBA_CACHE_DIR=OUT/cache/SIDE-TAG, emptied first. modax's on-disk kernel cache
# (cache=True, modax/_kernel_cache.py) keys a kernel on a digest of its generated source,
# closures, globals and the numba-cuda-mlir/numba-enzyme versions, not on modax's own version,
# so two trees that generate the same kernel would load each other's: a shared directory makes
# the comparison void. DISCO-EB solves with cache=True; the tests and solves.py use it too.
set -euo pipefail
mkdir -p "$4"
PY=$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1")
A=$(cd "$2" && pwd -P); B=$(cd "$3" && pwd -P); OUT=$(cd "$4" && pwd -P)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
mkdir -p "$OUT"/{logs,cache,tmp,xdg}
[ -x "$PY" ] || { echo "no python at $PY" >&2; exit 2; }
for t in "$A" "$B"; do [ -f "$t/modax/__init__.py" ] || { echo "no modax tree at $t" >&2; exit 2; }; done
# MC_LOCAL_TEST=1: both trees may be the branch, run on its CPU target (to test these scripts off a GPU)
if [ -z "${MC_LOCAL_TEST:-}" ]; then
  [ ! -f "$A/modax/_target.py" ] || { echo "$A has modax/_target.py: not upstream" >&2; exit 2; }
fi
[ -f "$B/modax/_target.py" ] || { echo "$B has no modax/_target.py: not the branch" >&2; exit 2; }
TIMEOUT=$(command -v timeout || true)

run_side() {
  local side=$1 tag=$2; shift 2
  local tree; [ "$side" = upstream ] && tree=$A || tree=$B
  local mc_side=$side allow=; [ -n "${MC_LOCAL_TEST:-}" ] && mc_side=branch allow=MC_ALLOW_CPU=1
  local cache="$OUT/cache/$side-$tag" tmp="$OUT/tmp/$side-$tag"
  rm -rf "$cache" "$tmp"; mkdir -p "$cache" "$tmp"
  echo "[d2] $(date -u +%Y-%m-%dT%H:%M:%SZ) $side/$tag: $* (cwd $tree, NUMBA_CACHE_DIR $cache)"
  local start=$(date +%s) status=0
  ( cd "$tree" && env -u PYTHONPATH \
      MC_SRC="$tree" MC_SIDE="$mc_side" $allow \
      NUMBA_CACHE_DIR="$cache" TMPDIR="$tmp" XDG_CACHE_HOME="$OUT/xdg" \
      PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      JAX_COMPILATION_CACHE_DIR= ${EXTRA_ENV:-} \
      ${TIMEOUT:+$TIMEOUT "${RUN_TIMEOUT:-3600}"} "$PY" "$HERE/runner.py" "$@" ) || status=$?
  echo "[d2] $side/$tag exit $status after $(( $(date +%s) - start )) s"
  return 0
}
