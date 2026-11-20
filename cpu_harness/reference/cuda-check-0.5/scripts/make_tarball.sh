#!/bin/bash
# The one file the GPU machine needs: a/ = upstream v0.5.0 and b/ = this branch at REV (default
# cpu-target-0.5), each modax/, tests/, pyproject.toml and the Python files the tests import
# (benchmarks/*.py, reference/**/*.py, examples/**/*.py); b/ also has these scripts and the
# DISCO-EB 2.1 record pair. Run from anywhere inside the modax repository:
#   make_tarball.sh OUT.tar.gz [REV]
set -euo pipefail
out=$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1")
rev=${2:-cpu-target-0.5}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
py=(':(glob)benchmarks/*.py' ':(glob)reference/**/*.py' ':(glob)examples/**/*.py')
git archive --format=tar --prefix=a/ v0.5.0 modax tests pyproject.toml "${py[@]}" > "$tmp/a.tar"
git archive --format=tar --prefix=b/ "$rev" modax tests pyproject.toml "${py[@]}" \
  cpu_harness/reference/cuda-check-0.5/scripts \
  cpu_harness/make_gpu_reference_21.py cpu_harness/compare_with_gpu_21.py > "$tmp/b.tar"
# bsdtar (macOS) has no -A: unpack both and pack again, without the AppleDouble ._* files and
# extended attributes it would otherwise add (the first tarball of 2026-11-20 had them)
export COPYFILE_DISABLE=1
tar -Af "$tmp/a.tar" "$tmp/b.tar" 2>/dev/null || {
  (cd "$tmp" && mkdir x && tar -xf a.tar -C x && tar -xf b.tar -C x &&
    { tar --no-mac-metadata --no-xattrs -cf a.tar -C x a b 2>/dev/null || tar -cf a.tar -C x a b; })
}
gzip -9 -c "$tmp/a.tar" > "$out"
echo "$out: $(du -h "$out" | cut -f1); a = v0.5.0 $(git rev-parse --short v0.5.0^{commit}), b = $rev $(git rev-parse --short "$rev")"
tar -tzf "$out" | awk -F/ '{print $1"/"$2}' | sort -u | tr '\n' ' '; echo
echo "AppleDouble entries: $(tar -tzf "$out" | grep -c '/\._' || true)"
