"""Whether the installed modax-solvers wheel is the upstream tree, file for file.

    check_installed.py UPSTREAM_TREE

Run with nothing of the trees on the path. Lists every modax/*.py whose
sha256 differs between site-packages and UPSTREAM_TREE/modax, and exits 1
if any does: the comparison runs upstream from its tree, and this says that
tree is what PyPI shipped as 0.5.0.
"""

import hashlib
import importlib.metadata as md
import importlib.util
import os
import sys

tree = os.path.join(sys.argv[1], "modax")
spec = importlib.util.find_spec("modax")
if spec is None or spec.origin is None:
    sys.exit("no installed modax on this interpreter's path")
installed = os.path.dirname(spec.origin)
print(
    f"installed modax-solvers {md.version('modax-solvers')} at {installed}; tree {tree}"
)


def digests(root):
    """sha256 of each ``*.py`` in ``root``, and the AppleDouble ``._*`` files a macOS tar adds."""
    names = sorted(n for n in os.listdir(root) if n.endswith(".py"))
    metadata = [n for n in names if n.startswith("._")]
    return {
        n: hashlib.sha256(open(os.path.join(root, n), "rb").read()).hexdigest()
        for n in names
        if not n.startswith("._")
    }, metadata


(a, _), (b, apple) = digests(installed), digests(tree)
if apple:
    print(
        f"  ignored in the tree: {len(apple)} AppleDouble files (._*) from a macOS tar, not Python"
    )
bad = [n for n in sorted(set(a) | set(b)) if a.get(n) != b.get(n)]
for n in bad:
    print(
        f"  differs: {n} (installed {a.get(n, 'absent')[:12]}, tree {b.get(n, 'absent')[:12]})"
    )
same = len([n for n in a if a[n] == b.get(n)])
print(f"{len(a)} installed, {len(b)} in the tree, {same} identical, {len(bad)} differ")
sys.exit(1 if bad else 0)
