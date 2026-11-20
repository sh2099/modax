"""Bitwise comparison of two .npz files written by solves.py.

    cmp_npz.py X.npz Y.npz [label]

Prints a table, one row per array, and exits 1 if any array is missing,
differs in shape or dtype, or differs in a single bit.
"""

import sys

import numpy as np

x, y = np.load(sys.argv[1]), np.load(sys.argv[2])
label = sys.argv[3] if len(sys.argv) > 3 else f"{sys.argv[1]} vs {sys.argv[2]}"
keys = sorted(set(x.files) | set(y.files))
print(f"## {label}: {len(x.files)} vs {len(y.files)} arrays")
print(
    "| array | dtype | shape | bitwise | max abs diff | max rel diff | n differing | finite |"
)
print("|---|---|---|---|---|---|---|---|")
same = 0
for k in keys:
    if k not in x.files or k not in y.files:
        print(
            f"| {k} | - | - | MISSING in {'first' if k not in x.files else 'second'} | | | | |"
        )
        continue
    a, b = x[k], y[k]
    if a.shape != b.shape or a.dtype != b.dtype:
        print(
            f"| {k} | {a.dtype}/{b.dtype} | {a.shape}/{b.shape} | SHAPE/DTYPE DIFFER | | | | |"
        )
        continue
    bit = a.tobytes() == b.tobytes()
    same += bit
    if bit:
        print(
            f"| {k} | {a.dtype} | {a.shape} | identical | 0 | 0 | 0 | {bool(np.isfinite(a).all())} |"
        )
    else:
        af, bf = a.astype(np.float64), b.astype(np.float64)
        d = np.abs(af - bf)
        rel = d / np.maximum(np.abs(af), 1e-300)
        print(
            f"| {k} | {a.dtype} | {a.shape} | DIFFERENT | {np.nanmax(d):.3e} | {np.nanmax(rel):.3e} | "
            f"{int((af != bf).sum())}/{a.size} | {bool(np.isfinite(a).all())}/{bool(np.isfinite(b).all())} |"
        )
print(f"\n{same}/{len(keys)} arrays bitwise identical\n")
sys.exit(0 if same == len(keys) else 1)
