"""modax 0.5.0: per-trajectory error weights under jax.vmap(jax.jacfwd(f)) are read past their end.

    python vmap_error_weights.py

Only modax's public API: runs on stock modax 0.5.0 on a GPU and on the
cpu-target branch's CPU target alike. A two-state cascade solved for one
trajectory, differentiated in its two parameters with jax.jacfwd, and
mapped over three parameter sets with jax.vmap, compared with the three
done one at a time, and the same for the values alone. The solve is given its error weights in each of the
forms modax documents: none, (n_vars,), and (N, n_vars) with N = 1.

Prints, per form, the largest difference between the mapped and the
one-at-a-time derivatives relative to their scale, per mapped row. With the
2-D form the rows after the first are wrong by whatever memory follows the
weights (from 4e-10 to 1 between two runs on the CPU target; 1e60 has been
seen on a GPU), with the other two forms they agree bitwise. Exits 1 if any
row differs at all, 0 if none does.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

import modax

jax.config.update("jax_enable_x64", True)

T = np.linspace(0.0, 2.0, 5)
Y0 = jnp.asarray([2.0, 0.5])
P0 = jnp.asarray([0.7, 1.3])
BATCH = jnp.asarray([[1.0, 1.0], [1.05, 0.9], [0.95, 1.1]])


def cascade(y, t, p):
    return (-p[0] * y[0], p[0] * y[0] - p[1] * y[1])


def final_state(weights):
    def f(scale):
        return modax.solve(
            cascade, Y0, T, P0 * scale, rtol=1e-8, atol=1e-10,
            method=modax.Rodas5P(lu_precision="fp64"), error_weights=weights,
        )[0, -1]  # fmt: skip

    return f


worst = 0.0
for label, weights in (
    ("no error_weights", None),
    ("error_weights (n_vars,)", np.ones(2)),
    ("error_weights (1, n_vars)", np.ones((1, 2))),
):
    f = final_state(weights)
    mapped = np.asarray(jax.vmap(jax.jacfwd(f))(BATCH))
    alone = np.stack([np.asarray(jax.jacfwd(f)(b)) for b in BATCH])
    scale = np.abs(alone).max()
    rows = [
        float(np.abs(mapped[i] - alone[i]).max() / scale) for i in range(len(BATCH))
    ]
    worst = max(worst, max(rows))
    print(
        f"{label:28s} jacfwd: mapped vs one at a time, per row: "
        + "  ".join(f"{r:.2e}" for r in rows)
    )
    mapped = np.asarray(jax.vmap(f)(BATCH))
    alone = np.stack([np.asarray(f(b)) for b in BATCH])
    rows = [
        float(np.abs(mapped[i] - alone[i]).max() / np.abs(alone).max())
        for i in range(len(BATCH))
    ]
    worst = max(worst, max(rows))
    print(
        f"{'':28s} values: mapped vs one at a time, per row: "
        + "  ".join(f"{r:.2e}" for r in rows)
    )
print(f"modax {modax.__file__}")
# One weight row, as given, is the same weights every row should see: right, the
# mapped rows match the single ones bitwise, as the other two forms do.
sys.exit(1 if worst > 0.0 else 0)
