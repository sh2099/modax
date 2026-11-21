"""A loud modax 0.5.0 reproducer: (N, n_vars) error weights under jax.vmap(jax.jacfwd(f)).

    python vmap_error_weights_loud.py [N] [B]

modax's public API only, no monkeypatching. Three Robertson systems side by
side (9 stiff states), N trajectories a call (default 32), mapped over B
parameter scalings (default 16) with ``jax.vmap(jax.jacfwd(f, has_aux=True))``,
so one launch has B * N trajectories and the weights N rows: the kernel reads
the weights of (B - 1) * N trajectories from past the array's end. The
weights are ones, which is what every trajectory should see; the
reference is the same calls one at a time, and ``(n_vars,)`` ones mapped
the same way is the control. Per mapped row: max |diff| / scale of the
jacobian, and the mean accepted steps a trajectory, mapped and alone.
Exits 1 if any row differs.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

import modax

jax.config.update("jax_enable_x64", True)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
B = int(sys.argv[2]) if len(sys.argv) > 2 else 16
rng = np.random.default_rng(7)
P0 = np.array([0.04, 1e4, 3e7]) * (1.0 + 0.2 * rng.random((N, 3)))
Y0 = np.tile([1.0, 0.0, 0.0] * 3, (N, 1))
T = np.array([0.0, 1e-3, 1e-1, 1e1, 1e3, 1e5])
SCALES = jnp.asarray(1.0 + 0.02 * np.arange(B))[:, None] * jnp.ones((B, 3))
TOL = dict(rtol=1e-6, atol=1e-10, method=modax.Rodas5P(lu_precision="fp64"))


def three(y, t, p):
    """Robertson three times, at rates 1, 0.5 and 2: states 0-2, 3-5, 6-8."""
    a, b, c = p[0], p[1], p[2]
    return (
        -a * y[0] + b * y[1] * y[2],
        a * y[0] - b * y[1] * y[2] - c * y[1] ** 2,
        c * y[1] ** 2,
        -0.5 * a * y[3] + 0.5 * b * y[4] * y[5],
        0.5 * a * y[3] - 0.5 * b * y[4] * y[5] - 0.5 * c * y[4] ** 2,
        0.5 * c * y[4] ** 2,
        -2.0 * a * y[6] + 2.0 * b * y[7] * y[8],
        2.0 * a * y[6] - 2.0 * b * y[7] * y[8] - 2.0 * c * y[7] ** 2,
        2.0 * c * y[7] ** 2,
    )


def make(weights):
    def f(scale):
        hist, stats = modax.solve(
            three, Y0, T, P0 * scale, error_weights=weights, return_stats=True, **TOL
        )
        return hist[:, -1, :], stats["accepted_steps"]

    return jax.jacfwd(f, has_aux=True)


print(
    f"modax {modax.__file__}; device {jax.devices()[0]}; N = {N} a call, B = {B} mapped, 9 states"
)
worst = 0.0
for label, weights in (
    ("(n_vars,) ones (control)", np.ones(9)),
    ("(N, n_vars) ones", np.ones((N, 9))),
):
    jac = make(weights)
    mapped, mapped_steps = jax.vmap(jac)(SCALES)
    mapped, mapped_steps = np.asarray(mapped), np.asarray(mapped_steps)
    alone = [jac(SCALES[b]) for b in range(B)]
    alone_j = np.stack([np.asarray(a[0]) for a in alone])
    alone_s = np.stack([np.asarray(a[1]) for a in alone])
    scale = np.abs(alone_j).max()
    rows = [
        float(np.abs(mapped[b] - alone_j[b]).max() / scale)
        if np.isfinite(mapped[b]).all()
        else float("inf")
        for b in range(B)
    ]
    print(f"\n{label}: per row max |diff| / scale ({scale:.2e})")
    print("  " + " ".join(f"{r:.1e}" for r in rows))
    print(
        "  mean accepted steps, mapped: "
        + " ".join(f"{s:.0f}" for s in mapped_steps.mean(axis=1))
    )
    print(
        "  mean accepted steps, alone:  "
        + " ".join(f"{s:.0f}" for s in alone_s.mean(axis=1))
    )
    print(
        f"  rows differing: {sum(r > 0 for r in rows)} of {B}; non-finite rows: {sum(not np.isfinite(r) for r in rows)}"
    )
    if "(N" in label:
        worst = max(rows)
sys.exit(1 if worst > 0 else 0)
