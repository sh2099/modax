"""Evidence for the cause of wrong jax.vmap rows with (N, n_vars) error weights, modax 0.5.0.

    python vmap_error_weights_evidence.py

modax's public API, plus one RUNTIME MONKEYPATCH (stated in the output): the
name ``build_error_weights`` that ``modax._jax_common.prepare_inputs`` calls
is wrapped, to print the shape it is given and the trajectory count ``n`` of
the launch, and, where a test says so, to change the array it returns before
the launch. Nothing in modax's files is edited.

Problem: Robertson's stiff kinetics, ``N = 2`` trajectories a call, mapped
over ``B = 4`` parameter scalings with ``jax.vmap(jax.jacfwd(f))`` (and, for
the values, ``jax.vmap(f)`` of single trajectories). The reference is the
same calls one at a time. Every comparison is bitwise, reported as the
largest |difference| per mapped row.

  1. observe    the weights' shape at launch against ``n``, loop and vmap;
  2. intervene  the same mapped call with the weights tiled to ``n`` rows in
                the order the rule flattens trajectories (row ``b * N + i``);
                and the discriminator: (N, n_vars) weights holding exactly the
                values of a (n_vars,) weights that passes;
  3. dose       the mapped call with the weights padded to ``n`` rows by a
                chosen value v (1, 0.5, 0, 1e300, NaN) against single launches
                given v explicitly: if the kernel reads its trajectories'
                weights from rows N.. of the array it is handed, the mapped rows
                b >= 1 equal those launches bitwise, whatever v does to them;
  4. rivals     no weights and (n_vars,) weights with max_directions_per_thread
                1 and 2 (the replica path, ``copies``), mapped against the loop.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

import modax
import modax._jax_common as JC

jax.config.update("jax_enable_x64", True)

REAL = JC.build_error_weights
LOG = []
MODE = {"fix": None}


def build_error_weights(error_weights, n, n_vars):
    """modax's own, then (under MODE) the launch's weights changed; every call logged."""
    out = REAL(error_weights, n, n_vars)
    given = None if error_weights is None else tuple(np.shape(error_weights))
    fix = MODE["fix"]
    if fix is not None and out.ndim == 2 and out.shape[0] != n:
        if fix == "tile":
            out = np.ascontiguousarray(np.tile(out, (n // out.shape[0], 1)))
        else:
            tail = np.full((n - out.shape[0], n_vars), float(fix))
            out = np.ascontiguousarray(np.concatenate([out, tail]))
    LOG.append((given, n, n_vars, out.shape, fix))
    return out


JC.build_error_weights = build_error_weights
print(
    "RUNTIME MONKEYPATCH: modax._jax_common.build_error_weights wrapped (logs shapes; changes the array only where a test says)"
)
print(f"modax {modax.__file__}; device {jax.devices()[0]}")

T = np.array([0.0, 1e-3, 1e-1, 1e1, 1e3])
N, B = 2, 4
P0 = np.array([[0.04, 1e4, 3e7], [0.05, 1.2e4, 2.5e7]])
Y0 = np.array([[1.0, 0.0, 0.0], [0.9, 0.0, 0.1]])
SCALES = jnp.asarray(1.0 + 0.05 * np.arange(B))[:, None] * jnp.ones((B, 3))
TOL = dict(rtol=1e-6, atol=1e-10, method=modax.Rodas5P(lu_precision="fp64"))


def robertson(y, t, p):
    return (
        -p[0] * y[0] + p[1] * y[1] * y[2],
        p[0] * y[0] - p[1] * y[1] * y[2] - p[2] * y[1] ** 2,
        p[2] * y[1] ** 2,
    )


def final(weights, y0=Y0, p0=P0, **kw):
    def f(scale):
        return modax.solve(
            robertson, y0, T, p0 * scale, error_weights=weights, **TOL, **kw
        )[..., -1, :]

    return f


def rows_vs_loop(f, label, single=None):
    """max |mapped row b - one-at-a-time b| for b in 0..B-1; ``single(b)`` overrides the reference."""
    LOG.clear()
    mapped = np.asarray(jax.vmap(jax.jacfwd(f))(SCALES))
    shapes = sorted(set(LOG))
    LOG.clear()
    alone = np.stack(
        [
            np.asarray(jax.jacfwd(single(b) if single else f)(SCALES[b]))
            for b in range(B)
        ]
    )
    rows = [
        float(np.nanmax(np.abs(mapped[b] - alone[b])))
        if np.isfinite(alone[b]).all()
        else (
            0.0 if np.array_equal(mapped[b], alone[b], equal_nan=True) else float("inf")
        )
        for b in range(B)
    ]
    scale = float(np.nanmax(np.abs(alone))) if np.isfinite(alone).any() else 1.0
    print(
        f"  {label:58s} per row |diff| / scale {scale:.1e}: "
        + "  ".join(f"{r / scale:.2e}" for r in rows)
    )
    return rows, shapes


print("\n1. observe: what build_error_weights is given and returns, and the launch's n")
ones2 = np.ones((N, 3))
for label, w in (("loop, (N, n_vars) weights", ones2),):
    LOG.clear()
    jax.jacfwd(final(w))(SCALES[0])
    print(
        f"  {label}: (given, n, n_vars, returned) = {sorted(set((g, n, v, r) for g, n, v, r, _ in LOG))}"
    )
LOG.clear()
jax.vmap(jax.jacfwd(final(ones2)))(SCALES)
print(
    f"  vmap(jacfwd) over B={B}, (N, n_vars) weights: {sorted(set((g, n, v, r) for g, n, v, r, _ in LOG))}"
)

print("\n   the bug as shipped (no change to the arrays):")
bad, _ = rows_vs_loop(final(ones2), "(N, n_vars) ones, unchanged")
ok1, _ = rows_vs_loop(final(np.ones(3)), "(n_vars,) ones, unchanged")
ok0, _ = rows_vs_loop(final(None), "no weights, unchanged")

print("\n2. intervene: the same mapped call, weights tiled to n rows at launch")
MODE["fix"] = "tile"
fixed, _ = rows_vs_loop(final(ones2), "(N, n_vars) ones, tiled to n at launch")
MODE["fix"] = None
w = np.array([1.0, 1e-3, 1e2])
d1, _ = rows_vs_loop(final(w), "discriminator: (n_vars,) = [1, 1e-3, 1e2], unchanged")
d2, _ = rows_vs_loop(
    final(np.tile(w, (N, 1))),
    "discriminator: the same values as (N, n_vars), unchanged",
)
MODE["fix"] = "tile"
d3, _ = rows_vs_loop(
    final(np.tile(w, (N, 1))), "discriminator: as (N, n_vars), tiled to n at launch"
)
MODE["fix"] = None

print(
    "\n3. dose: weights padded to n rows with v; reference: single launches given v explicitly for b >= 1"
)
dose = {}
for v in (1.0, 0.5, 0.0, 1e300, float("nan")):
    MODE["fix"] = v

    def single(b, v=v):
        return final(ones2 if b == 0 else np.full((N, 3), v))

    rows, _ = rows_vs_loop(final(ones2), f"pad v = {v!r}", single)
    MODE["fix"] = None
    LOG.clear()
    honest = np.asarray(jax.jacfwd(final(np.full((N, 3), v)))(SCALES[1]))
    plain = np.asarray(jax.jacfwd(final(ones2))(SCALES[1]))
    dose[v] = rows
    print(
        f"  {'':58s} what v does to an honest launch (row 1 vs weights 1): {np.nanmax(np.abs(honest - plain)) if np.isfinite(honest).all() else 'non-finite'}"
    )

print("\n4. rivals: the replica path and the tangent layout without 2-D weights")
riv = []
for per in (1, 2):
    for label, ww in (("no weights", None), ("(n_vars,) ones", np.ones(3))):
        rows, _ = rows_vs_loop(
            final(ww, max_directions_per_thread=per),
            f"{label}, max_directions_per_thread={per}",
        )
        riv += rows
    MODE["fix"] = "tile"
    rows, _ = rows_vs_loop(
        final(ones2, max_directions_per_thread=per),
        f"(N, n_vars) ones tiled, max_directions_per_thread={per}",
    )
    MODE["fix"] = None
    riv += rows

print("\n   values: jax.vmap(f) of single trajectories, N = 1")


def single_final(weights):
    def f(scale):
        return modax.solve(
            robertson, Y0[0], T, P0[0] * scale, error_weights=weights, **TOL
        )[0, -1]

    return f


for label, ww, fix in (
    ("(1, n_vars) ones, unchanged", np.ones((1, 3)), None),
    ("(1, n_vars) ones, tiled", np.ones((1, 3)), "tile"),
    ("(n_vars,) ones", np.ones(3), None),
):
    MODE["fix"] = fix
    LOG.clear()
    mapped = np.asarray(jax.vmap(single_final(ww))(SCALES))
    seen = sorted(set((g, n, r) for g, n, v, r, _ in LOG))
    alone = np.stack([np.asarray(single_final(ww)(SCALES[b])) for b in range(B)])
    MODE["fix"] = None
    print(
        f"  {label:30s} launch (given, n, returned) {seen}; per row |diff|: "
        + "  ".join(
            f"{float(np.abs(mapped[b] - alone[b]).max()):.2e}" for b in range(B)
        )
    )

# What must hold every time if the weights' row count is the cause: the controls and
# the intervened launches agree bitwise, and padding rows N.. with v gives exactly
# the launch given v. Whether the shipped call is wrong in a given run depends on
# what the memory past the weights holds, so it is reported, not required.
mechanism = (
    bad[0] == 0.0
    and max(fixed) == 0.0
    and max(ok1) == 0.0
    and max(ok0) == 0.0
    and max(d1) == 0.0
    and max(d3) == 0.0
    and max(riv) == 0.0
    and all(max(r) == 0.0 for r in dose.values())
)
print(
    "\nVERDICT: "
    + (
        "the prediction holds: right bitwise wherever the launch has n weight rows, and rows "
        "N.. padded with v give exactly the launch given v"
        if mechanism
        else "NOT as predicted: see the lines above"
    )
)
print(
    "  the shipped call this run: "
    + (
        "rows after the first WRONG"
        if max(bad[1:]) > 0.0
        else "rows right (the memory past the weights held usable values this time)"
    )
)
sys.exit(0 if mechanism else 1)
