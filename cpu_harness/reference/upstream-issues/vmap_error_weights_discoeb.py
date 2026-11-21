"""DISCO-EB 2.1.0 on modax 0.5.0: jax.vmap(jax.jacfwd(P(k))) over cosmologies against the loop.

    python vmap_error_weights_discoeb.py

DISCO-EB's public API only (solve_eb, matter_power_spectrum, LCDM). The
batch form is the one its docs recommend (docs/usage.md: "For a batch, map
over the cosmologies outside the derivative, jax.vmap(jax.jacfwd(f))").
Three LCDM cosmologies, P(k) at 8 modes, jacfwd in omega_c and omega_b.

It runs the mapped call twice:
  as shipped        no change to anything;
  weights tiled     with a RUNTIME MONKEYPATCH of the name build_error_weights
                    that modax._jax_common.prepare_inputs calls: it logs the
                    weights' shape against the launch's trajectory count n
                    and, where they have fewer rows than n, tiles them to n in
                    the order the vmap rule flattens trajectories. Nothing
                    else changes, and no file is edited.
For each, per cosmology: max |vmap - loop| / max |loop| of the jacobian.

On the CPU target the reference cosmology itself (the first) differs from
the loop by 6.4e-7 in both runs, tiled or not, and the difference follows
that cosmology to whichever position it is mapped at; the other two agree
to 2e-16 once tiled. Small beside rtol = 1e-4 and not the weights; not
traced further.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from discoeb import LCDM, solve_eb  # noqa: E402
from discoeb.observables.power_spectrum import matter_power_spectrum  # noqa: E402

import modax._jax_common as JC  # noqa: E402

PLANCK = LCDM.from_density_parameters(
    Omega_m=0.3142,
    Omega_b=0.0494,
    H_0=67.32,
    A_s=2.1e-9,
    n_s=0.966,
    T_cmb=2.7255,
    Y_He=0.2454,
    N_eff=3.046,
)
K = np.geomspace(1e-3, 0.3, 8)
THETA0 = np.array([float(PLANCK.omega_c), float(PLANCK.omega_b)])
BATCH = jnp.asarray(THETA0 * np.array([[1.0], [1.02], [0.98]]))


def pk(theta):
    c = PLANCK.replace(omega_c=theta[0], omega_b=theta[1])
    v = solve_eb(c, K, [1.0], keep=("delta_c", "delta_b"), thermo_weight=0.1)
    dc, db = jnp.unstack(v[0, :, :, -1])
    return matter_power_spectrum(
        K,
        delta_c=dc,
        delta_b=db,
        omega_c=c.omega_c,
        omega_b=c.omega_b,
        A_s=c.A_s,
        n_s=c.n_s,
        k_p=c.k_p,
    )


REAL = JC.build_error_weights
SEEN = []
TILE = {"on": False}


def build_error_weights(error_weights, n, n_vars):
    out = REAL(error_weights, n, n_vars)
    if TILE["on"] and out.ndim == 2 and out.shape[0] != n:
        out = np.ascontiguousarray(np.tile(out, (n // out.shape[0], 1)))
    SEEN.append(
        (
            None if error_weights is None else tuple(np.shape(error_weights)),
            n,
            tuple(out.shape),
        )
    )
    return out


JC.build_error_weights = build_error_weights
print(
    "RUNTIME MONKEYPATCH: modax._jax_common.build_error_weights wrapped (logs; tiles only in the second run)"
)
print(f"device {jax.devices()[0]}")

jac = jax.jacfwd(pk)
SEEN.clear()
loop = np.stack([np.asarray(jac(b)) for b in BATCH])
print(f"loop:   launches (weights given, n, weights returned): {sorted(set(SEEN))}")
for label, tile in (("as shipped", False), ("weights tiled", True)):
    TILE["on"] = tile
    SEEN.clear()
    mapped = np.asarray(jax.vmap(jac)(BATCH))
    TILE["on"] = False
    rows = [
        float(np.abs(mapped[i] - loop[i]).max() / np.abs(loop[i]).max())
        for i in range(len(BATCH))
    ]
    print(f"vmap, {label:13s}: launches {sorted(set(SEEN))}")
    print(
        "  per cosmology max |vmap - loop| / max |loop|: "
        + "  ".join(f"{r:.2e}" for r in rows)
    )
