"""DISCO-EB 2.1 derivative accuracy on the CPU target, against tight central differences.

    disco --modax 0.5 discoeb21_derivatives.py OUT.json

P(k) and R today at 64 modes of one LCDM cosmology, jacfwd in omega_c, omega_b,
n_s, A_s and N_eff at the default tolerance and at 1e-6, against central
differences of plain solves at rtol = atol = 1e-7 (h = 1e-3 of each
parameter), with that reference checked against h = 3e-4 and against
rtol = 1e-8. Then split_derivatives, a vmapped batch and jit against the
eager jacfwd, the warm cost of one parameter, and that reverse mode raises.
Writes results/discoeb_2.1.json's "derivatives" section.
"""

import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from discoeb import solve_eb
from discoeb.models import LCDM
from discoeb.observables.power_spectrum import matter_power_spectrum

planck = LCDM.from_density_parameters(
    Omega_m=0.3142, Omega_b=0.0494, H_0=67.32,
    A_s=2.1e-9, n_s=0.966, T_cmb=2.7255, Y_He=0.2454, N_eff=3.046,
)  # fmt: skip
K = np.geomspace(1e-4, 1.0, 64)
NAMES = ("omega_c", "omega_b", "n_s", "A_s", "N_eff")
THETA0 = np.array([float(getattr(planck, n)) for n in NAMES])


def cosmology(theta):
    return planck.replace(**{n: theta[i] for i, n in enumerate(NAMES)})


def pk(theta, **tol):
    c = cosmology(theta)
    v = solve_eb(c, K, [1.0], keep=("delta_c", "delta_b"), thermo_weight=0.1, **tol)
    dc, db = jnp.unstack(v[0, :, :, -1])
    return matter_power_spectrum(
        K, delta_c=dc, delta_b=db, omega_c=c.omega_c, omega_b=c.omega_b,
        A_s=c.A_s, n_s=c.n_s, k_p=c.k_p,
    )  # fmt: skip


def curvature(theta, **tol):
    return solve_eb(cosmology(theta), K, [1.0], keep=("R",), **tol)[0, 0, :, -1]


def central(f, h_rel, **tol):
    """Columns of central differences of plain solves, one per parameter."""
    cols = []
    for i in range(len(NAMES)):
        h = h_rel * abs(THETA0[i])
        up, down = THETA0.copy(), THETA0.copy()
        up[i] += h
        down[i] -= h
        cols.append((np.asarray(f(up, **tol)) - np.asarray(f(down, **tol))) / (2 * h))
    return np.stack(cols, axis=-1)


def column_errors(got, ref):
    """max |got - ref| / max |ref| per parameter column."""
    return [
        float(np.abs(got[:, i] - ref[:, i]).max() / np.abs(ref[:, i]).max())
        for i in range(ref.shape[1])
    ]


out = {"names": NAMES}
tight = dict(rtol=1e-7, atol=1e-7)
tighter = dict(rtol=1e-8, atol=1e-8)
for label, f in (("P(k)", pk), ("R", curvature)):
    start = time.perf_counter()
    ref = central(f, 1e-3, **tight)
    ref_h = central(f, 3e-4, **tight)
    ref_tol = central(f, 1e-3, **tighter)
    entry = {
        "reference: h 1e-3 vs 3e-4 at 1e-7": column_errors(ref_h, ref),
        "reference: tol 1e-7 vs 1e-8 at h 1e-3": column_errors(ref_tol, ref),
        "reference seconds": time.perf_counter() - start,
    }
    for tol_label, tol in (("default", {}), ("1e-6", dict(rtol=1e-6, atol=1e-6))):
        jac = jax.jacfwd(lambda t: f(t, **tol))
        start = time.perf_counter()
        got = np.asarray(jac(jnp.asarray(THETA0)))
        cold = time.perf_counter() - start
        start = time.perf_counter()
        got = np.asarray(jac(jnp.asarray(THETA0)))
        warm = time.perf_counter() - start
        entry[f"jacfwd {tol_label}"] = column_errors(got, ref_tol)
        entry[f"jacfwd {tol_label} seconds cold, warm"] = [cold, warm]
        if label == "P(k)" and tol_label == "default":
            jac_default = got
    out[label] = entry
    print(label, json.dumps(entry), flush=True)

# split_derivatives against one thread, default tolerance
split = jax.jacfwd(lambda t: pk(t, split_derivatives=True))
got_split = np.asarray(split(jnp.asarray(THETA0)))
out["split_derivatives vs one thread (P(k), default)"] = column_errors(
    got_split, jac_default
)

# a batch by vmap(jacfwd): three cosmologies around the reference
batch = jnp.asarray(THETA0 * np.array([[1.0], [1.02], [0.98]]))
got_batch = np.asarray(jax.vmap(jax.jacfwd(pk))(batch))
out["vmap(jacfwd) row 0 vs single (P(k), default)"] = column_errors(
    got_batch[0], jac_default
)
# under jit
got_jit = np.asarray(jax.jit(jax.jacfwd(pk))(jnp.asarray(THETA0)))
out["jit(jacfwd) vs eager (P(k), default)"] = column_errors(got_jit, jac_default)

# timing: jacfwd in one parameter, warm
one = jax.jacfwd(lambda x: pk(jnp.asarray(THETA0).at[0].set(x)))
one(jnp.asarray(THETA0[0]))
start = time.perf_counter()
one(jnp.asarray(THETA0[0]))
out["jacfwd omega_c alone seconds warm"] = time.perf_counter() - start
start = time.perf_counter()
pk(THETA0)
out["plain P(k) seconds warm"] = time.perf_counter() - start

# reverse mode raises
for name, transform in (("grad", jax.grad), ("vjp", None)):
    try:
        if transform is None:
            jax.vjp(lambda t: jnp.sum(pk(t)), jnp.asarray(THETA0))
        else:
            transform(lambda t: jnp.sum(pk(t)))(jnp.asarray(THETA0))
        out[f"{name} raises"] = "no"
    except Exception as exc:  # noqa: BLE001
        out[f"{name} raises"] = (
            f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
        )

json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
