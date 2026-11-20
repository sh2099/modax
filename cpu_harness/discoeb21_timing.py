"""DISCO-EB 2.1 timings on the CPU target: P(k) at 1 or N cosmologies x 64 modes, and jacfwd per parameter.

    disco --modax 0.5 discoeb21_timing.py N OUT.json

Run once with N = 1 and once with N = 128, each in a new process so the
first call includes compiling the kernel.
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
n = int(sys.argv[1])
batch = [
    planck.replace(omega_c=float(planck.omega_c) * (1 + 0.05 * x))
    for x in np.linspace(-1, 1, n)
]


def spectra(cosmologies):
    v = solve_eb(cosmologies, K, [1.0], keep=("delta_c", "delta_b"), thermo_weight=0.1)
    return np.asarray(v[:, :, :, -1])


def timed(f, *a):
    start = time.perf_counter()
    r = f(*a)
    return time.perf_counter() - start, r


out = {"n": n}
out["first call"], values = timed(spectra, batch)
out["warm"] = min(timed(spectra, batch)[0] for _ in range(3))
out["finite"] = bool(np.isfinite(values).all())

names = ("omega_c", "omega_b", "n_s", "A_s", "N_eff")
theta0 = jnp.asarray([float(getattr(planck, m)) for m in names])


def pk(theta, m):
    c = planck.replace(**{names[i]: theta[i] for i in range(m)})
    v = solve_eb(c, K, [1.0], keep=("delta_c", "delta_b"), thermo_weight=0.1)
    dc, db = jnp.unstack(v[0, :, :, -1])
    return matter_power_spectrum(
        K, delta_c=dc, delta_b=db, omega_c=c.omega_c, omega_b=c.omega_b,
        A_s=c.A_s, n_s=c.n_s, k_p=c.k_p,
    )  # fmt: skip


if n == 1:
    for m in (1, 2, 5):
        jac = jax.jacfwd(lambda t, m=m: pk(t, m))
        cold, _ = timed(lambda: np.asarray(jac(theta0[:m])))
        warm = min(timed(lambda: np.asarray(jac(theta0[:m])))[0] for _ in range(3))
        out[f"jacfwd {m} parameters cold, warm"] = [cold, warm]
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out))
