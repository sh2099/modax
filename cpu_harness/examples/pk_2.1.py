"""DISCO-EB 2.1.0 on the CPU target: P(k) today and its derivative in omega_c.

cpu_harness/disco --modax 0.5 cpu_harness/examples/pk_2.1.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from discoeb import LCDM, solve_eb
from discoeb.observables.power_spectrum import matter_power_spectrum

planck = LCDM.from_density_parameters(
    Omega_m=0.3142, Omega_b=0.0494, H_0=67.32,
    A_s=2.1e-9, n_s=0.966, T_cmb=2.7255, Y_He=0.2454, N_eff=3.046,
)  # fmt: skip
k = np.geomspace(1e-4, 1.0, 64)


def pk(omega_c):
    c = planck.replace(omega_c=omega_c)
    v = solve_eb(c, k, [1.0], keep=("delta_c", "delta_b"))
    dc, db = jnp.unstack(v[0, :, :, -1])
    return matter_power_spectrum(
        k, delta_c=dc, delta_b=db, omega_c=c.omega_c, omega_b=c.omega_b,
        A_s=c.A_s, n_s=c.n_s, k_p=c.k_p,
    )  # fmt: skip


def timed(f):
    start = time.perf_counter()
    out = np.asarray(f())
    return out, time.perf_counter() - start


p, first = timed(lambda: pk(planck.omega_c))
p, warm = timed(lambda: pk(planck.omega_c))
i = int(p.argmax())
print(
    f"P(k): peak {p[i]:.4e} Mpc^3 at k = {k[i]:.4f} /Mpc; first call {first:.1f} s, warm {warm:.3f} s"
)
dp, first = timed(lambda: jax.jacfwd(pk)(planck.omega_c))
dp, warm = timed(lambda: jax.jacfwd(pk)(planck.omega_c))
print(
    f"dP/d omega_c at the peak: {dp[i]:.4e}; first call {first:.1f} s, warm {warm:.3f} s"
)
