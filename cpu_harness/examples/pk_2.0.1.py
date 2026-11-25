"""DISCO-EB 2.0.1 on the CPU target: P(k) today and its derivative in the background densities.

cpu_harness/disco --modax 0.3 cpu_harness/examples/pk_2.0.1.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from discoeb.perturbations import (
    _as_cosmology,
    density_coefficients,
    matter_power_spectrum_jax,
    solve_matter_power_spectrum_batch,
)

cosmology = {
    "Omegam": (0.02238280 + 0.1201075) / 0.6732117**2,
    "Omegab": 0.02238280 / 0.6732117**2,
    "w_DE_0": -1.0, "w_DE_a": 0.0, "cs2_DE": 1.0, "Omegak": 0.0,
    "A_s": 2.100549e-9, "n_s": 0.9660499, "H0": 67.32117, "Tcmb": 2.7255,
    "YHe": 0.2454006, "Neff": 3.046, "Nmnu": 0.0, "mnu": 0.0, "k_p": 0.05,
}  # fmt: skip
k = np.geomspace(1e-4, 1.0, 64)


def timed(f):
    start = time.perf_counter()
    out = np.asarray(f())
    return out, time.perf_counter() - start


p, first = timed(lambda: solve_matter_power_spectrum_batch(k, [cosmology]))
p, warm = timed(lambda: solve_matter_power_spectrum_batch(k, [cosmology]))
i = int(p[0].argmax())
print(
    f"P(k): peak {p[0, i]:.4e} Mpc^3 at k = {k[i]:.4f} /Mpc; first call {first:.1f} s, warm {warm:.3f} s"
)

# the five background densities (photons, massless neutrinos, CDM, baryons, dark energy)
densities = jnp.asarray(
    np.asarray(density_coefficients(_as_cosmology(cosmology)), dtype=np.float64)
)
jac = jax.jacfwd(lambda d: matter_power_spectrum_jax(k, cosmology, d))
dp, first = timed(lambda: jac(densities))
dp, warm = timed(lambda: jac(densities))
print(
    f"dP/d(densities) at the peak: {np.array2string(dp[i], precision=3)}; first call {first:.1f} s, warm {warm:.3f} s"
)
