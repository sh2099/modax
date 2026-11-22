"""Stack C: DISCO-EB 2.0.1 on modax 0.3 (CPU). P(k) for the tests' cosmologies, and the gradient of summed P(k)."""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.expanduser("~/Documents/PhD/code/DISCO-EB-lawrence/tests"))
import jax
import jax.numpy as jnp
from discoeb.perturbations import (
    _as_cosmology,
    density_coefficients,
    matter_power_spectrum_jax,
    solve_matter_power_spectrum_batch,
)
from test_perturbations import MATTER_POWER_K, _perturbed_cosmologies

out, arrays = {}, {}


def timed(f, reps=3):
    t = time.perf_counter()
    r = f()
    first = time.perf_counter() - t
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        r = f()
        ts.append(time.perf_counter() - t)
    return r, first, min(ts)


for n in (1, 128, 1024):
    cos = _perturbed_cosmologies(n)
    pk, first, warm = timed(
        lambda: np.asarray(solve_matter_power_spectrum_batch(MATTER_POWER_K, cos))
    )
    arrays[f"pk_{n}"] = pk
    out[f"pk_{n}"] = [first, warm]
    print(n, first, warm, flush=True)
c = _perturbed_cosmologies(1)[0]
d = jnp.asarray(np.asarray(density_coefficients(_as_cosmology(c)), dtype=np.float64))
vg = jax.value_and_grad(
    lambda d: jnp.sum(matter_power_spectrum_jax(MATTER_POWER_K, c, d))
)
g, first, warm = timed(lambda: np.asarray(vg(d)[1]))
out["grad_sum_pk_5_densities"] = [first, warm]
jf = jax.jacfwd(lambda d: matter_power_spectrum_jax(MATTER_POWER_K, c, d))
j, first, warm = timed(lambda: np.asarray(jf(d)))
out["jacfwd_pk_5_densities"] = [first, warm]
arrays["jac_shape"] = np.array(j.shape)
np.savez(sys.argv[1] + ".npz", k=MATTER_POWER_K, **arrays)
json.dump(out, open(sys.argv[1] + ".json", "w"), indent=1)
print(json.dumps(out))
