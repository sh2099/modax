"""Stack D: DISCO-EB 2.1.0 on modax 0.5 (CPU). The same cosmologies; jacobians one at a time, vmapped and by a shared shift."""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.expanduser("~/Documents/PhD/code/DISCO-EB-2.1/tests"))
import jax
import jax.numpy as jnp
from discoeb import LCDM, solve_eb
from discoeb.observables.power_spectrum import matter_power_spectrum
from helpers import (
    MATTER_POWER_K,
    _matter_power,
    _perturbed_cosmologies,
    cosmologies_of,
)

import modax

K = MATTER_POWER_K
out, arrays = {"modax": modax.__file__}, {}


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
    cos = cosmologies_of(LCDM, _perturbed_cosmologies(n))
    pk, first, warm = timed(lambda: np.asarray(_matter_power(K, cos)))
    arrays[f"pk_{n}"] = pk
    out[f"pk_{n}"] = [first, warm]
    print(n, first, warm, flush=True)

ref = cosmologies_of(LCDM, _perturbed_cosmologies(1))[0]
names = ("omega_c", "omega_b", "n_s", "A_s", "N_eff")
theta0 = np.array([float(getattr(ref, m)) for m in names])


def pk_one(theta):
    c = ref.replace(**{m: theta[i] for i, m in enumerate(names)})
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


jac1 = jax.jacfwd(pk_one)
_, first, warm = timed(lambda: np.asarray(jac1(jnp.asarray(theta0))))
out["jacfwd_pk_5_params_1_cosmology"] = [first, warm]
for nb in (8, 128):
    batch = jnp.asarray(theta0 * (1.0 + 0.01 * np.linspace(-1, 1, nb))[:, None])

    def pk_batch(shift):
        th = batch + shift[None, :]
        cs = [
            ref.replace(**{m: th[i, j] for j, m in enumerate(names)}) for i in range(nb)
        ]
        v = solve_eb(cs, K, [1.0], keep=("delta_c", "delta_b"), thermo_weight=0.1)[
            ..., -1
        ]
        return jnp.stack(
            [
                matter_power_spectrum(
                    K,
                    delta_c=v[i, 0],
                    delta_b=v[i, 1],
                    omega_c=c.omega_c,
                    omega_b=c.omega_b,
                    A_s=c.A_s,
                    n_s=c.n_s,
                    k_p=c.k_p,
                )
                for i, c in enumerate(cs)
            ]
        )

    loop, f_l, w_l = timed(
        lambda: np.stack([np.asarray(jac1(b)) for b in batch]), reps=1
    )
    mapped, f_m, w_m = timed(lambda: np.asarray(jax.vmap(jac1)(batch)))
    shared, f_s, w_s = timed(
        lambda: np.asarray(jax.jacfwd(pk_batch)(jnp.zeros(len(names))))
    )
    scale = np.abs(loop).max(axis=(0, 1))  # per parameter
    out[f"jac_batch_{nb}"] = {
        "loop first, warm": [f_l, w_l],
        "vmap(jacfwd) first, warm": [f_m, w_m],
        "shared shift first, warm": [f_s, w_s],
        "vmap vs loop, max over rows of max|d|/scale per parameter": float(
            (np.abs(mapped - loop) / scale).max()
        ),
        "vmap vs loop, per row max": [
            float(x) for x in (np.abs(mapped - loop) / scale).max(axis=(1, 2))
        ][:8],
        "shared vs loop": float((np.abs(shared - loop) / scale).max()),
        "finite": bool(np.isfinite(mapped).all()),
    }
    print(nb, json.dumps(out[f"jac_batch_{nb}"]), flush=True)
np.savez(sys.argv[1] + ".npz", k=K, **arrays)
json.dump(out, open(sys.argv[1] + ".json", "w"), indent=1)
print(json.dumps(out))
