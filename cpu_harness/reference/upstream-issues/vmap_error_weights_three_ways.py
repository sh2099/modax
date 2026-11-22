"""DISCO-EB 2.1.0, no patching: per-cosmology jacobians of P(k) three ways.
loop of jacfwd (reference); jax.vmap(jax.jacfwd(f)); jacfwd in a shift shared by a batch given to solve_eb."""

import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from discoeb import LCDM, solve_eb
from discoeb.observables.power_spectrum import matter_power_spectrum

import modax

planck = LCDM.from_density_parameters(
    Omega_m=0.3142,
    Omega_b=0.0494,
    H_0=67.32,
    A_s=2.1e-9,
    n_s=0.966,
    T_cmb=2.7255,
    Y_He=0.2454,
    N_eff=3.046,
)
k = np.geomspace(1e-3, 0.3, 8)
theta0 = np.array([float(planck.omega_c), float(planck.omega_b)])
batch = jnp.asarray(theta0 * np.array([[1.0], [1.02], [0.98]]))
NB = batch.shape[0]


def pk_one(theta):
    c = planck.replace(omega_c=theta[0], omega_b=theta[1])
    v = solve_eb(c, k, [1.0], keep=("delta_c", "delta_b"))
    dc, db = jnp.unstack(v[0, :, :, -1])
    return matter_power_spectrum(
        k,
        delta_c=dc,
        delta_b=db,
        omega_c=c.omega_c,
        omega_b=c.omega_b,
        A_s=c.A_s,
        n_s=c.n_s,
        k_p=c.k_p,
    )


def pk_batch(shift):
    th = batch + shift[None, :]
    cs = [planck.replace(omega_c=th[i, 0], omega_b=th[i, 1]) for i in range(NB)]
    v = solve_eb(cs, k, [1.0], keep=("delta_c", "delta_b"))[..., -1]
    return jnp.stack(
        [
            matter_power_spectrum(
                k,
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


def rows(x, ref):
    return "  ".join(
        f"{float(np.abs(x[i] - ref[i]).max() / np.abs(ref[i]).max()):.2e}"
        for i in range(NB)
    )


def timed(f):
    np.asarray(f())
    t = time.perf_counter()
    out = np.asarray(f())
    return out, time.perf_counter() - t


print(f"modax {modax.__file__}; device {jax.devices()[0]}")
jac1 = jax.jacfwd(pk_one)
loop, t_loop = timed(lambda: jnp.stack([jac1(b) for b in batch]))
mapped, t_map = timed(lambda: jax.vmap(jac1)(batch))
shared, t_sh = timed(lambda: jax.jacfwd(pk_batch)(jnp.zeros(2)))
print("per cosmology max |x - loop| / max |loop|")
print(f"  vmap(jacfwd):        {rows(mapped, loop)}")
print(f"  shared-shift jacfwd: {rows(shared, loop)}")
print(
    f"warm seconds: loop {t_loop:.3f}, vmap(jacfwd) {t_map:.3f}, shared-shift {t_sh:.3f}"
)
