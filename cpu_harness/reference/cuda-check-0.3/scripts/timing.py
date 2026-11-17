"""Cold and warm wall time of a few 4096-trajectory solves through modax 0.3.0's interface.
Usage: timing.py"""
import time
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from modax import Rodas5P, Tsit5, solve

def rob(y, t, p):
    return (-p[0] * y[0] + p[1] * y[1] * y[2], p[0] * y[0] - p[1] * y[1] * y[2] - p[2] * y[1] ** 2, p[2] * y[1] ** 2)
def lorenz(y, t, p):
    return (10.0 * (y[1] - y[0]), y[0] * (p[0] - y[2]) - y[1], y[0] * y[1] - (8.0 / 3.0) * y[2])

N = 4096
rng = np.random.default_rng(0)
P = jnp.asarray(np.array([0.04, 1e4, 3e7]) * (1.0 + 0.1 * (2.0 * rng.random((N, 3)) - 1.0)))
Y0 = jnp.asarray(np.tile([1.0, 0.0, 0.0], (N, 1)))
T = jnp.asarray([0.0, 1e-6, 1e-2, 1e2, 1e5])
LP = jnp.asarray(28.0 + rng.random((N, 1)))
LY = jnp.asarray(np.ones((N, 3)))
LT = jnp.linspace(0.0, 5.0, 11)
F64 = Rodas5P(lu_precision="fp64")

def bench(name, fn, reps=7):
    t0 = time.perf_counter(); out = fn(); jax.block_until_ready(out); cold = time.perf_counter() - t0
    warm = []
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(); jax.block_until_ready(out); warm.append(time.perf_counter() - t0)
    print(f"[time] {name}: cold {cold:.2f} s; warm ms " + " ".join(f"{1e3 * w:.2f}" for w in warm) + f"; min {1e3 * min(warm):.2f} median {1e3 * float(np.median(warm)):.2f}", flush=True)

bench("rob4096_rodas5p_fp32", lambda: solve(rob, Y0, T, P, rtol=1e-6, atol=1e-8))
bench("rob4096_rodas5p_fp64", lambda: solve(rob, Y0, T, P, method=F64, rtol=1e-8, atol=1e-10))
bench("rob4096_rodas5p_fp64_jvp", lambda: jax.jvp(lambda p: solve(rob, Y0, T, p, method=F64, rtol=1e-8, atol=1e-10), (P,), (0.01 * P,)))
bench("lorenz4096_tsit5", lambda: solve(lorenz, LY, LT, LP, method=Tsit5(), rtol=1e-8, atol=1e-10))
