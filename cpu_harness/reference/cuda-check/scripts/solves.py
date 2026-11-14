"""Fixed problems through modax.solve; every output array saved to an .npz for bitwise comparison.
Usage: solves.py OUT.npz
The hierarchy and sparsity problems are copied from tests/test_hierarchy.py and tests/test_sparsity.py,
the hook problem from tests/test_save_hook.py (definitions inline so both revisions run the same text)."""
import math, sys, time, traceback

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
import modax
from modax import Auto, Rodas5P, Tsit5, solve
from modax._hierarchy import Block, Hierarchy

OUT = {}
ERR = {}


def keep(name, value):
    if isinstance(value, dict):
        for k, v in value.items():
            keep(f"{name}/{k}", v)
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            keep(f"{name}/{i}", v)
    else:
        OUT[name] = np.asarray(value)


def case(name):
    def deco(fn):
        t0 = time.time()
        try:
            keep(name, fn())
            print(f"[case] {name}: ok {time.time() - t0:.1f} s", flush=True)
        except Exception:
            ERR[name] = traceback.format_exc()
            print(f"[case] {name}: ERROR\n{ERR[name]}", flush=True)
        return fn
    return deco


# ---- Robertson ------------------------------------------------------------------
def rob(y, t, p):
    return (
        -p[0] * y[0] + p[1] * y[1] * y[2],
        p[0] * y[0] - p[1] * y[1] * y[2] - p[2] * y[1] ** 2,
        p[2] * y[1] ** 2,
    )

N = 64
rng = np.random.default_rng(42)
ROB_P = np.array([0.04, 1e4, 3e7]) * (1.0 + 0.1 * (2.0 * rng.random((N, 3)) - 1.0))
ROB_Y0 = np.tile(np.array([1.0, 0.0, 0.0]), (N, 1))
ROB_T = np.array([0.0, 1e-6, 1e-2, 1e2, 1e5])

@case("rob_rodas5p_fp64")
def _():
    return solve(rob, ROB_Y0, ROB_T, ROB_P, method=Rodas5P(lu_precision="fp64"), rtol=1e-8, atol=1e-10, return_stats=True)

@case("rob_rodas5p_default_fp32")
def _():
    return solve(rob, ROB_Y0, ROB_T, ROB_P, rtol=1e-6, atol=1e-8, return_stats=True)

@case("rob_auto")
def _():
    return solve(rob, ROB_Y0, ROB_T, ROB_P, method=Auto(), rtol=1e-6, atol=1e-9, return_stats=True)

@case("rob_rodas5p_fp64_cache")
def _():
    return solve(rob, ROB_Y0, ROB_T, ROB_P, method=Rodas5P(lu_precision="fp64"), rtol=1e-7, atol=1e-9, return_stats=True, cache=True)

@case("rob_jit_vmap_fp64")
def _():
    f = jax.jit(jax.vmap(lambda p: solve(rob, ROB_Y0[0], ROB_T, p, method=Rodas5P(lu_precision="fp64"))))
    return f(jnp.asarray(ROB_P[:16]))

@case("rob_jvp_params_fp64")
def _():
    dp = jnp.asarray(ROB_P[:8]) * 0.01
    f = lambda p: solve(rob, ROB_Y0[:8], ROB_T, p, method=Rodas5P(lu_precision="fp64"), rtol=1e-8, atol=1e-10, return_stats=True)
    (h, s), (dh, _) = jax.jvp(f, (jnp.asarray(ROB_P[:8]),), (dp,))
    return {"primal": h, "tangent": dh, "stats": s}

@case("rob_jvp_y0_fp32")
def _():
    dy = jnp.ones((8, 3)) * jnp.array([1.0, 0.5, -0.25])
    f = lambda y: solve(rob, y, ROB_T, ROB_P[:8], rtol=1e-6, atol=1e-8)
    return jax.jvp(f, (jnp.asarray(ROB_Y0[:8]),), (dy,))


# ---- Lorenz, non-stiff ------------------------------------------------------------
def lorenz(y, t, p):
    return (10.0 * (y[1] - y[0]), y[0] * (p[0] - y[2]) - y[1], y[0] * y[1] - (8.0 / 3.0) * y[2])

LOR_Y0 = np.tile(np.array([1.0, 1.0, 1.0]), (N, 1)) + 0.01 * rng.standard_normal((N, 3))
LOR_P = 28.0 + rng.random((N, 1))
LOR_T = np.linspace(0.0, 5.0, 11)

for backend in ("local", "shared", "auto"):
    @case(f"lorenz_tsit5_{backend}")
    def _(backend=backend):
        return solve(lorenz, LOR_Y0, LOR_T, LOR_P, method=Tsit5(backend=backend), rtol=1e-8, atol=1e-10, return_stats=True)

@case("lorenz_tsit5_local_cache")
def _():
    return solve(lorenz, LOR_Y0, LOR_T, LOR_P, method=Tsit5(backend="local"), rtol=1e-7, atol=1e-9, return_stats=True, cache=True)

@case("lorenz_tsit5_jvp_params")
def _():
    f = lambda p: solve(lorenz, LOR_Y0[:8], np.linspace(0.0, 1.0, 5), p, method=Tsit5(), rtol=1e-8, atol=1e-10, return_stats=True)
    (h, s), (dh, _) = jax.jvp(f, (jnp.asarray(LOR_P[:8]),), (jnp.ones((8, 1)),))
    return {"primal": h, "tangent": dh, "stats": s}

@case("lorenz_auto")
def _():
    return solve(lorenz, LOR_Y0, LOR_T, LOR_P, method=Auto(), rtol=1e-8, atol=1e-10, return_stats=True)


# ---- hierarchy (tests/test_hierarchy.py) -----------------------------------------------
T_SPAN = np.linspace(0.0, 3.0, 7)
TOL = dict(rtol=1e-9, atol=1e-11, method=Rodas5P(lu_precision="fp64"))

def rhs_a(y, t, p):
    return (-p[0] * y[0] + math.cos(t),)

def out_a(y, t, p):
    return (math.sin(y[0]), y[0] * t)

def rhs_b(y, t, p, q):
    return (-p[1] * (y[0] - q[0]) + q[1],)

def rhs_c(y, t, p, q):
    return (-p[2] * (y[0] - q[0]) + q[1],)

A = Block(rhs_a, 1, outputs=out_a, n_outputs=2, name="a")
B = Block(rhs_b, 1, reads=(A,), name="b")
C = Block(rhs_c, 1, reads=(B, A), name="c")
CHAIN = Hierarchy(C)
TWO = Hierarchy(B)
Y0 = jnp.array([[0.3, 0.1, -0.2], [1.0, -0.5, 0.4]])
P = jnp.array([[1.0, 25.0, 40.0], [0.5, 50.0, 10.0]])

@case("hier3_fp64")
def _():
    return solve(CHAIN, Y0, T_SPAN, P, return_stats=True, **TOL)

@case("hier2_fp64")
def _():
    return solve(TWO, Y0[:, :2], T_SPAN, P, return_stats=True, **TOL)

@case("hier3_default_method")
def _():
    return solve(CHAIN, Y0, T_SPAN, P, return_stats=True, rtol=1e-6, atol=1e-8)

@case("hier3_auto")
def _():
    return solve(CHAIN, Y0, T_SPAN, P, return_stats=True, method=Auto(), rtol=1e-6, atol=1e-9)

@case("hier3_cache")
def _():
    return solve(CHAIN, Y0, T_SPAN, P, return_stats=True, cache=True, rtol=1e-8, atol=1e-10, method=Rodas5P(lu_precision="fp64"))

for k in range(3):
    @case(f"hier3_jvp_p{k}")
    def _(k=k):
        dp = jnp.zeros((2, 3)).at[:, k].set(1.0)
        f = lambda p: solve(CHAIN, Y0, T_SPAN, p, return_stats=True, **TOL)
        (h, s), (dh, _) = jax.jvp(f, (P,), (dp,))
        return {"primal": h, "tangent": dh, "stats": s}

@case("hier3_jvp_y0")
def _():
    f = lambda y: solve(CHAIN, y, T_SPAN, P, **TOL)
    return jax.jvp(f, (Y0,), (jnp.ones((2, 3)),))


# ---- sparsity (tests/test_sparsity.py) ---------------------------------------------
def chain_ode(y, t, p):
    return (-p[0] * y[0], p[0] * y[0] - p[1] * y[1], p[1] * y[1] - p[2] * y[2], p[2] * y[2])

MASK = np.zeros((4, 4), dtype=bool)
for r, c in [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)]:
    MASK[r, c] = True
SP_Y0 = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (64, 1))
SP_P = np.tile(np.array([0.7, 0.4, 0.2]), (64, 1)) * (1.0 + 0.05 * rng.random((64, 3)))
SP_T = np.array([0.0, 0.5, 2.0])
SP_KW = dict(rtol=1e-11, atol=1e-13, first_step=1e-4, method=Rodas5P(lu_precision="fp64"))

@case("chain_sparse_fp64")
def _():
    return solve(chain_ode, SP_Y0, SP_T, SP_P, sparsity=MASK, return_stats=True, **SP_KW)

@case("chain_dense_fp64")
def _():
    return solve(chain_ode, SP_Y0, SP_T, SP_P, return_stats=True, **SP_KW)

@case("chain_sparse_fp32")
def _():
    return solve(chain_ode, SP_Y0, SP_T, SP_P, sparsity=MASK, return_stats=True, rtol=1e-6, atol=1e-8)

@case("chain_sparse_grad")
def _():
    def loss(p):
        out = solve(chain_ode, jnp.asarray(SP_Y0[:1]), jnp.asarray([0.0, 2.0]), jnp.asarray([p]), sparsity=MASK, **SP_KW)
        return jnp.sum(out[0, -1] ** 2)
    return jax.grad(loss)(jnp.asarray([0.7, 0.4, 0.2]))


# ---- save hook (tests/test_save_hook.py) --------------------------------------------
def damped_oscillator(y, t, p):
    return (y[1], -p[0] * y[0] - p[1] * y[1])

def energy_and_position(y, t, p):
    return (0.5 * (y[1] * y[1] + p[0] * y[0] * y[0]), y[0])

HK_T = np.linspace(0.0, 4.0, 33)
HK_Y0 = np.array([[1.0, 0.0], [0.5, 1.0], [-1.0, 0.3]])
HK_P = np.array([[4.0, 0.2], [9.0, 0.5], [1.0, 0.0]])

for mname, meth in (("rodas5p", Rodas5P(lu_precision="fp64")), ("tsit5", Tsit5())):
    @case(f"hook_{mname}")
    def _(meth=meth):
        return solve(damped_oscillator, HK_Y0, HK_T, HK_P, method=meth, save_hook=energy_and_position, return_stats=True, rtol=1e-8, atol=1e-10)

    @case(f"hook_nohistory_{mname}")
    def _(meth=meth):
        return solve(damped_oscillator, HK_Y0, HK_T, HK_P, method=meth, save_hook=energy_and_position, save_history=False, rtol=1e-8, atol=1e-10)

    @case(f"hook_jvp_{mname}")
    def _(meth=meth):
        f = lambda p: solve(damped_oscillator, HK_Y0, HK_T, p, method=meth, save_hook=energy_and_position, rtol=1e-8, atol=1e-10)
        return jax.jvp(f, (jnp.asarray(HK_P),), (jnp.ones((3, 2)),))

@case("hier3_hook")
def _():
    return solve(CHAIN, Y0, T_SPAN, P, save_hook=lambda y, t, p: (y[0] + y[2], y[1] * t), return_stats=True, **TOL)


np.savez(sys.argv[1], **OUT)
with open(sys.argv[1] + ".errors.txt", "w") as f:
    for k, v in ERR.items():
        f.write(f"=== {k}\n{v}\n")
print(f"saved {len(OUT)} arrays, {len(ERR)} errored cases -> {sys.argv[1]}")
