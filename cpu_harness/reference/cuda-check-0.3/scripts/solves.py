"""Fixed problems through modax 0.3.0's public interface; every output array saved to an .npz for
bitwise comparison. Usage: solves.py OUT.npz
modax.solve, modax.rodas5P.solve and modax.tsit5.solve; Rodas5P and Tsit5 plain solves, fp32 and
fp64 LU, Auto, JVPs and gradients (state and parameters), sparsity, save hooks (raw device hooks
and quantity_hook, with and without a SensitivitySpec), and two- and three-block hierarchies with
and without sensitivities, hooks and Auto. The hierarchy, sparsity and hook problems are copied
from v0.3.0's tests/test_hierarchy.py, tests/test_sparsity.py and tests/test_save_hook.py
(inline, so both revisions run the same text). A case that raises is recorded with its error."""
import math, sys, time, traceback

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
import modax
from modax import Auto, Block, Hierarchy, Rodas5P, Tsit5, solve
from modax import rodas5P as rodas5P_mod
from modax import tsit5 as tsit5_mod
from modax._sensitivity import SensitivitySpec
from modax.rodas5P import quantity_hook
from numba_cuda_mlir import cuda

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
        except Exception as e:
            ERR[name] = traceback.format_exc()
            print(f"[case] {name}: ERROR {type(e).__name__}: {e}", flush=True)
        return fn
    return deco


F64 = Rodas5P(lu_precision="fp64")

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
    return solve(rob, ROB_Y0, ROB_T, ROB_P, method=F64, rtol=1e-8, atol=1e-10, return_stats=True)

@case("rob_rodas5p_default_fp32")
def _():
    return solve(rob, ROB_Y0, ROB_T, ROB_P, rtol=1e-6, atol=1e-8, return_stats=True)

@case("rob_rodas5p_module_fp32")
def _():
    return rodas5P_mod.solve(rob, ROB_Y0, ROB_T, ROB_P, rtol=1e-6, atol=1e-8, return_stats=True)

@case("rob_rodas5p_module_fp64")
def _():
    return rodas5P_mod.solve(rob, ROB_Y0, ROB_T, ROB_P, rtol=1e-8, atol=1e-10, lu_precision="fp64", return_stats=True)

@case("rob_auto")
def _():
    return solve(rob, ROB_Y0, ROB_T, ROB_P, method=Auto(), rtol=1e-6, atol=1e-9, return_stats=True)

@case("rob_rodas5p_fp64_cache")
def _():
    return solve(rob, ROB_Y0, ROB_T, ROB_P, method=F64, rtol=1e-7, atol=1e-9, return_stats=True, cache=True)

@case("rob_jit_vmap_fp64")
def _():
    f = jax.jit(jax.vmap(lambda p: solve(rob, ROB_Y0[0], ROB_T, p, method=F64)))
    return f(jnp.asarray(ROB_P[:16]))

@case("rob_jvp_params_fp64")
def _():
    dp = jnp.asarray(ROB_P[:8]) * 0.01
    f = lambda p: solve(rob, ROB_Y0[:8], ROB_T, p, method=F64, rtol=1e-8, atol=1e-10, return_stats=True)
    (h, s), (dh, _) = jax.jvp(f, (jnp.asarray(ROB_P[:8]),), (dp,))
    return {"primal": h, "tangent": dh, "stats": s}

@case("rob_jvp_params_cols02_noerrctl")
def _():
    dp = jnp.asarray(ROB_P[:8]) * jnp.array([0.01, 0.0, -0.02])
    f = lambda p: solve(rob, ROB_Y0[:8], ROB_T, p, method=F64, rtol=1e-8, atol=1e-10,
                        sens_param_columns=(0, 2), sens_error_control=False)
    return jax.jvp(f, (jnp.asarray(ROB_P[:8]),), (dp,))

@case("rob_jvp_y0_fp32")
def _():
    dy = jnp.ones((8, 3)) * jnp.array([1.0, 0.5, -0.25])
    f = lambda y: solve(rob, y, ROB_T, ROB_P[:8], rtol=1e-6, atol=1e-8)
    return jax.jvp(f, (jnp.asarray(ROB_Y0[:8]),), (dy,))

@case("rob_jacfwd_params_one")
def _():
    f = lambda p: solve(rob, ROB_Y0[0], ROB_T, p, method=F64, rtol=1e-8, atol=1e-10)
    return jax.jacfwd(f)(jnp.asarray(ROB_P[0]))


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

@case("lorenz_tsit5_module_shared")
def _():
    return tsit5_mod.solve(lorenz, LOR_Y0, LOR_T, LOR_P, backend="shared", rtol=1e-8, atol=1e-10, return_stats=True)

@case("lorenz_tsit5_local_cache")
def _():
    return solve(lorenz, LOR_Y0, LOR_T, LOR_P, method=Tsit5(backend="local"), rtol=1e-7, atol=1e-9, return_stats=True, cache=True)

@case("lorenz_tsit5_jvp_params")
def _():
    f = lambda p: solve(lorenz, LOR_Y0[:8], np.linspace(0.0, 1.0, 5), p, method=Tsit5(), rtol=1e-8, atol=1e-10, return_stats=True)
    (h, s), (dh, _) = jax.jvp(f, (jnp.asarray(LOR_P[:8]),), (jnp.ones((8, 1)),))
    return {"primal": h, "tangent": dh, "stats": s}

@case("lorenz_tsit5_jvp_y0")
def _():
    f = lambda y: solve(lorenz, y, np.linspace(0.0, 1.0, 5), LOR_P[:8], method=Tsit5(), rtol=1e-8, atol=1e-10)
    return jax.jvp(f, (jnp.asarray(LOR_Y0[:8]),), (jnp.ones((8, 3)),))

@case("lorenz_tsit5_grad")
def _():
    loss = lambda p: jnp.sum(solve(lorenz, LOR_Y0[:4], np.linspace(0.0, 1.0, 5), p, method=Tsit5(), rtol=1e-8, atol=1e-10)[:, -1] ** 2)
    return jax.grad(loss)(jnp.asarray(LOR_P[:4]))

@case("lorenz_auto")
def _():
    return solve(lorenz, LOR_Y0, LOR_T, LOR_P, method=Auto(), rtol=1e-8, atol=1e-10, return_stats=True)

@case("lorenz_rodas5p_fp32")
def _():
    return solve(lorenz, LOR_Y0, LOR_T, LOR_P, rtol=1e-6, atol=1e-8, return_stats=True)


# ---- hierarchy (tests/test_hierarchy.py) -----------------------------------------------
T_SPAN = np.linspace(0.0, 3.0, 7)
TOL = dict(rtol=1e-9, atol=1e-11, method=F64)

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
A_T = Block(rhs_a, 1, outputs=out_a, n_outputs=2, name="a_tsit5", method=Tsit5())
B_T = Block(rhs_b, 1, reads=(A_T,), name="b_after_tsit5")
MIXED = Hierarchy(B_T)
Y0 = jnp.array([[0.3, 0.1, -0.2], [1.0, -0.5, 0.4]])
P = jnp.array([[1.0, 25.0, 40.0], [0.5, 50.0, 10.0]])

@case("hier3_fp64")
def _():
    return solve(CHAIN, Y0, T_SPAN, P, return_stats=True, **TOL)

@case("hier2_fp64")
def _():
    return solve(TWO, Y0[:, :2], T_SPAN, P, return_stats=True, **TOL)

@case("hier2_mixed_tsit5_root")
def _():
    return solve(MIXED, Y0[:, :2], T_SPAN, P, return_stats=True, **TOL)

@case("hier3_default_method")
def _():
    return solve(CHAIN, Y0, T_SPAN, P, return_stats=True, rtol=1e-6, atol=1e-8)

@case("hier3_auto")
def _():
    return solve(CHAIN, Y0, T_SPAN, P, return_stats=True, method=Auto(), rtol=1e-6, atol=1e-9)

@case("hier2_auto")
def _():
    return solve(TWO, Y0[:, :2], T_SPAN, P, return_stats=True, method=Auto(), rtol=1e-6, atol=1e-9)

@case("hier3_cache")
def _():
    return solve(CHAIN, Y0, T_SPAN, P, return_stats=True, cache=True, rtol=1e-8, atol=1e-10, method=F64)

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

for k in range(2):
    @case(f"hier2_jvp_p{k}")
    def _(k=k):
        dp = jnp.zeros((2, 3)).at[:, k].set(1.0)
        f = lambda p: solve(TWO, Y0[:, :2], T_SPAN, p, return_stats=True, **TOL)
        (h, s), (dh, _) = jax.jvp(f, (P,), (dp,))
        return {"primal": h, "tangent": dh, "stats": s}

@case("hier2_jvp_y0")
def _():
    f = lambda y: solve(TWO, y, T_SPAN, P, **TOL)
    return jax.jvp(f, (Y0[:, :2],), (jnp.ones((2, 2)),))

@case("hier2_auto_jvp_p1")
def _():
    f = lambda p: solve(TWO, Y0[:, :2], T_SPAN, p, method=Auto(), rtol=1e-6, atol=1e-9)
    return jax.jvp(f, (P,), (jnp.zeros((2, 3)).at[:, 1].set(1.0),))

@case("hier3_auto_jvp_p2")
def _():
    f = lambda p: solve(CHAIN, Y0, T_SPAN, p, method=Auto(), rtol=1e-6, atol=1e-9)
    return jax.jvp(f, (P,), (jnp.zeros((2, 3)).at[:, 2].set(1.0),))


# ---- sparsity (tests/test_sparsity.py) ---------------------------------------------
def chain_ode(y, t, p):
    return (-p[0] * y[0], p[0] * y[0] - p[1] * y[1], p[1] * y[1] - p[2] * y[2], p[2] * y[2])

MASK = np.zeros((4, 4), dtype=bool)
for r, c in [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)]:
    MASK[r, c] = True
SP_Y0 = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (64, 1))
SP_P = np.tile(np.array([0.7, 0.4, 0.2]), (64, 1)) * (1.0 + 0.05 * rng.random((64, 3)))
SP_T = np.array([0.0, 0.5, 2.0])
SP_KW = dict(rtol=1e-11, atol=1e-13, first_step=1e-4, method=F64)

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


# ---- save hooks (tests/test_save_hook.py) --------------------------------------------
def damped_oscillator(y, t, p):
    return (y[1], -p[0] * y[0] - p[1] * y[1])

def energy_and_position(y, t, p):
    return (0.5 * (y[1] * y[1] + p[0] * y[0] * y[0]), y[0])

N_SAVE = 33
HK_T = np.linspace(0.0, 4.0, N_SAVE)
HK_Y0 = np.array([[1.0, 0.0], [0.5, 1.0], [-1.0, 0.3]])
HK_P = np.array([[4.0, 0.2], [9.0, 0.5], [1.0, 0.0]])
HK_TOL = dict(rtol=1e-8, atol=1e-10)

_w = np.zeros_like(HK_T)
_w[1:] += 0.5 * np.diff(HK_T)
_w[:-1] += 0.5 * np.diff(HK_T)
WEIGHTS = cuda.to_device(_w)

def integrate_state(save_idx, y, t, p_row, acc):
    acc[0] += WEIGHTS[save_idx] * y[0]
    acc[1] += WEIGHTS[save_idx] * y[1]
    acc[2] += 1.0

def store_energy(save_idx, y, t, p_row, acc):
    acc[save_idx] = 0.5 * (y[1] * y[1] + p_row[0] * y[0] * y[0])

@case("hook_raw_integrate_nohistory_module")
def _():
    return rodas5P_mod.solve(damped_oscillator, HK_Y0, HK_T, HK_P, save_hook=integrate_state, hook_size=3,
                             save_history=False, lu_precision="fp64", **HK_TOL)

@case("hook_raw_store_energy")
def _():
    return solve(damped_oscillator, HK_Y0, HK_T, HK_P, method=F64, save_hook=store_energy, hook_size=N_SAVE,
                 return_stats=True, **HK_TOL)

QH = quantity_hook(energy_and_position, 2, 2, 2)
for mname, meth in (("rodas5p", F64), ("tsit5", Tsit5()), ("auto", Auto())):
    @case(f"qhook_{mname}")
    def _(meth=meth):
        return solve(damped_oscillator, HK_Y0, HK_T, HK_P, method=meth, save_hook=QH, hook_size=2 * N_SAVE,
                     return_stats=True, **HK_TOL)

    @case(f"qhook_nohistory_{mname}")
    def _(meth=meth):
        return solve(damped_oscillator, HK_Y0, HK_T, HK_P, method=meth, save_hook=QH, hook_size=2 * N_SAVE,
                     save_history=False, **HK_TOL)

for tag, kw in (("p", dict(wrt_y0=False, wrt_params=True)),
                ("y0p", dict(wrt_y0=True, wrt_params=True)),
                ("p0_errctl", dict(wrt_y0=False, wrt_params=True, param_columns=(0,), error_control=True))):
    @case(f"qhook_sens_{tag}")
    def _(kw=kw):
        kw = dict(kw)
        kw.setdefault("error_control", False)
        spec = SensitivitySpec(2, 2, **kw)
        hook = quantity_hook(energy_and_position, 2, 2, 2, spec)
        return solve(damped_oscillator, HK_Y0, HK_T, HK_P, method=F64, save_hook=hook,
                     hook_size=2 * (1 + spec.n_sens) * N_SAVE, save_history=False, sensitivity=spec,
                     return_stats=True, **HK_TOL)

@case("qhook_sens_p_module_history")
def _():
    spec = SensitivitySpec(2, 2, wrt_y0=False, wrt_params=True, error_control=False)
    hook = quantity_hook(energy_and_position, 2, 2, 2, spec)
    return rodas5P_mod.solve(damped_oscillator, HK_Y0, HK_T, HK_P, save_hook=hook, hook_size=2 * (1 + spec.n_sens) * N_SAVE,
                             sensitivity=spec, lu_precision="fp64", **HK_TOL)

def hier_quantities(y, t, p):
    return (y[0] + y[2], y[1] * t)

HQ = quantity_hook(hier_quantities, 2, 3, 3)

@case("hier3_qhook")
def _():
    return solve(CHAIN, Y0, T_SPAN, P, save_hook=HQ, hook_size=2 * len(T_SPAN), return_stats=True, **TOL)

@case("hier3_qhook_auto")
def _():
    return solve(CHAIN, Y0, T_SPAN, P, save_hook=HQ, hook_size=2 * len(T_SPAN), method=Auto(), rtol=1e-6, atol=1e-9)

@case("hier3_qhook_sens_p")
def _():
    spec = SensitivitySpec(3, 3, wrt_y0=False, wrt_params=True, error_control=False)
    hook = quantity_hook(hier_quantities, 2, 3, 3, spec)
    return solve(CHAIN, Y0, T_SPAN, P, save_hook=hook, hook_size=2 * (1 + spec.n_sens) * len(T_SPAN),
                 save_history=False, sensitivity=spec, **TOL)

def product_of_states(y, t, p):
    return (y[0] * y[1],)

@case("hier2_qhook_sens_y0p")
def _():
    spec = SensitivitySpec(2, 3, wrt_y0=True, wrt_params=True, error_control=False)
    hook = quantity_hook(product_of_states, 1, 2, 3, spec)
    return solve(TWO, Y0[:, :2], T_SPAN, P, save_hook=hook, hook_size=(1 + spec.n_sens) * len(T_SPAN),
                 save_history=False, sensitivity=spec, **TOL)


np.savez(sys.argv[1], **OUT)
with open(sys.argv[1] + ".errors.txt", "w") as f:
    for k, v in ERR.items():
        f.write(f"=== {k}\n{v}\n")
print(f"saved {len(OUT)} arrays, {len(ERR)} errored cases -> {sys.argv[1]}")
