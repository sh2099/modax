"""Fixed problems through modax.solve, every output saved to an .npz for a bitwise comparison.

    solves.py OUT.npz

The same text runs on upstream v0.5.0 and on cpu-target-0.5 (only the public
API is used), so any difference is the modax tree's. Problems are those of
modax's tests (Robertson, Lorenz, the three-block chain of
tests/test_hierarchy.py, the split of tests/test_split.py, the sparse chain
of tests/test_sparsity.py, the oscillator of tests/test_save_hook.py), plus
a small DISCO-EB 2.1 P(k) and its jacfwd when discoeb is importable. Each
case is independent; a failure is written to OUT.errors.txt.
"""

import math
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from modax import Auto, Block, Hierarchy, Rodas5P, Tsit5, solve  # noqa: E402

OUT = {}
ERR = {}
F64 = Rodas5P(lu_precision="fp64")


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
        except Exception:  # noqa: BLE001
            ERR[name] = traceback.format_exc()
            print(f"[case] {name}: ERROR\n{ERR[name]}", flush=True)
        return fn

    return deco


# ---- Robertson ----------------------------------------------------------------------
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
    return solve(
        rob, ROB_Y0, ROB_T, ROB_P, method=F64, rtol=1e-8, atol=1e-10, return_stats=True
    )


@case("rob_rodas5p_default_fp32")
def _():
    return solve(rob, ROB_Y0, ROB_T, ROB_P, rtol=1e-6, atol=1e-8, return_stats=True)


@case("rob_auto")
def _():
    return solve(
        rob,
        ROB_Y0,
        ROB_T,
        ROB_P,
        method=Auto(),
        rtol=1e-6,
        atol=1e-9,
        return_stats=True,
    )


@case("rob_rodas5p_fp64_cache")
def _():
    return solve(
        rob,
        ROB_Y0,
        ROB_T,
        ROB_P,
        method=F64,
        rtol=1e-7,
        atol=1e-9,
        return_stats=True,
        cache=True,
    )


@case("rob_jit_vmap_fp64")
def _():
    f = jax.jit(jax.vmap(lambda p: solve(rob, ROB_Y0[0], ROB_T, p, method=F64)))
    return f(jnp.asarray(ROB_P[:16]))


@case("rob_jvp_params_fp64")
def _():
    dp = jnp.asarray(ROB_P[:8]) * 0.01

    def f(p):
        return solve(
            rob,
            ROB_Y0[:8],
            ROB_T,
            p,
            method=F64,
            rtol=1e-8,
            atol=1e-10,
            return_stats=True,
        )

    (h, s), (dh, _) = jax.jvp(f, (jnp.asarray(ROB_P[:8]),), (dp,))
    return {"primal": h, "tangent": dh, "stats": s}


@case("rob_jvp_params_noerrctl")
def _():
    dp = jnp.asarray(ROB_P[:8]) * jnp.array([0.01, 0.0, -0.02])

    def f(p):
        return solve(
            rob,
            ROB_Y0[:8],
            ROB_T,
            p,
            method=F64,
            rtol=1e-8,
            atol=1e-10,
            sens_error_control=False,
        )

    return jax.jvp(f, (jnp.asarray(ROB_P[:8]),), (dp,))


@case("rob_jvp_y0_fp32")
def _():
    dy = jnp.ones((8, 3)) * jnp.array([1.0, 0.5, -0.25])

    def f(y):
        return solve(rob, y, ROB_T, ROB_P[:8], rtol=1e-6, atol=1e-8)

    return jax.jvp(f, (jnp.asarray(ROB_Y0[:8]),), (dy,))


@case("rob_jacfwd_params_one")
def _():
    def f(p):
        return solve(rob, ROB_Y0[0], ROB_T, p, method=F64, rtol=1e-8, atol=1e-10)

    return jax.jacfwd(f)(jnp.asarray(ROB_P[0]))


for per in (None, 2, 1):

    @case(f"rob_jacfwd_scaled_per_thread_{per}")
    def _(per=per):
        params = jnp.asarray(ROB_P[:4])

        def final(theta):
            return solve(rob, jnp.asarray(ROB_Y0[:4]) * theta[3], ROB_T, params * theta[:3], method=F64,
                         rtol=1e-8, atol=1e-10, max_directions_per_thread=per)  # fmt: skip

        return jax.jacfwd(final)(jnp.ones(4))


# ---- Lorenz, non-stiff ----------------------------------------------------------------
def lorenz(y, t, p):
    return (
        10.0 * (y[1] - y[0]),
        y[0] * (p[0] - y[2]) - y[1],
        y[0] * y[1] - (8.0 / 3.0) * y[2],
    )


LOR_Y0 = np.tile(np.array([1.0, 1.0, 1.0]), (N, 1)) + 0.01 * rng.standard_normal((N, 3))
LOR_P = 28.0 + rng.random((N, 1))
LOR_T = np.linspace(0.0, 5.0, 11)

for backend in ("local", "shared"):

    @case(f"lorenz_tsit5_{backend}")
    def _(backend=backend):
        return solve(
            lorenz,
            LOR_Y0,
            LOR_T,
            LOR_P,
            method=Tsit5(backend=backend),
            rtol=1e-8,
            atol=1e-10,
            return_stats=True,
        )


@case("lorenz_tsit5_jvp_params")
def _():
    def f(p):
        return solve(
            lorenz,
            LOR_Y0[:8],
            np.linspace(0.0, 1.0, 5),
            p,
            method=Tsit5(),
            rtol=1e-8,
            atol=1e-10,
            return_stats=True,
        )

    (h, s), (dh, _) = jax.jvp(f, (jnp.asarray(LOR_P[:8]),), (jnp.ones((8, 1)),))
    return {"primal": h, "tangent": dh, "stats": s}


@case("lorenz_tsit5_jacfwd_y0_per_thread_1")
def _():
    def f(y):
        return solve(
            lorenz,
            y,
            np.linspace(0.0, 1.0, 5),
            LOR_P[0],
            method=Tsit5(),
            rtol=1e-8,
            atol=1e-10,
            max_directions_per_thread=1,
        )

    return jax.jacfwd(f)(jnp.asarray(LOR_Y0[0]))


@case("lorenz_auto")
def _():
    return solve(
        lorenz,
        LOR_Y0,
        LOR_T,
        LOR_P,
        method=Auto(),
        rtol=1e-8,
        atol=1e-10,
        return_stats=True,
    )


# ---- hierarchy (tests/test_hierarchy.py) -------------------------------------------------
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


A_ = Block(rhs_a, 1, outputs=out_a, n_outputs=2, name="a")
B_ = Block(rhs_b, 1, reads=(A_,), name="b")
CHAIN = Hierarchy(Block(rhs_c, 1, reads=(B_, A_), name="c"))
TWO = Hierarchy(B_)
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
    return solve(
        CHAIN, Y0, T_SPAN, P, return_stats=True, method=Auto(), rtol=1e-6, atol=1e-9
    )


for k in range(3):

    @case(f"hier3_jvp_p{k}")
    def _(k=k):
        dp = jnp.zeros((2, 3)).at[:, k].set(1.0)

        def f(p):
            return solve(CHAIN, Y0, T_SPAN, p, return_stats=True, **TOL)

        (h, s), (dh, _) = jax.jvp(f, (P,), (dp,))
        return {"primal": h, "tangent": dh, "stats": s}


@case("hier2_jvp_y0")
def _():
    return jax.jvp(
        lambda y: solve(TWO, y, T_SPAN, P, **TOL), (Y0[:, :2],), (jnp.ones((2, 2)),)
    )


for per in (None, 1):

    @case(f"hier3_jacfwd_per_thread_{per}")
    def _(per=per):
        return jax.jacfwd(
            lambda t: solve(
                CHAIN, Y0, T_SPAN, P * t, max_directions_per_thread=per, **TOL
            )
        )(jnp.ones(3))


# ---- split hierarchy (tests/test_split.py) -----------------------------------------------
def split_ode(y, t, p):
    return (-p[0] * y[0] + math.cos(t), -p[1] * (y[1] - y[0]))


SPLIT = Hierarchy.from_sparsity(split_ode, 2, np.array([[0, 0], [1, 0], [1, 1]]))
SP_P = jnp.array([[1.0, 25.0], [0.5, 50.0]])


@case("split_fp64")
def _():
    return solve(SPLIT, Y0[:, :2], T_SPAN, SP_P, return_stats=True, **TOL)


@case("split_jvp")
def _():
    return jax.jvp(
        lambda p: solve(SPLIT, Y0[:, :2], T_SPAN, p, **TOL),
        (SP_P,),
        (jnp.ones((2, 2)),),
    )


# ---- sparsity (tests/test_sparsity.py) ---------------------------------------------------
def chain_ode(y, t, p):
    return (
        -p[0] * y[0],
        p[0] * y[0] - p[1] * y[1],
        p[1] * y[1] - p[2] * y[2],
        p[2] * y[2],
    )


MASK = np.zeros((4, 4), dtype=bool)
for r, c in [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)]:
    MASK[r, c] = True
CH_Y0 = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (64, 1))
CH_P = np.tile(np.array([0.7, 0.4, 0.2]), (64, 1)) * (1.0 + 0.05 * rng.random((64, 3)))
CH_T = np.array([0.0, 0.5, 2.0])
CH_KW = dict(rtol=1e-11, atol=1e-13, first_step=1e-4, method=F64)


@case("chain_sparse_fp64")
def _():
    return solve(
        chain_ode, CH_Y0, CH_T, CH_P, sparsity=MASK, return_stats=True, **CH_KW
    )


@case("chain_dense_fp64")
def _():
    return solve(chain_ode, CH_Y0, CH_T, CH_P, return_stats=True, **CH_KW)


@case("chain_sparse_fp32")
def _():
    return solve(
        chain_ode,
        CH_Y0,
        CH_T,
        CH_P,
        sparsity=MASK,
        return_stats=True,
        rtol=1e-6,
        atol=1e-8,
    )


@case("chain_sparse_jacfwd")
def _():
    def final(p):
        return solve(
            chain_ode,
            jnp.asarray(CH_Y0[:1]),
            jnp.asarray([0.0, 2.0]),
            jnp.asarray([p]),
            sparsity=MASK,
            **CH_KW,
        )[0, -1]

    return jax.jacfwd(final)(jnp.asarray([0.7, 0.4, 0.2]))


# ---- save hooks (tests/test_save_hook.py) ------------------------------------------------
def oscillator(y, t, p):
    return (y[1], -p[0] * y[0] - p[1] * y[1])


def energy_and_position(y, t, p):
    return (0.5 * (y[1] * y[1] + p[0] * y[0] * y[0]), y[0])


HK_T = np.linspace(0.0, 4.0, 33)
HK_Y0 = np.array([[1.0, 0.0], [0.5, 1.0], [-1.0, 0.3]])
HK_P = np.array([[4.0, 0.2], [9.0, 0.5], [1.0, 0.0]])
HK_TOL = dict(rtol=1e-8, atol=1e-10)

for mname, meth in (("rodas5p", F64), ("tsit5", Tsit5()), ("auto", Auto())):

    @case(f"hook_{mname}")
    def _(meth=meth):
        return solve(
            oscillator,
            HK_Y0,
            HK_T,
            HK_P,
            method=meth,
            save_hook=energy_and_position,
            return_stats=True,
            **HK_TOL,
        )

    @case(f"hook_nohistory_{mname}")
    def _(meth=meth):
        return solve(
            oscillator,
            HK_Y0,
            HK_T,
            HK_P,
            method=meth,
            save_hook=energy_and_position,
            save_history=False,
            **HK_TOL,
        )


@case("hook_jvp_params_rodas5p")
def _():
    def f(p):
        return solve(
            oscillator,
            HK_Y0,
            HK_T,
            p,
            method=F64,
            save_hook=energy_and_position,
            save_history=False,
            sens_error_control=False,
            **HK_TOL,
        )

    return jax.jvp(f, (jnp.asarray(HK_P),), (jnp.ones((3, 2)),))


def hier_quantities(y, t, p):
    return (y[0] + y[2], y[1] * t)


@case("hier3_hook")
def _():
    return solve(
        CHAIN, Y0, T_SPAN, P, save_hook=hier_quantities, return_stats=True, **TOL
    )


@case("hier3_hook_jvp")
def _():
    def f(p):
        return solve(
            CHAIN,
            Y0,
            T_SPAN,
            p,
            save_hook=hier_quantities,
            save_history=False,
            sens_error_control=False,
            **TOL,
        )

    return jax.jvp(f, (P,), (jnp.ones((2, 3)),))


# ---- DISCO-EB 2.1, small --------------------------------------------------------------------
try:
    import discoeb  # noqa: F401
except ImportError:
    print("[case] discoeb not importable: the DISCO-EB cases are skipped", flush=True)
else:
    from discoeb import solve_eb
    from discoeb.models import LCDM
    from discoeb.observables.power_spectrum import matter_power_spectrum

    PLANCK = LCDM.from_density_parameters(
        Omega_m=0.3142,
        Omega_b=0.0494,
        H_0=67.32,
        A_s=2.1e-9,
        n_s=0.966,
        T_cmb=2.7255,
        Y_He=0.2454,
        N_eff=3.046,
    )
    EB_K = np.geomspace(1e-4, 1.0, 16)

    def eb_pk(theta):
        c = PLANCK.replace(omega_c=theta[0], omega_b=theta[1])
        v = solve_eb(c, EB_K, [1.0], keep=("delta_c", "delta_b"), thermo_weight=0.1)
        dc, db = jnp.unstack(v[0, :, :, -1])
        return matter_power_spectrum(EB_K, delta_c=dc, delta_b=db, omega_c=c.omega_c, omega_b=c.omega_b,
                                     A_s=c.A_s, n_s=c.n_s, k_p=c.k_p)  # fmt: skip

    THETA = jnp.asarray([float(PLANCK.omega_c), float(PLANCK.omega_b)])

    @case("discoeb_pk")
    def _():
        return eb_pk(THETA)

    @case("discoeb_jacfwd_pk")
    def _():
        return jax.jacfwd(eb_pk)(THETA)


np.savez(sys.argv[1], **OUT)
with open(sys.argv[1] + ".errors.txt", "w") as f:
    for k, v in ERR.items():
        f.write(f"=== {k}\n{v}\n")
print(f"saved {len(OUT)} arrays, {len(ERR)} errored cases -> {sys.argv[1]}", flush=True)
