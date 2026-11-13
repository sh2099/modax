"""The CPU target: the same kernels under plain numba, with finite-difference Jacobians.

Everything else in the suite runs on whichever target the machine has; what
is checked here is what only the CPU target does. Its Jacobian writer is
compared with ``jax.jacfwd`` in both layouts (the dense grid and the sparse
factorisation's packed slots); the solves are compared with SciPy's
integrators, which share nothing with modax; and the things the target does
not have are checked to fail loudly rather than quietly.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from modax import _target, tsit5
from modax._numba_common import make_cuda_local_vector_writer
from modax._sparse_direct import sparse_direct_solver_for
from modax._sparsity import dense_jacobian, normalize_sparsity
from modax.rodas5P import _make_difference_jacobian, quantity_hook, solve

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.skipif(
    _target.TARGET != "cpu", reason="checks of the CPU target"
)

N_CHAIN = 12


def chain(y, t, p):
    """A nonlinear lower-bidiagonal chain with explicit time dependence."""
    return (
        -p[0] * y[0] + math.sin(t),
        p[0] * y[0] - (p[0] + 0.1) * y[1] * (1.0 + 0.01 * y[1]),
        p[0] * y[1] - (p[0] + 0.2) * y[2] * (1.0 + 0.01 * y[2]),
        p[0] * y[2] - (p[0] + 0.3) * y[3] * (1.0 + 0.01 * y[3]),
        p[0] * y[3] - (p[0] + 0.4) * y[4] * (1.0 + 0.01 * y[4]),
        p[0] * y[4] - (p[0] + 0.5) * y[5] * (1.0 + 0.01 * y[5]),
        p[0] * y[5] - (p[0] + 0.6) * y[6] * (1.0 + 0.01 * y[6]),
        p[0] * y[6] - (p[0] + 0.7) * y[7] * (1.0 + 0.01 * y[7]),
        p[0] * y[7] - (p[0] + 0.8) * y[8] * (1.0 + 0.01 * y[8]),
        p[0] * y[8] - (p[0] + 0.9) * y[9] * (1.0 + 0.01 * y[9]),
        p[0] * y[9] - (p[0] + 1.0) * y[10] * (1.0 + 0.01 * y[10]),
        p[0] * y[10] - (p[0] + 1.1) * y[11] * (1.0 + 0.01 * y[11]),
    )


def chain_jax(y, t, p):
    out = [-p[0] * y[0] + jnp.sin(t)]
    for i in range(1, N_CHAIN):
        out.append(p[0] * y[i - 1] - (p[0] + 0.1 * i) * y[i] * (1.0 + 0.01 * y[i]))
    return jnp.stack(out)


CHAIN_PATTERN = np.eye(N_CHAIN, dtype=bool) | np.eye(N_CHAIN, k=-1, dtype=bool)


def robertson(y, t, p):
    return (
        -p[0] * y[0] + p[1] * y[1] * y[2],
        p[0] * y[0] - p[1] * y[1] * y[2] - p[2] * y[1] ** 2,
        p[2] * y[1] ** 2,
    )


def van_der_pol(y, t, p):
    return (y[1], p[0] * ((1.0 - y[0] * y[0]) * y[1] - y[0]))


def forced_decay(y, t, p):
    """``y' = -lam (y - sin t) + cos t``: the solution through ``y(0) = 0`` is ``sin t``."""
    return (-p[0] * (y[0] - math.sin(t)) + math.cos(t),)


def lorenz(y, t, p):
    return (
        p[0] * (y[1] - y[0]),
        y[0] * (p[1] - y[2]) - y[1],
        y[0] * y[1] - p[2] * y[2],
    )


ROBERTSON_TIMES = np.array([0.0, 1e-6, 1e-2, 1e2, 1e5])
ROBERTSON_Y0 = np.array([1.0, 0.0, 0.0])
ROBERTSON_P = np.array([0.04, 1e4, 3e7])


def _reference(fn, y0, times, p, method="Radau"):
    solution = solve_ivp(
        lambda t, y: np.array(fn(y, t, p)),
        (times[0], times[-1]),
        y0,
        method=method,
        t_eval=times,
        rtol=1e-11,
        atol=1e-13,
    )
    assert solution.success
    return solution.y.T


# --- the Jacobian writer -----------------------------------------------------------------


@pytest.mark.parametrize("layout", ["grid", "packed"])
def test_difference_jacobian_matches_autodiff(layout):
    """``-df/dy`` in the layout's slots and ``df/dt``, against ``jax.jacfwd``.

    The state mixes sizes and holds exact zeros, which the step-size floor is
    there for; the right-hand side is nonlinear, so the differences are not
    exact and the agreement is the truncation error of a forward difference.
    """
    if layout == "grid":
        pattern, structure = None, dense_jacobian(N_CHAIN)
        assert structure.n_colours == N_CHAIN
    else:
        pattern = normalize_sparsity(CHAIN_PATTERN, N_CHAIN)
        structure = sparse_direct_solver_for(pattern, "natural").compressed
        assert structure.n_colours == 2

    size = structure.size

    @_target.cuda.jit(device=True)
    def clear_matrix(lu):
        for i in range(size):
            lu[i] = 0.0

    writer = _make_difference_jacobian(
        ode_write=make_cuda_local_vector_writer(chain, N_CHAIN),
        structure=structure,
        pattern=pattern,
        clear_matrix=clear_matrix,
        lu_dtype=np.float64,
    )
    rng = np.random.default_rng(0)
    y = rng.normal(size=N_CHAIN) * np.logspace(-6, 2, N_CHAIN)
    y[3] = y[8] = 0.0
    t, p = 0.7, np.array([3.0])
    lu, dT = np.full(structure.size, np.nan), np.empty(N_CHAIN)
    # rtol = atol: components below unit size are stepped as if of unit size
    writer(y, t, p, lu, dT, 1e-6, 1e-6, 1e-2)

    expected = np.asarray(jax.jacfwd(chain_jax)(jnp.asarray(y), t, p))
    expected_dt = np.asarray(jax.jacfwd(chain_jax, argnums=1)(jnp.asarray(y), t, p))
    rows, cols = np.nonzero(
        CHAIN_PATTERN if layout == "packed" else np.ones_like(CHAIN_PATTERN)
    )
    ours = np.array([-lu[structure.slot(r, c)] for r, c in zip(rows, cols)])
    scale = np.abs(expected).max(axis=1)[rows]
    np.testing.assert_array_less(np.abs(ours - expected[rows, cols]) / scale, 1e-6)
    np.testing.assert_allclose(dT, expected_dt, rtol=1e-6, atol=1e-8)
    assert np.isfinite(lu).all()  # every slot the factorisation reads was written


# --- solves against SciPy ------------------------------------------------------------------


@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
@pytest.mark.parametrize("sparse", [False, True])
def test_robertson_matches_radau(lu_precision, sparse):
    hist, stats = solve(
        robertson,
        ROBERTSON_Y0,
        ROBERTSON_TIMES,
        ROBERTSON_P,
        rtol=1e-8,
        atol=1e-10,
        lu_precision=lu_precision,
        sparsity=np.ones((3, 3), dtype=bool) if sparse else None,
        return_stats=True,
    )
    reference = _reference(robertson, ROBERTSON_Y0, ROBERTSON_TIMES, ROBERTSON_P)
    np.testing.assert_allclose(np.asarray(hist)[0], reference, rtol=2e-6, atol=1e-10)
    assert int(stats["loop_steps"][0]) < 2000


def test_error_falls_with_the_tolerance():
    reference = _reference(robertson, ROBERTSON_Y0, ROBERTSON_TIMES, ROBERTSON_P)
    errors, in_tolerances, steps = [], [], []
    for rtol in (1e-4, 1e-6, 1e-8):
        atol = rtol * 1e-4
        hist, stats = solve(
            robertson, ROBERTSON_Y0, ROBERTSON_TIMES, ROBERTSON_P,
            rtol=rtol, atol=atol, lu_precision="fp64", return_stats=True,
        )  # fmt: skip
        difference = np.abs(np.asarray(hist)[0] - reference)
        errors.append(difference.max())
        in_tolerances.append(np.max(difference / (atol + rtol * np.abs(reference))))
        steps.append(int(stats["accepted_steps"][0]))
    assert errors[0] > errors[1] > errors[2]
    assert max(in_tolerances) < 50.0
    assert steps[0] < steps[1] < steps[2]


def test_ensemble_is_independent_of_the_threads():
    """The parallel loop gives each trajectory the answer it gets alone."""
    mus = np.array([[10.0], [100.0], [1000.0]])
    y0 = np.array([2.0, 0.0])
    times = np.linspace(0.0, 20.0, 6)
    kwargs = {"rtol": 1e-8, "atol": 1e-10, "lu_precision": "fp64"}
    together = np.asarray(solve(van_der_pol, y0, times, mus, **kwargs))
    for i, mu in enumerate(mus):
        alone = np.asarray(solve(van_der_pol, y0, times, mu, **kwargs))
        np.testing.assert_array_equal(together[i], alone[0])
        np.testing.assert_allclose(
            together[i], _reference(van_der_pol, y0, times, mu), rtol=2e-5, atol=1e-7
        )


def test_stiff_forced_problem_is_accurate_at_the_step_ends():
    """``df/dt`` by a time difference holds the method's order on a non-autonomous problem."""
    for rtol in (1e-4, 1e-6, 1e-8):
        hist, stats = solve(
            forced_decay, np.zeros(1), np.array([0.0, 3.0]), np.array([1.0e4]),
            rtol=rtol, atol=rtol, lu_precision="fp64", return_stats=True,
        )  # fmt: skip
        assert abs(float(hist[0, -1, 0]) - math.sin(3.0)) < rtol
        assert int(stats["accepted_steps"][0]) < 60


def test_tsit5_matches_an_explicit_reference():
    times = np.linspace(0.0, 1.0, 5)
    y0, p = np.array([1.0, 1.0, 1.0]), np.array([10.0, 28.0, 8.0 / 3.0])
    hist = tsit5.solve(lorenz, y0, times, p, rtol=1e-9, atol=1e-11)
    reference = _reference(lorenz, y0, times, p, method="DOP853")
    np.testing.assert_allclose(np.asarray(hist)[0], reference, rtol=1e-6, atol=1e-8)


def test_shared_backend_is_the_local_one_on_the_cpu():
    times = np.linspace(0.0, 1.0, 5)
    y0, p = np.array([1.0, 1.0, 1.0]), np.array([10.0, 28.0, 8.0 / 3.0])
    kwargs = {"rtol": 1e-9, "atol": 1e-11}
    np.testing.assert_array_equal(
        np.asarray(tsit5.solve(lorenz, y0, times, p, backend="shared", **kwargs)),
        np.asarray(tsit5.solve(lorenz, y0, times, p, backend="local", **kwargs)),
    )


def inverse_rate_decay(y, t, p):
    return (-y[0] / p[0],)


def test_a_division_by_zero_does_not_cost_the_other_trajectories():
    """IEEE arithmetic, as on the device: a trajectory that blows up fails alone.

    Under numba's default error model the division raises, and a raise inside
    the parallel loop ends that thread's share of the ensemble without a word:
    the trajectories after the bad one came back as untouched zeros. Compiled
    as the device computes, the bad trajectory's steps are rejected until its
    step budget runs out and every other one is solved.
    """
    n = 64
    timescale = np.linspace(0.5, 2.0, n)[:, None]
    timescale[n // 2] = 0.0
    times = np.array([0.0, 1.0])
    hist, stats = solve(
        inverse_rate_decay, np.ones(1), times, timescale,
        rtol=1e-8, atol=1e-10, lu_precision="fp64", max_steps=200, return_stats=True,
    )  # fmt: skip
    hist, good = np.asarray(hist), np.arange(n) != n // 2
    np.testing.assert_allclose(
        hist[good, -1, 0], np.exp(-1.0 / timescale[good, 0]), rtol=1e-7
    )
    assert int(stats["accepted_steps"][n // 2]) == 0
    assert int(stats["loop_steps"][n // 2]) == 200


# --- the JAX side --------------------------------------------------------------------------


def test_solve_traces_under_jit_and_vmap():
    params = jnp.asarray(ROBERTSON_P) * jnp.linspace(0.9, 1.1, 4)[:, None]
    kwargs = {"rtol": 1e-8, "atol": 1e-10, "lu_precision": "fp64"}

    def one(p):
        return solve(robertson, ROBERTSON_Y0, ROBERTSON_TIMES, p, **kwargs)

    together = jax.jit(one)(params)
    assert together.shape == (4, len(ROBERTSON_TIMES), 3)
    mapped = jax.vmap(one)(params)
    np.testing.assert_array_equal(np.asarray(mapped)[:, 0], np.asarray(together))


def test_quantity_hook_stores_what_the_history_holds():
    def total_and_intermediate(y, t, p):
        return (y[0] + y[1] + y[2], y[1])

    kwargs = {"rtol": 1e-8, "atol": 1e-10, "lu_precision": "fp64"}
    n_save = len(ROBERTSON_TIMES)
    hook = quantity_hook(total_and_intermediate, 2, 3, 3)
    final, rows = solve(
        robertson, ROBERTSON_Y0, ROBERTSON_TIMES, ROBERTSON_P,
        save_hook=hook, hook_size=2 * n_save, save_history=False, **kwargs,
    )  # fmt: skip
    hist = np.asarray(
        solve(robertson, ROBERTSON_Y0, ROBERTSON_TIMES, ROBERTSON_P, **kwargs)
    )
    stored = np.asarray(rows).reshape(n_save, 2)
    np.testing.assert_allclose(stored[:, 0], hist[0].sum(axis=1), rtol=1e-13)
    np.testing.assert_allclose(stored[:, 1], hist[0, :, 1], rtol=1e-13, atol=0.0)
    assert final.shape == (1, 1, 3)
    np.testing.assert_array_equal(np.asarray(final)[0, 0], hist[0, -1])


# --- derivatives without Enzyme ------------------------------------------------------------


def mixed(y, t, p):
    """Nonlinear in the state, the parameters and the time, with all three coupled."""
    return (
        p[0] * y[0] * y[1] + math.sin(t) * y[2] + p[2] * y[0],
        math.exp(-p[1] * y[1]) * y[2] * y[2] + t * y[0] + p[2] * p[2],
        y[0] / (1.0 + y[2] * y[2]) + p[0] * p[1] * t * t + p[2] * y[1],
    )


def mixed_jax(y, t, p):
    return jnp.stack(
        [
            p[0] * y[0] * y[1] + jnp.sin(t) * y[2] + p[2] * y[0],
            jnp.exp(-p[1] * y[1]) * y[2] * y[2] + t * y[0] + p[2] * p[2],
            y[0] / (1.0 + y[2] * y[2]) + p[0] * p[1] * t * t + p[2] * y[1],
        ]
    )


# Sizes spread over four decades, one state and one parameter exactly zero.
MIXED_Y = np.array([3.0e-4, 2.0, 0.0])
MIXED_T = 0.7
MIXED_P = np.array([40.0, 0.25, 0.0])
MIXED_DIRECTIONS = [
    (np.array([1.0, 0.0, 0.0]), 0.0, np.zeros(3)),
    (np.array([0.0, 0.0, 1.0]), 0.0, np.zeros(3)),  # along the zero state
    (np.array([2.0e3, -5.0e-2, 7.0]), 0.0, np.zeros(3)),  # sizes unlike the state's
    (np.zeros(3), 0.0, np.array([0.0, 1.0, 0.0])),
    (np.zeros(3), 0.0, np.array([0.0, 0.0, 1.0])),  # along the zero parameter
    (np.zeros(3), 1.0, np.zeros(3)),
    (np.array([1.0e2, 0.3, -2.0]), 0.0, np.array([1.0, 0.0, 0.0])),  # a sensitivity's
]


def _directional(fn, y, t, p, direction):
    dy, dt, dp = direction
    return jax.jvp(
        fn,
        (jnp.asarray(y), jnp.asarray(t), jnp.asarray(p)),
        (jnp.asarray(dy), jnp.asarray(dt), jnp.asarray(dp)),
    )[1]


@pytest.mark.parametrize("direction", range(len(MIXED_DIRECTIONS)))
def test_difference_tangent_matches_autodiff(direction):
    """``Df[u]`` for state, parameter, time and mixed directions, zeros included."""
    from modax._differences import difference_tangent

    tangent = difference_tangent(mixed, 3, 3, 3)
    dy, dt, dp = MIXED_DIRECTIONS[direction]
    out = np.empty(3)
    tangent(out, MIXED_Y, MIXED_T, MIXED_P, dy, dt, dp)
    expected = np.asarray(
        _directional(mixed_jax, MIXED_Y, MIXED_T, MIXED_P, MIXED_DIRECTIONS[direction])
    )
    np.testing.assert_allclose(
        out, expected, rtol=1e-8, atol=1e-9 * np.abs(expected).max()
    )


def test_difference_tangent_reads_only_the_state_of_an_augmented_vector():
    """A caller hands over its whole ``[y, S]`` row; only the state is the state."""
    from modax._differences import difference_tangent

    tangent = difference_tangent(mixed, 3, 3, 3)
    dy, dt, dp = MIXED_DIRECTIONS[-1]
    alone, within = np.empty(3), np.empty(3)
    tangent(alone, MIXED_Y, MIXED_T, MIXED_P, dy, dt, dp)
    augmented = np.concatenate([MIXED_Y, 1.0e9 * np.ones(6)])
    tangent(within, augmented, MIXED_T, MIXED_P, dy, dt, dp)
    np.testing.assert_array_equal(within, alone)


@pytest.mark.parametrize("outer", ["state", "time"])
def test_difference_second_tangent_matches_autodiff(outer):
    """``D2f[u, v]`` with ``u`` a sensitivity's direction and ``v`` a stage increment or time."""
    from modax._differences import difference_second_tangent

    second = difference_second_tangent(mixed, 3, 3)
    u = MIXED_DIRECTIONS[-1]
    v = (
        (np.array([1.0e-5, 4.0e-2, 3.0e-3]), 0.0, np.zeros(3))
        if outer == "state"
        else (np.zeros(3), 1.0, np.zeros(3))
    )
    zero = (np.zeros(3), 0.0, np.zeros(3))
    out = np.empty(3)
    second(out, MIXED_Y, MIXED_T, MIXED_P, *u, *v, *zero)

    def along_u(y, t, p):
        return _directional(mixed_jax, y, t, p, u)

    expected = np.asarray(_directional(along_u, MIXED_Y, MIXED_T, MIXED_P, v))
    # In units of the term's largest component: an entry far below that is
    # noise, which is all a Jacobian block that only sets the order needs.
    assert np.abs(out - expected).max() < 5e-5 * np.abs(expected).max()
