"""The CPU target: the same kernels under plain numba, with differenced derivatives.

Everything else in the suite runs on whichever target the machine has. What is
checked here is what only the CPU target does: its Jacobian writer and its
stand-ins for Enzyme's sweeps against JAX autodiff, the launch loop, and the
choice of target.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from modax import Rodas5P, Tsit5, _target, solve
from modax._differences import (
    clear_directed_parameters,
    difference_second_tangent,
    difference_tangent,
    set_directed_parameter,
    set_parameter_groups,
)
from modax._numba_common import make_cuda_local_vector_writer
from modax.rodas5P import make_difference_jacobian
from modax._sparse_direct import sparse_direct_solver_for
from modax._sparsity import dense_jacobian, normalize_sparsity

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.skipif(
    _target.TARGET != "cpu", reason="checks of the CPU target"
)

N = 6
N_P = 3
TYPICAL = 1.0e-2
TIME_SCALE = 1.0e-3
RTOL = 1.0e-9


def chain(y, t, p):
    """A nonlinear lower-bidiagonal chain with explicit time dependence."""
    return (
        -p[0] * y[0] + math.sin(t) * p[2],
        p[0] * y[0] - (p[1] + 0.1) * y[1] * (1.0 + 0.01 * y[1]),
        p[0] * y[1] - (p[1] + 0.2) * y[2] * (1.0 + 0.01 * y[2]) + math.cos(p[2] * t),
        p[0] * y[2] - (p[1] + 0.3) * y[3] * (1.0 + 0.01 * y[3]),
        p[0] * y[3] - (p[1] + 0.4) * y[4] * (1.0 + 0.01 * y[4]),
        p[0] * y[4] - (p[1] + 0.5) * y[5] * math.exp(0.1 * y[5] * t),
    )


def chain_jax(y, t, p):
    return jnp.stack(
        [
            -p[0] * y[0] + jnp.sin(t) * p[2],
            p[0] * y[0] - (p[1] + 0.1) * y[1] * (1.0 + 0.01 * y[1]),
            p[0] * y[1] - (p[1] + 0.2) * y[2] * (1.0 + 0.01 * y[2]) + jnp.cos(p[2] * t),
            p[0] * y[2] - (p[1] + 0.3) * y[3] * (1.0 + 0.01 * y[3]),
            p[0] * y[3] - (p[1] + 0.4) * y[4] * (1.0 + 0.01 * y[4]),
            p[0] * y[4] - (p[1] + 0.5) * y[5] * jnp.exp(0.1 * y[5] * t),
        ]
    )


Y = np.array([1.3, -0.7, 2.0e-6, 0.4, 0.0, 1.1])
T = 0.37
P = np.array([2.0, 0.5, 1.5])


def row(p):
    """A parameter row as the launch hands it to a kernel: three more columns."""
    return np.concatenate([p, [TYPICAL, TIME_SCALE, RTOL]])


def robertson(y, t, p):
    return (
        -p[0] * y[0] + p[1] * y[1] * y[2],
        p[0] * y[0] - p[1] * y[1] * y[2] - p[2] * y[1] ** 2,
        p[2] * y[1] ** 2,
    )


ROBERTSON_Y0 = np.array([[1.0, 0.0, 0.0]])
ROBERTSON_P = np.array([[0.04, 1.0e4, 3.0e7]])
ROBERTSON_T = np.array([0.0, 1.0, 1.0e2, 1.0e4])


# --- the Jacobian writer ---------------------------------------------------------------


@pytest.mark.parametrize("layout", ["grid", "packed"])
def test_difference_jacobian_matches_autodiff(layout):
    pattern = normalize_sparsity(np.tril(np.triu(np.ones((N, N), dtype=bool), -1)), N)
    if layout == "grid":
        structure, used = dense_jacobian(N), None
    else:
        structure = sparse_direct_solver_for(pattern, "amd").compressed
        used = pattern

    @_target.cuda.jit(device=True)
    def clear(lu):
        for i in range(lu.shape[0]):
            lu[i] = 0.0

    writer = make_difference_jacobian(
        ode_write=make_cuda_local_vector_writer(chain, N),
        structure=structure,
        pattern=used,
        clear_matrix=clear,
        lu_dtype=np.float64,
        n_params=N_P,
        context=N_P,
    )
    lu = np.full(structure.size, np.nan)
    dT = np.full(N, np.nan)
    writer(Y, T, row(P), lu, dT)
    at = (jnp.asarray(Y), T, jnp.asarray(P))
    jac = np.asarray(jax.jacfwd(chain_jax)(*at))
    by_time = np.asarray(jax.jacfwd(chain_jax, argnums=1)(*at))
    scale = np.abs(jac).max()
    for r in range(N):
        columns = range(N) if used is None else used[r]
        for c in columns:
            assert abs(-lu[structure.slot(r, c)] - jac[r, c]) < 1e-6 * scale
    np.testing.assert_allclose(dT, by_time, rtol=1e-6, atol=1e-7)


def test_difference_jacobian_gives_the_rate_along_a_parameter_seed():
    structure = dense_jacobian(N)

    @_target.cuda.jit(device=True)
    def clear(lu):
        pass

    writer = make_difference_jacobian(
        ode_write=make_cuda_local_vector_writer(chain, N),
        structure=structure,
        pattern=None,
        clear_matrix=clear,
        lu_dtype=np.float64,
        n_params=N_P,
        context=N_P,
        coupling=2,
    )
    seeds = np.array([[1.0e-9, 0.0, 3.0e-9], [0.0, 0.0, 0.0]])
    coup = np.full((2, N), np.nan)
    writer(Y, T, row(P), np.zeros(structure.size), np.zeros(N), seeds, coup)
    at = (jnp.asarray(Y), T, jnp.asarray(P))
    by_p = np.asarray(jax.jacfwd(chain_jax, argnums=2)(*at))
    np.testing.assert_allclose(coup[0], by_p @ seeds[0], rtol=1e-7, atol=1e-18)
    np.testing.assert_array_equal(coup[1], 0.0)


# --- Enzyme's sweeps, differenced ------------------------------------------------------

DIRECTIONS = {
    "state": (np.array([0.3, -1.0, 2.0, 0.0, 0.5, 1.0]), 0.0, np.zeros(N_P)),
    "small state": (np.array([0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0]), 0.0, np.zeros(N_P)),
    "time": (np.zeros(N), 1.0, np.zeros(N_P)),
    "flow": (np.array([0.3, -1.0, 2.0e-6, 0.0, 0.5, 1.0]), 1.0, np.zeros(N_P)),
    "parameters": (np.zeros(N), 0.0, np.array([1.0, -2.0, 0.5])),
    "all": (np.array([0.1, 0.2, 0.0, 0.3, 0.0, -0.4]), 0.7, np.array([0.0, 1.0, 0.0])),
}
POINT = (jnp.asarray(Y), jnp.asarray(T), jnp.asarray(P))


def _as_jax(direction):
    return tuple(jnp.asarray(x) for x in direction)


def _first(direction):
    return np.asarray(jax.jvp(chain_jax, POINT, _as_jax(direction))[1])


def _second(u, v):
    def along_u(y, t, p):
        return jax.jvp(chain_jax, (y, t, p), _as_jax(u))[1]

    return np.asarray(jax.jvp(along_u, POINT, _as_jax(v))[1])


@pytest.mark.parametrize("name", list(DIRECTIONS))
def test_difference_tangent_matches_autodiff(name):
    tangent = difference_tangent(chain, N, N, N_P)
    out = np.full(N, np.nan)
    tangent(out, Y, T, row(P), *DIRECTIONS[name])
    exact = _first(DIRECTIONS[name])
    np.testing.assert_allclose(out, exact, rtol=1e-8, atol=1e-9 * np.abs(exact).max())


def test_difference_tangent_reads_only_the_state_of_an_augmented_vector():
    tangent = difference_tangent(chain, N, N, N_P)
    out = np.full(N, np.nan)
    augmented = np.concatenate([Y, np.full(2 * N, np.nan)])
    tangent(out, augmented, T, row(P), *DIRECTIONS["all"])
    np.testing.assert_allclose(out, _first(DIRECTIONS["all"]), rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize(
    "u, v, w",
    [
        ("state", "state", None),
        ("parameters", "state", None),
        ("all", "time", None),
        ("flow", "flow", None),
        ("flow", "flow", "state"),
        ("all", "flow", "all"),
    ],
)
def test_difference_second_tangent_matches_autodiff(u, v, w):
    second = difference_second_tangent(chain, N, N, N_P)
    out = np.full(N, np.nan)
    zero = (np.zeros(N), 0.0, np.zeros(N_P))
    varies = zero if w is None else DIRECTIONS[w]
    # the second direction's parameter part is not part of the contract
    second_direction = (*DIRECTIONS[v][:2], np.zeros(N_P))
    second(out, Y, T, row(P), *DIRECTIONS[u], *second_direction, *varies)
    exact = _second(DIRECTIONS[u], second_direction)
    if w is not None:
        exact = exact + _first(DIRECTIONS[w])
    np.testing.assert_allclose(out, exact, rtol=2e-5, atol=2e-6 * np.abs(exact).max())


def crossing(y, t, p):
    """A quantity read through two coefficients, one of them passing through zero."""
    return (math.sin(y[0]) * p[3] + p[4], y[1] * t)


def crossing_ungrouped(y, t, p):
    return (math.sin(y[0]) * p[3] + p[4], y[1] * t)


def test_grouped_parameters_are_sized_by_the_largest_of_the_group():
    """A coefficient passing through zero is stepped on the scale of its group.

    Sized on its own it is stepped by ``6e-6`` of ``1e-13``, and the difference
    of a function of order one over that step is rounding.
    """
    p = np.array([1.0, 2.0, 3.0, 1.0e-13, 0.7, TYPICAL, TIME_SCALE, RTOL])
    dp = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    y = np.array([0.4, 0.2])
    exact = [math.sin(0.4), 0.0]

    set_parameter_groups(crossing, 5, (0, 1, 2, 3, 3))
    out = np.full(2, np.nan)
    difference_tangent(crossing, 2, 2, 5)(out, y, 0.5, p, np.zeros(2), 0.0, dp)
    np.testing.assert_allclose(out, exact, rtol=1e-8, atol=1e-18)

    difference_tangent(crossing_ungrouped, 2, 2, 5)(
        out, y, 0.5, p, np.zeros(2), 0.0, dp
    )
    assert abs(out[0] - exact[0]) > 1e-3


def directed(y, t, p):
    """A model read along a direction of its input: ``x + eps * dir`` with ``eps`` zero."""
    return (math.sqrt(p[0] + p[2] * p[1]) * y[0],)


def directed_undeclared(y, t, p):
    return (math.sqrt(p[0] + p[2] * p[1]) * y[0],)


def test_a_direction_parameter_is_stepped_on_the_scale_of_what_it_moves():
    """``eps`` is zero and moves an input of ``1e-9``: a step sized for one leaves the input's domain."""
    p = np.array([1.0e-9, 1.0, 0.0, TYPICAL, TIME_SCALE, RTOL])
    dp = np.array([0.0, 0.0, 1.0])
    y = np.array([2.0])
    exact = 2.0 / (2.0 * math.sqrt(1.0e-9))
    out = np.full(1, np.nan)

    difference_tangent(directed_undeclared, 1, 1, 3)(out, y, 0.5, p, np.zeros(1), 0.0, dp)
    assert not np.isfinite(out[0])

    set_directed_parameter(2, (1,), (0,))
    try:
        difference_tangent(directed, 1, 1, 3)(out, y, 0.5, p, np.zeros(1), 0.0, dp)
    finally:
        clear_directed_parameters()
    np.testing.assert_allclose(out[0], exact, rtol=1e-8)


def beside_a_guard(y, t, p):
    """Defined for ``y[0] > 0`` only, beside a state a billion times larger."""
    return (math.log(y[0]) if y[0] > 0.0 else math.nan, y[1])


def test_a_second_difference_shortens_its_steps_off_a_non_finite_corner():
    """Every state counts as large as the largest, so the first corners are at ``y[0] = -120``."""
    second = difference_second_tangent(beside_a_guard, 2, 2, 0)
    p = np.array([TYPICAL, TIME_SCALE, RTOL])
    y = np.array([1.0e-3, 1.0e6])
    u = (np.array([1.0, 0.0]), 0.0, np.zeros(0))
    zero = (np.zeros(2), 0.0, np.zeros(0))
    out = np.full(2, np.nan)
    second(out, y, 0.5, p, *u, *u, *zero)
    # The first steps that land inside are within a factor of eight of the
    # guard, here 0.46 of ``y[0]``: finite, and no better than this.
    np.testing.assert_allclose(out[0], -1.0 / y[0] ** 2, rtol=0.25)
    assert out[1] == 0.0


# --- the launch --------------------------------------------------------------------------


@pytest.mark.parametrize("lu_precision", ["fp32", "fp64"])
def test_robertson_matches_radau(lu_precision):
    hist = solve(
        robertson,
        ROBERTSON_Y0,
        ROBERTSON_T,
        ROBERTSON_P,
        rtol=1e-8,
        atol=1e-10,
        method=Rodas5P(lu_precision=lu_precision),
    )
    reference = solve_ivp(
        lambda t, y: robertson(y, t, ROBERTSON_P[0]),
        (0.0, ROBERTSON_T[-1]),
        ROBERTSON_Y0[0],
        method="Radau",
        t_eval=ROBERTSON_T,
        rtol=1e-11,
        atol=1e-13,
    ).y.T
    np.testing.assert_allclose(np.asarray(hist)[0], reference, rtol=2e-6, atol=1e-10)


def decay(y, t, p):
    return (-p[0] * y[0], p[0] * y[0] - y[1])


@pytest.mark.parametrize("method", [Rodas5P(lu_precision="fp64"), Tsit5()])
def test_each_trajectory_gets_the_answer_it_gets_alone(method):
    """The kernel sees one trajectory per call; the loop must hand each its own rows."""
    rates = np.linspace(0.5, 3.0, 37)[:, None]
    y0 = np.column_stack([np.linspace(1.0, 2.0, 37), np.zeros(37)])
    times = np.array([0.0, 0.5, 1.0])
    together = np.asarray(solve(decay, y0, times, rates, method=method))
    for i in (0, 17, 36):
        alone = np.asarray(
            solve(decay, y0[i : i + 1], times, rates[i : i + 1], method=method)
        )
        np.testing.assert_array_equal(together[i], alone[0])
    np.testing.assert_allclose(
        together[:, -1, 0], y0[:, 0] * np.exp(-rates[:, 0]), rtol=1e-6
    )


def reciprocal(y, t, p):
    return (-y[0] / p[0], y[0] / p[0] - y[1])


def test_a_division_by_zero_does_not_cost_the_other_trajectories():
    """IEEE arithmetic, as on the device: an infinity the controller rejects.

    numba's default error model raises instead, and a raise inside the
    parallel loop silently ends that thread's share of the ensemble.
    """
    n = 64
    params = np.ones((n, 1))
    params[5, 0] = 0.0
    y0 = np.tile([1.0, 0.0], (n, 1))
    hist = np.asarray(
        solve(reciprocal, y0, np.array([0.0, 1.0]), params, max_steps=200)
    )
    good = np.delete(np.arange(n), 5)
    np.testing.assert_allclose(hist[good, -1, 0], np.exp(-1.0), rtol=1e-6)


def guarded(y, t, p):
    """Decay beside a component frozen under a guard, past which there is no value."""
    if y[1] < 1.0e-8:
        frozen = 0.0
    else:
        frozen = math.inf - math.inf
    return (-p[0] * y[0], frozen)


def test_a_difference_step_over_a_guard_is_taken_the_other_way():
    """A state resting just under a guard is stepped over it by the forward difference.

    The step for a component this small is ``sqrt(eps) atol / rtol = 1.5e-10``,
    three times the distance to the guard, and the function is NaN beyond it.
    Left in the matrix, the NaN rejects every step.
    """
    y0 = np.array([[1.0, 0.995e-8]])
    times = np.array([0.0, 1.0, 2.0])
    hist, stats = solve(
        guarded, y0, times, np.array([[1.0]]), rtol=1e-5, atol=1e-7, return_stats=True
    )
    hist = np.asarray(hist)[0]
    assert int(stats["rejected_steps"][0]) < 10
    np.testing.assert_allclose(hist[:, 0], np.exp(-times), rtol=1e-4)
    np.testing.assert_array_equal(hist[:, 1], y0[0, 1])


# --- which target is chosen --------------------------------------------------------------


@pytest.mark.parametrize(
    "missing",
    [["numba_cuda_mlir"], ["numba_enzyme"], ["numba_cuda_mlir", "numba_enzyme"]],
)
def test_an_incomplete_cuda_stack_is_the_cpu_target(monkeypatch, missing):
    """numba-CUDA without Enzyme imports, and then fails inside the first implicit solve."""
    monkeypatch.setattr(_target, "_missing_cuda_packages", lambda: missing)
    monkeypatch.delenv("MODAX_TARGET", raising=False)
    assert _target._choose() == "cpu"
    monkeypatch.setenv("MODAX_TARGET", "cuda")
    with pytest.raises(ImportError, match=missing[-1]):
        _target._choose()


def test_the_cpu_target_can_be_forced_beside_a_complete_cuda_stack(monkeypatch):
    monkeypatch.setattr(_target, "_missing_cuda_packages", list)
    monkeypatch.delenv("MODAX_TARGET", raising=False)
    assert _target._choose() == "cuda"
    monkeypatch.setenv("MODAX_TARGET", "cpu")
    assert _target._choose() == "cpu"
    monkeypatch.setenv("MODAX_TARGET", "gpu")
    with pytest.raises(ValueError, match="MODAX_TARGET"):
        _target._choose()


def test_a_cpu_stand_in_is_not_numba_cuda(monkeypatch):
    import sys
    import types

    stand_in = types.ModuleType("numba_cuda_mlir")
    stand_in.__cpu_shim__ = True
    monkeypatch.setitem(sys.modules, "numba_cuda_mlir", stand_in)
    assert "numba_cuda_mlir" in _target._missing_cuda_packages()
