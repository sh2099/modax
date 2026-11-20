"""Enzyme on the CPU target: its sweeps, its Jacobian and its stack arrays.

What is checked here is what only the Enzyme CPU path does: the forward
sweeps modax._enzyme_cpu builds against JAX autodiff (to rounding, where the
finite differences of tests/test_cpu_target.py are checked to 1e-8), the
CUDA branch's Jacobian writer fed with them, the conditions a function must
meet and the error when it does not, and that a derivative is built once per
function object. The rest of the suite runs on whichever engine
MODAX_CPU_DERIVATIVES chose; everything here skips without Enzyme.
"""

import math
import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numba import njit
from numba.core.errors import TypingError

from modax import Rodas5P, _target, solve
from modax._sparse_direct import sparse_direct_solver_for
from modax._sparsity import dense_jacobian, normalize_sparsity
from tests.test_cpu_target import (
    DIRECTIONS,
    N_P,
    N,
    P,
    T,
    Y,
    _first,
    _second,
    chain,
    chain_jax,
    row,
)

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.skipif(
    _target.TARGET != "cpu" or _target.CPU_DERIVATIVES != "enzyme",
    reason="checks of Enzyme on the CPU target",
)

if _target.TARGET == "cpu" and _target.CPU_DERIVATIVES == "enzyme":
    from modax import _enzyme_cpu
    from modax import _rodas5P as R
    from modax._sensitivity import make_quantity_second_tangent, make_quantity_tangent

# Rounding of a handful of operations on numbers of order one, not a step.
RTOL = 1e-13


def _close(out, exact):
    np.testing.assert_allclose(out, exact, rtol=RTOL, atol=RTOL * np.abs(exact).max())


# --- the sweeps --------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(DIRECTIONS))
def test_enzyme_tangent_matches_autodiff(name):
    tangent = make_quantity_tangent(chain, N, N, N_P)
    out = np.full(N, np.nan)
    tangent(out, Y, T, row(P), *DIRECTIONS[name])
    _close(out, _first(DIRECTIONS[name]))


def test_enzyme_tangent_reads_only_the_state_of_an_augmented_vector():
    """The state is the head of a longer vector and the row ends with the launch's columns."""
    tangent = make_quantity_tangent(chain, N, N, N_P)
    out = np.full(N, np.nan)
    augmented = np.concatenate([Y, np.full(2 * N, np.nan)])
    tangent(out, augmented, T, row(P), *DIRECTIONS["all"])
    _close(out, _first(DIRECTIONS["all"]))


@pytest.mark.parametrize(
    "u, v, w",
    [
        ("state", "state", None),
        ("parameters", "state", None),
        ("all", "time", None),
        ("flow", "flow", None),
        ("flow", "flow", "state"),
        ("all", "flow", "all"),
        ("all", "all", "parameters"),
    ],
)
def test_enzyme_second_tangent_matches_autodiff(u, v, w):
    """Unlike the differences, the second direction's parameter part counts too."""
    second = make_quantity_second_tangent(chain, N, N, N_P)
    out = np.full(N, np.nan)
    zero = (np.zeros(N), 0.0, np.zeros(N_P))
    varies = zero if w is None else DIRECTIONS[w]
    second(out, Y, T, row(P), *DIRECTIONS[u], *DIRECTIONS[v], *varies)
    exact = _second(DIRECTIONS[u], DIRECTIONS[v])
    if w is not None:
        exact = exact + _first(DIRECTIONS[w])
    _close(out, exact)


def one_value(y, t, p):
    return (math.sin(y[0] * p[0]) * t,)


def bare_value(y, t, p):
    return math.sin(y[0] * p[0]) * t


@pytest.mark.parametrize("fn", [one_value, bare_value])
def test_a_quantity_of_one_value(fn):
    """numba-enzyme lowers no one-tuple; the sweep differentiates its element."""
    y, p = np.array([0.3]), np.array([2.0, 0.0, 0.0, 0.0])
    dy, dp = np.array([1.5]), np.array([0.25])
    exact = math.cos(0.6) * (1.5 * 2.0 + 0.3 * 0.25) * 0.7 + math.sin(0.6) * 0.5
    out = np.full(1, np.nan)
    make_quantity_tangent(fn, 1, 1, 1)(out, y, 0.7, p, dy, 0.5, dp)
    np.testing.assert_allclose(out[0], exact, rtol=RTOL)
    second = np.full(1, np.nan)
    make_quantity_second_tangent(fn, 1, 1, 1)(
        second, y, 0.7, p, dy, 0.5, dp, np.zeros(1), 0.0, np.zeros(1),
        dy, 0.5, dp,
    )  # fmt: skip
    np.testing.assert_allclose(second[0], exact, rtol=RTOL)


def _scaled(factor):
    def scaled(y, t, p):
        return (factor * y[0] * y[0], p[0] * y[0])

    return scaled


def test_two_closures_of_the_same_source_get_their_own_derivatives():
    """numba-enzyme's own CPU cache is keyed on source text, and would give both the first's."""
    y, p = np.array([3.0]), np.array([1.0])
    one, zero = np.ones(1), np.zeros(1)
    results = []
    for factor in (2.0, 5.0):
        out = np.full(2, np.nan)
        make_quantity_tangent(_scaled(factor), 2, 1, 1)(out, y, 0.0, p, one, 0.0, zero)
        results.append(out[0])
    assert results == [2.0 * 2.0 * 3.0, 2.0 * 5.0 * 3.0]


def test_a_derivative_is_built_once_however_many_threads_ask():
    def raced(y, t, p):
        return (y[0] * p[0], y[0] - t)

    first = _enzyme_cpu.first_order(raced, 2)
    before = len(_enzyme_cpu.BUILDS)
    built = []
    threads = [
        threading.Thread(target=lambda: built.append(first.get())) for _ in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len({id(b) for b in built}) == 1
    assert len(_enzyme_cpu.BUILDS) == before + 1
    assert _enzyme_cpu.first_order(raced, 2) is first


def test_a_sweep_takes_contiguous_arrays():
    tangent = make_quantity_tangent(chain, N, N, N_P)

    @njit
    def strided(out, y, p):
        tangent(out, y[::2], 0.1, p, y[::2], 0.0, p)

    with pytest.raises(TypingError, match="contiguous 1-D float64"):
        strided(np.zeros(N), np.zeros(2 * N), row(P))


# --- what a differentiated function may contain -----------------------------------------


def _with_local(size):
    @_target.cuda.jit(device=True)
    def helper(y):
        buffer = _target.cuda.local.array(size, _target.types.float64)
        for j in range(buffer.shape[0]):
            buffer[j] = y[0] * (j + 1)
        total = 0.0
        for j in range(buffer.shape[0]):
            total += buffer[j]
        return total

    def uses_local(y, t, p):
        return (helper(y) * p[0], y[0])

    return uses_local


def test_a_constant_size_local_array_is_differentiated():
    """1 + 2 + 3 = 6 copies of ``y[0] p[0]``: a stack array, which Enzyme follows."""
    out = np.full(2, np.nan)
    make_quantity_tangent(_with_local(3), 2, 1, 1)(
        out, np.array([2.0]), 0.0, np.array([0.5]), np.ones(1), 0.0, np.zeros(1)
    )
    np.testing.assert_array_equal(out, [6.0 * 0.5, 1.0])


@njit
def _allocating_square(n, x):
    buffer = np.zeros(n)
    buffer[0] = x * x
    return buffer[0]


def allocates(y, t, p):
    buffer = np.zeros(int(p[1]))
    buffer[0] = y[0] * y[0]
    return (buffer[0], y[0])


def allocates_in_a_callee(y, t, p):
    return (_allocating_square(int(p[1]), y[0]), y[0])


@pytest.mark.parametrize("fn", [allocates, allocates_in_a_callee])
def test_an_allocation_is_refused_not_differentiated(fn):
    first = _enzyme_cpu.first_order(fn, 2)
    with pytest.raises(
        _enzyme_cpu.EnzymeBuildError, match="allocates from numba's heap"
    ):
        first.get()


def test_a_stack_array_starts_as_zeros_and_is_its_own():
    local = _target.cuda.local.array

    @njit
    def fill(k):
        a = local(4, np.float64)
        b = local((2, 3), np.float64)
        total = 0.0
        for i in range(4):
            total += a[i]
            a[i] = k
        for i in range(2):
            for j in range(3):
                total += b[i, j]
                b[i, j] = k
        return total, a[3], b[1, 2], b.shape[0], b.shape[1]

    assert fill(7.0) == (0.0, 7.0, 7.0, 2, 3)
    assert fill(8.0) == (0.0, 8.0, 8.0, 2, 3)

    @njit
    def runtime_size(n):
        return local(n, np.float64).sum()

    assert runtime_size(5) == 0.0


# --- the Jacobian writer, as the kernels build it ----------------------------------------


@pytest.mark.parametrize("layout", ["grid", "packed"])
def test_the_literal_seed_jacobian_matches_autodiff(layout):
    """The CUDA branch's writer, one direction a sweep, with the Enzyme tangent."""
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

    writer = R._make_literal_seed_jacobian(
        n_vars=N,
        n_params=N_P,
        n_colours=structure.n_colours,
        store_table=structure.store_slots(),
        namespace=dict(
            cuda=_target.cuda,
            float64=_target.types.float64,
            tangent_of=make_quantity_tangent(chain, N, N, N_P),
            colour_seeds=structure.seed_table(N_P),
            clear_matrix=clear,
            lu_dtype=np.float64,
        ),
        batch=1,
        coupling=1,
    )
    lu = np.full(structure.size, np.nan)
    dT = np.full(N, np.nan)
    seeds = np.array([[0.5, 0.0, -2.0]])
    coup = np.full((1, N), np.nan)
    writer(Y, T, row(P), lu, dT, seeds, coup)
    at = (jnp.asarray(Y), T, jnp.asarray(P))
    jac = np.asarray(jax.jacfwd(chain_jax)(*at))
    scale = np.abs(jac).max()
    for r in range(N):
        for c in range(N) if used is None else used[r]:
            assert abs(-lu[structure.slot(r, c)] - jac[r, c]) <= RTOL * scale
    _close(dT, np.asarray(jax.jacfwd(chain_jax, argnums=1)(*at)))
    _close(coup[0], np.asarray(jax.jacfwd(chain_jax, argnums=2)(*at)) @ seeds[0])


def test_sensitivities_of_a_stiff_solve_match_differenced_solves():
    """Robertson's dy/dp from the joint solve, against central differences of whole solves."""
    from tests.test_cpu_target import ROBERTSON_P, ROBERTSON_T, ROBERTSON_Y0, robertson

    def final(p):
        return solve(
            robertson, ROBERTSON_Y0, ROBERTSON_T, p, rtol=1e-10, atol=1e-14,
            method=Rodas5P(lu_precision="fp64"),
        )[0, -1]  # fmt: skip

    jac = np.asarray(jax.jacfwd(final)(jnp.asarray(ROBERTSON_P)))[:, 0, :]
    for k in range(3):
        h = 1e-4 * ROBERTSON_P[0, k]
        up, down = ROBERTSON_P.copy(), ROBERTSON_P.copy()
        up[0, k] += h
        down[0, k] -= h
        column = (np.asarray(final(up)) - np.asarray(final(down))) / (2.0 * h)
        np.testing.assert_allclose(jac[:, k], column, rtol=1e-5, atol=1e-12)
