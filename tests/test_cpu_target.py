"""The CPU target: the same kernels under plain numba, with differenced derivatives.

Everything else in the suite runs on whichever target the machine has, and on
the CPU with whichever derivatives MODAX_CPU_DERIVATIVES chose. What is
checked here is what only the CPU target does: its finite-difference Jacobian
writer and its stand-ins for Enzyme's sweeps against JAX autodiff (whichever
engine the solves use), the launch loop, and the choice of target and of
derivatives. tests/test_enzyme_cpu.py checks the Enzyme engine.
"""

import math
import os
import subprocess
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from modax import Rodas5P, Tsit5, _target, solve
from modax._differences import (
    difference_second_tangent,
    difference_tangent,
    set_parameter_groups,
)
from modax._numba_common import make_cuda_local_vector_writer
from modax._rodas5P import make_difference_jacobian
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


def test_the_launch_columns_are_read_at_the_rows_end():
    """A joint launch's row holds a ``dp`` per direction between the parameters and them.

    Zeros there, the worst case: read as the launch's columns they would be a
    zero floor and a zero time scale. The sweeps must give what they give on
    the plain row, and must not move the ``dp`` entries.
    """
    directions = np.zeros(2 * N_P)
    joint_row = np.concatenate([P, directions, [TYPICAL, TIME_SCALE, RTOL]])
    zero = (np.zeros(N), 0.0, np.zeros(N_P))
    for name in ("state", "time", "parameters", "all"):
        plain = np.full(N, np.nan)
        joint = np.full(N, np.nan)
        difference_tangent(chain, N, N, N_P)(plain, Y, T, row(P), *DIRECTIONS[name])
        difference_tangent(chain, N, N, N_P)(joint, Y, T, joint_row, *DIRECTIONS[name])
        np.testing.assert_array_equal(joint, plain)
        difference_second_tangent(chain, N, N, N_P)(
            plain, Y, T, row(P), *DIRECTIONS[name], *DIRECTIONS["flow"], *zero
        )
        difference_second_tangent(chain, N, N, N_P)(
            joint, Y, T, joint_row, *DIRECTIONS[name], *DIRECTIONS["flow"], *zero
        )
        np.testing.assert_array_equal(joint, plain)
    np.testing.assert_array_equal(joint_row[N_P : 3 * N_P], 0.0)


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


# The cascade of tests/test_sensitivity.py, differentiated with the finite
# differences whichever engine this process chose: the derivatives a joint
# launch takes on the plain kernels, against the closed form. Its directions
# have zeros in their first dp entries, which the sweeps once read as the
# launch's appended columns (a zero floor and time scale).
_DIFFERENCED_CASCADE = """
import json
import jax
import jax.numpy as jnp
import numpy as np
from modax import Rodas5P, Tsit5, _target, solve
from tests.test_sensitivity import A, B, T_SPAN, Y0, cascade

jax.config.update("jax_enable_x64", True)
p = jnp.asarray([A, B])
y0 = jnp.asarray(Y0)
errors = {"engine": _target.CPU_DERIVATIVES}
for name, method in (("rodas5P", Rodas5P(lu_precision="fp64")), ("tsit5", Tsit5())):
    def final(y, q):
        return solve(cascade, y, T_SPAN, q, rtol=1e-10, atol=1e-12, method=method)[0]
    for label, dy, dq in (
        ("b", jnp.zeros(2), jnp.asarray([0.0, 1.0])),
        ("c0", jnp.asarray([1.0, 0.0]), jnp.zeros(2)),
        ("c1", jnp.asarray([0.0, 1.0]), jnp.zeros(2)),
    ):
        value, tangent = jax.jvp(final, (y0, p), (dy, dq))
        errors[name + "/" + label] = [np.asarray(value).tolist(), np.asarray(tangent).tolist()]
print(json.dumps(errors))
"""


def test_differenced_joint_solves_on_the_plain_kernels_match_the_closed_form():
    """Rodas5P and Tsit5 tangents along one parameter or one initial state, by differences."""
    import json

    from tests.test_sensitivity import (
        T_SPAN,
        Y0,
        A,
        B,
        cascade_dparams,
        cascade_dy0,
        cascade_exact,
    )

    env = dict(
        os.environ, MODAX_TARGET="cpu", MODAX_CPU_DERIVATIVES="differences",
        JAX_PLATFORMS="cpu",
    )  # fmt: skip
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [root, *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    done = subprocess.run(
        [sys.executable, "-c", _DIFFERENCED_CASCADE],
        env=env, cwd=root, capture_output=True, text=True, timeout=900,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    got = json.loads(done.stdout.strip().splitlines()[-1])
    assert got.pop("engine") == "differences"
    value = cascade_exact(T_SPAN, Y0, (A, B))
    exact = {
        "b": cascade_dparams(T_SPAN, Y0, (A, B))[..., 1],
        "c0": cascade_dy0(T_SPAN, Y0, (A, B))[..., 0],
        "c1": cascade_dy0(T_SPAN, Y0, (A, B))[..., 1],
    }
    for key, (primal, tangent) in got.items():
        np.testing.assert_allclose(primal, value, rtol=1e-7, atol=1e-9, err_msg=key)
        np.testing.assert_allclose(
            tangent, exact[key.split("/")[1]], rtol=1e-6, atol=1e-8, err_msg=key
        )


@pytest.mark.parametrize("per_thread, copies", [(2, 2), (1, 3)])
def test_hierarchy_directions_spread_over_replicas(per_thread, copies, monkeypatch):
    """``max_directions_per_thread`` on the hierarchy kernel, the CPU's way.

    Three directions (each parameter column scaled) of two trajectories: with
    ``k`` a thread each trajectory runs as ``ceil(3 / k)`` copies, the last
    padded with zero directions. The tangents are those of one thread
    carrying all three.
    """
    from tests.test_hierarchy import CHAIN, T_SPAN, TOL, Y0, P
    from tests.test_sensitivity import _launches

    def final(per):
        def at(theta):
            return solve(
                CHAIN, Y0, T_SPAN, P * theta,
                max_directions_per_thread=per, **TOL,
            )  # fmt: skip

        return at

    reference = jax.jacfwd(final(None))(jnp.ones(3))
    seen = _launches(monkeypatch)
    spread = jax.jacfwd(final(per_thread))(jnp.ones(3))
    n, spec = seen[-1]
    assert n == 2 * copies and spec.n_sens == per_thread
    # as tests/test_sensitivity.py's replicas of a plain solve: the copies'
    # step control sees fewer sensitivities, so the steps differ
    np.testing.assert_allclose(
        np.asarray(spread), np.asarray(reference), rtol=1e-6, atol=1e-8
    )
    assert np.abs(np.asarray(reference)).max() > 1e-3


# --- which derivatives the CPU target takes ----------------------------------------------


def _unavailable(monkeypatch, reason):
    from modax import _enzyme_cpu

    monkeypatch.setattr(_enzyme_cpu, "unavailable", lambda: reason)


def test_without_enzyme_the_cpu_target_differences_and_says_why(monkeypatch):
    _unavailable(monkeypatch, "no tools here")
    monkeypatch.delenv("MODAX_CPU_DERIVATIVES", raising=False)
    with pytest.warns(RuntimeWarning, match="no tools here") as caught:
        assert _target._choose_derivatives("cpu") == "differences"
    assert len(caught) == 1
    monkeypatch.setenv("MODAX_CPU_DERIVATIVES", "enzyme")
    with pytest.raises(ImportError, match="no tools here"):
        _target._choose_derivatives("cpu")
    monkeypatch.setenv("MODAX_CPU_DERIVATIVES", "differences")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _target._choose_derivatives("cpu") == "differences"


def test_with_enzyme_the_cpu_target_takes_it(monkeypatch):
    _unavailable(monkeypatch, None)
    monkeypatch.delenv("MODAX_CPU_DERIVATIVES", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _target._choose_derivatives("cpu") == "enzyme"
        monkeypatch.setenv("MODAX_CPU_DERIVATIVES", "enzyme")
        assert _target._choose_derivatives("cpu") == "enzyme"
    monkeypatch.setenv("MODAX_CPU_DERIVATIVES", "exact")
    with pytest.raises(ValueError, match="MODAX_CPU_DERIVATIVES"):
        _target._choose_derivatives("cpu")


def test_cuda_derivatives_are_always_enzyme(monkeypatch):
    _unavailable(monkeypatch, "not asked")
    monkeypatch.setenv("MODAX_CPU_DERIVATIVES", "differences")
    assert _target._choose_derivatives("cuda") == "enzyme"


def test_a_configured_tool_directory_must_hold_the_tools(monkeypatch, tmp_path):
    from modax import _enzyme_cpu

    monkeypatch.setenv("MODAX_ENZYME_TOOLS", str(tmp_path))
    with pytest.raises(_enzyme_cpu.EnzymeUnavailable, match="lacks opt.*Enzyme plugin"):
        _enzyme_cpu.find_tools()


def test_a_missing_toolchain_falls_back_with_one_warning(tmp_path):
    """The whole import, in a process of its own: one warning, and differences."""
    code = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import modax\n"
        "from modax import _target\n"
        "said = [w for w in caught if 'Enzyme cannot run' in str(w.message)]\n"
        "print(_target.TARGET, _target.CPU_DERIVATIVES, len(said))\n"
    )
    env = dict(
        os.environ,
        MODAX_TARGET="cpu",
        MODAX_CPU_DERIVATIVES="auto",
        MODAX_ENZYME_TOOLS=str(tmp_path),
        JAX_PLATFORMS="cpu",
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["cpu", "differences", "1"]


# --- the source generated for CUDA -------------------------------------------------------

# sha256[:16] of every Jacobian writer and hierarchy kernel these builds
# generate for the CUDA target, from upstream v0.5.0 (imported with
# numba-cuda stand-ins, its Enzyme sweeps stubbed: nothing here reads them).
# The CPU target's branches must leave that text alone.
V050_CUDA_SOURCE = {
    "hier3_plain": [
        "05c96f922f29396b",
        "b632d2b59cbb371e",
        "0a0a04b5c624509c",
        "4da668a74a796d15",
    ],
    "hier2_plain": ["b69731bbdd0805c9", "9472756311cbbfdd", "5c4cae320718a1e8"],
    "hier3_fp32_default": [
        "05c96f922f29396b",
        "b632d2b59cbb371e",
        "0a0a04b5c624509c",
        "c6b12fa25f1913ea",
    ],
    "hier3_auto": [
        "05c96f922f29396b",
        "b632d2b59cbb371e",
        "0a0a04b5c624509c",
        "6f8b4e4eb74b4885",
    ],
    "hier3_sens": [
        "84df6af1f6b052d8",
        "a18b5be36cbee460",
        "447d0a25af49ef49",
        "dd6ed873160df04e",
    ],
    "hier3_hook": [
        "05c96f922f29396b",
        "b632d2b59cbb371e",
        "0a0a04b5c624509c",
        "5ea8eb0a6e2d2729",
    ],
    "rodas_dense": ["bcad10e56d455c5c"],
    "rodas_dense_sens": ["bcad10e56d455c5c"],
    "rodas_sparse": ["a767317abd83d7ef"],
}


def rhs_a(y, t, p):
    return (-p[0] * y[0] + math.cos(t),)


def out_a(y, t, p):
    return (math.sin(y[0]), y[0] * t)


def rhs_b(y, t, p, q):
    return (-p[1] * (y[0] - q[0]) + q[1],)


def rhs_c(y, t, p, q):
    return (-p[2] * (y[0] - q[0]) + q[1],)


def sum_hook(save_idx, y, t, p_row, acc):
    acc[save_idx] = y[0] + y[2]


def bidiagonal(y, t, p):
    return tuple(
        (-p[0] * y[0] if j == 0 else p[0] * y[j - 1] - (p[1] + 0.1 * j) * y[j])
        for j in range(6)
    )


def test_the_source_generated_for_cuda_is_upstreams(monkeypatch):
    """Built with this branch's modules told the target is CUDA; nothing is compiled."""
    import hashlib

    from modax import Auto
    from modax import _hierarchy as H
    from modax import _rodas5P as R
    from modax._hierarchy import Block, Hierarchy
    from modax._options import KernelOptions
    from modax._sensitivity import SensitivitySpec

    monkeypatch.setattr(H, "TARGET", "cuda")
    monkeypatch.setattr(R, "TARGET", "cuda")
    sources: dict[str, list[str]] = {}
    current = [""]
    generate = R._make_literal_seed_jacobian

    def recording(**kw):
        writer = generate(**kw)
        sources.setdefault(current[0], []).append(writer.py_func._generated_source)
        return writer

    class Generated(Exception):
        pass

    def kernel_source(source, namespace, jit_options):
        sources.setdefault(current[0], []).append(source)
        raise Generated

    monkeypatch.setattr(R, "_make_literal_seed_jacobian", recording)
    monkeypatch.setattr(H, "_compile_kernel", kernel_source)

    a = Block(rhs_a, 1, outputs=out_a, n_outputs=2, name="a")
    b = Block(rhs_b, 1, reads=(a,), name="b")
    chain3 = Hierarchy(Block(rhs_c, 1, reads=(b, a), name="c"))
    fp64 = Rodas5P(lu_precision="fp64")
    # Two directions: the row is the three parameters and a dp for each.
    spec = SensitivitySpec(3, 3, n_dirs=2)
    for name, hierarchy, width, options in [
        ("hier3_plain", chain3, 3, KernelOptions(method=fp64)),
        ("hier2_plain", Hierarchy(b), 3, KernelOptions(method=fp64)),
        ("hier3_fp32_default", chain3, 3, KernelOptions()),
        ("hier3_auto", chain3, 3, KernelOptions(method=Auto())),
        ("hier3_sens", chain3, spec.row_width, KernelOptions(method=fp64, spec=spec)),
        ("hier3_hook", chain3, 3,
         KernelOptions(method=fp64, save_hook=sum_hook, hook_size=7)),
    ]:  # fmt: skip
        current[0] = name
        with pytest.raises(Generated):
            H.make_hierarchy_kernel.__wrapped__(hierarchy, width, options)
    pattern = normalize_sparsity(np.tril(np.triu(np.ones((6, 6), dtype=bool), -1)), 6)
    for name, fn, n, width, options in [
        ("rodas_dense", robertson, 3, 3, KernelOptions(method=fp64)),
        ("rodas_dense_sens", robertson, 3, spec.row_width,
         KernelOptions(method=fp64, spec=spec)),
        ("rodas_sparse", bidiagonal, 6, 3, KernelOptions(sparsity=pattern)),
    ]:  # fmt: skip
        current[0] = name
        R.make_kernel.__wrapped__(fn, n, width, options)
    digests = {
        name: [hashlib.sha256(s.encode()).hexdigest()[:16] for s in texts]
        for name, texts in sources.items()
    }
    assert digests == V050_CUDA_SOURCE
