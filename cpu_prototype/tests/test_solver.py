"""The CPU Rodas5P against stiff problems with known answers.

Run from ``cpu_prototype/`` with the modax checkout on the path::

    PYTHONPATH=..:. .venv/bin/python -m pytest tests -q
"""

import math
import re
from pathlib import Path

import numpy as np
import pytest
import rodas5p_cpu
from rodas5p_cpu import make_solver
from scipy.integrate import solve_ivp

MODAX = Path(__file__).resolve().parents[2] / "modax"


# --- the copied constants are modax's ---------------------------------------------------


def _source_constants(text: str) -> dict:
    return {
        name: float(value)
        for name, value in re.findall(
            r"^([A-Z][A-Z0-9_]*) = +(-?\d[\d.e-]*)\s*(?:#.*)?$",
            text,
            flags=re.MULTILINE,
        )
    }


def test_tableau_matches_modax_source():
    source = _source_constants((MODAX / "rodas5P.py").read_text())
    names = [
        n for n in source if re.fullmatch(r"GAMMA|[ACD]\d+|ORDER|FACTOR_M(IN|AX)", n)
    ]
    assert len(names) >= 15 + 28 + 4 + 4 + 3
    for name in names:
        assert getattr(rodas5p_cpu, name) == source[name], name
    assert rodas5p_cpu.D1 == rodas5p_cpu.GAMMA
    common = _source_constants((MODAX / "_numba_common.py").read_text())
    assert rodas5p_cpu.SAFETY == common["SAFETY"]


def test_dense_output_rows_match_modax_source():
    text = (MODAX / "rodas5P.py").read_text()
    for name, row in (
        ("h1", rodas5p_cpu.H1),
        ("h2", rodas5p_cpu.H2),
        ("h3", rodas5p_cpu.H3),
    ):
        block = re.search(rf"{name} = \((.*?)\n\s+\)", text, flags=re.DOTALL).group(1)
        signed = re.findall(r"([+-]?)\s*(\d+\.\d+) \* k_stages\[(\d), j\]", block)
        assert len(signed) == 8, name
        for sign, value, k in signed:
            expected = -float(value) if sign == "-" else float(value)
            assert row[int(k)] == expected, (name, k)


# --- test problems -----------------------------------------------------------------------


def robertson(y, t, p):
    return (
        -p[0] * y[0] + p[1] * y[1] * y[2],
        p[0] * y[0] - p[1] * y[1] * y[2] - p[2] * y[1] ** 2,
        p[2] * y[1] ** 2,
    )


def van_der_pol(y, t, p):
    return (y[1], p[0] * ((1.0 - y[0] * y[0]) * y[1] - y[0]))


def prothero_robinson(y, t, p):
    """``y' = -lam (y - sin t) + cos t``: the solution through ``y(0) = 0`` is ``sin t``."""
    return (-p[0] * (y[0] - math.sin(t)) + math.cos(t),)


ROBERTSON_TIMES = np.array([0.0, 1e-6, 1e-2, 1e2, 1e5])
ROBERTSON_Y0 = np.array([1.0, 0.0, 0.0])
ROBERTSON_P = np.array([0.04, 1e4, 3e7])


def _radau(fn, y0, times, p, rtol=1e-11, atol=1e-13):
    sol = solve_ivp(
        lambda t, y: np.array(fn(y, t, p)),
        (times[0], times[-1]),
        y0,
        method="Radau",
        t_eval=times,
        rtol=rtol,
        atol=atol,
    )
    assert sol.success
    return sol.y.T


@pytest.mark.parametrize("lu_precision", ["fp64", "fp32"])
def test_robertson_matches_radau(lu_precision):
    solver = make_solver(robertson, 3, 3, lu_precision=lu_precision)
    hist, stats = solver.solve(
        ROBERTSON_Y0, ROBERTSON_TIMES, ROBERTSON_P, rtol=1e-8, atol=1e-10, fd_floor=1e-8
    )
    reference = _radau(robertson, ROBERTSON_Y0, ROBERTSON_TIMES, ROBERTSON_P)
    np.testing.assert_allclose(hist[0], reference, rtol=2e-6, atol=1e-10)
    assert stats["loop_steps"][0] < 2000
    # mass is conserved, to the precision of the stage solves
    np.testing.assert_allclose(
        hist[0].sum(axis=1), 1.0, rtol=1e-9 if lu_precision == "fp64" else 1e-6
    )


def test_robertson_error_falls_with_tolerance():
    reference = _radau(robertson, ROBERTSON_Y0, ROBERTSON_TIMES, ROBERTSON_P)
    solver = make_solver(robertson, 3, 3)
    errors, in_tolerances, steps = [], [], []
    for rtol in (1e-4, 1e-6, 1e-8):
        atol = rtol * 1e-4
        hist, stats = solver.solve(
            ROBERTSON_Y0,
            ROBERTSON_TIMES,
            ROBERTSON_P,
            rtol=rtol,
            atol=atol,
            fd_floor=1e-8,
        )
        difference = np.abs(hist[0] - reference)
        errors.append(np.max(difference))
        in_tolerances.append(np.max(difference / (atol + rtol * np.abs(reference))))
        steps.append(int(stats["accepted_steps"][0]))
    assert errors[0] > errors[1] > errors[2]
    # the global error stays a modest multiple of the local tolerance
    assert max(in_tolerances) < 50.0
    assert steps[0] < steps[1] < steps[2]


def test_stiff_van_der_pol_ensemble_matches_radau():
    mus = np.array([[10.0], [100.0], [1000.0]])
    y0 = np.array([2.0, 0.0])
    times = np.linspace(0.0, 20.0, 6)
    solver = make_solver(van_der_pol, 2, 1)
    hist, stats = solver.solve(y0, times, mus, rtol=1e-8, atol=1e-10)
    for i, mu in enumerate(mus):
        reference = _radau(van_der_pol, y0, times, mu)
        np.testing.assert_allclose(hist[i], reference, rtol=2e-5, atol=1e-7)
    serial, _ = solver.solve(y0, times, mus, rtol=1e-8, atol=1e-10, parallel=False)
    np.testing.assert_array_equal(hist, serial)
    assert np.all(stats["accepted_steps"] > 0)


def test_stiff_nonautonomous_problem_converges():
    """``df/dt`` enters every stage; without it the stiff forced problem loses its order.

    Checked at the step ends, where the method is accurate far beyond the
    tolerance however stiff the problem (10 steps across the interval at
    ``rtol = 1e-4``).
    """
    lam = np.array([1.0e4])
    ends = np.array([0.0, 3.0])
    solver = make_solver(prothero_robinson, 1, 1)
    for rtol in (1e-4, 1e-6, 1e-8):
        hist, stats = solver.solve(np.zeros(1), ends, lam, rtol=rtol, atol=rtol)
        assert abs(hist[0, -1, 0] - math.sin(3.0)) < rtol
        # smooth forcing: the step is set by accuracy, not by the stiffness 1/lam
        assert stats["accepted_steps"][0] < 60


def test_dense_output_of_the_stiff_problem_is_less_accurate_than_the_steps():
    """Pins a measured property of the continuous extension, to compare with CUDA.

    Between step ends the interpolant of a very stiff forced problem is far
    less accurate than the steps themselves: at ``lam = 1e4`` and ``rtol =
    1e-4`` the step ends are good to 3e-8 and the saves inside the (0.3-wide)
    steps to 4e-3. The coefficients and the formula are modax's
    (``test_dense_output_rows_match_modax_source``), so the CUDA kernel is
    expected to show the same; ``dump_gpu_reference.py`` records it. It
    converges with the tolerance, and at ``lam = 1`` it is as accurate as the
    steps.
    """
    times = np.linspace(0.0, 3.0, 301)
    solver = make_solver(prothero_robinson, 1, 1)
    errors = []
    for rtol in (1e-4, 1e-6, 1e-8):
        hist, _ = solver.solve(
            np.zeros(1), times, np.array([1.0e4]), rtol=rtol, atol=rtol
        )
        errors.append(np.max(np.abs(hist[0, :, 0] - np.sin(times))))
    assert errors[0] > errors[1] > errors[2]
    assert 1e-3 < errors[0] < 1e-2 and errors[2] < 1e-6
    mild, _ = solver.solve(np.zeros(1), times, np.array([1.0]), rtol=1e-4, atol=1e-4)
    assert np.max(np.abs(mild[0, :, 0] - np.sin(times))) < 5e-5


def test_per_trajectory_end_time_freezes_the_final_state():
    def decay(y, t, p):
        return (-p[0] * y[0],)

    solver = make_solver(decay, 1, 2, tf_index=1)
    params = np.array([[1.0, 1.0], [1.0, 2.5]])
    times = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    hist, _ = solver.solve(np.ones(1), times, params, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(hist[0, :3, 0], np.exp(-times[:3]), rtol=1e-7)
    np.testing.assert_allclose(hist[0, 3:, 0], math.exp(-1.0), rtol=1e-7)
    np.testing.assert_allclose(hist[1, :4, 0], np.exp(-times[:4]), rtol=1e-7)
    np.testing.assert_allclose(hist[1, 4, 0], math.exp(-2.5), rtol=1e-7)


def test_error_weights_tighten_a_component():
    solver = make_solver(robertson, 3, 3)
    kwargs = {"rtol": 1e-4, "atol": 1e-6, "fd_floor": 1e-8}
    _, plain = solver.solve(ROBERTSON_Y0, ROBERTSON_TIMES, ROBERTSON_P, **kwargs)
    _, weighted = solver.solve(
        ROBERTSON_Y0,
        ROBERTSON_TIMES,
        ROBERTSON_P,
        error_weights=[1.0, 1e3, 1.0],
        **kwargs,
    )
    assert weighted["accepted_steps"][0] > plain["accepted_steps"][0]


# --- the finite-difference Jacobian --------------------------------------------------------


def _banded_pattern(n):
    """The lower-bidiagonal pattern of the chain below: two colours however long."""
    pattern = np.zeros((n, n), dtype=bool)
    for i in range(n):
        pattern[i, i] = True
        if i:
            pattern[i, i - 1] = True
    return pattern


def test_coloured_jacobian_matches_the_dense_one_and_autodiff():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    n = 12
    pattern = _banded_pattern(n)
    # numba needs a fixed-size tuple: build the right-hand side as source
    lines = ["def chain_n(y, t, p):", "    return ("]
    lines.append("        -p[0] * y[0] + math.sin(t),")
    for i in range(1, n):
        lines.append(
            f"        p[0] * y[{i - 1}] - (p[0] + {0.1 * i}) * y[{i}] * (1.0 + 0.01 * y[{i}]),"
        )
    lines.append("    )")
    scope = {"math": math}
    exec("\n".join(lines), scope)  # noqa: S102
    chain_n = scope["chain_n"]

    dense = make_solver(chain_n, n, 1)
    coloured = make_solver(chain_n, n, 1, sparsity=pattern)
    assert dense.n_colours == n and coloured.n_colours == 2

    rng = np.random.default_rng(0)
    y, t, p = rng.normal(size=n), 0.7, np.array([3.0])
    j_dense, dt_dense, _ = dense.jacobian(y, t, p, fd_floor=1.0, t_scale=1.0)
    j_col, dt_col, f_col = coloured.jacobian(y, t, p, fd_floor=1.0, t_scale=1.0)

    def jax_chain(yv, tv):
        out = [-p[0] * yv[0] + jnp.sin(tv)]
        for i in range(1, n):
            out.append(
                p[0] * yv[i - 1] - (p[0] + 0.1 * i) * yv[i] * (1.0 + 0.01 * yv[i])
            )
        return jnp.stack(out)

    j_ad = np.asarray(jax.jacfwd(jax_chain)(jnp.asarray(y), t))
    dt_ad = np.asarray(jax.jacfwd(jax_chain, argnums=1)(jnp.asarray(y), t))
    np.testing.assert_allclose(
        f_col, np.asarray(jax_chain(jnp.asarray(y), t)), rtol=1e-13
    )
    np.testing.assert_allclose(j_dense, j_ad, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(j_col, j_ad, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dt_col, dt_ad, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(dt_dense, dt_col, rtol=1e-12)

    times = np.linspace(0.0, 5.0, 4)
    h_dense, s_dense = dense.solve(np.ones(n), times, p, rtol=1e-7, atol=1e-9)
    h_col, s_col = coloured.solve(np.ones(n), times, p, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(h_col, h_dense, rtol=1e-6, atol=1e-9)
    assert abs(int(s_col["accepted_steps"][0]) - int(s_dense["accepted_steps"][0])) <= 2
