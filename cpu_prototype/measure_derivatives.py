"""Derivatives of DISCO-EB on modax's CPU target: accuracy, cost, and the two layouts.

Run from the DISCO-EB checkout (its tests are imported for their cosmologies)::

    PYTHONPATH=<modax>:<modax>/cpu_prototype:src python <modax>/cpu_prototype/measure_derivatives.py

Three things, written to ``results/derivatives.json``:

* ``test_solve_eb_derivatives`` with its first assertion removed. That
  assertion requires two launches to agree to 1e-12, which they do only when
  their densities are bitwise equal; here NumPy and XLA's CPU ``pow`` differ by
  one ulp in ``Tcmb**4``, and an adaptive solve turns one ulp into ~1e-6 of a
  value's scale. The derivative checks that follow it are run unchanged.
* The gradient of P(k) in the five densities by the two sensitivity layouts
  (``one_column_per_trajectory``): the same derivative through a parameter
  that is exactly zero (``eps``) and through the nonzero densities, so their
  difference measures the zero-parameter step rule of ``modax._differences``.
* Wall times of the plain solve and of value-and-gradient.
"""

from __future__ import annotations

import inspect
import json
import sys
import textwrap
import time
from pathlib import Path

import discoeb_cpu

discoeb_cpu.install()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "tests")
import test_perturbations as TP  # noqa: E402
from discoeb import perturbations as P  # noqa: E402

RESULTS = Path(__file__).parent / "results"
out: dict = {}

# --- solve_eb's derivative checks, without the bitwise-equality assertion -----------------
source = textwrap.dedent(inspect.getsource(TP.test_solve_eb_derivatives))
dropped = "    np.testing.assert_allclose(np.asarray(plain.values), base, rtol=1e-12)\n"
assert source.count(dropped) == 1
namespace = dict(vars(TP))
exec(source.replace(dropped, ""), namespace)  # noqa: S102
start = time.perf_counter()
try:
    namespace["test_solve_eb_derivatives"]()
    out["solve_eb_derivative_checks"] = "pass"
except AssertionError as error:
    # Report which check and by how much, rather than stopping the measurements.
    frame = error.__traceback__
    while (
        frame.tb_next is not None
        and frame.tb_frame.f_code.co_name != "test_solve_eb_derivatives"
    ):
        frame = frame.tb_next
    line = (
        source.splitlines()[frame.tb_lineno - 1].strip()
        if frame.tb_lineno <= len(source.splitlines())
        else "?"
    )
    summary = [
        text.strip()
        for text in str(error).splitlines()
        if "difference among violations" in text or "relative" in text
    ]
    out["solve_eb_derivative_checks"] = {"failed_at": line, "detail": summary[:3]}
out["solve_eb_derivative_checks_s"] = time.perf_counter() - start
print(
    f"solve_eb derivative checks: {out['solve_eb_derivative_checks']} ({out['solve_eb_derivative_checks_s']:.1f} s)",
    flush=True,
)

# --- P(k): the two layouts against each other, and cost --------------------------------------
cosmology = P._as_cosmology(TP.REFERENCE_COSMOLOGY)
k = TP.MATTER_POWER_K
densities = jnp.asarray(P.density_coefficients(cosmology))


def log_pk(d, one_column):
    return jnp.log(
        P.matter_power_spectrum_jax(
            k, cosmology, d, one_column_per_trajectory=one_column
        )
    )


def timed(fn):
    begin = time.perf_counter()
    result = jax.block_until_ready(fn())
    return time.perf_counter() - begin, result


jacobians = {}
for one_column in (True, False):
    label = "eps column (one per replica)" if one_column else "five density columns"
    timed(lambda: P.matter_power_spectrum_jax(k, cosmology, densities))
    t_plain = min(
        timed(lambda: P.matter_power_spectrum_jax(k, cosmology, densities))[0]
        for _ in range(3)
    )
    jac = jax.jacfwd(lambda d, one_column=one_column: log_pk(d, one_column))
    timed(lambda: jac(densities))
    t_jac, value = timed(lambda: jac(densities))
    # d ln P / d ln density: order-one numbers, comparable across the five
    jacobians[one_column] = np.asarray(value) * np.asarray(densities)[None, :]
    out[label] = {"plain_s": t_plain, "jacobian_s": t_jac, "ratio": t_jac / t_plain}
    print(
        f"{label}: plain {t_plain * 1e3:.0f} ms, Jacobian {t_jac * 1e3:.0f} ms ({t_jac / t_plain:.1f}x)",
        flush=True,
    )

difference = np.abs(jacobians[True] - jacobians[False])
names = ("grhog", "grhornomass", "grhoc", "grhob", "grhov")
out["layout_difference_dlnP_dlnrho"] = {
    name: {
        "max_abs_difference": float(difference[:, j].max()),
        "max_abs_value": float(np.abs(jacobians[False][:, j]).max()),
    }
    for j, name in enumerate(names)
}
for j, name in enumerate(names):
    print(
        f"   d ln P / d ln {name:12s}: layouts differ by at most {difference[:, j].max():.1e} "
        f"(values up to {np.abs(jacobians[False][:, j]).max():.2f})"
    )
RESULTS.mkdir(exist_ok=True)
(RESULTS / "derivatives.json").write_text(json.dumps(out, indent=1))
