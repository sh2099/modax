"""A few perturbation modes against SciPy's Radau, an independent stiff integrator.

The same numba right-hand side and thermal tables, integrated by
``scipy.integrate.solve_ivp(method="Radau")`` at tight tolerance with the
finite-difference Jacobian, against modax's Rodas5P on the CPU target at
DISCO-EB's settings.

    PYTHONPATH=..:.:<DISCO-EB>/src .venv/bin/python check_radau.py
"""

from __future__ import annotations

import sys

import discoeb_cpu
import numpy as np
from scipy.integrate import solve_ivp

discoeb_cpu.install()

from discoeb import perturbations as P  # noqa: E402
from jacobian_tools import difference_jacobian  # noqa: E402
from run_pk import COSMOLOGIES  # noqa: E402

from modax._numba_common import as_cuda_device  # noqa: E402


def check(name: str) -> bool:
    from modax.rodas5P import solve

    cosmology = COSMOLOGIES[name]
    prepared = P._prepare_solve((cosmology,))
    prepared.bind()
    layout = prepared.layout
    k = np.array([1.0e-3, 1.0e-2, 1.0e-1, 0.5])
    params, y0, t_span, _, _ = P._pack_batch(prepared, k)
    hist, stats = solve(
        prepared.ode_fn, y0, t_span, params,
        sparsity=prepared.sparsity, rtol=P.PERTURB_RTOL, atol=P.PERTURB_ATOL,
        first_step=P.PERTURB_FIRST_STEP, max_steps=P.PERTURB_MAX_STEPS,
        pcoeff=0.3, icoeff=0.4, tf_index=P.IX_TAU_END, lu_precision="fp64",
        return_stats=True,
    )  # fmt: skip
    hist = np.asarray(hist)
    rhs = as_cuda_device(prepared.ode_fn)
    jacobian = difference_jacobian(prepared.ode_fn, layout.nvar, prepared.sparsity)
    ok = True
    for i, kv in enumerate(k):
        p = params[i]
        reference = solve_ivp(
            lambda t, y, p=p: np.array(rhs(y, t, p)),
            (t_span[0], p[P.IX_TAU_END]),
            y0[i],
            method="Radau",
            jac=lambda t, y, p=p: jacobian(y, t, p)[0],
            rtol=1e-9,
            atol=1e-11,
        )
        assert reference.success
        ours, theirs = hist[i, -1], reference.y[:, -1]
        matter = max(
            abs(ours[j] / theirs[j] - 1.0) for j in (layout.ix_clxc, layout.ix_clxb)
        )
        ok &= matter < 10 * P.PERTURB_RTOL
        print(
            f"{name} k={kv:6.0e}: matter contrasts within {matter:.1e} of Radau "
            f"(rtol 1e-9, {reference.t.size - 1} steps) in "
            f"{int(stats['accepted_steps'][i])} Rodas5P steps"
        )
    return ok


if __name__ == "__main__":
    results = [check(name) for name in (sys.argv[1:] or list(COSMOLOGIES))]
    sys.exit(0 if all(results) else 1)
