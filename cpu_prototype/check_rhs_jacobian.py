"""The numba right-hand side and its finite-difference Jacobian against JAX autodiff.

Both are built from the same DISCO-EB component model: the numba one is what
the CPU solver integrates (``build_numba_rhs`` under the CUDA stand-in), the
JAX one is the model emitted as traced ``jax.numpy`` arithmetic, whose
``jacfwd`` is exact. Compared at states taken along real trajectories:

* the right-hand sides (the same equations through two compilers);
* the coloured finite-difference Jacobian and ``df/dt`` the solver uses
  against ``jacfwd``, in units of each row's scale;
* the *uncoloured* finite-difference Jacobian against the traced sparsity
  pattern, which must contain every nonzero for the colouring to be valid.

    PYTHONPATH=..:.:<DISCO-EB>/src .venv/bin/python check_rhs_jacobian.py
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import discoeb_cpu
import numpy as np

discoeb_cpu.install()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from discoeb import perturbations as P  # noqa: E402
from discoeb.component import Component, Model  # noqa: E402
from discoeb.eb_model import perturbation_model  # noqa: E402
from jacobian_tools import difference_jacobian  # noqa: E402
from run_pk import COSMOLOGIES  # noqa: E402

from modax._numba_common import as_cuda_device  # noqa: E402
from modax.rodas5P import solve  # noqa: E402


class JaxThermalHistory(Component):
    """``ThermalHistory``'s Horner spline over the same float32 tables, in JAX."""

    def __init__(self, tau_min, inv_dtau, values, rescaled_seconds):
        values = jnp.asarray(values[0], dtype=jnp.float64)
        seconds = jnp.asarray(rescaled_seconds[0], dtype=jnp.float64)
        n_grid = values.shape[-1]
        x_min, inv_dx = float(tau_min[0]), float(inv_dtau[0])

        def spline(x, channel):
            i = (x - x_min) * inv_dx
            idx = jnp.clip(jnp.floor(i).astype(int), 0, n_grid - 2)
            t = i - idx
            y0, y1 = values[channel, idx], values[channel, idx + 1]
            k0, k1 = seconds[channel, idx], seconds[channel, idx + 1]
            return y0 + t * (
                (y1 - y0) - (2.0 * k0 + k1) + t * (3.0 * k0 + t * (k1 - k0))
            )

        self.log_tau = lambda tau: jnp.log(tau)
        self.a = lambda log_tau: spline(log_tau, 0)
        self.opacity = lambda log_tau: jnp.maximum(spline(log_tau, 1), 1e-30)
        self.cs2 = lambda log_tau: jnp.maximum(spline(log_tau, 2), 0.0)
        self.rhonu = lambda log_tau: spline(log_tau, 3)
        self.pnu = lambda log_tau: spline(log_tau, 4)


def jax_rhs(layout, stacked):
    """The component model's right-hand side as ``f(y, tau, p) -> array`` in JAX."""

    def object_zeros(shape, dtype):
        array = np.empty(shape, dtype=object)
        array.fill(0.0)
        return array

    original = Model._device

    def device_with_jax_math(self, jit, cuda, float64):
        jit, names = original(self, jit, cuda, float64)
        return jit, {**names, "math": jnp}

    Model._device = device_with_jax_math
    try:
        model = perturbation_model(layout, JaxThermalHistory(*stacked))
        as_tuple = model.build(
            P.PARAM_COLUMNS,
            jit=lambda f: f,
            cuda=SimpleNamespace(local=SimpleNamespace(array=object_zeros)),
            float64=object,
        )
    finally:
        Model._device = original
    return lambda y, tau, p: jnp.stack(as_tuple(y, tau, p))


def check(name: str) -> bool:
    cosmology = COSMOLOGIES[name]
    prepared = P._prepare_solve((cosmology,))
    prepared.bind()
    layout, nvar = prepared.layout, prepared.layout.nvar
    k = np.geomspace(1.0e-4, 1.0, 9)
    tau_save = np.geomspace(P.TAU_START, float(prepared.tau0[0]), 12)
    params, y0, t_span, _, _ = P._pack_batch(prepared, k, tau_save)

    coloured = difference_jacobian(prepared.ode_fn, nvar, prepared.sparsity)
    dense = difference_jacobian(prepared.ode_fn, nvar)
    rhs = as_cuda_device(prepared.ode_fn)
    hist = np.asarray(
        solve(
            prepared.ode_fn,
            y0,
            t_span,
            params,
            sparsity=prepared.sparsity,
            rtol=1e-6,
            atol=1e-6,
            first_step=1e-2,
            tf_index=P.IX_TAU_END,
            lu_precision="fp64",
        )  # fmt: skip
    )

    f_jax = jax.jit(jax_rhs(layout, prepared.stacked))
    j_jax = jax.jit(jax.jacfwd(jax_rhs(layout, prepared.stacked), argnums=0))
    dt_jax = jax.jit(jax.jacfwd(jax_rhs(layout, prepared.stacked), argnums=1))

    pattern = np.zeros((nvar, nvar), dtype=bool)
    pattern[tuple(np.asarray(prepared.sparsity).T)] = True
    worst = {"rhs": 0.0, "jac": 0.0, "dT": 0.0, "outside": 0.0}
    seen = np.zeros((nvar, nvar), dtype=bool)
    for i in range(len(k)):
        for s in range(1, len(tau_save) - 1):
            y, tau, p = hist[i, s], float(tau_save[s]), params[i]
            f_ref = np.asarray(f_jax(y, tau, p))
            j_ref = np.asarray(j_jax(y, tau, p))
            dt_ref = np.asarray(dt_jax(y, tau, p))
            j_fd, dt_fd = coloured(y, tau, p)
            j_dense, _ = dense(y, tau, p)
            f_nb = np.array(rhs(y, tau, p))
            scale = np.abs(f_ref).max()
            worst["rhs"] = max(worst["rhs"], np.abs(f_nb - f_ref).max() / scale)
            row_scale = np.abs(j_ref).max(axis=1, keepdims=True) + 1e-300
            worst["jac"] = max(worst["jac"], (np.abs(j_fd - j_ref) / row_scale).max())
            worst["dT"] = max(
                worst["dT"],
                np.abs(dt_fd - dt_ref).max() / (np.abs(dt_ref).max() + 1e-300),
            )
            worst["outside"] = max(
                worst["outside"],
                (np.abs(np.where(pattern, 0.0, j_ref)) / row_scale).max(),
            )
            seen |= j_ref != 0.0
            # the uncoloured sweep agrees with the coloured one where the pattern is
            assert np.allclose(
                np.where(pattern, j_dense, 0.0),
                j_fd,
                rtol=1e-5,
                atol=1e-7 * np.abs(j_ref).max(),
            )
    ok = (
        worst["rhs"] < 1e-6
        and worst["jac"] < 1e-5
        and worst["dT"] < 1e-4
        and worst["outside"] == 0.0
        and not (seen & ~pattern).any()
    )
    print(
        f"{name} (nvar {nvar}, {coloured.n_colours} colours): "
        f"rhs numba vs JAX {worst['rhs']:.1e}; FD Jacobian vs jacfwd {worst['jac']:.1e} of the row scale; "
        f"df/dt {worst['dT']:.1e}; autodiff nonzeros {int(seen.sum())} of {int(pattern.sum())} "
        f"pattern entries, outside the pattern {int((seen & ~pattern).sum())}  ->  {'ok' if ok else 'FAIL'}"
    )
    return ok


if __name__ == "__main__":
    results = [check(name) for name in (sys.argv[1:] or list(COSMOLOGIES))]
    sys.exit(0 if all(results) else 1)
