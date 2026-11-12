"""``modax.rodas5P.solve`` on the CPU prototype, so unedited callers reach it.

DISCO-EB imports ``modax.rodas5P.solve`` at the point of each launch. The real
module cannot be imported without CUDA; :func:`install` puts a stand-in under
that name whose ``solve`` has modax's signature and runs
:mod:`rodas5p_cpu` behind ``jax.pure_callback``. That makes it traceable
under ``jax.jit`` and, like modax's ``custom_vmap`` rule, turns an enclosing
``jax.vmap`` over scalar solves into one ensemble solve.

This is a prototype bridge, the smallest thing that lets DISCO-EB's own JAX
code (the thermal-history post-processing around the RECFAST solve) run
unchanged. Not supported, and refused loudly: save hooks, sensitivities and
differentiation through the solve.
"""

from __future__ import annotations

import sys
import types as _pytypes

import numpy as np
import rodas5p_cpu

LAST_STATS: dict = {}
"""The step counters of the most recent solve, for inspection."""

DEFAULTS = {"parallel": True, "threads": None, "lu_precision": None}
"""Prototype-wide overrides: ``lu_precision`` here wins over the caller's."""


def solve(
    ode_fn,
    y0,
    t_span,
    params,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    first_step: float | None = None,
    max_steps: int = 100000,
    return_stats: bool = False,
    error_weights=None,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
    lu_precision: str = "fp32",
    trajectories_per_block=None,
    sens_error_control: bool = True,
    sens_param_columns=None,
    sparsity=None,
    ordering: str = "amd",
    tf_index: int | None = None,
    max_registers=None,
    array_rhs=None,
    save_hook=None,
    hook_size: int = 0,
    save_history: bool = True,
    sensitivity=None,
):
    """modax's ``solve``, forward history only, on the CPU; see the module docstring.

    ``trajectories_per_block``, ``max_registers`` and ``ordering`` are CUDA
    and sparse-LU settings with no meaning here and are ignored.
    """
    import jax
    import jax.numpy as jnp

    if save_hook is not None or not save_history or hook_size:
        raise NotImplementedError("the CPU prototype has no save hooks")
    if sensitivity is not None or sens_param_columns is not None:
        raise NotImplementedError("the CPU prototype has no sensitivities")
    if array_rhs is not None:
        raise NotImplementedError("the CPU prototype takes the tuple-form ode_fn only")

    y0 = jnp.asarray(y0, dtype=jnp.float64)
    params = jnp.asarray(params, dtype=jnp.float64)
    times = jnp.asarray(t_span, dtype=jnp.float64)
    y0_ndim, params_ndim = y0.ndim, params.ndim
    n_vars, n_params, n_save = y0.shape[-1], params.shape[-1], times.shape[-1]
    n = max(
        y0.shape[0] if y0_ndim == 2 else 1, params.shape[0] if params_ndim == 2 else 1
    )

    solver = rodas5p_cpu.make_solver(
        ode_fn,
        n_vars,
        n_params,
        sparsity=sparsity,
        lu_precision=DEFAULTS["lu_precision"] or lu_precision,
        tf_index=-1 if tf_index is None else int(tf_index),
        pcoeff=pcoeff,
        icoeff=icoeff,
        dcoeff=dcoeff,
    )
    have_weights = error_weights is not None
    weights = (
        jnp.asarray(error_weights, dtype=jnp.float64)
        if have_weights
        else jnp.zeros((1, 1))
    )
    weights_ndim = weights.ndim

    def host(y0_b, times_b, params_b, weights_b):
        y0_b, params_b = np.asarray(y0_b), np.asarray(params_b)
        # Leading axes beyond the traced shapes are vmap lanes: one ensemble.
        lanes = np.broadcast_shapes(
            y0_b.shape[: y0_b.ndim - y0_ndim],
            params_b.shape[: params_b.ndim - params_ndim],
        )
        n_lanes = int(np.prod(lanes, dtype=np.int64))
        y_all = np.broadcast_to(
            y0_b.reshape(y0_b.shape[: y0_b.ndim - y0_ndim] + (-1, n_vars)),
            lanes + (n, n_vars),
        ).reshape(n_lanes * n, n_vars)
        p_all = np.broadcast_to(
            params_b.reshape(
                params_b.shape[: params_b.ndim - params_ndim] + (-1, n_params)
            ),
            lanes + (n, n_params),
        ).reshape(n_lanes * n, n_params)
        w_all = None
        if have_weights:
            w = np.asarray(weights_b)
            w = w.reshape(w.shape[: w.ndim - weights_ndim] + (-1, n_vars))
            w_all = np.broadcast_to(w, lanes + (n, n_vars)).reshape(n_lanes * n, n_vars)
        hist, stats = solver.solve(
            y_all,
            np.asarray(times_b).reshape(-1, n_save)[0],
            p_all,
            rtol=rtol,
            atol=atol,
            first_step=first_step,
            max_steps=max_steps,
            error_weights=w_all,
            parallel=DEFAULTS["parallel"],
            threads=DEFAULTS["threads"],
        )
        LAST_STATS.clear()
        LAST_STATS.update(stats)
        shape = lanes + (n,)
        return (
            hist.reshape(shape + (n_save, n_vars)),
            stats["accepted_steps"].reshape(shape),
            stats["rejected_steps"].reshape(shape),
            stats["loop_steps"].reshape(shape),
        )

    result_shapes = (
        jax.ShapeDtypeStruct((n, n_save, n_vars), jnp.float64),
        jax.ShapeDtypeStruct((n,), jnp.int32),
        jax.ShapeDtypeStruct((n,), jnp.int32),
        jax.ShapeDtypeStruct((n,), jnp.int32),
    )
    hist, accepted, rejected, loops = jax.pure_callback(
        host, result_shapes, y0, times, params, weights, vmap_method="expand_dims"
    )
    if return_stats:
        return hist, {
            "accepted_steps": accepted,
            "rejected_steps": rejected,
            "loop_steps": loops,
        }
    return hist


def _unsupported(what: str):
    def refuse(*args, **kwargs):
        raise NotImplementedError(f"the CPU prototype has no {what}")

    return refuse


def install() -> None:
    """Make ``from modax.rodas5P import solve`` resolve to :func:`solve`; idempotent."""
    existing = sys.modules.get("modax.rodas5P")
    if existing is not None:
        if getattr(existing, "__cpu_stub__", False):
            return
        raise RuntimeError("the real modax.rodas5P is already imported")
    import modax  # the package itself imports without CUDA

    stub = _pytypes.ModuleType("modax.rodas5P")
    stub.solve = solve
    stub.quantity_hook = _unsupported("save hooks (quantity_hook)")
    stub.SensitivitySpec = _unsupported("sensitivities (SensitivitySpec)")
    stub.__cpu_stub__ = True
    sys.modules["modax.rodas5P"] = stub
    modax.rodas5P = stub
