"""Record what modax on CUDA produces, for comparison with the CPU prototype.

Run on the NVIDIA machine, in the stock DISCO-EB v2 environment (``uv sync``
in the DISCO-EB checkout, then from that checkout)::

    uv run python /path/to/modax/cpu_prototype/dump_gpu_reference.py gpu_reference.npz

Needs nothing from ``cpu_prototype`` but this file. For the flat-LCDM
(50-variable) and massive-neutrino (98-variable) layouts it saves the thermal
tables, the recombination history, the packed solver inputs, the final states
and step counters of the perturbation solve at both LU precisions, and P(k);
and two solver-only checks that involve no cosmology (Robertson, and the
dense output of a stiff forced problem). Copy the ``.npz`` back and run
``compare_with_gpu.py`` on it.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

REFERENCE_COSMOLOGY = {
    "Omegam": (0.02238280 + 0.1201075) / 0.6732117**2,
    "Omegab": 0.02238280 / 0.6732117**2,
    "w_DE_0": -1.0,
    "w_DE_a": 0.0,
    "cs2_DE": 1.0,
    "Omegak": 0.0,
    "A_s": 2.100549e-9,
    "n_s": 0.9660499,
    "H0": 67.32117,
    "Tcmb": 2.7255,
    "YHe": 0.2454006,
    "Neff": 3.046,
    "Nmnu": 0.0,
    "mnu": 0.0,
    "k_p": 0.05,
}
COSMOLOGIES = {
    "flat": REFERENCE_COSMOLOGY,
    "mnu": {**REFERENCE_COSMOLOGY, "Neff": 2.046, "Nmnu": 1.0, "mnu": 0.3},
}
K = np.geomspace(1.0e-4, 1.0, 128)


def robertson(y, t, p):
    return (
        -p[0] * y[0] + p[1] * y[1] * y[2],
        p[0] * y[0] - p[1] * y[1] * y[2] - p[2] * y[1] ** 2,
        p[2] * y[1] ** 2,
    )


def prothero_robinson(y, t, p):
    return (-p[0] * (y[0] - math.sin(t)) + math.cos(t),)


def timed(fn):
    start = time.perf_counter()
    out = jax.block_until_ready(fn())
    return time.perf_counter() - start, out


def main(path: str) -> None:
    from discoeb import perturbations as P

    from modax.rodas5P import solve

    out: dict = {}
    info = [
        f"platform={platform.platform()}",
        f"python={sys.version.split()[0]}",
        f"jax={jax.__version__}",
    ]
    try:
        from importlib.metadata import version

        import modax

        info.append(f"modax={version('modax-solvers')}")
        from numba_cuda_mlir import cuda

        info.append(f"gpu={cuda.get_current_device().name.decode()}")
        del modax
    except Exception as error:  # noqa: BLE001
        info.append(f"gpu-info-error={error!r}")
    print("\n".join(info))
    out["info"] = np.array(info)
    out["k"] = K

    # --- solver-only checks -------------------------------------------------------
    times = np.array([0.0, 1e-6, 1e-2, 1e2, 1e5])
    for lu in ("fp32", "fp64"):
        hist, stats = solve(
            robertson, np.array([1.0, 0.0, 0.0]), times, np.array([0.04, 1e4, 3e7]),
            rtol=1e-8, atol=1e-10, lu_precision=lu, return_stats=True,
        )  # fmt: skip
        out[f"robertson_{lu}_hist"] = np.asarray(hist)
        out[f"robertson_{lu}_accepted"] = np.asarray(stats["accepted_steps"])
        out[f"robertson_{lu}_rejected"] = np.asarray(stats["rejected_steps"])
    out["robertson_times"] = times
    dense = np.linspace(0.0, 3.0, 301)
    out["pr_times"] = dense
    for tol in (1e-4, 1e-6, 1e-8):
        hist, stats = solve(
            prothero_robinson, np.zeros(1), dense, np.array([1.0e4]),
            rtol=tol, atol=tol, lu_precision="fp64", return_stats=True,
        )  # fmt: skip
        out[f"pr_hist_{tol:.0e}"] = np.asarray(hist)
        out[f"pr_accepted_{tol:.0e}"] = np.asarray(stats["accepted_steps"])
        print(
            f"stiff forced problem, rtol={tol:.0e}: max dense-output error "
            f"{np.max(np.abs(np.asarray(hist)[0, :, 0] - np.sin(dense))):.2e}, "
            f"{int(np.asarray(stats['accepted_steps'])[0])} steps"
        )

    # --- DISCO-EB ---------------------------------------------------------------------
    for name, cosmology in COSMOLOGIES.items():
        print(f"== {name}")
        history = P.thermal_history_batch([cosmology])
        for key, value in history.items():
            out[f"{name}_thermal_{key}"] = np.asarray(value)
        prepared = P._prepare_solve((cosmology,))
        tau_grid, values, seconds, tau0 = prepared.tables[0]
        out[f"{name}_tau_grid"] = tau_grid
        out[f"{name}_values"] = values
        out[f"{name}_seconds"] = seconds
        out[f"{name}_tau0"] = np.float64(tau0)
        out[f"{name}_nvar"] = np.int64(prepared.layout.nvar)
        params, y0, t_span, order, _ = P._pack_batch(prepared, K)
        out[f"{name}_params"] = params
        out[f"{name}_y0"] = y0
        out[f"{name}_t_span"] = t_span
        out[f"{name}_order"] = order
        out[f"{name}_sparsity"] = np.asarray(prepared.sparsity)

        prepared.bind()
        for lu in ("fp32", "fp64"):

            def launch(lu=lu, prepared=prepared, y0=y0, t_span=t_span, params=params):
                return solve(
                    prepared.ode_fn, y0, t_span, params,
                    sparsity=prepared.sparsity, rtol=P.PERTURB_RTOL, atol=P.PERTURB_ATOL,
                    first_step=P.PERTURB_FIRST_STEP, max_steps=P.PERTURB_MAX_STEPS,
                    pcoeff=0.3, icoeff=0.4, trajectories_per_block=P.TRAJECTORIES_PER_BLOCK,
                    tf_index=P.IX_TAU_END, max_registers=P.CUDA_MAX_REGISTERS,
                    lu_precision=lu, return_stats=True,
                )  # fmt: skip

            t_first, _ = timed(launch)
            t_warm, (hist, stats) = timed(launch)
            out[f"{name}_{lu}_final"] = np.asarray(hist)[:, -1, :]
            for key, value in stats.items():
                out[f"{name}_{lu}_{key}"] = np.asarray(value)
            out[f"{name}_{lu}_seconds"] = np.array([t_first, t_warm])
            accepted = np.asarray(stats["accepted_steps"])
            print(
                f"   {lu} LU: first launch {t_first:.1f} s, warm {t_warm * 1e3:.1f} ms; "
                f"steps per mode min {accepted.min()} median {int(np.median(accepted))} max {accepted.max()}"
            )
        t_pk, pk = timed(
            lambda cosmology=cosmology: P.solve_matter_power_spectrum_batch(
                K, (cosmology,)
            )
        )
        out[f"{name}_pk"] = np.asarray(pk)[0]
        out[f"{name}_pk_seconds"] = np.float64(t_pk)
        print(f"   solve_matter_power_spectrum_batch (warm): {t_pk * 1e3:.1f} ms")

    np.savez_compressed(path, **out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gpu_reference.npz")
