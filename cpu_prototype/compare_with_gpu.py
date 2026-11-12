"""The CPU prototype against modax on CUDA, on identical inputs.

Takes the ``.npz`` written by ``dump_gpu_reference.py`` on the NVIDIA machine::

    PYTHONPATH=..:.:<DISCO-EB>/src .venv/bin/python compare_with_gpu.py gpu_reference.npz

Three comparisons, in order of how much they isolate:

1. **Solver only**: Robertson and the dense output of the stiff forced
   problem, which involve no cosmology.
2. **Perturbation solve on the GPU's own inputs**: the right-hand side is
   built here from the GPU's thermal tables, and the GPU's packed ``params``,
   ``y0`` and ``t_span`` are integrated, so any difference is the solver's
   (finite-difference against Enzyme Jacobian, dense against sparse LU).
3. **Thermal history**: the CPU's RECFAST tables against the GPU's.
"""

from __future__ import annotations

import math
import sys

import discoeb_cpu
import numpy as np

discoeb_cpu.install()

import rodas5p_cpu
from discoeb import perturbations as P
from run_pk import COSMOLOGIES


def robertson(y, t, p):
    return (
        -p[0] * y[0] + p[1] * y[1] * y[2],
        p[0] * y[0] - p[1] * y[1] * y[2] - p[2] * y[1] ** 2,
        p[2] * y[1] ** 2,
    )


def prothero_robinson(y, t, p):
    return (-p[0] * (y[0] - math.sin(t)) + math.cos(t),)


def relative(ours, theirs, floor):
    return float(np.max(np.abs(ours - theirs) / (np.abs(theirs) + floor)))


def main(path: str) -> None:
    gpu = np.load(path)
    print("GPU run:", "; ".join(str(s) for s in gpu["info"]))

    print("\n1. Solver only")
    for lu in ("fp32", "fp64"):
        solver = rodas5p_cpu.make_solver(robertson, 3, 3, lu_precision=lu)
        hist, stats = solver.solve(
            np.array([1.0, 0.0, 0.0]), gpu["robertson_times"], np.array([0.04, 1e4, 3e7]),
            rtol=1e-8, atol=1e-10,
        )  # fmt: skip
        print(
            f"   Robertson, {lu} LU: max difference {relative(hist, gpu[f'robertson_{lu}_hist'], 1e-10):.2e}; "
            f"accepted steps CPU {int(stats['accepted_steps'][0])} / GPU {int(gpu[f'robertson_{lu}_accepted'][0])}, "
            f"rejected {int(stats['rejected_steps'][0])} / {int(gpu[f'robertson_{lu}_rejected'][0])}"
        )
    solver = rodas5p_cpu.make_solver(prothero_robinson, 1, 1)
    dense = gpu["pr_times"]
    for tol in (1e-4, 1e-6, 1e-8):
        hist, stats = solver.solve(
            np.zeros(1), dense, np.array([1.0e4]), rtol=tol, atol=tol
        )
        theirs = gpu[f"pr_hist_{tol:.0e}"]
        print(
            f"   stiff forced problem, rtol={tol:.0e}: dense-output error CPU "
            f"{np.max(np.abs(hist[0, :, 0] - np.sin(dense))):.2e} / GPU "
            f"{np.max(np.abs(theirs[0, :, 0] - np.sin(dense))):.2e}; steps "
            f"{int(stats['accepted_steps'][0])} / {int(gpu[f'pr_accepted_{tol:.0e}'][0])}"
        )

    for name, cosmology in COSMOLOGIES.items():
        if f"{name}_params" not in gpu:
            continue
        print(
            f"\n2. Perturbation solve on the GPU's inputs: {name} (nvar {int(gpu[f'{name}_nvar'])})"
        )
        cosmo = P._as_cosmology(cosmology)
        layout = P._layout_for(cosmo)
        tables = (
            gpu[f"{name}_tau_grid"], gpu[f"{name}_values"], gpu[f"{name}_seconds"],
            float(gpu[f"{name}_tau0"]),
        )  # fmt: skip
        tau_min, inv_dtau, values, rescaled, _ = P._stack_thermo_tables([tables])
        rhs = P.build_numba_rhs(layout, tau_min, inv_dtau, values, rescaled)
        params, y0, t_span = (
            gpu[f"{name}_params"],
            gpu[f"{name}_y0"],
            gpu[f"{name}_t_span"],
        )
        for lu in ("fp32", "fp64"):
            solver = rodas5p_cpu.make_solver(
                rhs, layout.nvar, P.N_PARAM, sparsity=gpu[f"{name}_sparsity"],
                lu_precision=lu, tf_index=P.IX_TAU_END, pcoeff=0.3, icoeff=0.4,
            )  # fmt: skip
            hist, stats = solver.solve(
                y0, t_span, params, rtol=P.PERTURB_RTOL, atol=P.PERTURB_ATOL,
                first_step=P.PERTURB_FIRST_STEP, max_steps=P.PERTURB_MAX_STEPS,
            )  # fmt: skip
            final, theirs = hist[:, -1, :], gpu[f"{name}_{lu}_final"]
            ix_c, ix_b = layout.ix_clxc, layout.ix_clxb
            matter = max(
                relative(final[:, ix_c], theirs[:, ix_c], 0.0),
                relative(final[:, ix_b], theirs[:, ix_b], 0.0),
            )
            a_cpu, a_gpu = stats["accepted_steps"], gpu[f"{name}_{lu}_accepted_steps"]
            r_cpu, r_gpu = stats["rejected_steps"], gpu[f"{name}_{lu}_rejected_steps"]
            print(
                f"   {lu} LU: matter contrasts differ by at most {matter:.2e} "
                f"(solver tolerance {P.PERTURB_RTOL:.0e}); whole state, in units of each "
                f"mode's largest component, {float(np.max(np.abs(final - theirs) / np.abs(theirs).max(axis=1, keepdims=True))):.2e}"
            )
            print(
                f"            accepted steps CPU {int(a_cpu.sum())} / GPU {int(a_gpu.sum())} "
                f"(per mode max {int(a_cpu.max())} / {int(a_gpu.max())}), "
                f"rejected {int(r_cpu.sum())} / {int(r_gpu.sum())}; "
                f"GPU warm launch {float(gpu[f'{name}_{lu}_seconds'][1]) * 1e3:.1f} ms"
            )

        print(f"3. Thermal history: {name}")
        ours = P.build_thermo_tables(cosmology)
        labels = ("a", "opacity", "cs2", "rhonu", "pnu")
        print(f"   tau0 CPU {ours[3]:.6f} / GPU {float(gpu[f'{name}_tau0']):.6f}")
        for channel, label in enumerate(labels):
            theirs = gpu[f"{name}_values"][channel]
            scale = np.abs(theirs).max() * 1e-12
            print(
                f"   {label:8s} max relative difference "
                f"{relative(ours[1][channel], theirs, scale):.2e}"
            )
        history = P.thermal_history_batch([cosmology])
        z = 1.0 / history["a"] - 1.0
        inside = (z >= 10.0) & (z <= 5000.0)
        print(
            f"   x_e(z) on the RECFAST grid, 10 < z < 5000: "
            f"{relative(history['xe'][0][inside], gpu[f'{name}_thermal_xe'][0][inside], 0.0):.2e}"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gpu_reference.npz")
