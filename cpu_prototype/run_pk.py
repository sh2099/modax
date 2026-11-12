"""P(k) of DISCO-EB v2 on modax's CPU target: accuracy, step counts and timings.

Solves the flat-LCDM layout (50 variables) and the massive-neutrino layout
(98 variables) with modax's CPU target, from the unedited DISCO-EB checkout's
own thermal tables, packed inputs and solver settings, and writes
``results/pk_<layout>.json``. The launch is made here rather than through
``solve_matter_power_spectrum_batch`` only so that the LU (sparse or dense,
float32 or float64) and the thread count can be varied and the step counters
read; ``run_discoeb_tests.py`` exercises DISCO-EB's own entry points.

    PYTHONPATH=..:.:<DISCO-EB>/src .venv/bin/python run_pk.py [flat] [mnu]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import discoeb_cpu
import numpy as np

discoeb_cpu.install()

import numba  # noqa: E402
from discoeb import perturbations as P  # noqa: E402

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
"""``tests/test_perturbations.py``'s reference (Planck 2018 flat LCDM, massless neutrinos)."""

COSMOLOGIES = {
    "flat": REFERENCE_COSMOLOGY,
    "mnu": {**REFERENCE_COSMOLOGY, "Neff": 2.046, "Nmnu": 1.0, "mnu": 0.3},
}
K = np.geomspace(1.0e-4, 1.0, 128)
RESULTS = Path(__file__).parent / "results"


def class_pk(param, k_values):
    """CLASS's linear P(k) of cold dark matter plus baryons at z = 0, or ``None`` without classy."""
    try:
        from classy import Class
    except ImportError:
        return None
    h = param["H0"] / 100.0
    settings = {
        "h": h,
        "omega_b": param["Omegab"] * h**2,
        "omega_cdm": (param["Omegam"] - param["Omegab"]) * h**2,
        "T_cmb": param["Tcmb"],
        "YHe": param["YHe"],
        "N_ur": param["Neff"],
        "N_ncdm": 0,
        "Omega_k": 0.0,
        "A_s": param["A_s"],
        "n_s": param["n_s"],
        "k_pivot": 0.05,
        "output": "mPk",
        "P_k_max_1/Mpc": float(k_values[-1]) * 1.1,
        "z_max_pk": 0.0,
    }
    massive = bool(param.get("Nmnu", 0))
    if massive:
        settings.update(
            N_ncdm=1,
            deg_ncdm=param["Nmnu"],
            m_ncdm=param["mnu"],
            T_ncdm=(4.0 / 11.0) ** (1.0 / 3.0),
        )
    cosmo = Class()
    cosmo.set(settings)
    try:
        cosmo.compute()
        # DISCO-EB's delta_m weights CDM and baryons only.
        pk = cosmo.pk_cb_lin if massive else cosmo.pk_lin
        return np.array([pk(float(k), 0.0) for k in k_values])
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


def timed(fn):
    start = time.perf_counter()
    out = fn()
    return time.perf_counter() - start, out


def run(name: str) -> dict:
    from modax.rodas5P import solve

    cosmology = COSMOLOGIES[name]
    n_cores = numba.config.NUMBA_NUM_THREADS
    t_thermo, prepared = timed(lambda: P._prepare_solve((cosmology,)))
    cosmo, layout = prepared.cosmologies[0], prepared.layout
    params, y0, t_span, order, k_sorted = P._pack_batch(prepared, K)
    prepared.bind()
    print(f"== {name}: nvar = {layout.nvar}, {len(K)} modes", flush=True)
    print(f"   thermal tables (compile + solve): {t_thermo:.1f} s", flush=True)
    out: dict = {
        "layout": name,
        "nvar": layout.nvar,
        "k": K.tolist(),
        "thermo_tables_first_call_s": t_thermo,
        "threads_available": n_cores,
        "cpu": os.uname().machine,
        "runs": {},
    }

    def launch(*, sparse=True, lu="fp32", tol=P.PERTURB_RTOL, threads=None):
        """DISCO-EB's launch (``perturbations._launch``) with the LU and threads chosen here."""
        numba.set_num_threads(threads or n_cores)
        try:
            hist, stats = solve(
                prepared.ode_fn, y0, t_span, params,
                sparsity=prepared.sparsity if sparse else None,
                rtol=tol, atol=tol, first_step=P.PERTURB_FIRST_STEP,
                max_steps=P.PERTURB_MAX_STEPS, pcoeff=0.3, icoeff=0.4,
                tf_index=P.IX_TAU_END, lu_precision=lu, return_stats=True,
            )  # fmt: skip
        finally:
            numba.set_num_threads(n_cores)
        final = np.asarray(hist)[:, -1, :]
        densities = P.density_coefficients(cosmo)
        delta_m = P.total_matter_density_contrast(
            densities[2], final[:, P.IX_CLXC], densities[3], final[:, P.IX_CLXB]
        )
        pk_sorted = P.matter_power_spectrum(
            k_sorted, delta_m, cosmo.A_s, cosmo.n_s, cosmo.k_pivot
        )
        pk = np.empty_like(pk_sorted)
        pk[order] = pk_sorted
        return pk, {key: np.asarray(value) for key, value in stats.items()}

    # The configuration DISCO-EB itself launches: sparse fp32 LU, rtol = atol = 1e-4.
    t_first, (pk, stats) = timed(launch)
    accepted, rejected = stats["accepted_steps"], stats["rejected_steps"]
    out["first_call_s"] = t_first
    out["pk"] = pk.tolist()
    out["accepted_steps"] = accepted.tolist()  # in ascending k
    out["rejected_steps"] = rejected.tolist()
    print(
        f"   first call (compiles the kernel): {t_first:.1f} s; steps per mode "
        f"min {accepted.min()} median {int(np.median(accepted))} max {accepted.max()}, "
        f"rejected {rejected.sum()} of {accepted.sum() + rejected.sum()}",
        flush=True,
    )

    for label, kwargs in (
        ("sparse fp32", {}),
        ("sparse fp64", {"lu": "fp64"}),
        ("dense  fp32", {"sparse": False}),
        ("dense  fp64", {"sparse": False, "lu": "fp64"}),
    ):
        launch(**kwargs)  # compile
        row = {}
        for threads in sorted({1, n_cores}):
            row[f"threads{threads}"] = min(
                timed(
                    lambda kwargs=kwargs, threads=threads: launch(
                        threads=threads, **kwargs
                    )
                )[0]
                for _ in range(3)
            )
        pk_variant, stats_variant = launch(**kwargs)
        row["accepted"] = int(stats_variant["accepted_steps"].sum())
        row["rejected"] = int(stats_variant["rejected_steps"].sum())
        row["pk_max_rel_vs_default"] = float(np.max(np.abs(pk_variant / pk - 1.0)))
        out["runs"][label.replace("  ", " ")] = row
        print(
            f"   {label} LU: {row['threads1'] * 1e3:8.1f} ms on 1 thread, "
            f"{row[f'threads{n_cores}'] * 1e3:7.1f} ms on {n_cores}; "
            f"{row['accepted']} accepted, {row['rejected']} rejected; "
            f"P(k) within {row['pk_max_rel_vs_default']:.1e} of the default",
            flush=True,
        )

    sweep = {}
    reference, _ = launch(lu="fp64", tol=1e-7)
    for tol in (1e-3, 1e-4, 1e-5, 1e-6):
        t, (pk_tol, stats_tol) = timed(lambda tol=tol: launch(lu="fp64", tol=tol))
        sweep[f"{tol:.0e}"] = {
            "max_rel_vs_1e-7": float(np.max(np.abs(pk_tol / reference - 1))),
            "seconds": t,
            "accepted": int(stats_tol["accepted_steps"].sum()),
        }
        print(
            f"   rtol = atol = {tol:.0e}: P(k) within {sweep[f'{tol:.0e}']['max_rel_vs_1e-7']:.2e} "
            f"of the 1e-7 solve, {sweep[f'{tol:.0e}']['accepted']} steps, {t * 1e3:.0f} ms",
            flush=True,
        )
    out["tolerance_sweep"] = sweep

    pk_class = class_pk(cosmology, K)
    if pk_class is not None:
        rel = np.abs(pk / pk_class - 1.0)
        out["class_max_rel"] = float(rel.max())
        out["class_max_rel_k"] = float(K[rel.argmax()])
        out["pk_class"] = pk_class.tolist()
        print(
            f"   against CLASS: max relative error {rel.max():.3e} at k = {K[rel.argmax()]:.3g} "
            f"(DISCO-EB's gate is {5e-3:.0e})",
            flush=True,
        )
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"pk_{name}.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    for which in sys.argv[1:] or list(COSMOLOGIES):
        run(which)
