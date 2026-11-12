"""P(k) of DISCO-EB v2 on the CPU prototype: accuracy, step counts and timings.

Runs the unedited ``discoeb.perturbations.solve_matter_power_spectrum_batch``
(thermal history included) for the flat-LCDM layout (50 variables) and the
massive-neutrino layout (98 variables), with the CUDA and modax stand-ins of
:mod:`discoeb_cpu` underneath, and writes ``results/pk_<layout>.json``.

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

import modax_stub
import numba
from discoeb import perturbations as P

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
    cosmology = COSMOLOGIES[name]
    layout = P._layout_for(P._as_cosmology(cosmology))
    print(f"== {name}: nvar = {layout.nvar}, {len(K)} modes", flush=True)
    out: dict = {"layout": name, "nvar": layout.nvar, "k": K.tolist(), "runs": {}}

    def solve(**kwargs):
        return P.solve_matter_power_spectrum_batch(K, (cosmology,), **kwargs)[0]

    n_cores = numba.config.NUMBA_NUM_THREADS
    modax_stub.DEFAULTS.update(parallel=True, threads=None, lu_precision=None)
    t_thermo, _ = timed(lambda: P.build_thermo_tables(cosmology))
    out["thermo_tables_first_call_s"] = t_thermo
    print(f"   thermal tables (compile + solve): {t_thermo:.1f} s", flush=True)

    # The configuration DISCO-EB itself launches: fp32 LU, rtol = atol = 1e-4.
    t_first, pk = timed(solve)
    stats = {k: v.copy() for k, v in modax_stub.LAST_STATS.items()}
    order = np.argsort(K, kind="stable")
    accepted = np.empty(len(K), int)
    rejected = np.empty(len(K), int)
    accepted[order] = stats["accepted_steps"]
    rejected[order] = stats["rejected_steps"]
    out["first_call_s"] = t_first
    out["pk"] = pk.tolist()
    out["accepted_steps"] = accepted.tolist()
    out["rejected_steps"] = rejected.tolist()
    print(
        f"   first call (compiles the kernel): {t_first:.1f} s; steps per mode "
        f"min {accepted.min()} median {int(np.median(accepted))} max {accepted.max()}, "
        f"rejected {rejected.sum()} of {accepted.sum() + rejected.sum()}",
        flush=True,
    )

    for threads in sorted({1, 4, 8, n_cores}):
        if threads > n_cores:
            continue
        modax_stub.DEFAULTS.update(parallel=threads > 1, threads=threads)
        best = min(timed(solve)[0] for _ in range(3))
        out["runs"][f"fp32_threads{threads}"] = best
        print(f"   fp32 LU, {threads:2d} thread(s): {best * 1e3:8.1f} ms", flush=True)
    modax_stub.DEFAULTS.update(parallel=True, threads=None)

    # fp64 LU and tighter tolerances, as accuracy references.
    modax_stub.DEFAULTS.update(lu_precision="fp64")
    timed(solve)
    t64 = min(timed(solve)[0] for _ in range(3))
    pk64 = solve()
    steps64 = int(modax_stub.LAST_STATS["accepted_steps"].sum())
    out["runs"][f"fp64_threads{n_cores}"] = t64
    out["pk_fp64"] = pk64.tolist()
    print(
        f"   fp64 LU, {n_cores:2d} thread(s): {t64 * 1e3:8.1f} ms; "
        f"P(k) fp32 vs fp64 LU: {np.max(np.abs(pk / pk64 - 1)):.2e}; "
        f"accepted steps {steps64} against {accepted.sum()} (fp32)",
        flush=True,
    )
    sweep = {}
    reference = solve(rtol=1e-7, atol=1e-7)
    for tol in (1e-3, 1e-4, 1e-5, 1e-6):
        t, pk_tol = timed(lambda tol=tol: solve(rtol=tol, atol=tol))
        sweep[f"{tol:.0e}"] = {
            "max_rel_vs_1e-7": float(np.max(np.abs(pk_tol / reference - 1))),
            "seconds": t,
            "accepted": int(modax_stub.LAST_STATS["accepted_steps"].sum()),
        }
        print(
            f"   rtol = atol = {tol:.0e}: P(k) within {sweep[f'{tol:.0e}']['max_rel_vs_1e-7']:.2e} "
            f"of the 1e-7 solve, {sweep[f'{tol:.0e}']['accepted']} steps, {t * 1e3:.0f} ms",
            flush=True,
        )
    out["tolerance_sweep"] = sweep
    modax_stub.DEFAULTS.update(lu_precision=None)

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
    out["threads_available"] = n_cores
    out["cpu"] = os.uname().machine
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"pk_{name}.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    for which in sys.argv[1:] or list(COSMOLOGIES):
        run(which)
