"""Which input the end-to-end lensing gate of DISCO-EB's tests is missing on.

Run from the DISCO-EB checkout::

    PYTHONPATH=<modax>:<modax>/cpu_prototype:src python <modax>/cpu_prototype/check_lensing.py

``tests/test_lensing.py::test_lensed_spectra_end_to_end_match_class`` asks the
lensing correction, lensed over unlensed, to match CLASS's to 1e-3 in TT for
``l <= 1500``. On the CPU target it comes out at 1.7e-3 near ``l = 1492``.
The correction is built from three things, and this script swaps each for
CLASS's own to see which one carries the error:

* the unlensed spectra at the test's multipoles (the perturbation solve and
  the projection);
* their spline onto every multipole (``cl_on_every_ell``; the test samples
  every 40th multipole above ``l = 200``);
* the lensing potential spectrum (the solve again, and its own spline);
* the arithmetic of the lensing sum itself: the test takes ``lensing_tables``
  at its default, float32, where the convolution test beside it uses float64.

The CLASS reference is then varied too (its lensing accuracy settings), and
the result written to ``results/lensing_gate.json``. The solve is the test's,
with its grids; nothing here needs a GPU except the comparison it leaves open.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import discoeb_cpu

discoeb_cpu.install()

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "tests")
import test_lensing as TL  # noqa: E402
from discoeb.cmb import (  # noqa: E402
    cmb_grids,
    cmb_k_grid,
    compute_cl_power_spectrum_batch,
)
from discoeb.lensing import (  # noqa: E402
    cl_on_every_ell,
    lensed_spectra,
    lensing_tables,
)
from test_perturbations import _perturbed_cosmologies  # noqa: E402

RESULTS = Path(__file__).parent / "results"
LMAX, LCHECK = TL.E2E_LMAX, TL.E2E_LCHECK
CHECK = slice(2, LCHECK + 1)
ELL = np.arange(2, LCHECK + 1)


def test_ells(step=40):
    """The multipoles of the end-to-end test; ``step`` is its spacing above ``l = 200``."""
    return np.unique(
        np.concatenate(
            [
                np.arange(2, 12),
                np.arange(12, 40, 2),
                np.arange(40, 200, 10),
                np.arange(200, LMAX + 1, step),
            ]
        )
    )


def solve(cosmology, ells, ells_pp):
    """The test's solve: unlensed spectra at ``ells`` (muK^2) and ``C_l^pp`` at ``ells_pp``."""
    k_values = cmb_k_grid(cosmology, n=128, mode="ode", k_min=1.0e-5, k_max=0.5)
    grids = cmb_grids(
        cosmology, k_values=k_values, ells=ells, n_save=1000, n_k_fine=1500
    )
    grids = grids._replace(
        k_values=np.sort(np.concatenate([k_values, np.geomspace(0.55, 1.2, 16)]))
    )
    _, cl, _, cl_pp = compute_cl_power_spectrum_batch(
        [cosmology], grids=grids, lensing_potential_ells=ells_pp
    )
    norm = (cosmology["Tcmb"] * 1.0e6) ** 2
    samples = {name: np.asarray(getattr(cl, name)) * norm for name in TL.UNLENSED}
    return samples, np.asarray(cl_pp)


def every_ell(ells, samples, power=1):
    return np.asarray(cl_on_every_ell(ells, samples, LMAX, power=power))


def lens(unlensed, pp, tables):
    out = lensed_spectra(unlensed["tt"], unlensed["te"], unlensed["ee"], pp, tables)
    return {name: np.asarray(getattr(out, name))[0] for name in TL.LENSED}


def correction_error(unlensed, lensed, unlensed_ref, lensed_ref, nodes):
    """The gate's quantity for TT and EE: worst value, where, and the worst at the sampled multipoles."""
    report = {}
    at_nodes = np.isin(ELL, nodes)
    for name in ("tt", "ee"):
        ours = lensed[name][CHECK] / unlensed[name][0][CHECK]
        reference = lensed_ref[name][CHECK] / unlensed_ref[name][CHECK]
        error = np.abs(ours / reference - 1.0)
        report[name] = {
            "worst": float(error.max()),
            "at_l": int(ELL[np.argmax(error)]),
            "worst_at_sampled_l": float(error[at_nodes].max()),
            "worst_above_1000": float(error[ELL > 1000].max()),
            "worst_below_1000": float(error[ELL <= 1000].max()),
        }
    return report


def class_spectra(cosmology, **settings):
    """``TL._class_spectra`` with extra CLASS settings."""
    if not settings:
        return TL._class_spectra(cosmology)
    import types

    import classy

    class Tightened(classy.Class):
        def set(self, *args, **kwargs):
            super().set(*args, **kwargs)
            return super().set(settings)

    real = TL.pytest
    TL.pytest = types.SimpleNamespace(
        importorskip=lambda name: types.SimpleNamespace(Class=Tightened)
    )
    try:
        return TL._class_spectra(cosmology)
    finally:
        TL.pytest = real


def main():
    cosmology = _perturbed_cosmologies(1)[0]
    tables = lensing_tables(LMAX)
    tables64 = lensing_tables(LMAX, dtype=jnp.float64)
    ells = test_ells()
    ells_pp = np.unique(np.round(np.geomspace(2, LMAX, 40)).astype(np.int64))
    unlensed_class, lensed_class = class_spectra(cosmology)
    cut = slice(None, LMAX + 1)

    class_every = {name: unlensed_class[name][None, cut] for name in ("tt", "te", "ee")}
    class_pp_every = unlensed_class["pp"][None, cut]
    class_sampled = {
        name: every_ell(ells, unlensed_class[name][None, ells]) for name in TL.UNLENSED
    }
    class_pp_sampled = every_ell(ells_pp, unlensed_class["pp"][None, ells_pp], power=2)

    # The reference: CLASS's lensed-over-unlensed ratio under tighter settings.
    base_ratio = lensed_class["tt"][CHECK] / unlensed_class["tt"][CHECK]
    reference_shift = {}
    for label, settings in {
        "accurate_lensing": {"accurate_lensing": 1},
        "delta_l_max 1000 (default 500)": {"delta_l_max": 1000},
        "sampling stepsize 0.005": {"perturbations_sampling_stepsize": 0.005},
    }.items():
        unlensed_alt, lensed_alt = class_spectra(cosmology, **settings)
        ratio = lensed_alt["tt"][CHECK] / unlensed_alt["tt"][CHECK]
        reference_shift[label] = float(np.abs(ratio / base_ratio - 1.0).max())

    print("CLASS reference shift (TT correction):", reference_shift, flush=True)

    samples, cl_pp = solve(cosmology, ells, ells_pp)
    ours = {name: every_ell(ells, samples[name]) for name in TL.UNLENSED}
    ours_pp = every_ell(ells_pp, cl_pp, power=2)

    def case(unlensed, pp, tables=tables):
        return correction_error(
            unlensed, lens(unlensed, pp, tables), unlensed_class, lensed_class, ells
        )

    record = {
        "class_reference_shift_tt": reference_shift,
        "gate_tt": 1.0e-3,
        "gate_ee": 5.0e-3,
        "cases": {
            "the test: our spectra, our potential": case(ours, ours_pp),
            "the test with float64 lensing tables": case(ours, ours_pp, tables64),
            "CLASS spectra and potential at every l, float64 lensing tables": case(
                class_every, class_pp_every, tables64
            ),
            "CLASS spectra at every l, CLASS potential at every l": case(
                class_every, class_pp_every
            ),
            "CLASS spectra splined from the test's multipoles, CLASS potential at every l": case(
                class_sampled, class_pp_every
            ),
            "CLASS spectra at every l, CLASS potential splined from the test's multipoles": case(
                class_every, class_pp_sampled
            ),
            "CLASS spectra at every l, our potential": case(class_every, ours_pp),
            "our spectra, CLASS potential at every l": case(ours, class_pp_every),
        },
    }

    # The unlensed spectra themselves at the sampled multipoles, and the potential.
    in_check = ells[(ells >= 2) & (ells <= LCHECK)]
    position = np.searchsorted(ells, in_check)
    record["our unlensed spectra against CLASS at the sampled multipoles"] = {
        name: float(
            np.abs(
                samples[name][0][position] / unlensed_class[name][in_check] - 1.0
            ).max()
        )
        for name in ("tt", "ee")
    }
    record["our potential against CLASS at its sampled multipoles"] = float(
        np.abs(cl_pp[0] / unlensed_class["pp"][ells_pp] - 1.0).max()
    )

    # The test with its multipoles twice as dense above l = 200.
    dense = test_ells(step=20)
    dense_samples, dense_pp = solve(cosmology, dense, ells_pp)
    dense_ours = {name: every_ell(dense, dense_samples[name]) for name in TL.UNLENSED}
    record["cases"]["the test with multipoles every 20 above l = 200"] = (
        correction_error(
            dense_ours,
            lens(dense_ours, every_ell(ells_pp, dense_pp, power=2), tables),
            unlensed_class,
            lensed_class,
            dense,
        )
    )

    import classy

    record["classy"] = classy.__version__
    for label, report in record["cases"].items():
        tt = report["tt"]
        print(
            f"{label}\n    TT worst {tt['worst']:.2e} at l={tt['at_l']}, "
            f"at sampled l {tt['worst_at_sampled_l']:.2e}; "
            f"EE worst {report['ee']['worst']:.2e}"
        )
    for key in (
        "our unlensed spectra against CLASS at the sampled multipoles",
        "our potential against CLASS at its sampled multipoles",
        "class_reference_shift_tt",
    ):
        print(key, record[key])
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "lensing_gate.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
