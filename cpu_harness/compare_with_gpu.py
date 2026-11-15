"""Compare the CPU target with a GPU record of DISCO-EB on identical inputs.

Run from the DISCO-EB checkout, with the record made on a CUDA machine
(``gpu_reference_2.0.1.npz``: DISCO-EB 2.0.1 on modax 0.3.0)::

    PYTHONPATH=<modax>:<modax>/cpu_harness:src python <modax>/cpu_harness/compare_with_gpu.py gpu_reference.npz

Only what DISCO-EB can compute on the installed modax is compared: the
matter power spectrum and the perturbation history. The two solves share the
equations and the method and nothing below that (the derivatives, the step
sequence and, here, the modax version), so agreement is expected to the
solve's tolerance, not to rounding. Writes ``results/gpu_comparison.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import discoeb_cpu

discoeb_cpu.install()

import numpy as np  # noqa: E402

sys.path.insert(0, "tests")
from discoeb import perturbations as P  # noqa: E402
from test_perturbations import _perturbed_cosmologies  # noqa: E402

from modax import _target  # noqa: E402

RESULTS = Path(__file__).parent / "results"


def summary(ours, theirs):
    relative = np.abs(ours / theirs - 1.0)
    worst = np.unravel_index(np.argmax(relative), relative.shape)
    return {
        "max": float(relative.max()),
        "median": float(np.median(relative)),
        "at": [int(i) for i in worst],
    }


def main():
    reference = np.load(sys.argv[1], allow_pickle=False)
    meta = json.loads(str(reference["meta"]))
    k = reference["k"]
    record = {
        "reference": {key: meta.get(key) for key in ("commit", "gpu", "versions")}
    }

    counters = {}
    inner = _target.run_ensemble

    def run(driver, *args):
        inner(driver, *args)
        counters["accepted"], counters["rejected"], counters["loop"] = (
            np.array(args[10]),
            np.array(args[11]),
            np.array(args[12]),
        )

    _target.run_ensemble = run

    for label, n in (("c1", 1), ("c128", 128)):
        cosmologies = _perturbed_cosmologies(n)
        P.solve_matter_power_spectrum_batch(k, cosmologies)
        start = time.perf_counter()
        ours = np.asarray(P.solve_matter_power_spectrum_batch(k, cosmologies))
        seconds = time.perf_counter() - start
        entry = summary(ours, reference[f"pk_{label}"])
        entry["cpu_seconds"] = round(seconds, 4)
        for name in ("accepted", "rejected", "loop"):
            theirs = reference[f"stats_{label}_{name}_steps"]
            entry[f"{name}_steps_mean"] = {
                "cpu": float(counters[name].mean()),
                "gpu": float(theirs.mean()),
            }
        record[f"P(k), {n} cosmologies x {len(k)} modes"] = entry
        print(f"P(k) {label}: {entry}")

    hist, k_sorted, _ = P.solve_perturbation_history_batch(
        reference["hist_k"], _perturbed_cosmologies(1)
    )
    ours = np.asarray(hist)
    theirs = reference["hist_c1_default"]
    assert ours.shape == theirs.shape, (ours.shape, theirs.shape)
    np.testing.assert_array_equal(k_sorted, reference["hist_c1_default_k_sorted"])
    # In units of the solve's own tolerance (rtol = atol = PERTURB_RTOL), which
    # is what two solves on different step sequences can be asked to share.
    final_ours, final_theirs = ours[:, 0, -1, :], theirs[:, 0, -1, :]
    tolerance = P.PERTURB_ATOL + P.PERTURB_RTOL * np.abs(final_theirs)
    difference = np.abs(final_ours - final_theirs) / tolerance
    names = [str(n) for n in reference["hist_state_names"]]
    worst = np.argsort(difference.max(axis=0))[::-1][:5]
    record["final state of 16 modes, difference in tolerances"] = {
        "max": float(difference.max()),
        "99th percentile": float(np.percentile(difference, 99)),
        "median": float(np.median(difference)),
        "worst_states": {names[j]: float(difference[:, j].max()) for j in worst},
    }
    print(record["final state of 16 modes, difference in tolerances"])

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "gpu_comparison.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
