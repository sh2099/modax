"""Where the time of a DISCO-EB CMB call goes on the CPU target.

Run from the DISCO-EB checkout (its tests are imported for their cosmologies)::

    PYTHONPATH=<modax>:<modax>/cpu_prototype:src python <modax>/cpu_prototype/profile_cmb.py [n_cosmologies] [thermal_chunk]

Times the stages of ``compute_cl_power_spectrum_batch`` for the first call in
the process, a batch it has not seen and the same batch again, and writes
``results/cmb_profile_n<N>.json`` (``..._chunk<C>.json`` with a thermal chunk,
see :mod:`discoeb_cpu`). The stages are DISCO-EB's own functions and
modax's launch, wrapped with timers; compile time is what numba and JAX report
of themselves. Nested stages are listed under the stage that calls them, so
the numbers of one level add up to at most their parent.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import discoeb_cpu

THERMAL_CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else None
discoeb_cpu.install(thermal_chunk=THERMAL_CHUNK)

import jax  # noqa: E402
import numpy as np  # noqa: E402
from numba.core import event as numba_event  # noqa: E402

sys.path.insert(0, "tests")
import test_perturbations as TP  # noqa: E402
from discoeb import cmb  # noqa: E402
from discoeb import perturbations as P  # noqa: E402

from modax import _target  # noqa: E402

RESULTS = Path(__file__).parent / "results"

seconds: dict[str, float] = defaultdict(float)
calls: dict[str, int] = defaultdict(int)


def timed(module, name, label):
    """Replace ``module.name`` with a version that adds its wall time to ``seconds[label]``."""
    inner = getattr(module, name)

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return inner(*args, **kwargs)
        finally:
            seconds[label] += time.perf_counter() - start
            calls[label] += 1

    setattr(module, name, wrapper)


timed(P, "_prepare_solve", "prepare")
timed(P, "_build_thermo_tables_batch", "prepare/thermal tables")
timed(cmb, "_source_tables", "source tables")
timed(cmb, "_solve_with_hook", "solve")
timed(_target, "run_ensemble", "solve/kernels")
timed(cmb, "projection_operator", "projection operator")
timed(cmb, "theta_ell_batch", "projection")
timed(cmb, "spectra_from_transfer", "spectra")


class _CompileTimer(numba_event.Listener):
    """Wall time numba spends holding its compiler lock (outermost entries only)."""

    def __init__(self):
        self.depth = 0
        self.start = 0.0

    def on_start(self, event):
        if self.depth == 0:
            self.start = time.perf_counter()
        self.depth += 1

    def on_end(self, event):
        self.depth -= 1
        if self.depth == 0:
            seconds["compile: numba"] += time.perf_counter() - self.start


numba_event.register("numba:compiler_lock", _CompileTimer())


def _jax_duration(name, duration, **_):
    if "compile" in name or "trace" in name:
        seconds["compile: jax " + name.rsplit("/", 1)[-1]] += duration


jax.monitoring.register_event_duration_secs_listener(_jax_duration)


def measure(label, cosmologies, grids):
    seconds.clear()
    calls.clear()
    start = time.perf_counter()
    out = cmb.compute_cl_power_spectrum_batch(cosmologies, grids=grids)
    np.asarray(out[1].tt)
    total = time.perf_counter() - start
    stages = {name: round(value, 4) for name, value in sorted(seconds.items())}
    print(f"{label}: {total:.3f} s")
    for name, value in stages.items():
        print(f"    {name:<44s} {value:9.3f} s  ({calls.get(name, '-')} calls)")
    return {"total_s": round(total, 4), "stages_s": stages}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cosmologies = TP._perturbed_cosmologies(3 * n)
    start = time.perf_counter()
    grids = cmb.cmb_grids(cosmologies[0])
    grid_time = time.perf_counter() - start
    record = {
        "n_cosmologies": n,
        "thermal_chunk": THERMAL_CHUNK,
        "threads": int(__import__("numba").get_num_threads()),
        "threading_layer": None,
        "grids_s": round(grid_time, 4),
        "first": measure("first call", cosmologies[:n], grids),
        "new": measure("new batch", cosmologies[n : 2 * n], grids),
        "new_2": measure("another new batch", cosmologies[2 * n :], grids),
        "repeat": measure("same batch again", cosmologies[2 * n :], grids),
    }
    record["threading_layer"] = __import__("numba").threading_layer()
    RESULTS.mkdir(exist_ok=True)
    suffix = "" if THERMAL_CHUNK is None else f"_chunk{THERMAL_CHUNK}"
    path = RESULTS / f"cmb_profile_n{n}{suffix}.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
