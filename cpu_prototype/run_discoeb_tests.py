"""Run DISCO-EB's own test suite against the CPU prototype, without editing it.

Installs the stand-ins of :mod:`discoeb_cpu`, then hands the given arguments
to pytest from inside the DISCO-EB checkout (its conftest reads resources by
relative path)::

    PYTHONPATH=..:.:<DISCO-EB>/src .venv/bin/python run_discoeb_tests.py <DISCO-EB> \
        tests/test_thermo.py tests/test_perturbations.py -k "matches_class and N1"

``--benchmark-disable`` is always passed: the timing rounds warm up on a
second batch of the same shape, which on the CPU would need the frozen tables
rebuilt (see :mod:`cuda_shim`). For the same reason the prepared solves are
dropped before every test, so each builds its functions on its own tables.
Tests that need what the prototype lacks (save hooks, sensitivities) fail
with ``NotImplementedError``; the CUDA-gated ones skip themselves.
"""

from __future__ import annotations

import os
import sys

import discoeb_cpu

discoeb_cpu.install()

import pytest


class FreshSolves:
    @pytest.fixture(autouse=True)
    def _fresh_prepared_solves(self):
        from discoeb import perturbations

        perturbations.clear_prepared_solve_cache()
        yield


if __name__ == "__main__":
    checkout, *arguments = sys.argv[1:]
    os.chdir(checkout)
    sys.path.insert(0, os.path.join(checkout, "tests"))
    sys.exit(
        pytest.main(
            ["--benchmark-disable", "-p", "no:cacheprovider", *arguments],
            plugins=[FreshSolves()],
        )
    )
