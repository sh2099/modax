"""Run DISCO-EB's own test suite on modax's CPU target, without editing it.

Installs the stand-ins of :mod:`discoeb_cpu`, then hands the given arguments
to pytest from inside the DISCO-EB checkout (its conftest reads resources by
relative path)::

    PYTHONPATH=..:.:<DISCO-EB>/src .venv/bin/python run_discoeb_tests.py <DISCO-EB> \
        tests/test_thermo.py tests/test_perturbations.py -k "matches_class and N1"

``--benchmark-disable`` is always passed, to keep the runs short: the tests'
checks are unaffected, only their timing rounds are skipped. Tests that need
what the CPU target lacks (sensitivities) fail with ``NotImplementedError``.
"""

from __future__ import annotations

import os
import sys

import discoeb_cpu

discoeb_cpu.install()

import pytest  # noqa: E402

if __name__ == "__main__":
    checkout, *arguments = sys.argv[1:]
    os.chdir(checkout)
    sys.path.insert(0, os.path.join(checkout, "tests"))
    sys.exit(pytest.main(["--benchmark-disable", "-p", "no:cacheprovider", *arguments]))
