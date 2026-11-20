"""Run SCRIPT [args] as __main__, or pytest when SCRIPT is the word 'pytest', on a chosen modax tree.

    MC_SRC=<tree> MC_SIDE=upstream|branch python runner.py SCRIPT [args]

The tree is put first on sys.path (and the cwd should be the tree, for the
tests). Before anything runs it refuses a modax that is not the one in
MC_SRC, and checks the side's marker: the branch has ``modax._target`` with
``TARGET == "cuda"`` and numba-cuda-mlir's own ``cuda``; upstream v0.5.0 has
no ``_target`` module at all (``MC_ALLOW_CPU=1`` lets the branch run on its
CPU target, to test these scripts off a GPU). At exit it prints which modax ran and which of
the branch's CPU-target modules got imported (``_differences`` always is, by
``_sensitivity``, and unused on CUDA; ``_enzyme_cpu`` must not be).
"""

import atexit
import importlib.metadata as md
import os
import runpy
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
SRC = os.path.realpath(os.environ["MC_SRC"])
SIDE = os.environ["MC_SIDE"]
assert SIDE in ("upstream", "branch"), SIDE
sys.path.insert(0, SRC)

import modax  # noqa: E402

where = os.path.realpath(modax.__file__)
if not where.startswith(SRC + os.sep):
    print(f"[runner] ERROR: modax is {where}, not from MC_SRC={SRC}", flush=True)
    os._exit(3)
has_target = os.path.exists(os.path.join(SRC, "modax", "_target.py"))
if SIDE == "upstream" and has_target:
    print(
        "[runner] ERROR: MC_SIDE=upstream but the tree has modax/_target.py", flush=True
    )
    os._exit(3)
if SIDE == "branch":
    from modax import _target

    try:
        from numba_cuda_mlir import cuda
    except ImportError:  # the CPU target, testing these scripts (MC_ALLOW_CPU)
        cuda = None

    ok = _target.TARGET == "cuda" and _target.cuda is cuda
    print(
        f"[runner] branch: TARGET={_target.TARGET}, cuda is numba_cuda_mlir.cuda: {_target.cuda is cuda}",
        flush=True,
    )
    if not ok and not os.environ.get("MC_ALLOW_CPU"):
        print("[runner] ERROR: the branch did not choose the CUDA target", flush=True)
        os._exit(3)
versions = dict.fromkeys(
    (
        "modax-solvers",
        "numba",
        "llvmlite",
        "numba-cuda-mlir",
        "numba-enzyme-cuda",
        "jax",
        "jaxlib",
        "numpy",
        "disco-eb",
    )
)
for p in versions:
    try:
        versions[p] = md.version(p)
    except Exception:  # noqa: BLE001
        pass
print(
    f"[runner] side {SIDE}, modax {where}, python {sys.version.split()[0]}, {versions}",
    flush=True,
)


def _report():
    cpu = sorted(
        k
        for k in sys.modules
        if k in ("modax._enzyme_cpu", "modax._differences", "modax._target")
    )
    print(
        f"[runner] at exit: modax {where} ({SIDE}); CPU-target modules imported: {cpu or 'none'}",
        flush=True,
    )
    target = sys.modules.get("modax._target")
    if (
        "modax._enzyme_cpu" in sys.modules
        and getattr(target, "TARGET", "cuda") == "cuda"
    ):
        print("[runner] ERROR: modax._enzyme_cpu was imported on CUDA", flush=True)


atexit.register(_report)
sys.argv = sys.argv[1:]
if not sys.argv:
    sys.exit(__doc__)
if sys.argv[0] == "pytest":
    import pytest

    sys.path.insert(0, os.getcwd())
    sys.exit(pytest.main(sys.argv[1:]))
sys.path.insert(0, os.getcwd())
runpy.run_path(sys.argv[0], run_name="__main__")
