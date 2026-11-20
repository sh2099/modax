"""Print the device and the versions; runner.py has already checked which modax this is."""

import importlib.metadata as md
import sys

import jax

print("python", sys.version.split()[0])
print("jax", jax.__version__, jax.devices())
try:
    from numba_cuda_mlir import cuda

    d = cuda.get_current_device()
    print("device", d.name, "cc", d.compute_capability)
except Exception as e:  # noqa: BLE001
    print("numba-cuda-mlir device query:", repr(e))
for p in (
    "modax-solvers",
    "disco-eb",
    "numba",
    "llvmlite",
    "numba-cuda-mlir",
    "numba-enzyme-cuda",
    "jax",
    "jaxlib",
    "numpy",
    "cuda-bindings",
):
    try:
        print(p, md.version(p))
    except Exception:  # noqa: BLE001
        print(p, "not installed")
