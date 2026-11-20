"""Make an unedited DISCO-EB checkout solve on modax's CPU target.

DISCO-EB compiles its own device functions with numba-CUDA, which it imports
lazily under its own name: ``cuda.jit(device=True)``, ``cuda.local.array`` and
the ``float64`` / ``float32`` types, and ``cuda.is_available()`` in its tests.
``install()`` puts modax's CPU meanings of those names (``modax._target``)
where ``import numba_cuda_mlir`` finds them, and enables float64 in JAX. Call
it before anything from ``discoeb`` builds a kernel.
"""

from __future__ import annotations

import os
import sys
import types


def install() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax

    jax.config.update("jax_enable_x64", True)

    from modax import _target

    if _target.TARGET != "cpu":
        raise RuntimeError("modax chose the CUDA target; set MODAX_TARGET=cpu")
    existing = sys.modules.get("numba_cuda_mlir")
    if existing is not None and not getattr(existing, "__cpu_shim__", False):
        raise RuntimeError("the real numba_cuda_mlir is already imported")
    package = types.ModuleType("numba_cuda_mlir")
    package.cuda = _target.cuda
    package.types = _target.types
    package.__cpu_shim__ = True
    sys.modules["numba_cuda_mlir"] = package
    sys.modules["numba_cuda_mlir.cuda"] = _target.cuda
    sys.modules["numba_cuda_mlir.types"] = _target.types
