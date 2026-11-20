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
    _declare_direction_parameter()


def _declare_direction_parameter() -> None:
    """Tell the finite differences how large DISCO-EB's ``eps`` is.

    DISCO-EB 2.0.x differentiates along a direction of its inputs: the model
    reads each as ``x + eps * dir_x`` at ``eps = 0``. A difference along
    ``eps`` needs a step small beside the inputs it moves, which run from the
    densities (about 1e-8) to ``H0``, and nothing in a zero says so. A DISCO-EB
    without these columns needs no declaration, and a modax without
    ``set_directed_parameter`` (0.4 and later) cannot take one.
    """
    from modax import _differences

    if not hasattr(_differences, "set_directed_parameter"):
        return
    try:
        from discoeb import perturbations as P

        columns = (P.IX_EPS, P.IX_DIR, tuple(P.DIRECTED_COLUMNS))
    except (ImportError, AttributeError):
        return
    eps, first_direction, targets = columns
    _differences.set_directed_parameter(
        eps, tuple(first_direction + i for i in range(len(targets))), targets
    )
