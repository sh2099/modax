"""A stand-in for ``numba_cuda_mlir`` that compiles device code for the CPU.

DISCO-EB's device functions name three things from numba-CUDA: the decorator
``cuda.jit(device=True)``, thread-local scratch ``cuda.local.array(shape,
float64)``, and the device-array helpers its tables go through. Installed
into ``sys.modules`` *before* ``discoeb`` builds anything, this module gives
each a CPU meaning, so the unedited DISCO-EB sources compile with plain numba:

==============================  ==========================================
``cuda.jit(device=True)``       ``numba.njit``
``cuda.local.array(n, dtype)``  ``np.empty(n, dtype)``
``cuda.to_device(a)``           a contiguous host array
``cuda.device_array(s, d)``     ``np.empty(s, d)``
``cuda.is_cuda_array(a)``       ``False`` (so tables are "uploaded")
``cuda.synchronize()``          nothing
``types.float64``               ``np.float64``
==============================  ==========================================

One difference in behaviour matters: numba freezes an array a compiled
function closes over at compile time, where a CUDA kernel reads a device
array's contents at launch. A function built against one set of tables must
therefore be rebuilt for another; DISCO-EB's ``DeviceTableSlot`` (one kernel,
refilled buffers) does not carry over.

Usage::

    import cuda_shim
    cuda_shim.install()
    import discoeb.perturbations  # now CPU-compilable
"""

from __future__ import annotations

import sys
import types as _pytypes

import numpy as np
from numba import njit


@njit
def _local_array(shape, dtype):
    return np.empty(shape, dtype)


def _jit(*args, **options):
    """``cuda.jit``: ``njit``, whatever CUDA options (``device``, ``max_registers``) were given."""
    if args and callable(args[0]) and len(args) == 1 and not options:
        return njit(args[0])
    return njit


def _to_device(array):
    return np.ascontiguousarray(array)


class _HostArray(np.ndarray):
    """A host array with the one device-array method DISCO-EB calls.

    It may be filled once. Compiled CPU code holds a frozen copy of the
    contents it was compiled against, so refilling the buffer with other data
    would leave every function built on it silently reading the old tables;
    that is refused instead.
    """

    _filled = False

    def copy_to_device(self, source):
        source = np.asarray(source, dtype=self.dtype)
        if self._filled and not np.array_equal(self, source, equal_nan=True):
            raise RuntimeError(
                "a table buffer was refilled with different data: CPU-compiled code "
                "froze the first contents, so the functions built on it must be rebuilt "
                "(discoeb.perturbations.clear_prepared_solve_cache() drops them)"
            )
        self[...] = source
        self._filled = True


def _device_array(shape, dtype):
    return np.empty(shape, dtype).view(_HostArray)


def build() -> _pytypes.ModuleType:
    """The stand-in package: ``numba_cuda_mlir`` with ``cuda`` and ``types`` submodules."""
    package = _pytypes.ModuleType("numba_cuda_mlir")
    cuda = _pytypes.ModuleType("numba_cuda_mlir.cuda")
    local = _pytypes.ModuleType("numba_cuda_mlir.cuda.local")
    types = _pytypes.ModuleType("numba_cuda_mlir.types")

    local.array = _local_array
    cuda.local = local
    cuda.jit = _jit
    cuda.to_device = _to_device
    cuda.device_array = _device_array
    cuda.is_cuda_array = lambda array: False
    cuda.is_available = lambda: False
    cuda.synchronize = lambda: None
    types.float64 = np.float64
    types.float32 = np.float32
    package.cuda = cuda
    package.types = types
    package.__cpu_shim__ = True
    return package


def install() -> _pytypes.ModuleType:
    """Put the stand-in where ``import numba_cuda_mlir`` finds it; idempotent."""
    existing = sys.modules.get("numba_cuda_mlir")
    if existing is not None:
        if getattr(existing, "__cpu_shim__", False):
            return existing
        raise RuntimeError("the real numba_cuda_mlir is already imported")
    package = build()
    sys.modules["numba_cuda_mlir"] = package
    sys.modules["numba_cuda_mlir.cuda"] = package.cuda
    sys.modules["numba_cuda_mlir.cuda.local"] = package.cuda.local
    sys.modules["numba_cuda_mlir.types"] = package.types
    return package
