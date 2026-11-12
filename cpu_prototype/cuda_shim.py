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
``cuda.device_array(s, d)``     a host buffer read live
``cuda.is_cuda_array(a)``       whether ``a`` is such a buffer
``cuda.synchronize()``          nothing
``types.float64``               ``np.float64``
==============================  ==========================================

A CUDA kernel reads a device array's contents at launch; numba freezes an
ordinary array a compiled function closes over. ``cuda.device_array`` buffers
are therefore typed so that compiled code reads them live
(:class:`_HostArray`), which is what lets DISCO-EB's ``DeviceTableSlot`` refill
one kernel's tables per batch. Arrays made by ``cuda.to_device`` are frozen
as usual; nothing refills those.

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
from numba import types as nb_types
from numba.core import cgutils
from numba.core.imputils import lower_constant
from numba.extending import models, register_model, typeof_impl
from numba.np.numpy_support import from_dtype

# IEEE arithmetic, as on the device: no exception on a division by zero.
_device = njit(error_model="numpy")


@_device
def _local_array(shape, dtype):
    return np.empty(shape, dtype)


def _jit(*args, **options):
    """``cuda.jit``: ``njit``, whatever CUDA options (``device``, ``max_registers``) were given."""
    if args and callable(args[0]) and len(args) == 1 and not options:
        return _device(args[0])
    return _device


def _to_device(array):
    return np.ascontiguousarray(array)


class _HostArray(np.ndarray):
    """A table buffer that compiled code reads live, as a CUDA kernel reads a device array.

    numba freezes an ordinary array a compiled function closes over: the
    function carries a copy of the contents at compile time. A CUDA kernel
    instead reads a device array's contents at launch, which DISCO-EB relies
    on (``DeviceTableSlot``: one kernel, its table buffers refilled per
    batch). Arrays of this class are typed as :class:`_LiveArrayType`, whose
    constant is the buffer's address and not its contents, so a refill in
    place is seen by every function already compiled against the buffer. The
    buffer must outlive those functions and never be reallocated.
    """

    def copy_to_device(self, source):
        self[...] = source


class _LiveArrayType(nb_types.Array):
    def __init__(self, dtype, ndim, layout):
        super().__init__(
            dtype, ndim, layout, name=f"live_array({dtype}, {ndim}d, {layout})"
        )

    def copy(self, dtype=None, ndim=None, layout=None, readonly=None):
        # Type inference marks a global array read-only through copy(), which
        # would otherwise hand back a plain (frozen) array type.
        return _LiveArrayType(
            dtype or self.dtype,
            self.ndim if ndim is None else ndim,
            layout or self.layout,
        )


register_model(_LiveArrayType)(models.ArrayModel)


@typeof_impl.register(_HostArray)
def _typeof_host_array(value, context):
    return _LiveArrayType(from_dtype(value.dtype), value.ndim, "C")


@lower_constant(_LiveArrayType)
def _live_array_constant(context, builder, ty, value):
    array = context.make_array(ty)(context, builder)
    intp = nb_types.intp
    context.populate_array(
        array,
        data=context.get_constant(intp, value.ctypes.data).inttoptr(array.data.type),
        shape=cgutils.pack_array(
            builder, [context.get_constant(intp, n) for n in value.shape]
        ),
        strides=cgutils.pack_array(
            builder, [context.get_constant(intp, n) for n in value.strides]
        ),
        itemsize=context.get_constant(intp, value.itemsize),
        meminfo=None,
    )
    return array._getvalue()


def _device_array(shape, dtype):
    return np.zeros(shape, dtype).view(_HostArray)


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
    cuda.is_cuda_array = lambda array: isinstance(array, _HostArray)
    # "The device target is usable": what DISCO-EB's tests gate on.
    cuda.is_available = lambda: True
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
