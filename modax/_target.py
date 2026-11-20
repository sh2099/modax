"""Which device the kernels are compiled for, and the names they are written in.

The kernels are written against numba-CUDA: ``cuda.jit(device=True)``
functions over thread-local ``cuda.local.array`` buffers, one trajectory per
thread. Nothing in them is collective -- no shared memory a step depends on,
no barrier -- so the same source is also a correct CPU program once those
names mean something there. This module decides the target once, at import,
and exports the names every other module uses:

``TARGET``
    ``"cuda"`` when ``numba_cuda_mlir`` and ``numba_enzyme`` are both
    installed, ``"cpu"`` otherwise. ``MODAX_TARGET=cpu`` forces the CPU target
    where both are possible; ``MODAX_TARGET=cuda`` insists on CUDA and fails
    without it. The choice looks at what is installed, not at the machine: an
    installed CUDA stack with no usable GPU is still the CUDA target, and
    needs ``MODAX_TARGET=cpu`` to run here.

``cuda``, ``types``, ``DeviceFunction``
    On CUDA, ``numba_cuda_mlir.cuda``, ``numba_cuda_mlir.types`` and its
    ``MLIRDispatcher`` themselves: the CUDA path differs from one that
    imported them directly by this indirection and nothing else. On the CPU
    they are stand-ins with plain-numba meanings:

    ==============================  ======================================
    ``cuda.jit(device=True)``       ``numba.njit(error_model="numpy")``
    ``cuda.local.array(n, dtype)``  ``np.zeros(n, dtype)``
    ``cuda.const.array_like(a)``    ``a`` (numba freezes a closed-over array)
    ``cuda.shared.array(s, dtype)``   ``np.zeros(s, dtype)``
    ``cuda.blockIdx.x`` etc.        ``0``
    ``cuda.grid(1)``                ``0``
    ``cuda.ballot_sync(m, p)``      ``1 if p else 0``
    ``cuda.any_sync(m, p)``         ``p``
    ``types``                       ``numba.types``
    ``DeviceFunction``              numba's CPU dispatcher
    ==============================  ======================================

A kernel finds its trajectory from its thread's index (``cuda.blockIdx``,
``cuda.threadIdx``, ``cuda.grid``) and, in the hierarchy kernel, votes with the
other lanes of its warp (``cuda.ballot_sync``, ``cuda.any_sync``). On the CPU a
kernel is called once per trajectory on that trajectory's own rows of every
array, so the index is always zero and a warp is one lane: the index names are
zero, a ballot is the lane's own predicate, and the kernels' source is the
same text on both targets. :func:`ensemble_driver` makes those calls from a
parallel loop.

On the CPU there is no Enzyme (it has no macOS build), so every derivative of
a user's function is a finite difference: the Jacobian in
``modax/rodas5P.py`` (``_make_difference_jacobian``) and the directional
derivatives the forward sensitivities are built from in
[`modax._differences`][]. They are accurate to rounding over a step, not to
rounding, and they cannot see past a switch in the function the way a
derivative does; both modules say what that costs.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types as _pytypes
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np

_ENV_VAR = "MODAX_TARGET"
_SERIAL_ENV_VAR = "MODAX_SERIAL"
_POISON_ENV_VAR = "MODAX_POISON"


def _missing_cuda_packages() -> list[str]:
    """The packages of the CUDA stack that are not installed, by import name.

    A CPU stand-in under ``numba_cuda_mlir``'s name counts as missing. Enzyme is
    part of the stack: every implicit solve differentiates through it, and the
    package imports it with the solvers.
    """
    missing = []
    try:
        import numba_cuda_mlir
    except ImportError:
        missing.append("numba_cuda_mlir")
    else:
        if getattr(numba_cuda_mlir, "__cpu_shim__", False):
            missing.append("numba_cuda_mlir")
    try:
        found = importlib.util.find_spec("numba_enzyme") is not None
    except (ImportError, ValueError):
        found = False
    if not found:
        missing.append("numba_enzyme")
    return missing


def _choose() -> Literal["cuda", "cpu"]:
    requested = os.environ.get(_ENV_VAR, "auto").strip().lower()
    if requested not in ("auto", "cuda", "cpu"):
        raise ValueError(
            f"{_ENV_VAR} must be 'auto', 'cuda' or 'cpu'; got {requested!r}"
        )
    if requested == "cpu":
        return "cpu"
    missing = _missing_cuda_packages()
    if not missing:
        return "cuda"
    if requested == "cuda":
        raise ImportError(
            f"{_ENV_VAR}=cuda, but {' and '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} not installed"
        )
    return "cpu"


TARGET: Literal["cuda", "cpu"] = _choose()
"""The device the kernels are compiled for."""


def _cpu_namespace() -> tuple[Any, Any, Any, Any, Any]:
    """The CPU meanings of ``cuda``, ``types``, ``DeviceFunction``, ``Type``, ``Signature``."""
    from numba import njit
    from numba import types as numba_types
    from numba.core.registry import CPUDispatcher
    from numba.core.typing.templates import Signature as NumbaSignature

    # CUDA arithmetic is IEEE: a division by zero is an infinity the step
    # controller rejects, not an exception. numba's default error model would
    # raise there instead, and inside a parallel loop the raise silently ends
    # the thread's share of the ensemble.
    device = njit(error_model="numpy")

    if os.environ.get(_POISON_ENV_VAR):
        # For finding a read of a local array before it is written: every one
        # starts as NaN, which then shows in whatever read it.
        @device
        def local_array(shape, dtype):
            array = np.empty(shape, dtype)
            array.reshape(-1)[:] = np.nan
            return array

    else:
        # Zeroed, not empty. With three blocks the hierarchy kernel hands a
        # reader the second-derivative coefficients of an upstream
        # sensitivity before anything has written them (found by starting
        # every local array as NaN, MODAX_POISON=1). A differenced sweep
        # sizes its step by the stale entry, and three blocks' sensitivities
        # came out NaN on x86-64 until the entry was a number that does no
        # harm. The read is in the kernel on either target: unmodified
        # v0.4.0 on a Tesla P100 gives NaN for the same three-block
        # sensitivities (cpu_harness/reference/cuda-check), though whether
        # that is this read or that card's unsupported stack was not
        # established.
        @device
        def local_array(shape, dtype):
            return np.zeros(shape, dtype)

    @device
    def array_like(array):
        return array

    @device
    def grid(ndim):
        return 0

    @device
    def ballot_sync(mask, predicate):
        return 1 if predicate else 0

    @device
    def any_sync(mask, predicate):
        return predicate

    def jit(*args: Any, **options: Any) -> Any:
        """``cuda.jit``: ``njit``, whatever CUDA options (``device``, ``max_registers``) it was given."""
        if len(args) == 1 and callable(args[0]) and not options:
            return device(args[0])
        return device

    name = f"{__name__}.cuda"
    namespace = _pytypes.ModuleType(name)
    local = _pytypes.ModuleType(f"{name}.local")
    const = _pytypes.ModuleType(f"{name}.const")
    local.array = local_array  # type: ignore[attr-defined]
    const.array_like = array_like  # type: ignore[attr-defined]
    namespace.local = local  # type: ignore[attr-defined]
    namespace.const = const  # type: ignore[attr-defined]
    # modules, so that numba resolves ``cuda.blockIdx.x`` to the constant
    for index_name in ("blockIdx", "threadIdx", "blockDim", "shared"):
        index = _pytypes.ModuleType(f"{name}.{index_name}")
        index.x = 0  # type: ignore[attr-defined]
        setattr(namespace, index_name, index)
        sys.modules[index.__name__] = index
    namespace.shared.array = local_array  # type: ignore[attr-defined]
    namespace.grid = grid  # type: ignore[attr-defined]
    namespace.ballot_sync = ballot_sync  # type: ignore[attr-defined]
    namespace.any_sync = any_sync  # type: ignore[attr-defined]
    namespace.jit = jit  # type: ignore[attr-defined]
    # "a solve can run here", which is what callers ask it for
    namespace.is_available = lambda: True  # type: ignore[attr-defined]
    namespace.to_device = np.ascontiguousarray  # type: ignore[attr-defined]
    # numba resolves a module's attributes at compile time only for modules it
    # can find again by name.
    sys.modules[name] = namespace
    sys.modules[local.__name__] = local
    sys.modules[const.__name__] = const
    return namespace, numba_types, CPUDispatcher, numba_types.Type, NumbaSignature


if TYPE_CHECKING or TARGET == "cuda":
    from numba_cuda_mlir import cuda, types
    from numba_cuda_mlir.descriptor import MLIRDispatcher as DeviceFunction
    from numba_cuda_mlir.numba_cuda.types import Type
    from numba_cuda_mlir.numba_cuda.typing.templates import Signature
else:
    cuda, types, DeviceFunction, Type, Signature = _cpu_namespace()


def available() -> bool:
    """Whether a solve can run here: a CUDA device is present, or the target is the CPU."""
    return bool(cuda.is_available())


def _claim_next() -> Any:
    """``claim(counter)``: add one to ``counter[0]`` atomically and return what it held."""
    from llvmlite import ir
    from numba import types as numba_types
    from numba.extending import intrinsic

    @intrinsic
    def claim(typingctx, counter):
        def codegen(context, builder, signature, args):
            array = context.make_array(signature.args[0])(context, builder, args[0])
            one = ir.Constant(ir.IntType(64), 1)
            return builder.atomic_rmw("add", array.data, one, "seq_cst")

        return numba_types.int64(counter), codegen

    return claim


N_DRIVER_COLUMNS = 3
"""Columns :func:`ensemble_driver` appends to a trajectory's parameter row."""


def device_return_type(function: Any, argument_types: tuple[Any, ...]) -> Any:
    """The type a device function returns for arguments of ``argument_types``."""
    if TARGET == "cuda":
        return function.compile_device(argument_types).signature.return_type
    function.compile(tuple(argument_types))
    return function.overloads[tuple(argument_types)].signature.return_type


def ensemble_driver(kernel: Any) -> Callable[..., None]:
    """The CPU launch of a kernel: one call per trajectory, in parallel.

    ``kernel`` takes the fourteen arguments of
    :py:obj:`SOLVER_ARGTYPES <modax._numba_common.SOLVER_ARGTYPES>`. Each call
    gets one trajectory's rows of every array (a save grid of its own, one row
    of ``y0``, of the weights and of each output), so the kernel finds its
    trajectory at index zero, which is what the index names mean here.

    The parameter row is a copy with :data:`N_DRIVER_COLUMNS` more columns
    than the kernel was built for. A right-hand side reads only its own
    columns; the finite-difference Jacobian (``modax/_rodas5P.py``) reads the
    ones after them, which it has no other way to be told without the kernels'
    source differing between the targets:

    * ``atol / rtol``, the size at which the two tolerances meet, as the
      typical size of a state that is zero or far below it;
    * the first step (the one given, or the kernel's own default of ``1e-6`` of
      the window), as the scale of time where time itself is near zero;
    * ``rtol``, which bounds how fine a difference step is worth taking.

    Trajectories take very different numbers of steps, and numba's default
    threading layer hands each thread a fixed share of a loop, so a loop over
    the trajectories leaves most threads idle while one finishes the expensive
    share. The loop is therefore over the threads, each of which claims the
    next unclaimed trajectory from a shared counter until none is left. The
    caller runs it with ``numba.parallel_chunksize(1)`` (:func:`run_ensemble`
    does), so that each thread gets one iteration.

    An exception raised inside the parallel loop is lost, and with it whatever
    the thread had left to do. Device code raises none of its own, but numba's
    bounds checking does (``NUMBA_BOUNDSCHECK=1``), so an out-of-bounds index
    in a kernel shows up only when the loop is serial: ``MODAX_SERIAL=1`` makes
    it one, for debugging. Without bounds checking such an index reads or
    writes whatever is there, as it does on the device.
    """
    from numba import get_num_threads, njit, prange

    serial = bool(os.environ.get(_SERIAL_ENV_VAR))
    extra = N_DRIVER_COLUMNS

    @njit(error_model="numpy")
    def one(
        i, row, y0, times, params, dt0, rtol, atol, max_steps, time_rows,
        weights, hist, accepted, rejected, loops, hook,
    ):  # fmt: skip
        n_save = times.shape[0] // time_rows
        start = (i if time_rows > 1 else 0) * n_save
        own_times = times[start : start + n_save]
        n_params = params.shape[1]
        for j in range(n_params):
            row[0, j] = params[i, j]
        row[0, n_params] = atol / rtol if rtol > 0.0 else 1.0
        window = own_times[n_save - 1] - own_times[0]
        row[0, n_params + 1] = dt0 if dt0 > 0.0 else window * 1e-6
        row[0, n_params + 2] = rtol
        kernel(
            y0[i : i + 1], own_times, row, dt0, rtol, atol, max_steps, 1,
            weights[i : i + 1], hist[i : i + 1], accepted[i : i + 1],
            rejected[i : i + 1], loops[i : i + 1], hook[i : i + 1],
        )  # fmt: skip

    claim = _claim_next()

    if serial:

        @njit(error_model="numpy")
        def driver(
            y0, times, params, dt0, rtol, atol, max_steps, time_rows,
            weights, hist, accepted, rejected, loops, hook,
        ):  # fmt: skip
            row = np.empty((1, params.shape[1] + extra))
            for i in range(y0.shape[0]):
                one(
                    i, row, y0, times, params, dt0, rtol, atol, max_steps,
                    time_rows, weights, hist, accepted, rejected, loops, hook,
                )  # fmt: skip

        return driver

    @njit(parallel=True, error_model="numpy")
    def driver(  # noqa: F811
        y0, times, params, dt0, rtol, atol, max_steps, time_rows,
        weights, hist, accepted, rejected, loops, hook,
    ):  # fmt: skip
        n = y0.shape[0]
        counter = np.zeros(1, np.int64)
        for thread in prange(min(n, get_num_threads())):
            row = np.empty((1, params.shape[1] + extra))
            while True:
                i = claim(counter)
                if i >= n:
                    break
                one(
                    i, row, y0, times, params, dt0, rtol, atol, max_steps,
                    time_rows, weights, hist, accepted, rejected, loops, hook,
                )  # fmt: skip

    return driver


def run_ensemble(driver: Callable[..., None], *args: Any) -> None:
    """Call a CPU :func:`ensemble_driver`, one iteration of its loop to a thread."""
    import numba

    with numba.parallel_chunksize(1):
        driver(*args)
