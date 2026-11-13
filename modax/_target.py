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
    ``cuda.local.array(n, dtype)``  ``np.empty(n, dtype)``
    ``cuda.const.array_like(a)``    ``a`` (numba freezes a closed-over array)
    ``types``                       ``numba.types``
    ``DeviceFunction``              numba's CPU dispatcher
    ==============================  ======================================

What has no CPU meaning is a thread's index (``cuda.blockIdx``,
``cuda.threadIdx``, ``cuda.grid``): each solver keeps one trajectory's
integration in a device function of the trajectory index, which a CUDA kernel
calls with its thread's index and :func:`ensemble_driver` calls from a
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

    @device
    def local_array(shape, dtype):
        return np.empty(shape, dtype)

    @device
    def array_like(array):
        return array

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
    namespace.jit = jit  # type: ignore[attr-defined]
    namespace.is_available = lambda: False  # type: ignore[attr-defined]
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
    return TARGET == "cpu" or bool(cuda.is_available())


def ensemble_driver(trajectory: Any, n_args: int) -> Callable[..., None]:
    """The CPU launch of a kernel: ``trajectory(i, *args)`` for every ``i``, in parallel.

    ``trajectory`` is the device function integrating trajectory ``i`` and
    ``args`` the kernel's ``n_args`` arguments, the first of which has the
    ensemble as its leading axis. Trajectories take very different numbers of
    steps, so the caller should run the loop with ``numba.parallel_chunksize(1)``
    (:func:`run_ensemble` does).

    An exception raised inside the parallel loop is lost, and with it whatever
    the thread had left to do. Device code raises none of its own, but numba's
    bounds checking does (``NUMBA_BOUNDSCHECK=1``), so an out-of-bounds index
    in a kernel shows up only when the loop is serial: ``MODAX_SERIAL=1`` makes
    it one, for debugging. Without bounds checking such an index reads or
    writes whatever is there, as it does on the device.
    """
    from numba import njit, prange

    serial = bool(os.environ.get(_SERIAL_ENV_VAR))
    names = ", ".join(f"a{k}" for k in range(n_args))
    source = (
        f"def driver({names}):\n"
        f"    for i in {'range' if serial else 'prange'}(a0.shape[0]):\n"
        f"        trajectory(i, {names})\n"
    )
    scope: dict[str, Any] = {}
    exec(source, {"prange": prange, "trajectory": trajectory}, scope)  # noqa: S102
    return njit(parallel=not serial, error_model="numpy")(scope["driver"])


def run_ensemble(driver: Callable[..., None], *args: Any) -> None:
    """Call a CPU :func:`ensemble_driver`, handing the threads one trajectory at a time."""
    import numba

    with numba.parallel_chunksize(1):
        driver(*args)
