"""Make an unedited DISCO-EB v2 checkout solve on modax's CPU target.

DISCO-EB names ``numba_cuda_mlir`` itself for its device functions and table
buffers; ``install()`` puts the stand-in of :mod:`cuda_shim` under that name
and enables float64 in JAX. modax, seeing no real numba-CUDA, compiles for
the CPU (``modax._target``). Call it before importing anything from
``discoeb``.
"""

from __future__ import annotations

import os


def install() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax

    jax.config.update("jax_enable_x64", True)

    import cuda_shim

    cuda_shim.install()

    from modax import _target

    if _target.TARGET != "cpu":
        raise RuntimeError("modax chose the CUDA target; set MODAX_TARGET=cpu")
