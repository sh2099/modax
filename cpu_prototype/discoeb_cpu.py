"""Make an unedited DISCO-EB v2 checkout solve on modax's CPU target.

DISCO-EB names ``numba_cuda_mlir`` itself for its device functions and table
buffers; ``install()`` puts the stand-in of :mod:`cuda_shim` under that name
and enables float64 in JAX. modax, seeing no real numba-CUDA, compiles for
the CPU (``modax._target``). Call it before importing anything from
``discoeb``.

``install(thermal_chunk=16)`` also shrinks the batch DISCO-EB pads its
thermal-history program to (``discoeb.perturbations.THERMO_BATCH_CHUNK``, 128).
On a GPU the padding lanes are free; on the CPU each one is computed, and the
128-lane program touches several GB of fresh memory per call, so one new
cosmology costs 0.3 to 1.7 s where a 16-lane program takes 0.03 s. The tables
were bitwise identical between 128, 32 and 16 lanes for the 128 test
cosmologies (8 lanes differed in the last digit). It is off by default so that
the harness runs DISCO-EB as it is.
"""

from __future__ import annotations

import os


def install(thermal_chunk: int | None = None) -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax

    jax.config.update("jax_enable_x64", True)

    import cuda_shim

    cuda_shim.install()

    from modax import _target

    if _target.TARGET != "cpu":
        raise RuntimeError("modax chose the CUDA target; set MODAX_TARGET=cpu")

    if thermal_chunk is not None:
        from discoeb import perturbations

        perturbations.THERMO_BATCH_CHUNK = int(thermal_chunk)
