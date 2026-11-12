"""Make an unedited DISCO-EB v2 checkout solve on the CPU prototype.

``install()`` puts the CUDA stand-in (:mod:`cuda_shim`) and the solver
stand-in (:mod:`modax_stub`) in place and enables float64 in JAX. Call it
before importing anything from ``discoeb``.
"""

from __future__ import annotations

import os


def install() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax

    jax.config.update("jax_enable_x64", True)

    import cuda_shim
    import modax_stub

    cuda_shim.install()
    modax_stub.install()
