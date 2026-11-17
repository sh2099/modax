"""Which target modax picks, whether its names are numba-cuda-mlir's own, and whether anything of
the CPU target's derivative toolchain is imported or warned about, before and after solves that
need Enzyme (a Rodas5P Jacobian, a parameter JVP, a quantity hook with tangents).
Usage: check_target.py"""
import importlib.metadata as md, sys, warnings

warnings.simplefilter("always")
with warnings.catch_warnings(record=True) as caught:
    import jax, jax.numpy as jnp, numpy as np
    import modax
print("python", sys.version.split()[0], "modax", modax.__file__)
CPU_MODULES = ("modax._enzyme_cpu", "modax._differences")


def cpu_modules():
    return [k for k in CPU_MODULES if k in sys.modules]


try:
    import modax._target as t
    print("TARGET", t.TARGET, "| CPU_DERIVATIVES", t.CPU_DERIVATIVES, "| missing:", t._missing_cuda_packages())
    import numba_cuda_mlir
    print("cuda is numba_cuda_mlir.cuda:", t.cuda is numba_cuda_mlir.cuda, "| types is numba_cuda_mlir.types:", t.types is numba_cuda_mlir.types)
    from numba_cuda_mlir.descriptor import MLIRDispatcher
    print("DeviceFunction is MLIRDispatcher:", t.DeviceFunction is MLIRDispatcher)
    import modax.rodas5P as r, modax.tsit5 as ts, modax._hierarchy as h
    for mod in (r, ts, h):
        c = getattr(mod, "cuda", None)
        print(f"  {mod.__name__}.cuda is numba_cuda_mlir.cuda:", c is numba_cuda_mlir.cuda if c is not None else "no attribute")
except ImportError as e:
    print("no modax._target:", e)
print("after import: CPU-target modules imported:", cpu_modules() or "none")
print("warnings at import:", [f"{w.category.__name__}: {w.message}" for w in caught] or "none")

from numba_cuda_mlir import cuda, tools
print("cuda.is_available", cuda.is_available(), "cc", tools.get_gpu_compute_capability(tuple))
print("jax", jax.__version__, jax.devices())
print({p: md.version(p) for p in ("numba", "llvmlite", "numba-cuda-mlir", "numba-enzyme-cuda", "jax", "jaxlib", "numpy")})

from modax import Rodas5P, solve
from modax._sensitivity import SensitivitySpec
from modax.rodas5P import quantity_hook


def rob(y, t, p):
    return (-p[0] * y[0] + p[1] * y[1] * y[2], p[0] * y[0] - p[1] * y[1] * y[2] - p[2] * y[1] ** 2, p[2] * y[1] ** 2)


def osc(y, t, p):
    return (y[1], -p[0] * y[0] - p[1] * y[1])


def energy(y, t, p):
    return (0.5 * (y[1] * y[1] + p[0] * y[0] * y[0]),)


with warnings.catch_warnings(record=True) as caught:
    P = jnp.asarray([[0.04, 1e4, 3e7]] * 4)
    Y0 = jnp.asarray([[1.0, 0.0, 0.0]] * 4)
    T = jnp.asarray([0.0, 1e-2, 1e2])
    h = solve(rob, Y0, T, P, method=Rodas5P(lu_precision="fp64"))
    _, dh = jax.jvp(lambda p: solve(rob, Y0, T, p, method=Rodas5P(lu_precision="fp64")), (P,), (P * 0.01,))
    spec = SensitivitySpec(2, 2, wrt_y0=True, wrt_params=True, error_control=False)
    hook = quantity_hook(energy, 1, 2, 2, spec)
    _, acc = solve(osc, jnp.asarray([[1.0, 0.0]] * 3), jnp.linspace(0.0, 1.0, 5), jnp.asarray([[4.0, 0.2]] * 3),
                   method=Rodas5P(lu_precision="fp64"), save_hook=hook, hook_size=5 * 5, save_history=False, sensitivity=spec)
    print("solve ok:", bool(np.isfinite(np.asarray(h)).all()), bool(np.isfinite(np.asarray(dh)).all()), bool(np.isfinite(np.asarray(acc)).all()))
print("after solves: CPU-target modules imported:", cpu_modules() or "none")
print("warnings during solves:", [f"{w.category.__name__}: {w.message}" for w in caught] or "none")
