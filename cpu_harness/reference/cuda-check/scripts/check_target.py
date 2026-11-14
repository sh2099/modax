import importlib.metadata as md, sys
import jax, modax
print("python", sys.version.split()[0], "modax", modax.__file__)
try:
    import modax._target as t
    print("TARGET", t.TARGET, "| missing:", t._missing_cuda_packages())
    import numba_cuda_mlir
    print("cuda is numba_cuda_mlir.cuda:", t.cuda is numba_cuda_mlir.cuda, "| types is:", t.types is numba_cuda_mlir.types)
    from numba_cuda_mlir.descriptor import MLIRDispatcher
    print("DeviceFunction is MLIRDispatcher:", t.DeviceFunction is MLIRDispatcher)
except ImportError as e:
    print("no modax._target:", e)
from numba_cuda_mlir import cuda, tools
print("cuda.is_available", cuda.is_available(), "cc", tools.get_gpu_compute_capability(tuple))
print("jax", jax.__version__, jax.devices())
print({p: md.version(p) for p in ("numba", "llvmlite", "numba-cuda-mlir", "numba-enzyme-cuda", "jax", "numpy")})
