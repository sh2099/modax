# CPU prototype of the Rodas5P kernel

A standalone experiment: modax's Rodas5P compiled with plain numba for the CPU,
to find out whether a CPU target is worth building into the package. Nothing
under `modax/` is changed, and nothing here is imported by it.

| File | What it is |
|---|---|
| `rodas5p_cpu.py` | The kernel of `modax/rodas5P.py` transcribed for `numba.njit`, one trajectory per `prange` iteration. Dense LU; `df/dy` and `df/dt` by coloured finite differences instead of Enzyme. No save hooks, sensitivities or sparse LU. |
| `cuda_shim.py` | A stand-in for `numba_cuda_mlir` under which device code written for CUDA compiles for the CPU. |
| `modax_stub.py` | `modax.rodas5P.solve` with modax's signature, running the CPU solver behind `jax.pure_callback`. |
| `discoeb_cpu.py` | Installs the two stand-ins, so an unedited DISCO-EB v2 checkout solves on the CPU. |
| `tests/test_solver.py` | The solver against SciPy Radau on stiff problems; the copied tableau against `modax/rodas5P.py`. |
| `check_rhs_jacobian.py` | DISCO-EB's numba right-hand side and the finite-difference Jacobian against JAX autodiff of the same model. |
| `check_radau.py` | A few perturbation modes against SciPy Radau. |
| `run_pk.py` | DISCO-EB's P(k) on the CPU: accuracy against CLASS, step counts, timings. Writes `results/`. |
| `run_discoeb_tests.py` | Runs DISCO-EB's own test suite against the prototype. |
| `dump_gpu_reference.py` | Run on a CUDA machine: records what modax produces there. |
| `compare_with_gpu.py` | Compares the prototype with that record on identical inputs. |

## Setup

```bash
cd cpu_prototype
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install numba numpy scipy jax networkx pytest pytest-benchmark classy
export PYTHONPATH=..:.:/path/to/DISCO-EB/src     # modax and DISCO-EB by path, not installed
```

`modax-solvers` itself cannot be installed without CUDA; `import modax` works
from the checkout because the package's `__init__` only configures JAX.

## Running

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python check_rhs_jacobian.py
.venv/bin/python check_radau.py
.venv/bin/python run_pk.py                      # flat (50 variables) and mnu (98)
.venv/bin/python run_discoeb_tests.py /path/to/DISCO-EB tests/test_thermo.py tests/test_perturbations.py
```

On the CUDA machine, from the DISCO-EB checkout with its own environment:

```bash
uv run python /path/to/modax/cpu_prototype/dump_gpu_reference.py gpu_reference.npz
```

then here: `.venv/bin/python compare_with_gpu.py gpu_reference.npz`.

## Known limits

- numba freezes arrays a compiled function closes over. DISCO-EB's
  `DeviceTableSlot` (one kernel, table buffers refilled per batch) therefore
  cannot be reused across batches: `cuda_shim` refuses a refill with different
  data, and `discoeb.perturbations.clear_prepared_solve_cache()` forces a
  rebuild.
- Forward solves only. Anything that needs a save hook or sensitivities (the
  CMB path, `solve_eb`, the derivative tests) raises `NotImplementedError`.
