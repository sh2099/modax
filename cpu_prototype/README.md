# DISCO-EB on modax's CPU target

modax now compiles its kernels for the CPU when numba-CUDA is not installed
(`modax/_target.py`, `tests/test_cpu_target.py`). This directory is the harness
around that for DISCO-EB v2, which names `numba_cuda_mlir` in its own device
code: a stand-in that lets an unedited DISCO-EB checkout run, the checks that
were used to validate the target on it, and the scripts for comparing with
CUDA. Nothing here is imported by the package.

It began as a standalone copy of the Rodas5P kernel (commit `a337d6d`), which
the in-package target has replaced.

| File | What it is |
|---|---|
| `cuda_shim.py` | A stand-in for `numba_cuda_mlir`: device functions compile with `numba.njit`, and `cuda.device_array` buffers are read live by compiled code, as a CUDA kernel reads a device array. |
| `discoeb_cpu.py` | Installs the stand-in; call before importing `discoeb`. |
| `run_discoeb_tests.py` | Runs DISCO-EB's own test suite on the CPU target. |
| `run_pk.py` | P(k): accuracy against CLASS, step counts, timings for sparse/dense and float32/float64 LU. Writes `results/`. |
| `check_rhs_jacobian.py` | DISCO-EB's compiled right-hand side and the finite-difference Jacobian against JAX autodiff of the same model. |
| `check_radau.py` | A few perturbation modes against SciPy Radau. |
| `measure_derivatives.py` | Derivative accuracy and cost: the two sensitivity layouts against each other, and the checks of `test_solve_eb_derivatives` that do not depend on bitwise-equal inputs. |
| `jacobian_tools.py` | The target's Jacobian writer as a host-callable function, for the two checks. |
| `dump_gpu_reference.py` | Run on a CUDA machine: records what modax produces there. |
| `compare_with_gpu.py` | Compares the CPU target with that record on identical inputs. |

## Setup

```bash
cd cpu_prototype
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install numba numpy scipy jax networkx cvxopt pytest pytest-benchmark classy
export PYTHONPATH=..:.:/path/to/DISCO-EB/src     # modax and DISCO-EB by path
```

(`uv sync --no-dev` at the repository root also installs modax itself on
macOS; the environment above adds what DISCO-EB's tests need.)

## Running

```bash
.venv/bin/python -m pytest ../tests -q --benchmark-disable     # modax's own suite
.venv/bin/python check_rhs_jacobian.py
.venv/bin/python check_radau.py
.venv/bin/python run_pk.py                                     # flat (50 variables) and mnu (98)
.venv/bin/python run_discoeb_tests.py /path/to/DISCO-EB tests -k "not derivatives"
```

On the CUDA machine, from the DISCO-EB checkout with its own environment:

```bash
uv run python /path/to/modax/cpu_prototype/dump_gpu_reference.py gpu_reference.npz
```

then here: `.venv/bin/python compare_with_gpu.py gpu_reference.npz`.

## Known limits

- Derivatives on the CPU target are finite differences (`modax/_differences.py`).
  DISCO-EB's `matter_power_spectrum_jax` and `cl_power_spectrum_jax` pass their
  tests; a Jacobian of P(k) in the five densities costs about 25 to 30 plain
  solves. `test_solve_eb_derivatives` fails: it requires two launches whose
  densities differ by one unit in the last place (NumPy against XLA's
  `Tcmb**4`) to agree to 1e-12, and with a finite-difference Jacobian the
  solve moves by about 1e-6 of its scale for such a change.
  `measure_derivatives.py` runs the rest of that test's checks and records
  costs.
- Tables made with `cuda.to_device` are frozen into the code compiled against
  them; only `cuda.device_array` buffers can be refilled.
