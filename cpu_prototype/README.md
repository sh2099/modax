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
| `profile_cmb.py` | Stage timings of DISCO-EB's CMB call (first call, new batch, repeated batch) and what numba and JAX compile. |
| `check_lensing.py` | Which input the end-to-end lensing gate is missed on: each of the test's inputs swapped for CLASS's own. |
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

On Linux, from the lock and without the two CUDA packages (checked on RHEL 9,
glibc 2.34, x86-64, where `numba-enzyme-cuda` cannot be installed):

```bash
uv sync --no-dev --python 3.12 --no-install-package numba-enzyme-cuda --no-install-package numba-cuda-mlir
VIRTUAL_ENV=$PWD/.venv uv pip install pytest pytest-benchmark classy
```

The small NVIDIA toolkit wheels the lock lists still install and are unused.

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
  tests; a Jacobian of P(k) in the five densities costs about 25 plain solves.
  Each sensitivity column costs about 12 right-hand-side evaluations a stage
  (4 for its first derivative, 8 for the second-order coupling term), against
  about 20 a step for the plain solve.
- `test_solve_eb_derivatives` fails on its first assertion, which requires two
  launches to agree to 1e-12. Their photon densities differ by one unit in the
  last place: one is formed in NumPy and one in JAX, and XLA's CPU `pow`
  rounds `Tcmb**4` differently (on Linux x86-64 too). The solve then moves by
  about 1e-6 of its scale, and that is the float32 linear solve, not the
  differenced Jacobian: with `lu_precision="fp64"` it moves by 1e-9, and on
  Robertson an exact Jacobian with the float32 solve moves as much as the
  differenced one (1e-7). So the assertion holds only where the two densities
  are bitwise equal; whether they are on a GPU is not yet checked.
  `measure_derivatives.py` runs the rest of that test's checks.
- `test_lensed_spectra_end_to_end_match_class` misses one gate (TT lensing
  correction 1.7e-3 against 1e-3 near l = 1490). The test takes
  `lensing_tables` at its float32 default, and the lensing sum alone, given
  CLASS's own spectra, is then off by 1.1e-3 there (1.9e-4 with float64
  tables). With float64 tables the test's calculation gives 8.2e-4.
  `check_lensing.py` records this in `results/lensing_gate.json`.
- A first CMB call takes about 95 s with DISCO-EB's default grids, of which
  85 s is DISCO-EB building its projection tables in eager JAX and 9 s is
  numba compiling the kernel. numba cannot cache the kernels on disk: they
  and the right-hand sides are generated source with no file behind them, and
  the harness compiles table buffers in by address.
- A cosmology not seen before costs 1.4 to 1.8 s more than one seen, because
  DISCO-EB pads its thermal-history program to 128 cosmologies;
  `discoeb_cpu.install(thermal_chunk=16)` brings that to 0.04 s (see its
  docstring).
- Tables made with `cuda.to_device` are frozen into the code compiled against
  them; only `cuda.device_array` buffers can be refilled.
