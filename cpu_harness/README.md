# DISCO-EB on modax's CPU target

modax compiles its kernels for the CPU when numba-CUDA is not installed
(`modax/_target.py`, `tests/test_cpu_target.py`). DISCO-EB also names
`numba_cuda_mlir` in its own device code, so this directory holds what an
unedited DISCO-EB checkout needs to run on that target. Nothing here is
imported by the package.

| File | What it is |
|---|---|
| `discoeb_cpu.py` | Puts modax's CPU meanings of `cuda.jit`, `cuda.local.array`, `cuda.is_available` and the float types under the name `numba_cuda_mlir`. Call `install()` before `discoeb` builds a kernel. |
| `run_discoeb_tests.py` | Runs DISCO-EB's own test suite on the CPU target. |

## Running

With an environment that has modax's dependencies without the two CUDA
packages (see the README's Development section), plus `pytest`,
`pytest-benchmark` and `classy`:

```bash
export PYTHONPATH=/path/to/modax:/path/to/modax/cpu_harness:/path/to/DISCO-EB/src
python -m pytest /path/to/modax/tests -q --benchmark-disable     # modax's own suite
python run_discoeb_tests.py /path/to/DISCO-EB tests -q
```

`MODAX_SERIAL=1 NUMBA_BOUNDSCHECK=1` runs the trajectories in one thread
with bounds checking, which is the only way an out-of-bounds index or an
exception inside a kernel shows up: in the parallel loop both are silent.

## State on 2026-11-13

modax 0.4.0, DISCO-EB 2.0.1 (`6c48e2d`), Apple M4 Pro, 14 threads.

- modax's suite: every test passes except
  `test_an_upstream_auto_block_switches_where_its_problem_turns_stiff`, which
  asks an interpolated upstream state for 1e-4 and gets 3e-4. With exact
  derivatives in place of differences the same solve gives between 8e-6 and
  3e-4 as the tolerance is varied by one percent, so the gate is met or
  missed on the step sequence, on any target.
- DISCO-EB's suite: everything that does not import `modax.rodas5P` passes,
  including P(k) against CLASS for 1 and 128 cosmologies. DISCO-EB 2.0.1 is
  written for modax 0.3.0; `solve_eb` and the derivative paths wait for the
  DISCO-EB that uses modax 0.4.0.
- P(k), 128 modes: 0.03 s for one cosmology, 2.8 s for 128, about 22 s to
  compile in a new process.

## How the target differs from CUDA

- A kernel is called once per trajectory, so a warp is one lane: where the
  lanes of a GPU warp advance an upstream block together, each trajectory
  here advances it when it needs to. Step sequences differ from a GPU's.
- Every derivative is a finite difference (`make_difference_jacobian` in
  `modax/_rodas5P.py`, `modax/_differences.py`): good to about 1e-8 of the
  quantities' scale, not to rounding. Sensitivities at tolerances tighter
  than about 1e-8 do not improve.
- `cache=True` is accepted and ignored: numba cannot cache generated source.
