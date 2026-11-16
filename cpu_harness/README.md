# DISCO-EB on modax's CPU target

modax compiles its kernels for the CPU when numba-CUDA is not installed
(`modax/_target.py`, `tests/test_cpu_target.py`). DISCO-EB also names
`numba_cuda_mlir` in its own device code, so this directory holds what an
unedited DISCO-EB checkout needs to run on that target. Nothing here is
imported by the package.

| File | What it is |
|---|---|
| `disco`, `disco_run.py` | The launcher: `disco [--modax 0.3\|0.4] script.py \| -m module \| -c code`, with the chosen modax, this directory and the DISCO-EB / DISCO-CMB sources on `PYTHONPATH`. Prints what ran on stderr, derivative engine included. |
| `build_enzyme_toolchain.sh` | Builds, on an Apple Silicon Mac, the LLVM 15 tools, the Enzyme plugin and a numba 0.61 environment that Enzyme on the CPU needs. See below. |
| `discoeb_cpu.py` | Puts modax's CPU meanings of `cuda.jit`, `cuda.local.array`, `cuda.is_available` and the float types under the name `numba_cuda_mlir`. Call `install()` before `discoeb` builds a kernel. |
| `run_discoeb_tests.py` | Runs DISCO-EB's own test suite on the CPU target. |
| `make_gpu_reference.py` | Run on a CUDA machine with DISCO-EB's own environment: records P(k), histories, `solve_eb`, derivatives, thermal histories and step counts on the test suite's cosmologies. |
| `compare_with_gpu.py` | Compares the CPU target with that record on identical inputs. Writes `results/gpu_comparison.json`. |
| `reference/` | `cuda-check/`: see below. Also the record from DISCO-EB 2.0.1 on modax 0.3.0 on a Tesla P100 (2026-11-13), with its package list and test log. The `.npz` (14 MB) is not in git. |

## Two derivative engines

The CPU target differentiates a user's function in one of two ways, chosen
at import by `MODAX_CPU_DERIVATIVES`:

| | Enzyme (`modax/_enzyme_cpu.py`) | Finite differences (`modax/_differences.py`) |
|---|---|---|
| Chosen by | `auto` (the default) where Enzyme can run; `enzyme` | `auto` where it cannot (with one warning saying why); `differences` |
| Jacobian | the CUDA branch's literal-seed writer, one direction a sweep | `make_difference_jacobian` in `modax/rodas5P.py` |
| Sensitivity sweeps | Enzyme forward mode, second order forward over forward | central differences |
| Accuracy | rounding, as on CUDA | about 1e-8 of a quantity's scale |
| `cuda.local.array` | a zeroed stack array | `np.zeros` |
| Needs | numba 0.61.x, llvmlite 0.44.x, `numba-enzyme-cuda`, LLVM 15 tools and the Enzyme plugin | nothing more |

Enzyme needs numba 0.61: numba-enzyme lowers through numba 0.61's
typed-pointer IR, which the LLVM 15 tools read, and fails on the opaque
pointers of numba 0.62 and later. That is why there are two environments on
the Mac: the numba 0.68 one (`.venv-discoeb-lock` in the modax 0.4
checkout) runs with finite differences, and the toolchain's numba 0.61 one
with Enzyme.

`MODAX_ENZYME_TOOLS` is a directory with `opt`, `llvm-link` and
`llvm-extract` (in `bin/` or `llvm15/bin/`) and `LLVMEnzyme-15.dylib` or
`.so` (in it, `enzyme/` or `lib/`). Without it modax looks in the
`numba-enzyme-cuda` wheel's `_vendor/` (Linux), then for `opt-15` etc. on
`PATH` with `NUMBA_ENZYME_PLUGIN_PATH`. `MODAX_ENZYME_KEEP=dir` keeps each
build's IR. `python -c "from modax import _enzyme_cpu; print(_enzyme_cpu.unavailable())"`
says what is missing (`None` when nothing is).

## The Mac toolchain

```bash
cpu_harness/build_enzyme_toolchain.sh            # into ~/Documents/PhD/code/enzyme-toolchain
```

downloads LLVM 15.0.7's arm64 macOS release (about 650 MB, unpacked 4.5 GB
during the build), builds Enzyme v0.0.289 against it (under a minute),
keeps the three tools and the plugin (about 115 MB), deletes the rest, and
makes `venv/` (Python 3.12, numba 0.61.2, llvmlite 0.44.0, numpy 2.2.6,
`numba-enzyme-cuda` 0.3.1 built from its sdist with
`NUMBA_ENZYME_VENDOR_FROM_PYPI=0`, and the rest of what DISCO-EB, DISCO-CMB
and the test suites import, at the versions of the numba 0.68
environment). It needs `cmake`, `uv` and the Xcode command-line tools, and
takes about six minutes. It is outside every checkout and nothing in a
repository points into it except the launcher's default.

## Running

The launcher does it all:

```bash
cd ~/Documents/PhD/code/DISCO-EB-lawrence
../modax-0.3/cpu_harness/disco -m pytest tests -q -p no:cacheprovider --benchmark-disable
MODAX_CPU_DERIVATIVES=differences ../modax-0.3/cpu_harness/disco -m pytest tests -q   # the fallback
DISCO_PYTHON=../modax/cpu_harness/.venv-discoeb-lock/bin/python ../modax-0.3/cpu_harness/disco ...  # numba 0.68
```

With `--modax 0.3` (the default) its interpreter is the toolchain's
`venv/bin/python` where that exists, with `MODAX_ENZYME_TOOLS` set to the
toolchain; otherwise, and with `--modax 0.4`, the numba 0.68 environment.
`DISCO_PYTHON` overrides either. The banner says which engine ran, e.g.
`disco: modax 0.3.0 (abc1234, cpu, enzyme derivatives, numba 0.61.2), ...`.

By hand, with an environment that has modax's dependencies without
numba-CUDA (see the README's Development section), plus `pytest`,
`pytest-benchmark` and `classy`:

```bash
export PYTHONPATH=/path/to/modax:/path/to/modax/cpu_harness:/path/to/DISCO-EB/src
python -m pytest /path/to/modax/tests -q --benchmark-disable     # modax's own suite
python run_discoeb_tests.py /path/to/DISCO-EB tests -q
```

Run modax's suite under both engines (`MODAX_CPU_DERIVATIVES=enzyme` and
`=differences`): `tests/test_enzyme_cpu.py` skips without Enzyme, and the
finite-difference checks in `tests/test_cpu_target.py` run under either.

`MODAX_SERIAL=1 NUMBA_BOUNDSCHECK=1` runs the trajectories in one thread
with bounds checking, which is the only way an out-of-bounds index or an
exception inside a kernel shows up: in the parallel loop both are silent.

## State on 2026-11-16

modax 0.3.0 with this branch, DISCO-EB 2.0.1 (`6c48e2d`), DISCO-CMB
(`41a8a5c`); Apple M4 Pro, 14 threads (numba 0.61.2, the toolchain's
environment) and an Ubuntu 24.04 sandbox on calx059 (6 threads, CPU only,
the `numba-enzyme-cuda` wheel's own tools).

- modax's suite, Enzyme: 215 passed, 5 skipped, 1 failed on the Mac and on
  calx059; the Enzyme, sensitivity, hierarchy and CPU-target files also in
  the serial bounds-checked mode (94 passed). Finite differences: 189
  passed, 31 skipped (the Enzyme file), 1 failed, the same as before the
  Enzyme path (183, 5, 1) plus the new checks; also in the numba 0.68
  environment. The failure is
  `test_an_upstream_auto_block_switches_where_its_problem_turns_stiff`,
  which asks an interpolated upstream state for 1e-4 and gets 3e-4 under
  either engine: the gate is met or missed on the step sequence (between
  8e-6 and 3e-4 as the tolerance is varied by one percent).
- DISCO-EB's suite with Enzyme: 74 passed (Mac and calx059). With
  `DISCO_THERMO_FP32_MATH=0`, `test_recfast_matches_class[N128]` still fails
  (as on every CPU variant so far); the default single-precision RECFAST
  passes.
- DISCO-CMB's suite with Enzyme: 11 passed, 2 failed. The derivative test,
  which was NaN with finite differences, now misses its gate narrowly
  (`dD_l^EE/dgrhog` at l = 1200 off by 0.0022 of the scale against 0.002)
  and passes with `DISCO_THERMO_FP32_MATH=0`; the lensing gate fails as
  before (float32 tables in the test).
- Derivatives against the P100 record on identical inputs: with double
  precision RECFAST the P(k) gradient agrees to 3e-7 and the `solve_eb`
  Jacobian's weighted gradient to 1.1e-5; with the default single-precision
  RECFAST to 9e-5 and 3.5e-2 (the GPU's own central differences disagree
  with it by 1e-3 there).
- Times on the Mac (`deriv_record.py` of the Enzyme investigation, first
  call in a new process and then warm):

  | | Enzyme | finite differences | P100 (warm) |
  |---|---|---|---|
  | `value_and_grad` of summed P(k) | 48 s, 0.24 s | 68 s, 29.3 s | 1.8 s |
  | `jacfwd` of `solve_eb` | 37 s, 0.027 s | 27 s, 1.70 s | 1.3 s |

  The warm gap is the step controller: in the earlier calx059 measurements
  the differenced joint solves rejected 3861 steps per trajectory where
  Enzyme's rejected 3.9. Building the
  twelve Enzyme derivatives is 6.8 s of the cold times (3.9 s numba
  lowering, 1.7 s Enzyme); the rest is compiling kernels, which with the
  derivative inlined take longer than with the differences.

## The CUDA path, checked on a GPU

`reference/cuda-check/` holds a comparison of the modax 0.4 CPU-target
branch (`ba31b7e`) with upstream `v0.4.0` on one Tesla P100, the two
environments differing in the modax source alone. It has not been redone for
this branch; here `test_the_source_generated_for_cuda_is_upstreams` checks,
without a GPU, that the source generated for CUDA is v0.3.0's. On 0.4:

- the branch selects the CUDA target, and its `cuda`, `types` and dispatcher
  class are numba-cuda-mlir's own objects;
- every test present in both has the same outcome (216 passed, 2 failed, 2
  skipped in each; `tests/test_cpu_target.py` skips on CUDA);
- 126 output arrays of 33 solves are bitwise identical, step counts included;
- the generated hierarchy kernel source is byte-identical for six
  hierarchies;
- warm solve times are within run-to-run scatter.

Two limits. That card needs a sandbox, Python 3.10, CUDA 12, a newer
numba-cuda-mlir and a patched Enzyme capability check, so this says nothing
of the stock path (Python 3.11+, CUDA 13, capability 7.0+). And the
three-block hierarchy sensitivities are NaN there in upstream and branch
alike (one of the two failures), so on that path "identical" means "fails
identically".

## How the target differs from CUDA

- A kernel is called once per trajectory, so a warp is one lane: where the
  lanes of a GPU warp advance an upstream block together, each trajectory
  here advances it when it needs to. Step sequences differ from a GPU's.
- With Enzyme the derivatives are CUDA's to rounding, but not bitwise: the
  IR, the math library and FMA contraction differ. With finite differences
  (`make_difference_jacobian` in `modax/rodas5P.py`,
  `modax/_differences.py`) they are good to about 1e-8 of the quantities'
  scale, and sensitivities at tolerances tighter than that do not improve.
- The CUDA Jacobian puts up to eight directions in one Enzyme call; the CPU
  one sweep per direction.
- `cache=True` is accepted and ignored: numba cannot cache generated source.
