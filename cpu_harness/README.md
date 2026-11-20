# DISCO-EB on modax's CPU target

modax compiles its kernels for the CPU when numba-CUDA is not installed
(`modax/_target.py`, `tests/test_cpu_target.py`). DISCO-EB also names
`numba_cuda_mlir` in its own device code, so this directory holds what an
unedited DISCO-EB checkout needs to run on that target. Nothing here is
imported by the package.

| File | What it is |
|---|---|
| `disco`, `disco_run.py` | The launcher: `disco [--modax 0.3\|0.4\|0.5] script.py \| -m module \| -c code`, with the chosen modax, this directory and the DISCO-EB / DISCO-CMB sources on `PYTHONPATH` (with 0.5, DISCO-EB 2.1 and no DISCO-CMB unless `DISCO_CMB` names it). Prints what ran on stderr, derivative engine included. |
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
| Jacobian | the CUDA branch's literal-seed writer, one direction a sweep | `make_difference_jacobian` in `modax/_rodas5P.py` |
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
cd ~/Documents/PhD/code/DISCO-EB-2.1
../modax-0.5/cpu_harness/disco --modax 0.5 -m pytest tests -q -p no:cacheprovider --benchmark-disable
MODAX_CPU_DERIVATIVES=differences ../modax-0.5/cpu_harness/disco --modax 0.5 -m pytest tests -q   # the fallback
DISCO_PYTHON=../modax/cpu_harness/.venv-discoeb-lock/bin/python ../modax-0.5/cpu_harness/disco --modax 0.5 ...  # numba 0.68
```

`--modax 0.5` runs the `modax-0.5` worktree (`MODAX_05`) with DISCO-EB 2.1
(`DISCO_EB` defaults to `DISCO-EB-2.1`); DISCO-CMB does not run on 2.1 and
is on the path only when `DISCO_CMB` names it. `--modax 0.3` (the default)
and `0.4` are as on cpu-target-0.3: DISCO-EB 2.0.1 and DISCO-CMB. With 0.3
and 0.5 the interpreter is the toolchain's `venv/bin/python` where that
exists, with `MODAX_ENZYME_TOOLS` set to the toolchain; otherwise, and with
`--modax 0.4`, the numba 0.68 environment. `DISCO_PYTHON` overrides either.
The banner says which engine ran, e.g. `disco: modax 0.5.0 (abc1234, cpu,
enzyme derivatives, numba 0.61.2), ...`. Whichever modax runs, it is this
directory's `disco_run.py` and `discoeb_cpu.py` that start it (the 0.4
checkout has neither; the 0.3 launcher runs the chosen checkout's own and
so fails for 0.4).

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

## State on 2026-11-20

modax 0.5.0 with this branch (cpu-target-0.4 merged, the Enzyme engine and
this launcher from cpu-target-0.3), DISCO-EB 2.1.0 (`6385553`); Apple M4
Pro.

- modax's suite, Enzyme (the toolchain's numba 0.61.2): 255 passed, 5
  skipped, 1 failed. Finite differences (numba 0.68.1): 211 passed, 31
  skipped (the Enzyme file among them), 19 failed. The failure common to
  both is `test_an_upstream_auto_block_switches_where_its_problem_turns_stiff`,
  as on 0.3 and 0.4. The other 18 are the differentiated solves on the
  plain Rodas5P and Tsit5 kernels (`tests/test_sensitivity.py`,
  `test_save_hook.py`, `test_params.py`, and `test_hierarchy.py`'s joint
  reference): v0.5.0 puts the directions in the parameter row after the
  parameters and narrows `n_params` to the parameters, while the
  differences read the launch's appended columns at `p[n_params + c]`, so
  they read a direction instead. Not yet fixed.
- DISCO-EB 2.1's P(k) for one LCDM cosmology at 64 modes
  (`solve_eb(..., keep=("delta_c", "delta_b"), thermo_weight=0.1)`) is
  finite under either engine, and the two agree to about 1e-5.

## The CUDA path, checked on a GPU

Not yet for this branch. `reference/cuda-check/` holds the comparison of
the modax 0.4 CPU-target branch (`ba31b7e`) with upstream `v0.4.0` on one
Tesla P100, made before the Enzyme engine; cpu-target-0.3 keeps the same
for its own branch with Enzyme (`reference/cuda-check-0.3/` there). What
this branch has is `test_the_source_generated_for_cuda_is_upstreams`, which
needs no GPU and pins the text generated for CUDA to v0.5.0's.

That card needed a sandbox, Python 3.10, CUDA 12, a newer numba-cuda-mlir
and a patched Enzyme capability check, so neither record says anything of
the stock path (Python 3.11+, CUDA 13, capability 7.0+).

## How the target differs from CUDA

- A kernel is called once per trajectory, so a warp is one lane: where the
  lanes of a GPU warp advance an upstream block together, each trajectory
  here advances it when it needs to. Step sequences differ from a GPU's.
- With Enzyme the derivatives are CUDA's to rounding, but not bitwise: the
  IR, the math library and FMA contraction differ. With finite differences
  (`make_difference_jacobian` in `modax/_rodas5P.py`,
  `modax/_differences.py`) they are good to about 1e-8 of the quantities'
  scale, and sensitivities at tolerances tighter than that do not improve.
- The CUDA Jacobian puts up to eight directions in one Enzyme call; the CPU
  one sweep per direction.
- `cache=True` is accepted and ignored: numba cannot cache generated source.
