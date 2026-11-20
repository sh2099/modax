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
| `discoeb21_derivatives.py`, `discoeb21_timing.py` | DISCO-EB 2.1's derivatives against tight central differences, and its times; the numbers are in `results/discoeb_2.1.json` and below. |
| `make_gpu_reference_21.py`, `compare_with_gpu_21.py` | The same for DISCO-EB 2.1 on modax 0.5: one script that runs unchanged on a GPU (stock environment, no stand-in) and on the CPU target, and a numpy-only comparison of two records. See "DISCO-EB 2.1 against a GPU". |
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

- modax's suite: 260 passed, 5 skipped, 1 failed with Enzyme (the
  toolchain's numba 0.61.2); 234 passed, 31 skipped (the Enzyme file), 1
  failed with finite differences, in the numba 0.68 environment, in the
  toolchain's, and serial with bounds checking. The failure is
  `test_an_upstream_auto_block_switches_where_its_problem_turns_stiff`, as on
  0.3 and 0.4. With `MODAX_POISON=1` the hierarchy's sensitivities are NaN
  too (`test_hierarchy.py` and the replica test in `test_cpu_target.py`), as
  on the 0.4 branch: the read of an unwritten upstream coefficient
  `modax/_target.py` describes.
- The differences once read the launch's columns at `p[n_params + c]`,
  which in 0.5's joint rows is the first direction; they now read them at
  the row's end (`_differences.driver_column`). On the sensitivity tests'
  problems the two engines' tangents agree to 1e-9 of their scale.

## DISCO-EB 2.1 on this branch (2026-11-20)

DISCO-EB 2.1.0 (`6385553`, unedited), the toolchain's environment, Apple M4
Pro; `results/discoeb_2.1.json` has every number. Its tests decide "GPU or
not" by importing `numba_cuda_mlir.cuda` (`tests/helpers._cuda_available`),
which is the stand-in, so the 28 tests marked `gpu` run on the CPU.
`PYTHONDONTWRITEBYTECODE=1` keeps the checkout clean.

- **Its suite with Enzyme: 138 passed, 1 skipped** (124 in the default
  selection, 14 with `-m slow`; the skip needs `ty`).
- **Derivatives with Enzyme.** `jacfwd` of P(k) and of `R` today at 64
  modes in `omega_c`, `omega_b`, `n_s`, `A_s` and `N_eff`, against central
  differences of plain solves at `rtol = atol = 1e-7`: within 1.5e-3 of each
  column's scale at the default tolerance and at 1e-6 alike (P(k): 4.8e-4,
  9.7e-4, 3.4e-5, 6.5e-5, 9.7e-4). That is the reference's own convergence
  (h 1e-3 against 3e-4: up to 5.6e-3; `rtol` 1e-7 against 1e-8: up to
  2.5e-3), not the tangents'. `split_derivatives=True` gives the same
  numbers bitwise (the launch has `sens_error_control=False`, so the steps
  are the plain solve's either way), `jax.vmap(jax.jacfwd(f))` the single
  cosmology's to 3e-7, `jit` to rounding. `jax.grad` and `jax.vjp` raise
  modax's forward-mode-only `NotImplementedError`.
- **Times** (first call in a new process, then warm): P(k) of one
  cosmology at 64 modes 16 s, 0.009 s; 128 cosmologies 17 s, 0.47 s.
  `jacfwd` in 1, 2 and 5 parameters 26-28 s cold, then 0.022, 0.030 and
  0.049 s warm: about 0.007 s a parameter beyond the first. With finite
  differences: 0.016 s, 1.15 s, and 0.096, 0.16, 0.27 s.
- **Finite differences are values only for DISCO-EB 2.1.** Its suite: 130
  passed, 8 failed. Six derivative tests miss their gates, by from 10
  percent (`test_matter_power_spectrum_derivatives`, both widths) to 33
  orders of magnitude (`test_more_parameters_matter_power[z_re]`); two value
  tests miss by step-sequence margins
  (`test_recfast_matches_class[N128]`: three cosmologies straddle the T_m
  switch against a gate of two; a background field read at eight modes
  agrees to 1.02e-6 against 1e-6). The derivatives go wrong in the
  recombination block: alone, with DISCO-EB's own inputs, its `x_H`
  tangent is 34 times its scale off at the onset of hydrogen recombination,
  where Enzyme is at the reference's accuracy (5e-2 there). RECFAST
  switches on the state (`x_H` clamped at 1, `x_H` thresholds at 0.99999 and
  0.9999999, the helium freeze at 1e-8, the Compton lock of `T_m`), and the
  second differences step a state by `eps**(1/4)` of the *largest* state
  (here `kappa`, about 1e6), straddling them: a hundred-times-shorter
  second step brings `x_H` to 4e-2, a ten-times-longer one to 1, and
  neither helps the other states. DISCO-EB runs its
  joint launches with `sens_error_control=False`, so nothing rejects such a
  step; with it on, the recombination block's tangents are Enzyme's, but the
  whole system's differenced joint solve then never finishes. The error
  depends on each mode's step sequence, which is why it is erratic in `k`.
  No change local to the CPU layer fixed it (one-sided differences across a
  jump, per-state step floors and other step sizes were each tried and
  measured); use Enzyme for DISCO-EB 2.1 derivatives.

## DISCO-EB 2.1 against a GPU

`make_gpu_reference_21.py OUT.npz` writes one record of fixed inputs, all
in the file (a seeded generator, fixed wave numbers and scale factors): P(k)
of 32 perturbed LCDM cosmologies x 64 modes at the default thermal weight and
at 0.1, of the reference cosmology, one `LCDM_MNU` and one `W0WACDM`; 18 kept
quantities of the reference over 48 scale factors at 8 modes (states, the
metric's `R`, `alpha`, `aH`, and the thermal history `tau`, `xe`, `Tmat`,
`cs2`, `opacity`, `kappa`), and of `LCDM_MNU` with neutrino multipoles;
`jacfwd` of P(k) in `omega_c`, `omega_b`, `n_s`, `A_s`, `N_eff` with and
without `split_derivatives`, a `jvp`, `vmap(jacfwd)` and `vmap` over three
cosmologies beside the same one at a time; a reionization case and its
`jvp` in `z_re`; per-trajectory step counts of the plain and the joint
launch; versions, device, target and cold and warm times per section. On a
4 GB card it stays at 2048 trajectories a launch and five columns.
On the Mac it takes about 4 minutes (Enzyme); `results/cpu_reference_2.1.npz`
(280 kB) is that record, the CPU side of the comparison.

```bash
python make_gpu_reference_21.py gpu_reference_2.1.npz                                # on the GPU
../modax-0.5/cpu_harness/disco --modax 0.5 ../modax-0.5/cpu_harness/make_gpu_reference_21.py cpu.npz
python compare_with_gpu_21.py cpu.npz gpu_reference_2.1.npz [out.json]               # numpy only
```

The comparison measures each element against the largest magnitude of its
series (one quantity at one mode over the scale factors, one P(k) curve,
one derivative column) and reports it in units of `rtol` (1e-4); its
docstring gives and justifies the gates. Two Enzyme records made on the Mac
are bitwise identical except where noted below; the differences engine
against Enzyme passes the values and flags the derivatives, as above.

**`vmap` over cosmologies reads error weights past their end.** solve_eb
gives modax `(n_traj, n_vars)` error weights for one cosmology's 64 modes;
`jax.vmap` over three cosmologies folds 192 trajectories into one launch,
and modax's `prepare_inputs` passes a 2-D array through without checking it
against the ensemble (upstream code, `_jax_common.py` and
`build_error_weights` in `_numba_common.py`, unchanged here). The kernel then
reads the weights of trajectories 64 and beyond from past the array: on the
CPU target the vmapped `jacfwd`'s rows after the first are garbage, different
in every process. modax alone shows it with any 2-D `error_weights` under
`jax.vmap(jax.jacfwd(f))`. `jax.vmap` of the plain P(k) over the same three
raises modax's "vmap over an already-ensembled y0 is not supported", so the
derivative is the only batched path, and it is DISCO-EB's documented one.
The record keeps both forms, and the comparison prints the self-check; what
a GPU reads there is for its record to say.

## The CUDA path of this branch against upstream (`reference/cuda-check-0.5/`)

`scripts/` holds what shows that on CUDA this branch is upstream v0.5.0: one
interpreter (DISCO-EB 2.1's stock environment) and two source trees put
first on the path, `a/` = v0.5.0 and `b/` = this branch, from one tarball
(`scripts/make_tarball.sh OUT.tar.gz`). `runner.py` refuses a modax that is
not the chosen tree and checks the side (`b` has `modax._target` with
`TARGET == "cuda"` and numba-cuda-mlir's own `cuda`; `a` has no `_target`).
Every run has an empty `NUMBA_CACHE_DIR` of its own under the output
directory: modax's on-disk kernel cache keys a kernel on a digest of its
generated source, closures, globals and the numba-cuda-mlir/numba-enzyme
versions, not on modax's version, so two trees generating the same kernel
would load each other's. Drivers, each `PY A B OUT`:

| | What | Compared |
|---|---|---|
| `d2_env.sh` | which modax, versions, device; the installed wheel against `a/` file by file (`check_installed.py`) | |
| `d2_solves.sh` | `solves.py` (Robertson, Lorenz, the hierarchy chain, a split, a sparse chain, save hooks, jvp and jacfwd with and without `max_directions_per_thread`, and a small DISCO-EB P(k) and jacfwd) upstream, branch, upstream, branch | bitwise, `cmp_npz.py`: across sides and run to run |
| `d2_kernels.sh` | `dump_kernels.py`: hierarchy kernel source, every generated device function, and a sha256 of each launched kernel's `inspect_asm` / `inspect_llvm` if numba-cuda-mlir gives one | `diff -r` |
| `d2_tests.sh` | each side's own `pytest tests -v -rA` | test by test, `cmp_tests.py` |

`MC_LOCAL_TEST=1` lets both trees be the branch on its CPU target, which is
how these were tested on the Mac: the logic, the cache separation and the
comparisons, not anything CUDA does.

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
