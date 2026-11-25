# DISCO-EB on modax's CPU target

To set it up and run it from nothing, see [`QUICKSTART.md`](QUICKSTART.md).

modax compiles its kernels for the CPU when numba-CUDA is not installed
(`modax/_target.py`, `tests/test_cpu_target.py`). DISCO-EB also names
`numba_cuda_mlir` in its own device code, so this directory holds what an
unedited DISCO-EB checkout needs to run on that target. Nothing here is
imported by the package.

This branch, `cpu-target-0.5`, is modax 0.5.0 with the CPU target of
cpu-target-0.4 merged in and the Enzyme engine and launcher of
cpu-target-0.3 ported. It runs DISCO-EB 2.1.0, which is written for modax
0.5. The launcher still runs the 0.3 and 0.4 branches with DISCO-EB 2.0.1.

| File | What it is |
|---|---|
| `disco`, `disco_run.py` | The launcher: `disco [--modax 0.3\|0.4\|0.5] script.py \| -m module \| -c code`. See "Running". |
| `discoeb_cpu.py` | Puts modax's CPU meanings of `cuda.jit`, `cuda.local.array`, `cuda.is_available` and the float types under the name `numba_cuda_mlir`. Call `install()` before `discoeb` builds a kernel; the launcher does. |
| `build_enzyme_toolchain.sh` | Builds, on an Apple Silicon Mac, the LLVM 15 tools, the Enzyme plugin and a numba 0.61 environment that Enzyme on the CPU needs. |
| `run_discoeb_tests.py` | Runs DISCO-EB's own test suite on the CPU target without the launcher. |
| `discoeb21_derivatives.py`, `discoeb21_timing.py` | DISCO-EB 2.1's derivatives against tight central differences, and its times. Numbers in `results/discoeb_2.1.json`. |
| `make_gpu_reference_21.py`, `compare_with_gpu_21.py` | A DISCO-EB 2.1 record that runs unchanged on a GPU and on the CPU target, and a numpy-only comparison of two records. |
| `make_gpu_reference.py`, `compare_with_gpu.py` | The same for DISCO-EB 2.0.1 on modax 0.3 (from cpu-target-0.4). |
| `results/` | `discoeb_2.1.json`, `cpu_reference_2.1.npz` and `gpu_comparison_2.1.json` (this branch); `gpu_comparison.json` (DISCO-EB 2.0.1, from cpu-target-0.4). |
| `reference/` | `gpu_reference_2.1.npz`, the GPU record of DISCO-EB 2.1; `cuda-check-0.5/`, the check of this branch's CUDA path against v0.5.0 (scripts, `RESULTS.md`, the calx069 logs); `upstream-issues/`, problems found in upstream modax; and, from cpu-target-0.4, `cuda-check/` and the DISCO-EB 2.0.1 / modax 0.3.0 P100 record's package list and test log. |

## Two derivative engines

The CPU target differentiates a user's function in one of two ways, chosen
at import by `MODAX_CPU_DERIVATIVES`:

| | Enzyme (`modax/_enzyme_cpu.py`) | Finite differences (`modax/_differences.py`) |
|---|---|---|
| Chosen by | `auto` (the default) where Enzyme can run; `enzyme` | `auto` where it cannot (with one warning saying why); `differences` |
| Jacobian | the CUDA branch's literal-seed writer, one direction a sweep | `make_difference_jacobian` in `modax/_rodas5P.py` |
| Sensitivity sweeps | Enzyme forward mode, second order forward over forward | central differences |
| Accuracy | rounding, as on CUDA | about 1e-8 of a quantity's scale on smooth functions |
| `cuda.local.array` | a zeroed stack array | `np.zeros` |
| Needs | numba 0.61.x, llvmlite 0.44.x, `numba-enzyme-cuda`, LLVM 15 tools and the Enzyme plugin | nothing more |
| DISCO-EB 2.1 | values and derivatives | **values only** (below) |

Enzyme needs numba 0.61: numba-enzyme lowers through numba 0.61's
typed-pointer IR, which the LLVM 15 tools read, and fails on the opaque
pointers of numba 0.62 and later. So there are two environments on the
Mac: the numba 0.68 one (`.venv-discoeb-lock` in the modax 0.4 checkout),
which runs with finite differences, and the toolchain's numba 0.61 one,
which runs Enzyme.

`MODAX_ENZYME_TOOLS` is a directory with `opt`, `llvm-link` and
`llvm-extract` (in `bin/` or `llvm15/bin/`) and `LLVMEnzyme-15.dylib` or
`.so` (in it, `enzyme/` or `lib/`). Without it modax looks in the
`numba-enzyme-cuda` wheel's `_vendor/` (Linux), then for `opt-15` etc. on
`PATH` with `NUMBA_ENZYME_PLUGIN_PATH`. `MODAX_ENZYME_KEEP=dir` keeps each
build's IR. `python -c "from modax import _enzyme_cpu; print(_enzyme_cpu.unavailable())"`
says what is missing (`None` when nothing is).

In a joint (differentiated) launch, modax 0.5 stores each direction's `dp`
in the parameter row after the parameters. The launch appends three columns
for the differences (`atol / rtol`, the first step, `rtol`), and they are
read from the row's end (`_differences.driver_column`). Before that fix they
were read at `p[n_params + c]`, which is a direction, and every differenced
derivative on the plain kernels was wrong.

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

```bash
cd ~/Documents/PhD/code/DISCO-EB-2.1
export PYTHONDONTWRITEBYTECODE=1        # keeps the checkout clean
../modax-0.5/cpu_harness/disco --modax 0.5 -m pytest tests -p no:cacheprovider --benchmark-disable
../modax-0.5/cpu_harness/disco --modax 0.5 -m pytest tests -m slow -p no:cacheprovider --benchmark-disable
MODAX_CPU_DERIVATIVES=differences ../modax-0.5/cpu_harness/disco --modax 0.5 script.py
```

- `--modax 0.5` runs the `modax-0.5` worktree (`MODAX_05` overrides it) with
  DISCO-EB 2.1 (`DISCO_EB` defaults to `DISCO-EB-2.1`). DISCO-CMB does not
  run on 2.1 and is on the path only when `DISCO_CMB` names it.
- `--modax 0.3` (the default) and `0.4` run those branches' worktrees with
  DISCO-EB 2.0.1 (`DISCO-EB-2.0.1` where that exists, else
  `DISCO-EB-lawrence`) and DISCO-CMB, as cpu-target-0.3's launcher does.
- With 0.3 and 0.5 the interpreter is the toolchain's `venv/bin/python`
  where that exists, with `MODAX_ENZYME_TOOLS` set to the toolchain.
  Otherwise, and with 0.4, it is the numba 0.68 environment. `DISCO_PYTHON`
  overrides either.
- The banner says what ran, e.g. `disco: modax 0.5.0 (abc1234, cpu, enzyme
  derivatives, numba 0.61.2), discoeb source (6385553)`.
- This directory's `disco_run.py` and `discoeb_cpu.py` start every version.
  The 0.4 checkout has neither, so cpu-target-0.3's own launcher fails for
  `--modax 0.4`.

DISCO-EB's tests decide whether there is a GPU by importing
`numba_cuda_mlir.cuda` (`tests/helpers._cuda_available`). That import finds
the stand-in, so the 28 tests marked `gpu` run on the CPU.

modax's own suite, in either environment:

```bash
cd ~/Documents/PhD/code/modax-0.5
MODAX_TARGET=cpu python -m pytest tests -q -p no:cacheprovider --benchmark-disable
```

Run it under both engines. `tests/test_enzyme_cpu.py` skips without
Enzyme, and the finite-difference checks in `tests/test_cpu_target.py` run
under either. `MODAX_SERIAL=1 NUMBA_BOUNDSCHECK=1` runs the trajectories in
one thread with bounds checking. That is the only way an out-of-bounds
index or an exception inside a kernel shows up: in the parallel loop both
are silent. `MODAX_POISON=1` starts every local array as NaN.

## Results (2026-11-20, Apple M4 Pro)

modax 0.5.0 with this branch, and DISCO-EB 2.1.0 (`6385553`, unedited).
Every number is also in `results/discoeb_2.1.json` and
`results/gpu_comparison_2.1.json`.

**modax's suite.**
- Enzyme (the toolchain's numba 0.61.2): 260 passed, 5 skipped, 1 failed.
- Finite differences: 234 passed, 31 skipped (the Enzyme file among them),
  1 failed. This holds in the numba 0.68 environment, in the toolchain's,
  and serial with bounds checking.
- The one failure is `test_an_upstream_auto_block_switches_where_its_problem_turns_stiff`,
  as on 0.3 and 0.4. It asks an interpolated upstream state for 1e-4 and
  gets 3e-4, a gate met or missed on the step sequence.
- With `MODAX_POISON=1` the hierarchy's sensitivities are NaN too
  (`test_hierarchy.py` and a replica test in `test_cpu_target.py`), as on
  the 0.4 branch. This is the read of an unwritten upstream coefficient
  that `modax/_target.py` describes.

**DISCO-EB 2.1's suite with Enzyme: 138 passed, 1 skipped.** That is 124
in the default selection and 14 with `-m slow`. The skip needs `ty`.

**Derivatives with Enzyme.**
- What was measured: `jacfwd` of P(k) and of `R` today, at 64 modes, in
  `omega_c`, `omega_b`, `n_s`, `A_s` and `N_eff`. The reference is central
  differences of plain solves at `rtol = atol = 1e-7`.
- Agreement: within 1.5e-3 of each column's scale, at the default tolerance
  and at 1e-6 alike. For P(k) the columns agree to 4.8e-4, 9.7e-4, 3.4e-5,
  6.5e-5 and 9.7e-4.
- What limits the check: the reference's own convergence (h 1e-3 against
  3e-4: up to 5.6e-3; `rtol` 1e-7 against 1e-8: up to 2.5e-3), not the
  tangents.
- `split_derivatives=True` gives the same numbers bitwise. The launch has
  `sens_error_control=False`, so its steps are the plain solve's.
- `jit` agrees to rounding. `jax.grad` and `jax.vjp` raise modax's
  forward-mode-only `NotImplementedError`.
- `jax.vmap(jax.jacfwd(f))` over cosmologies is wrong after the first
  cosmology, on the CPU and on a GPU alike ("Upstream issues").

**Times** (first call in a new process, then warm).

| | Enzyme | finite differences |
|---|---|---|
| P(k), 1 cosmology x 64 modes | 16 s, 0.009 s | 18 s, 0.016 s |
| P(k), 128 cosmologies x 64 modes | 17 s, 0.47 s | 19 s, 1.15 s |
| `jacfwd`, 1 / 2 / 5 parameters, warm | 0.022 / 0.030 / 0.049 s | 0.096 / 0.16 / 0.27 s |

Each `jacfwd` takes 26-30 s on its first call. Each parameter after the
first adds about 0.007 s warm with Enzyme.

**Finite differences give DISCO-EB 2.1's values but not its derivatives.**

- The suite with `MODAX_CPU_DERIVATIVES=differences`: 130 passed, 8 failed,
  1 skipped.
- Six of the failures are derivative tests. They miss their gates by 10
  percent (`test_matter_power_spectrum_derivatives`, both widths), by
  factors of 2.4 (`test_components_from_outside_leave_the_solve_unchanged`),
  9.5 (`test_reionization_derivative`) and 190
  (`test_solve_eb_derivatives`), and by 33 orders of magnitude
  (`test_more_parameters_matter_power[z_re]`).
- Two are value tests, missed by step-sequence margins.
  `test_recfast_matches_class[N128]` has three cosmologies straddling the
  T_m switch against a gate of two. A background field read at eight modes
  agrees to 1.02e-6 against 1e-6.
- Where it goes wrong: the recombination block. Alone, with DISCO-EB's own
  inputs, its `x_H` tangent is 34 times its scale off at the onset of
  hydrogen recombination. Enzyme is at the reference's accuracy there (5e-2).
- Why: RECFAST switches on the state (`x_H` clamped at 1, `x_H` thresholds
  at 0.99999 and 0.9999999, the helium freeze at 1e-8, the Compton lock of
  `T_m`). The second differences step a state by `eps**(1/4)` of the
  *largest* state (here `kappa`, about 1e6), so they straddle those
  switches. A hundred-times-shorter second step brings `x_H` to 4e-2 of its
  scale and a ten-times-longer one to 1.0, and neither helps the other
  states.
- DISCO-EB launches with `sens_error_control=False`, so nothing rejects such
  a step. With error control on, the recombination block's tangents match
  Enzyme's, but the whole system's differenced joint solve breaks down.
- The error depends on each mode's step sequence, which is why it is
  erratic in `k`.
- No change local to the CPU layer fixed it; one-sided differences across a
  jump, per-state step floors and other step sizes were each tried and
  measured. Use Enzyme for DISCO-EB 2.1 derivatives. Whether the launcher
  should warn about this is still to be decided.

## DISCO-EB 2.1 against a GPU

**What the record holds.** `make_gpu_reference_21.py OUT.npz` writes one
record from fixed inputs, all stored in the file: a seeded generator and
fixed wave numbers and scale factors.
- P(k): 32 perturbed LCDM cosmologies x 64 modes, at the default thermal
  weight and at 0.1; the reference cosmology; one `LCDM_MNU` and one
  `W0WACDM` cosmology.
- Histories: 18 kept quantities of the reference over 48 scale factors at 8
  modes, namely states, `R`, `alpha`, `aH`, and the thermal history (`tau`,
  `xe`, `Tmat`, `cs2`, `opacity`, `kappa`). Also an `LCDM_MNU` history.
- Derivatives: `jacfwd` of P(k) in five parameters, with and without
  `split_derivatives`; a `jvp`; `vmap(jacfwd)` over three cosmologies beside
  the same three one at a time.
- A reionization case and its `jvp` in `z_re`.
- Step counts, versions and timings.

```bash
python make_gpu_reference_21.py gpu_reference_2.1.npz                                # on the GPU
../modax-0.5/cpu_harness/disco --modax 0.5 ../modax-0.5/cpu_harness/make_gpu_reference_21.py cpu.npz
python compare_with_gpu_21.py cpu.npz gpu_reference_2.1.npz [out.json]               # numpy only
```

**How the comparison scores.** It reports each array in units of `rtol`
(1e-4):
- kept quantities are measured against the quantity's largest magnitude over
  all modes and scale factors, with the per-curve number printed beside;
- P(k) is measured per curve, and derivatives per column;
- values may differ by 10 `rtol` at most (median 1), derivatives by 50
  (median 5). Its docstring justifies both.

Two Enzyme records made on the Mac are bitwise identical, except for the
`vmap(jacfwd)` rows that read memory past an array.

**The GPU record** (`reference/gpu_reference_2.1.npz`):
- Machine: calx069, NVIDIA RTX A400 (cc 8.6, 4 GB).
- Environment: DISCO-EB 2.1.0's stock CUDA environment, `uv sync --extra
  cuda13` (Python 3.13, modax-solvers 0.5.0 from PyPI, jax 0.10.2, numba
  0.68.1, numba-cuda-mlir 0.5.1).
- 811 s.

**Against the CPU record** (`results/gpu_comparison_2.1.json`): 46 arrays,
two flagged.
- P(k): 3.5 `rtol` at most (32 cosmologies at `thermo_weight = 0.1`), 0.3
  at the default weight.
- `jacfwd` of P(k): 0.71. The reionization tangent: 1.0 (2.8 per curve).
- The kept quantities, each against its own scale: `xe` 1.6; the
  contrasts, `theta_b`, `R`, `alpha`, `tau`, `Tmat`, `kappa` and `cs2`
  under 0.2; the reionization values 0.05.
- Accepted steps per trajectory: 96.4 against 96.2 (plain) and 113.9 against
  113.7 (joint).
- Flag 1, `vmap_jacfwd_pk`: the error-weights bug below, garbage on both
  sides. vmap against one at a time is 1.0 of the scale on the CPU and
  1.5e60 on the GPU.
- Flag 2, `hist_planck` (11.9 `rtol`): `F_4` at k = 0.035, a = 0.01, where
  the two records are 5.2e-4 apart at a value of 0.022. That is five times
  the local `atol + rtol |y|`, and the largest difference of any kept
  quantity. Measured per curve, near-zero series read worse (`E_2` at the
  lowest mode peaks at 2e-5 of its scale, `theta_b` at 4e-11), but those
  are rounding.

`results/gpu_comparison_2.1.json` also keeps the plain vmap attempt's
error.

## The CUDA path of this branch against upstream (`reference/cuda-check-0.5/`)

`scripts/` shows that on CUDA this branch is upstream v0.5.0.
- The setup: one interpreter (DISCO-EB 2.1's stock environment) and two
  source trees put first on the path, `a/` = v0.5.0 and `b/` = this branch.
  Both come from one tarball (`scripts/make_tarball.sh OUT.tar.gz`).
- `runner.py` refuses a modax that is not the chosen tree and checks the
  side: `b` has `modax._target` with `TARGET == "cuda"` and numba-cuda-mlir's
  own `cuda`; `a` has no `_target`.
- Every run has an empty `NUMBA_CACHE_DIR` of its own. modax's on-disk
  kernel cache keys a kernel on a digest of its generated source, closures,
  globals and the numba-cuda-mlir/numba-enzyme versions, not on modax's
  version. Two trees generating the same kernel would otherwise load each
  other's.

| Driver (`PY A B OUT`) | What | Compared |
|---|---|---|
| `d2_env.sh` | which modax, versions, device; the installed wheel against `a/` file by file | |
| `d2_solves.sh` | `solves.py`: Robertson, Lorenz, the hierarchy chain, a split, a sparse chain, save hooks, jvp and jacfwd with and without `max_directions_per_thread`, a small DISCO-EB P(k) and jacfwd; upstream, branch, upstream, branch | bitwise: across sides and run to run |
| `d2_kernels.sh` | generated hierarchy kernels and device functions, and a digest of each launched kernel's device code with its per-process ids masked | `diff -r` |
| `d2_tests.sh` | each side's own `pytest tests -v -rA` | test by test |

`MC_LOCAL_TEST=1` lets both trees be the branch on its CPU target. That is
how the scripts were tested on the Mac: 136 of 136 arrays bitwise
identical, 48 dump files identical, and the suites' outcomes the same test
for test. That exercises the logic, not CUDA.

**On calx069** (RTX A400, DISCO-EB 2.1's stock environment, 2026-11-20;
`reference/cuda-check-0.5/RESULTS.md`, logs in `calx069/`), the branch on
CUDA is upstream v0.5.0:
- each side imports its own tree, and the branch selects the CUDA target
  without importing `modax._enzyme_cpu`; the installed modax-solvers 0.5.0
  is `a/` file for file;
- the 138 arrays of `solves.py` are bitwise identical across the sides, in
  both runs, and run to run;
- the generated source of all 16 dump cases is identical, and so is their
  device code once numba-cuda-mlir's per-process ids are masked
  (`<qualname>_<id(dispatcher)>` in every called function's symbol; raw, 11
  of 16 differ between any two processes);
- each side's suite: 233 passed on both; the only differences are the
  branch's own CPU-target test files, skipped on CUDA.

Older records: `reference/cuda-check/` compares the modax 0.4 CPU-target
branch with v0.4.0 on a Tesla P100, made before the Enzyme engine;
cpu-target-0.3 keeps its own `cuda-check-0.3/`. That card needed Python
3.10, CUDA 12 and a patched Enzyme capability check, so those records say
nothing of the stock path. Independently of any GPU,
`test_the_source_generated_for_cuda_is_upstreams` pins the text this branch
generates for CUDA to v0.5.0's.

## Upstream issues

`reference/upstream-issues/vmap_error_weights.{md,py}`: given
`(N, n_vars)` error weights, which `solve_eb` always passes,
`jax.vmap(jax.jacfwd(f))` or `jax.vmap(f)` folds `axis_size * N`
trajectories into one launch while the weights keep `N` rows. The kernel
reads the rest from past the array. This is modax v0.5.0's code, left as
it is on this branch. The `.py` reproduces it with modax alone; the `.md`
says where the code is and suggests a fix. In DISCO-EB, `jax.vmap` of a
plain `solve_eb` raises first ("vmap over an already-ensembled y0"), so
`vmap(jacfwd)`, the batch form its docs recommend, is the path that hits
it.

## Not covered

- DISCO-CMB: it does not run on DISCO-EB 2.1. Its results on the CPU target
  are for DISCO-EB 2.0.1 on cpu-target-0.3.
- A P100 or another pre-Volta card with this branch.
- Finite-difference derivatives of DISCO-EB 2.1 (above).

## How the target differs from CUDA

- A kernel is called once per trajectory, so a warp is one lane. Where the
  lanes of a GPU warp advance an upstream block together, each trajectory
  here advances it when it needs to, so step sequences differ from a GPU's.
- With Enzyme the derivatives are CUDA's to rounding, but not bitwise: the
  IR, the math library and FMA contraction differ.
- The CUDA Jacobian puts up to eight directions in one Enzyme call; the CPU
  makes one sweep per direction.
- `cache=True` is accepted and ignored: numba cannot cache generated source.
