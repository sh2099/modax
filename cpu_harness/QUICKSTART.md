# Quick start: DISCO-EB on the CPU (Apple Silicon Mac)

Two stacks, one launcher. Neither edits DISCO-EB.

| Stack | DISCO-EB | modax branch (this fork) |
|---|---|---|
| latest | v2.1.0 | `cpu-target-0.5` (modax 0.5.0 + CPU target) |
| earlier | v2.0.1, with DISCO-CMB 0.1.0 | `cpu-target-0.3` (modax 0.3.0 + CPU target) |

Needs: an Apple Silicon Mac, `git`, `uv`, `cmake` and the Xcode command-line
tools. About 1 GB of disk once built (4.5 GB during the build) and about ten
minutes.

## Set up

Everything goes side by side in one directory, which is where the launcher
looks by default.

```bash
mkdir disco-cpu && cd disco-cpu

# modax: both CPU branches
git clone -b cpu-target-0.5 https://github.com/sh2099/modax.git modax-0.5
git -C modax-0.5 worktree add ../modax-0.3 cpu-target-0.3

# DISCO-EB: both releases, unedited
git clone https://github.com/cosmo-sims/DISCO-EB.git DISCO-EB-2.1
git -C DISCO-EB-2.1 checkout -q v2.1.0
git -C DISCO-EB-2.1 worktree add ../DISCO-EB-2.0.1 v2.0.1

# LLVM 15 tools, the Enzyme plugin and a Python environment, into ./enzyme-toolchain
modax-0.5/cpu_harness/build_enzyme_toolchain.sh
```

For CMB spectra on the earlier stack, also clone DISCO-CMB beside them as
`DISCO-CMB`.

## Run

```bash
export PYTHONDONTWRITEBYTECODE=1     # keeps the checkouts clean

# latest: DISCO-EB 2.1.0
modax-0.5/cpu_harness/disco --modax 0.5 modax-0.5/cpu_harness/examples/pk_2.1.py

# earlier: DISCO-EB 2.0.1
modax-0.5/cpu_harness/disco --modax 0.3 modax-0.5/cpu_harness/examples/pk_2.0.1.py
```

Each prints P(k) at its peak and a derivative, with the time of the first
call (which compiles the kernel) and of a warm one. On an M4 Pro the first
P(k) call takes about 17 s and a warm one about 0.01 s. The first line says
what ran, e.g. `disco: modax 0.5.0 (81d5417, cpu, enzyme derivatives, numba
0.61.2), discoeb source (6385553)`.

`disco` takes what `python` takes: a script, `-m module` or `-c code`. Your
own scripts need no change: the launcher supplies the `numba_cuda_mlir` name
DISCO-EB imports before the script starts.

The test suites:

```bash
(cd DISCO-EB-2.1   && ../modax-0.5/cpu_harness/disco --modax 0.5 -m pytest tests -p no:cacheprovider --benchmark-disable)
(cd DISCO-EB-2.0.1 && ../modax-0.5/cpu_harness/disco --modax 0.3 -m pytest tests -p no:cacheprovider --benchmark-disable)
```

Expected on an M4 Pro: 138 passed and 1 skipped for 2.1.0, 74 passed for
2.0.1.

## Know before use

- **Batched derivatives on 2.1.0:** `jax.vmap(jax.jacfwd(f))` over
  cosmologies is wrong on modax 0.5.0, on a GPU too
  ([lawrenceberry/modax#2](https://github.com/lawrenceberry/modax/issues/2)).
  This branch does not patch it. A Python loop, or a list of cosmologies
  given to `solve_eb`, is right.
- **Derivatives need the toolchain.** Without it the CPU target falls back
  to finite differences: fine for 2.0.1, values only for 2.1.0.
- **`jax.grad` raises on 2.1.0**, as on a GPU: modax 0.5 is forward mode
  only.
- **DISCO-CMB 0.1.0 does not run on DISCO-EB 2.1.0**, on any platform.
- **Linux:** `build_enzyme_toolchain.sh` is for macOS. The 0.5 stack has
  only been run on a Mac.
- **Other layouts:** `MODAX_03`, `MODAX_05`, `DISCO_EB`, `DISCO_CMB`,
  `ENZYME_TOOLCHAIN` and `DISCO_PYTHON` override each location.

`README.md` beside this file has the details: the two derivative engines,
measured times, agreement with a GPU, and the check that the fork leaves
modax's CUDA path unchanged.
