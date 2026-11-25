# Quick start: DISCO-EB on the CPU

Two stacks, one launcher. Neither edits DISCO-EB.

| Stack | DISCO-EB | modax branch (this fork) |
|---|---|---|
| latest | v2.1.0 | `cpu-target-0.5` (modax 0.5.0 + CPU target) |
| earlier | v2.0.1, with DISCO-CMB 0.1.0 | `cpu-target-0.3` (modax 0.3.0 + CPU target) |

On an Apple Silicon Mac (Linux is at the end). Needs: `git`, `uv`, `cmake` and the Xcode command-line
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

Expected on an M4 Pro: 124 passed and 1 skipped for 2.1.0 (138 with its
slow tests, `-m slow`), 74 passed for 2.0.1.

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
- **DISCO-EB after v2.1.0:** `main` at `a7aeff0` (2026-11-20) runs too. On
  the Mac one test of its suite fails, `test_recfast_matches_class[N128]`:
  one cosmology's matter-temperature step straddles the RECFAST switch just
  outside the window the test allows. It passes on Linux and on a GPU.
- **Other layouts:** `MODAX_03`, `MODAX_05`, `DISCO_EB`, `DISCO_CMB`,
  `ENZYME_TOOLCHAIN` and `DISCO_PYTHON` override each location.

## Linux (x86-64)

No toolchain to build: `numba-enzyme-cuda`'s Linux wheel carries the LLVM 15
tools and the Enzyme plugin, and modax finds them there. The wheel needs
glibc 2.38 or newer (Ubuntu 24.04; not RHEL 9). Clone as above, then in
place of `build_enzyme_toolchain.sh`:

```bash
uv venv --python 3.12 venv
uv pip install --python venv/bin/python \
    numba==0.61.2 llvmlite==0.44.0 numpy==2.2.6 jax==0.10.2 jaxlib==0.10.2 \
    scipy==1.18.0 networkx==3.6.1 cvxopt==1.3.3 classy==3.3.4.0 \
    pytest==9.1.1 pytest-benchmark==5.2.3 pytest-regressions==2.11.0
uv pip install --python venv/bin/python --no-deps numba-enzyme-cuda==0.3.1
export DISCO_PYTHON=$PWD/venv/bin/python
```

and run as above. `--no-deps` keeps the CUDA packages out, so modax takes
the CPU target; beside a CUDA stack set `MODAX_TARGET=cpu`. Run on
2026-11-25 on a 20-thread x86 desktop in an Ubuntu 24.04 sandbox, from this
branch on GitHub: both examples give the Mac's numbers with Enzyme
derivatives, and DISCO-EB 2.1.0's suite gives 124 passed, 1 skipped (its 14
slow tests not run).

`README.md` beside this file has the details: the two derivative engines,
measured times, agreement with a GPU, and the check that the fork leaves
modax's CUDA path unchanged.
