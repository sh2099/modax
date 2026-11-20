# cpu-target-0.5 against upstream v0.5.0 on CUDA (calx069, 2026-11-20)

The same interpreter for both sides, DISCO-EB 2.1.0's stock environment
(`uv sync --extra cuda13`: Python 3.13.18, modax-solvers 0.5.0 from PyPI,
numba 0.68.1, llvmlite 0.50.0, numba-cuda-mlir 0.5.1, numba-enzyme-cuda
0.3.1, jax 0.10.2, numpy 2.5.4, cuda-bindings 13.4.1), on an NVIDIA RTX A400
(cc 8.6, 4 GB), in a bwrap Ubuntu sandbox. Two source trees from one
tarball (`scripts/make_tarball.sh`): `a/` = v0.5.0 (`11991ee`), `b/` = this
branch at `b57f7f9`. Before the env and kernel steps were rerun, `b/`'s
harness scripts were replaced by those of `ff56554` (`scripts/` and the two
`*_21.py` files; `modax/` and `tests/` untouched, checked file by file
against the commit). The logs are in `calx069/`.

| Step | Result |
|---|---|
| which modax | each side imports its own tree; the branch selects `TARGET == "cuda"` with numba-cuda-mlir's own `cuda`, and imports `modax._target` and `modax._differences` (unused on CUDA) but not `modax._enzyme_cpu` (`env_*.txt`) |
| installed wheel | the 18 `modax/*.py` of modax-solvers 0.5.0 are those of `a/`, sha256 for sha256 (`installed_vs_upstream_tree.txt`); the first run had reported "18 differ", keying the 18 AppleDouble `._*` files the macOS tarball carried as Python |
| solves (`solves.py`, 138 arrays) | bitwise identical upstream against branch in both runs, and run to run on each side; no case errored (`solves_comparison.md`) |
| generated source (16 cases) | the hierarchy kernels and every generated device function identical (`dump_*.txt`, `dump_diff.txt` empty) |
| device code (16 kernels) | identical once numba-cuda-mlir's per-process ids are masked (`asm_digests_*.txt`); raw, 11 of 16 differ between the sides, as they do between two runs of one side |
| tests (each side's own suite) | 233 passed on both; upstream 3 skipped, branch 68; the 65 differences are all the branch's `tests/test_cpu_target.py` and `tests/test_enzyme_cpu.py`, absent upstream and skipped on CUDA (`test_comparison.txt`, `pytest_*.txt`) |

**The device-code mask.** numba-cuda-mlir names each device function a
kernel calls `<qualname>_<id(dispatcher)>` (`mlir_lowering.py`,
`f"{fn.py_func.__qualname__}_{id(fn)}"`, and the same for overloads), so
symbols carry a memory address of the process, and the first run's raw
digests differed for every kernel that calls a generated device function
(all hierarchies) while the five plain Rodas5P / Tsit5 kernels agreed.
`dump_kernels.py` now replaces each live dispatcher's id by `ID<n>`, in
order of first appearance, inside the Itanium-mangled identifier it sits in
(length prefix rewritten): 0 to 12 ids a kernel, 17 labels in all, and no
`_<12+ digits>` left in either side's 11 MB of masked PTX. The remaining
12-plus-digit runs the script counts are PTX double literals and
numba-enzyme's content hashes, identical on both sides.

Two cases (`hier3_jacfwd_per_thread_1`, `onebl_tsit5_hook`) print no new
source: the kernel they launch was built earlier in the same process
(`hier3_jvp_params`'s, a plain Tsit5 kernel). Their device code is still
digested and agrees.

Times on the A400: solves about 11 min a run, the dump 3.4 min a side,
each suite under 10 min.
