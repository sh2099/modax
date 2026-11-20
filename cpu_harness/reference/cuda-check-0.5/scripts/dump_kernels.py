"""Dump the generated kernel source of fixed problems, and the device code where it can be read.

    dump_kernels.py OUTDIR

Per case:
  <case>.py.txt   the hierarchy kernel's source (MODAX_HIERARCHY_DUMP makes
                  make_hierarchy_kernel print it; stdout is captured), one-block
                  hierarchies (Auto) included;
  <case>.dev.txt  every device function built from source through
                  modax._codegen.compile_device_source during the solve (the
                  literal-seed Jacobian writers, the sparse LU), wrapped where
                  each module imported it;
  <case>.asm.txt  per kernel launched and signature, the sha256 of the device
                  code numba-cuda-mlir gives (``inspect_asm``, else
                  ``inspect_llvm``) with its per-process ids masked, the number
                  of ids masked and of 12-plus-digit runs left (the raw sha256s,
                  which differ between processes, go to OUTDIR/../<dir>_raw_digests.txt);
  <case>.ptx.txt  that masked device code itself, to diff where digests differ.

numba-cuda-mlir names a device function it calls ``<qualname>_<id(dispatcher)>``
(mlir_lowering.py, ``f"{fn.py_func.__qualname__}_{id(fn)}"``, and the same for
overloads), so every symbol carries a memory address of this process and the
raw device code differs between two runs of the same tree. The mask finds
every live dispatcher, and replaces its id, wherever it appears, by ``ID<n>``
in order of first appearance, rewriting the Itanium-mangled identifier
(``<length><name>``) it sits in so the length prefix stays consistent.
Anything else that changed would still change the masked digest.
The source dumps are the same text whichever target compiles them; the asm
lines exist only on CUDA (the CPU target builds no CudaLaunch).
"""

import contextlib
import gc
import hashlib
import io
import math
import os
import re
import sys

os.environ["MODAX_HIERARCHY_DUMP"] = "1"
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)
import modax._codegen as cg  # noqa: E402
import modax._solve as solve_module  # noqa: E402
from modax import Auto, Block, Hierarchy, Rodas5P, Tsit5, solve  # noqa: E402

DEV = []
ASM = []
RAW = []
PTX = []
LONG_DIGITS = re.compile(r"(?<![0-9])[0-9]{12,}(?![0-9])")


def _mangle(name):
    """numba's Itanium encoding of one identifier, ``<length><escaped name>``."""
    try:
        from numba.core.itanium_mangler import mangle_identifier
    except ImportError:  # pragma: no cover - older layouts
        from numba_cuda_mlir.numba_cuda.core.itanium_mangler import mangle_identifier
    return mangle_identifier(name)


def mask_ids(text):
    """``text`` with each live dispatcher's id replaced by ``ID<n>``, and the count replaced."""
    found = {}
    for obj in gc.get_objects():
        if "Dispatcher" in type(obj).__name__ and hasattr(obj, "py_func"):
            ident = str(id(obj))
            if ident in text:
                found[ident] = getattr(obj.py_func, "__qualname__", "")
    order = sorted(found, key=text.index)
    for n, ident in enumerate(order):
        label = f"ID{n}"
        qualname = found[ident]
        text = text.replace(
            _mangle(f"{qualname}_{ident}"), _mangle(f"{qualname}_{label}")
        )
        text = text.replace(ident, label)
    return text, len(order)


_compile = cg.compile_device_source


def capture(name, lines, namespace=None, inline="always"):
    DEV.append(f"### {name} inline={inline}\n" + "\n".join(lines) + "\n")
    return _compile(name, lines, namespace, inline)


for modname in (
    "modax._hierarchy",
    "modax._sparse_direct",
    "modax._split",
    "modax._rodas5P",
):
    mod = sys.modules.get(modname) or __import__(modname, fromlist=["x"])
    if getattr(mod, "compile_device_source", None) is _compile:
        mod.compile_device_source = capture
        print(f"[dump] wrapped compile_device_source in {modname}", flush=True)

_make_launch = getattr(solve_module, "make_launch", None)


def recording_launch(kernel, argtypes, **kw):
    launch = _make_launch(kernel, argtypes, **kw)
    for attr in ("inspect_asm", "inspect_llvm"):
        fn = getattr(kernel, attr, None)
        if fn is None:
            continue
        try:
            text = fn()
            items = text.items() if isinstance(text, dict) else [("", text)]
            for sig, body in items:
                raw = str(body)
                masked, n_ids = mask_ids(raw)
                left = len(LONG_DIGITS.findall(masked))
                ASM.append(
                    f"{attr} {getattr(kernel, '__name__', type(kernel).__name__)} {sig}"
                    f" masked {hashlib.sha256(masked.encode()).hexdigest()} ids {n_ids}"
                    f" long-digit-runs-left {left} chars {len(raw)}"
                )
                RAW.append(
                    f"{attr} {sig} raw {hashlib.sha256(raw.encode()).hexdigest()}"
                )
                PTX.append(f"### {attr} {sig}\n{masked}\n")
            break
        except Exception as e:  # noqa: BLE001
            ASM.append(f"{attr} failed: {e!r}")
    else:
        ASM.append(f"no inspect_asm / inspect_llvm on {type(kernel).__name__}")
    return launch


if _make_launch is not None:
    solve_module.make_launch = recording_launch

outdir = sys.argv[1]
os.makedirs(outdir, exist_ok=True)
open(
    os.path.join(outdir, "..", f"{os.path.basename(outdir)}_raw_digests.txt"), "w"
).close()
T_SPAN = np.linspace(0.0, 3.0, 7)
F64 = Rodas5P(lu_precision="fp64")
TOL = dict(rtol=1e-9, atol=1e-11, method=F64)


def rhs_a(y, t, p):
    return (-p[0] * y[0] + math.cos(t),)


def out_a(y, t, p):
    return (math.sin(y[0]), y[0] * t)


def rhs_b(y, t, p, q):
    return (-p[1] * (y[0] - q[0]) + q[1],)


def rhs_c(y, t, p, q):
    return (-p[2] * (y[0] - q[0]) + q[1],)


A_ = Block(rhs_a, 1, outputs=out_a, n_outputs=2, name="a")
B_ = Block(rhs_b, 1, reads=(A_,), name="b")
CHAIN = Hierarchy(Block(rhs_c, 1, reads=(B_, A_), name="c"))
TWO = Hierarchy(B_)
MIXED = Hierarchy(
    Block(
        rhs_b,
        1,
        reads=(
            Block(rhs_a, 1, outputs=out_a, n_outputs=2, name="a_tsit5", method=Tsit5()),
        ),
        name="b",
    )
)
Y0 = jnp.array([[0.3, 0.1, -0.2], [1.0, -0.5, 0.4]])
P = jnp.array([[1.0, 25.0, 40.0], [0.5, 50.0, 10.0]])


def hier_quantities(y, t, p):
    return (y[0] + y[2], y[1] * t)


def osc(y, t, p):
    return (y[1], -p[0] * y[0] - p[1] * y[1])


def energy_and_position(y, t, p):
    return (0.5 * (y[1] * y[1] + p[0] * y[0] * y[0]), y[0])


def chain_ode(y, t, p):
    return (
        -p[0] * y[0],
        p[0] * y[0] - p[1] * y[1],
        p[1] * y[1] - p[2] * y[2],
        p[2] * y[2],
    )


def rob(y, t, p):
    return (
        -p[0] * y[0] + p[1] * y[1] * y[2],
        p[0] * y[0] - p[1] * y[1] * y[2] - p[2] * y[1] ** 2,
        p[2] * y[1] ** 2,
    )


MASK = np.zeros((4, 4), dtype=bool)
for r, c in [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)]:
    MASK[r, c] = True
HK_T = np.linspace(0.0, 4.0, 33)
HK_Y0 = np.array([[1.0, 0.0], [0.5, 1.0], [-1.0, 0.3]])
HK_P = np.array([[4.0, 0.2], [9.0, 0.5], [1.0, 0.0]])
CH_Y0 = np.tile([1.0, 0.0, 0.0, 0.0], (4, 1))
CH_P = np.tile([0.7, 0.4, 0.2], (4, 1))
ROB_Y0 = np.tile([1.0, 0.0, 0.0], (4, 1))
ROB_P = np.tile([0.04, 1e4, 3e7], (4, 1))
ROB_T = np.array([0.0, 1e-2, 1e2])


def dump(name, fn):
    DEV.clear()
    ASM.clear()
    RAW.clear()
    PTX.clear()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        jax.block_until_ready(fn())
    text = buf.getvalue()
    for suffix, body in (
        (".py.txt", text),
        (".dev.txt", "".join(DEV)),
        (".asm.txt", "\n".join(ASM) + "\n"),
        (".ptx.txt", "".join(PTX)),
    ):
        with open(os.path.join(outdir, name + suffix), "w") as f:
            f.write(body)
    # the raw digests differ between processes by construction: kept apart, so that
    # the comparison of the dump directories reads the masked ones only
    with open(
        os.path.join(outdir, "..", f"{os.path.basename(outdir)}_raw_digests.txt"), "a"
    ) as f:
        f.write("".join(f"{name} {line}\n" for line in RAW))
    print(
        f"[dump] {name}: kernel {text.count(chr(10))} lines; {len(DEV)} generated device functions; {len(ASM)} asm lines",
        flush=True,
    )


dump("hier3_plain", lambda: solve(CHAIN, Y0, T_SPAN, P, **TOL))
dump("hier2_plain", lambda: solve(TWO, Y0[:, :2], T_SPAN, P, **TOL))
dump("hier2_mixed_tsit5_root", lambda: solve(MIXED, Y0[:, :2], T_SPAN, P, **TOL))
dump("hier3_fp32_default", lambda: solve(CHAIN, Y0, T_SPAN, P, rtol=1e-6, atol=1e-8))
dump(
    "hier3_auto",
    lambda: solve(CHAIN, Y0, T_SPAN, P, method=Auto(), rtol=1e-6, atol=1e-9),
)
dump(
    "hier3_jvp_params",
    lambda: jax.jvp(
        lambda p: solve(CHAIN, Y0, T_SPAN, p, **TOL), (P,), (jnp.ones((2, 3)),)
    ),
)
dump(
    "hier3_jacfwd_per_thread_1",
    lambda: jax.jacfwd(
        lambda t: solve(CHAIN, Y0, T_SPAN, P * t, max_directions_per_thread=1, **TOL)
    )(jnp.ones(3)),
)
dump(
    "hier2_jvp_y0",
    lambda: jax.jvp(
        lambda y: solve(TWO, y, T_SPAN, P, **TOL), (Y0[:, :2],), (jnp.ones((2, 2)),)
    ),
)
dump(
    "hier3_hook", lambda: solve(CHAIN, Y0, T_SPAN, P, save_hook=hier_quantities, **TOL)
)
dump(
    "hier3_hook_jvp",
    lambda: jax.jvp(
        lambda p: solve(
            CHAIN, Y0, T_SPAN, p, save_hook=hier_quantities, save_history=False, **TOL
        ),
        (P,),
        (jnp.ones((2, 3)),),
    ),
)
dump(
    "onebl_tsit5_hook",
    lambda: solve(
        osc, HK_Y0, HK_T, HK_P, method=Tsit5(), save_hook=energy_and_position
    ),
)
dump("onebl_auto", lambda: solve(osc, HK_Y0, HK_T, HK_P, method=Auto()))
dump("plain_rodas5p_dense", lambda: solve(rob, ROB_Y0, ROB_T, ROB_P, method=F64))
dump(
    "plain_rodas5p_hook_jvp",
    lambda: jax.jvp(
        lambda p: solve(
            osc,
            HK_Y0,
            HK_T,
            p,
            method=F64,
            save_hook=energy_and_position,
            save_history=False,
        ),
        (jnp.asarray(HK_P),),
        (jnp.ones((3, 2)),),
    ),
)
dump(
    "plain_sparse_fp64",
    lambda: solve(
        chain_ode, CH_Y0, np.array([0.0, 0.5, 2.0]), CH_P, sparsity=MASK, method=F64
    ),
)
dump(
    "plain_sparse_jvp",
    lambda: jax.jvp(
        lambda p: solve(
            chain_ode, CH_Y0, np.array([0.0, 0.5, 2.0]), p, sparsity=MASK, method=F64
        ),
        (jnp.asarray(CH_P),),
        (jnp.ones((4, 3)),),
    ),
)
print("[dump] done", flush=True)
