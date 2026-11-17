"""Dump the generated kernel source for fixed problems, for a byte-for-byte comparison.
Two kinds of generated text, both what numba-cuda-mlir then compiles:
  <case>.py.txt   the hierarchy kernel (MODAX_HIERARCHY_DUMP makes make_hierarchy_kernel print
                  it; stdout during the solve is captured), one-block hierarchies included;
  <case>.dev.txt  every device function built from source through
                  modax._codegen.compile_device_source during the solve (the literal-seed
                  Jacobian writer, the sparse LU), wrapped where each module imported it.
Usage: dump_hier.py OUTDIR"""
import contextlib, io, math, os, sys

os.environ["MODAX_HIERARCHY_DUMP"] = "1"
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
import modax._codegen as cg
from modax import Auto, Block, Hierarchy, Rodas5P, Tsit5, solve
from modax._sensitivity import SensitivitySpec
from modax.rodas5P import quantity_hook

DEV = []
_orig = cg.compile_device_source


def capture(name, lines, namespace=None, inline="always"):
    DEV.append(f"### {name} inline={inline}\n" + "\n".join(lines) + "\n")
    return _orig(name, lines, namespace, inline)


for modname in ("modax._hierarchy", "modax._sparse_direct", "modax._split", "modax.rodas5P"):
    mod = sys.modules.get(modname) or __import__(modname, fromlist=["x"])
    if getattr(mod, "compile_device_source", None) is _orig:
        mod.compile_device_source = capture
        print(f"[dump] wrapped compile_device_source in {modname}")

outdir = sys.argv[1]
os.makedirs(outdir, exist_ok=True)
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

A = Block(rhs_a, 1, outputs=out_a, n_outputs=2, name="a")
B = Block(rhs_b, 1, reads=(A,), name="b")
C = Block(rhs_c, 1, reads=(B, A), name="c")
CHAIN = Hierarchy(C)
TWO = Hierarchy(B)
A_T = Block(rhs_a, 1, outputs=out_a, n_outputs=2, name="a_tsit5", method=Tsit5())
MIXED = Hierarchy(Block(rhs_b, 1, reads=(A_T,), name="b_after_tsit5"))
Y0 = jnp.array([[0.3, 0.1, -0.2], [1.0, -0.5, 0.4]])
P = jnp.array([[1.0, 25.0, 40.0], [0.5, 50.0, 10.0]])

def hier_quantities(y, t, p):
    return (y[0] + y[2], y[1] * t)

def osc(y, t, p):
    return (y[1], -p[0] * y[0] - p[1] * y[1])

def energy_and_position(y, t, p):
    return (0.5 * (y[1] * y[1] + p[0] * y[0] * y[0]), y[0])

def chain_ode(y, t, p):
    return (-p[0] * y[0], p[0] * y[0] - p[1] * y[1], p[1] * y[1] - p[2] * y[2], p[2] * y[2])

MASK = np.zeros((4, 4), dtype=bool)
for r, c in [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)]:
    MASK[r, c] = True
HK_T = np.linspace(0.0, 4.0, 33)
HK_Y0 = np.array([[1.0, 0.0], [0.5, 1.0], [-1.0, 0.3]])
HK_P = np.array([[4.0, 0.2], [9.0, 0.5], [1.0, 0.0]])


def dump(name, fn):
    DEV.clear()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    text = buf.getvalue()
    with open(os.path.join(outdir, name + ".py.txt"), "w") as f:
        f.write(text)
    dev = "".join(DEV)
    with open(os.path.join(outdir, name + ".dev.txt"), "w") as f:
        f.write(dev)
    print(f"[dump] {name}: kernel {text.count(chr(10))} lines; {len(DEV)} generated device functions, {dev.count(chr(10))} lines", flush=True)


S3 = SensitivitySpec(3, 3, wrt_y0=False, wrt_params=True, error_control=False)
S2 = SensitivitySpec(2, 2, wrt_y0=True, wrt_params=True, error_control=False)
dump("hier3_plain", lambda: solve(CHAIN, Y0, T_SPAN, P, **TOL))
dump("hier2_plain", lambda: solve(TWO, Y0[:, :2], T_SPAN, P, **TOL))
dump("hier2_mixed_tsit5_root", lambda: solve(MIXED, Y0[:, :2], T_SPAN, P, **TOL))
dump("hier3_fp32_default", lambda: solve(CHAIN, Y0, T_SPAN, P, rtol=1e-6, atol=1e-8))
dump("hier3_auto", lambda: solve(CHAIN, Y0, T_SPAN, P, method=Auto(), rtol=1e-6, atol=1e-9))
dump("hier3_jvp_params", lambda: jax.jvp(lambda p: solve(CHAIN, Y0, T_SPAN, p, **TOL), (P,), (jnp.ones((2, 3)),)))
dump("hier2_jvp_y0", lambda: jax.jvp(lambda y: solve(TWO, y, T_SPAN, P, **TOL), (Y0[:, :2],), (jnp.ones((2, 2)),)))
dump("hier3_qhook", lambda: solve(CHAIN, Y0, T_SPAN, P, save_hook=quantity_hook(hier_quantities, 2, 3, 3), hook_size=14, **TOL))
dump("hier3_qhook_sens", lambda: solve(CHAIN, Y0, T_SPAN, P, save_hook=quantity_hook(hier_quantities, 2, 3, 3, S3),
                                       hook_size=2 * 4 * 7, save_history=False, sensitivity=S3, **TOL))
dump("onebl_tsit5_qhook", lambda: solve(osc, HK_Y0, HK_T, HK_P, method=Tsit5(), save_hook=quantity_hook(energy_and_position, 2, 2, 2), hook_size=66))
dump("onebl_auto", lambda: solve(osc, HK_Y0, HK_T, HK_P, method=Auto()))
dump("plain_rodas5p_qhook_sens", lambda: solve(osc, HK_Y0, HK_T, HK_P, method=F64, save_hook=quantity_hook(energy_and_position, 2, 2, 2, S2),
                                               hook_size=2 * 5 * 33, save_history=False, sensitivity=S2))
dump("plain_sparse_fp64", lambda: solve(chain_ode, np.tile([1.0, 0, 0, 0], (4, 1)), np.array([0.0, 0.5, 2.0]),
                                        np.tile([0.7, 0.4, 0.2], (4, 1)), sparsity=MASK, method=F64))
dump("plain_sparse_jvp", lambda: jax.jvp(lambda p: solve(chain_ode, np.tile([1.0, 0, 0, 0], (4, 1)), np.array([0.0, 0.5, 2.0]), p,
                                                         sparsity=MASK, method=F64), (jnp.tile(jnp.array([0.7, 0.4, 0.2]), (4, 1)),), (jnp.ones((4, 3)),)))
