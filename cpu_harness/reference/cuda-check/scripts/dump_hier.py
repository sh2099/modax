"""Dump the generated hierarchy kernel source (MODAX_HIERARCHY_DUMP makes make_hierarchy_kernel
print it) for fixed hierarchies; stdout during each solve is captured to OUTDIR/<name>.py.txt.
Usage: dump_hier.py OUTDIR"""
import contextlib, io, math, os, sys

os.environ["MODAX_HIERARCHY_DUMP"] = "1"
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from modax import Auto, Rodas5P, Tsit5, solve
from modax._hierarchy import Block, Hierarchy

outdir = sys.argv[1]
os.makedirs(outdir, exist_ok=True)
T_SPAN = np.linspace(0.0, 3.0, 7)
TOL = dict(rtol=1e-9, atol=1e-11, method=Rodas5P(lu_precision="fp64"))

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
Y0 = jnp.array([[0.3, 0.1, -0.2], [1.0, -0.5, 0.4]])
P = jnp.array([[1.0, 25.0, 40.0], [0.5, 50.0, 10.0]])

def dump(name, fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    text = buf.getvalue()
    with open(os.path.join(outdir, name + ".py.txt"), "w") as f:
        f.write(text)
    print(f"[dump] {name}: {len(text)} chars, {text.count(chr(10))} lines", flush=True)

dump("hier3_plain", lambda: solve(CHAIN, Y0, T_SPAN, P, **TOL))
dump("hier2_plain", lambda: solve(Hierarchy(B), Y0[:, :2], T_SPAN, P, **TOL))
dump("hier3_fp32_default", lambda: solve(CHAIN, Y0, T_SPAN, P, rtol=1e-6, atol=1e-8))
dump("hier3_auto", lambda: solve(CHAIN, Y0, T_SPAN, P, method=Auto(), rtol=1e-6, atol=1e-9))
dump("hier3_jvp_params", lambda: jax.jvp(lambda p: solve(CHAIN, Y0, T_SPAN, p, **TOL), (P,), (jnp.ones((2, 3)),)))
dump("hier3_hook", lambda: solve(CHAIN, Y0, T_SPAN, P, save_hook=lambda y, t, p: (y[0] + y[2],), **TOL))
