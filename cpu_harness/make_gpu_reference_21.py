"""A record of DISCO-EB 2.1.0 on modax 0.5, the same script on a GPU and on the CPU target.

On a CUDA machine, in DISCO-EB 2.1.0's own environment (``uv sync --extra
cuda13``), with nothing of this repository on the path::

    python make_gpu_reference_21.py gpu_reference_2.1.npz

On the CPU target, through the launcher (which installs the stand-in)::

    cpu_harness/disco --modax 0.5 cpu_harness/make_gpu_reference_21.py cpu_reference_2.1.npz

``compare_with_gpu_21.py`` compares two such records. Everything that
decides the arrays is in this file and stored in the record: the
cosmologies come from a seeded generator, the wave numbers and scale factors
are fixed. Sized for a 4 GB display card: at most 2048 trajectories a launch
(32 cosmologies x 64 modes), five sensitivity columns, three save grids.

Each section is independent: a failure is recorded in ``meta["errors"]`` and
the rest still runs, and the file is rewritten after every section. Every
section's first call (with whatever it compiles) and a second, warm call are
timed. Step counts come from separate launches that ask modax for them, so
asking does not touch the recorded arrays.
"""

import datetime
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

T_SCRIPT = time.perf_counter()
OUT = sys.argv[1] if len(sys.argv) > 1 else "reference_2.1.npz"
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

import discoeb  # noqa: E402
from discoeb import solve_eb  # noqa: E402
from discoeb.components.recombination import ReionizationDepth  # noqa: E402
from discoeb.models import LCDM, LCDM_MNU, W0WACDM  # noqa: E402
from discoeb.observables.power_spectrum import matter_power_spectrum  # noqa: E402

import modax  # noqa: E402

A = {}
META = {"errors": {}, "timings_s": {}, "notes": {}}


def log(*a):
    print(f"[ref21 +{time.perf_counter() - T_SCRIPT:7.1f}s]", *a, flush=True)


def f64(x):
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


def save():
    META["script_elapsed_s"] = time.perf_counter() - T_SCRIPT
    tmp = OUT + ".tmp.npz"
    np.savez(tmp, meta=np.array(json.dumps(META, indent=1, sort_keys=True)), **A)
    os.replace(tmp, OUT)


def section(name):
    def deco(fn):
        log("BEGIN", name)
        t0 = time.perf_counter()
        try:
            fn()
            log("OK   ", name, f"{time.perf_counter() - t0:.1f} s")
        except BaseException as e:  # noqa: BLE001 - one failure must not block the rest
            if isinstance(e, KeyboardInterrupt):
                raise
            META["errors"][name] = traceback.format_exc()
            log("FAIL ", name, repr(e))
            traceback.print_exc()
            sys.stderr.flush()
        META["timings_s"][f"section {name}"] = time.perf_counter() - t0
        save()
        return fn

    return deco


def cold_and_warm(name, fn):
    """``fn()`` twice; the first call's and the second's seconds under ``name``; the second's value."""
    t0 = time.perf_counter()
    jax.block_until_ready(fn())
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn())
    META["timings_s"][name] = {"cold": cold, "warm": time.perf_counter() - t0}
    log(f"  {name}: cold {cold:.2f} s, warm {META['timings_s'][name]['warm']:.3f} s")
    return out


def version(p):
    try:
        return md.version(p)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- meta
PKGS = (
    "disco-eb", "modax-solvers", "numba-cuda-mlir", "numba-enzyme-cuda", "numba",
    "llvmlite", "jax", "jaxlib", "jax-cuda13-plugin", "jax-cuda13-pjrt",
    "jax-cuda12-plugin", "jax-cuda12-pjrt", "numpy", "scipy", "cuda-bindings",
    "cuda-core", "cuda-pathfinder", "nvidia-nvjitlink", "nvidia-cuda-nvrtc",
)  # fmt: skip
META["versions"] = {p: version(p) for p in PKGS}
META["python"] = sys.version
META["platform"] = platform.platform()
META["host"] = platform.node()
META["date_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
META["argv"] = sys.argv
META["env"] = {
    k: os.environ.get(k)
    for k in (
        "CUDA_VISIBLE_DEVICES", "NUMBA_CACHE_DIR", "XLA_PYTHON_CLIENT_PREALLOCATE",
        "MODAX_TARGET", "MODAX_CPU_DERIVATIVES", "MODAX_ENZYME_TOOLS",
        "DISCO_THERMO_FP32_MATH", "DISCO_MAX_SENSITIVITY_STATE",
        "DISCO_CUDA_MAX_REGISTERS", "NUMBA_NUM_THREADS",
    )
}  # fmt: skip
META["discoeb_file"] = discoeb.__file__
META["modax_file"] = modax.__file__
try:
    from modax import _target

    META["target"] = {
        "TARGET": _target.TARGET,
        "CPU_DERIVATIVES": _target.CPU_DERIVATIVES,
    }
except ImportError:
    META["target"] = {"TARGET": "cuda", "note": "no modax._target: upstream modax"}


def git_describe(path):
    try:
        return subprocess.run(
            ["git", "-C", path, "describe", "--tags", "--always", "--dirty"],
            capture_output=True, text=True, check=True, timeout=20,
        ).stdout.strip()  # fmt: skip
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e!r}"


META["discoeb_git"] = git_describe(os.path.dirname(discoeb.__file__))
META["modax_git"] = git_describe(os.path.dirname(modax.__file__))
try:
    dev = jax.devices()[0]
    META["jax_device"] = {
        "repr": repr(dev),
        "platform": dev.platform,
        "kind": getattr(dev, "device_kind", None),
    }
except Exception as e:  # noqa: BLE001
    META["jax_device"] = repr(e)
if META["target"]["TARGET"] == "cuda":
    try:
        from numba_cuda_mlir import cuda as _cuda

        d = _cuda.get_current_device()
        name = d.name.decode() if isinstance(d.name, bytes) else str(d.name)
        META["gpu"] = {"model": name, "compute_capability": list(d.compute_capability)}
    except Exception as e:  # noqa: BLE001
        META["gpu"] = {"query_error": repr(e)}
else:
    META["cpu"] = {
        "processor": platform.processor(),
        "machine": platform.machine(),
        "count": os.cpu_count(),
    }
META["discoeb_defaults"] = {
    k: getattr(discoeb, k) for k in ("A_START", "PERTURB_RTOL", "PERTURB_ATOL", "PERTURB_FIRST_STEP", "PERTURB_MAX_STEPS", "THERMO_ERROR_WEIGHT")
}  # fmt: skip
log(
    json.dumps(
        {
            k: META[k]
            for k in ("versions", "target", "discoeb_git", "jax_device", "env")
        },
        indent=1,
    )
)

# ---------------------------------------------------------------- inputs
REFERENCE = {
    "Omega_m": (0.02238280 + 0.1201075) / 0.6732117**2,
    "Omega_b": 0.02238280 / 0.6732117**2,
    "A_s": 2.100549e-9,
    "n_s": 0.9660499,
    "H_0": 67.32117,
    "T_cmb": 2.7255,
    "Y_He": 0.2454006,
    "N_eff": 3.046,
}
"""tests/helpers.py's REFERENCE_COSMOLOGY, the keys LCDM reads."""
SPREAD = {"Omega_m": 0.03, "Omega_b": 0.03, "A_s": 0.05, "n_s": 0.01, "H_0": 0.02}
"""Relative half-widths of the uniform perturbations of the 32 LCDM cosmologies."""
RNG = np.random.default_rng(20261120)
K = np.geomspace(1.0e-4, 1.0, 64)
A_GRID = np.geomspace(discoeb.A_START, 1.0, 48)
N_COSMO = 32


def of(model, params):
    """A cosmology of ``model`` from a dictionary in Omega_m, Omega_b and H_0, as tests/helpers.cosmologies_of."""
    return model.from_dict(model.physical_parameters(params))


draws = [
    {
        k: v * (1.0 + SPREAD.get(k, 0.0) * (2.0 * RNG.random() - 1.0))
        for k, v in REFERENCE.items()
    }
    for _ in range(N_COSMO)
]
LCDM32 = [of(LCDM, d) for d in draws]
PLANCK = of(LCDM, REFERENCE)
MNU = of(LCDM_MNU, {**REFERENCE, "N_mnu": 1.0, "m_nu": 0.1, "N_eff": 2.0328})
W0WA = of(W0WACDM, {**REFERENCE, "w_DE_0": -0.9, "w_DE_a": 0.1})
LCDM_REIO = LCDM.add(ReionizationDepth, name="LCDM+reio")
REIO = of(LCDM_REIO, {**REFERENCE, "z_re": 7.67})
INPUT_KEYS = list(REFERENCE)
A["k"] = f64(K)
A["a_grid"] = f64(A_GRID)
A["input_keys"] = np.array(INPUT_KEYS)
A["lcdm32_inputs"] = f64([[d[k] for k in INPUT_KEYS] for d in draws])
for label, c in (("planck", PLANCK), ("mnu", MNU), ("w0wa", W0WA), ("reio", REIO)):
    d = c.to_dict()
    A[f"{label}_param_names"] = np.array(sorted(d))
    A[f"{label}_param_values"] = f64([float(d[n]) for n in sorted(d)])
A["lcdm32_omega_c"] = f64([float(c.omega_c) for c in LCDM32])
META["notes"]["inputs"] = (
    "lcdm32: 32 LCDM cosmologies, REFERENCE x (1 + SPREAD u), u uniform in [-1, 1] from "
    "default_rng(20261120) in dict order; planck = REFERENCE; mnu = REFERENCE with N_mnu 1, "
    "m_nu 0.1, N_eff 2.0328; w0wa = REFERENCE with w_DE_0 -0.9, w_DE_a 0.1; reio = LCDM + "
    "ReionizationDepth with z_re 7.67. k = geomspace(1e-4, 1, 64) 1/Mpc; a_grid = "
    "geomspace(A_START, 1, 48). Arrays from solve_eb are (cosmology, name, k, a)."
)
save()

PK_KEEP = ("delta_c", "delta_b")


def spectrum(c, values):
    dc, db = jnp.unstack(values[:, :, -1])
    return matter_power_spectrum(
        K, delta_c=dc, delta_b=db, omega_c=c.omega_c, omega_b=c.omega_b,
        A_s=c.A_s, n_s=c.n_s, k_p=c.k_p,
    )  # fmt: skip


def pk(cosmologies, **options):
    """``(n, 64)`` P(k) of a list of cosmologies, the docs' recipe."""
    values = solve_eb(cosmologies, K, [1.0], keep=PK_KEEP, **options)
    return jnp.stack([spectrum(c, values[i]) for i, c in enumerate(cosmologies)])


# ---------------------------------------------------------------- P(k)
@section("pk")
def _pk():
    A["pk_lcdm32_default"] = f64(cold_and_warm("pk lcdm32 default", lambda: pk(LCDM32)))
    A["pk_lcdm32_tw01"] = f64(
        cold_and_warm(
            "pk lcdm32 thermo_weight 0.1", lambda: pk(LCDM32, thermo_weight=0.1)
        )
    )
    A["pk_planck_tw01"] = f64(
        cold_and_warm(
            "pk planck thermo_weight 0.1", lambda: pk([PLANCK], thermo_weight=0.1)
        )
    )
    A["pk_mnu_default"] = f64(cold_and_warm("pk mnu default", lambda: pk([MNU])))
    A["pk_w0wa_default"] = f64(cold_and_warm("pk w0wa default", lambda: pk([W0WA])))


# ---------------------------------------------------------------- histories
HIST_KEEP = (
    "delta_c", "delta_b", "theta_b", "eta", "F_0", "F_2", "F_4", "E_2", "N_2",
    "R", "alpha", "aH", "opacity", "tau", "xe", "Tmat", "cs2", "kappa",
)  # fmt: skip
MNU_KEEP = ("delta_c", "psi_0_0", "psi_0_2", "psi_1_0", "R", "xe")
HIST_K = K[::8]


@section("histories")
def _histories():
    A["hist_keep"] = np.array(HIST_KEEP)
    A["hist_k"] = f64(HIST_K)
    A["hist_planck"] = f64(
        cold_and_warm(
            "history planck", lambda: solve_eb(PLANCK, HIST_K, A_GRID, keep=HIST_KEEP)
        )
    )
    A["hist_mnu_keep"] = np.array(MNU_KEEP)
    A["hist_mnu"] = f64(
        cold_and_warm(
            "history mnu", lambda: solve_eb(MNU, HIST_K, A_GRID, keep=MNU_KEEP)
        )
    )


# ---------------------------------------------------------------- derivatives
NAMES = ("omega_c", "omega_b", "n_s", "A_s", "N_eff")
THETA0 = np.array([float(getattr(PLANCK, n)) for n in NAMES])
A["deriv_names"] = np.array(NAMES)
A["deriv_theta0"] = f64(THETA0)


def pk_of(theta, base=PLANCK, **options):
    c = base.replace(**{n: theta[i] for i, n in enumerate(NAMES)})
    return spectrum(
        c, solve_eb(c, K, [1.0], keep=PK_KEEP, thermo_weight=0.1, **options)[0]
    )


@section("derivatives")
def _derivatives():
    theta = jnp.asarray(THETA0)
    A["jacfwd_pk"] = f64(
        cold_and_warm("jacfwd pk 5 parameters", lambda: jax.jacfwd(pk_of)(theta))
    )
    A["jacfwd_pk_split"] = f64(
        cold_and_warm(
            "jacfwd pk 5 parameters split",
            lambda: jax.jacfwd(lambda t: pk_of(t, split_derivatives=True))(theta),
        )
    )
    direction = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0])
    value, tangent = cold_and_warm(
        "jvp pk omega_c", lambda: jax.jvp(pk_of, (theta,), (direction,))
    )
    A["jvp_pk_value"], A["jvp_pk_tangent"] = f64(value), f64(tangent)
    batch = jnp.asarray(THETA0 * np.array([[1.0], [1.02], [0.98]]))
    A["vmap_jacfwd_theta"] = f64(batch)
    A["vmap_jacfwd_pk"] = f64(
        cold_and_warm(
            "vmap(jacfwd) pk 3 cosmologies", lambda: jax.vmap(jax.jacfwd(pk_of))(batch)
        )
    )
    # The same three one at a time, and the values both ways: solve_eb hands modax
    # (n_traj, n_vars) error weights, which a vmap folds three cosmologies past.
    A["loop_jacfwd_pk"] = f64(
        np.stack([np.asarray(jax.jacfwd(pk_of)(b)) for b in batch])
    )


@section("vmap_values")
def _vmap_values():
    """``vmap`` of the plain solve over the same three; modax 0.5 may refuse it."""
    batch = jnp.asarray(A["vmap_jacfwd_theta"])
    A["loop_pk"] = f64(np.stack([np.asarray(pk_of(b)) for b in batch]))
    A["vmap_pk"] = f64(
        cold_and_warm("vmap pk 3 cosmologies", lambda: jax.vmap(pk_of)(batch))
    )


# ---------------------------------------------------------------- reionization
REIO_KEEP = ("kappa_reio", "kappa", "xe", "delta_c", "delta_b")


@section("reionization")
def _reionization():
    def kept(z_re):
        return solve_eb(REIO.replace(z_re=z_re), HIST_K, [1.0], keep=REIO_KEEP)

    z0 = jnp.asarray(float(REIO.z_re))
    A["reio_keep"] = np.array(REIO_KEEP)
    A["reio_values"] = f64(cold_and_warm("reio values", lambda: kept(z0)))
    value, tangent = cold_and_warm(
        "reio jvp z_re", lambda: jax.jvp(kept, (z0,), (jnp.ones_like(z0),))
    )
    A["reio_jvp_value"], A["reio_jvp_tangent"] = f64(value), f64(tangent)


# ---------------------------------------------------------------- step statistics
@section("stats")
def _stats():
    """The step counters of the plain and the joint launch, from launches that ask for them."""
    real = modax.solve
    seen = []

    def counting(*args, **kwargs):
        out = real(*args, **kwargs, return_stats=True)
        jax.debug.callback(
            lambda s: seen.append({k: np.asarray(v) for k, v in s.items()}), out[-1]
        )
        return out[0] if len(out) == 2 else out[:-1]

    modax.solve = counting
    try:
        for label, fn in (
            ("pk_lcdm32_default", lambda: pk(LCDM32)),
            ("pk_lcdm32_tw01", lambda: pk(LCDM32, thermo_weight=0.1)),
            (
                "jvp_pk",
                lambda: jax.jvp(
                    pk_of, (jnp.asarray(THETA0),), (jnp.asarray([1.0, 0, 0, 0, 0]),)
                ),
            ),
        ):
            seen.clear()
            jax.block_until_ready(fn())
            jax.effects_barrier()
            for key in ("accepted_steps", "rejected_steps", "loop_steps"):
                A[f"stats_{label}_{key}"] = (
                    np.asarray(seen[-1][key]).astype(np.int64).ravel()
                )
            log(
                f"  stats {label}: accepted mean {A[f'stats_{label}_accepted_steps'].mean():.1f}"
            )
    finally:
        modax.solve = real
    META["notes"]["stats"] = (
        "per trajectory, modax's row order (k-major: row = k_idx * n_cosmo + cosmology_idx)"
    )


log(f"wrote {OUT}: {len(A)} arrays, {len(META['errors'])} failed sections")
