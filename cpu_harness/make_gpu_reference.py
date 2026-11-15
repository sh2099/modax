"""GPU reference outputs of DISCO-EB 2.0.1 (modax-solvers 0.3.0) for comparison with a CPU build.

Run on cap006a through the rebuild kit (which applies the Pascal workarounds first):

    ~/disco-eb-p100/run.sh GPU LOGNAME /tmp/sh2099-disco-eb/refrun/make_gpu_reference.py [OUT.npz] [mode]

mode: "all" (default) or "cold" (only the P(k) timings, for a run with an empty NUMBA_CACHE_DIR;
writes OUT.npz with just `meta`).

The cosmologies are the test suite's own: tests/test_perturbations.py::_perturbed_cosmologies(n),
a deterministic construction with no random seed. Note _perturbed_cosmologies(1) is
[REFERENCE_COSMOLOGY], which is NOT _perturbed_cosmologies(128)[0] (that one is the phase = -1
end of the perturbed family). Every array below says in its name which set it came from:
  c1    = _perturbed_cosmologies(1)       (the reference cosmology)
  c2    = _perturbed_cosmologies(2)       (phase -1 and +1)
  c128  = _perturbed_cosmologies(128)     (phase linspace(-1, 1, 128))
Each section is independent: a failure is recorded in meta["errors"] and the rest still runs.
The file is rewritten after every section.
"""

import datetime
import importlib.metadata as md
import json
import os
import platform
import sys
import time
import traceback

import numpy as np

T_SCRIPT = time.perf_counter()
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("DISCO_ROOT", "."), "refrun", "gpu_reference_2.0.1.npz")
MODE = sys.argv[2] if len(sys.argv) > 2 else "all"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

sys.path.insert(0, os.path.join(os.getcwd(), "tests"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

import discoeb  # noqa: E402
import discoeb.perturbations as P  # noqa: E402
import modax  # noqa: E402
import test_perturbations as TP  # noqa: E402

A = {}  # arrays to save
META = {"errors": {}, "timings_s": {}, "notes": {}}


def log(*a):
    print(f"[ref +{time.perf_counter() - T_SCRIPT:7.1f}s]", *a, flush=True)


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
        META["timings_s"][f"section_{name}"] = time.perf_counter() - t0
        save()
        return fn

    return deco


def version(p):
    try:
        return md.version(p)
    except Exception:  # noqa: BLE001
        return None


def ulps(a, b):
    """Units in the last place between two float64 arrays of the same sign (signed: b - a)."""
    a, b = f64(a), f64(b)
    return (b.view(np.int64) - a.view(np.int64)).astype(np.int64)


# ---------------------------------------------------------------- meta
PKGS = ("disco-eb", "modax-solvers", "numba-cuda-mlir", "numba-enzyme-cuda", "numba", "llvmlite", "jax", "jaxlib",
        "jax-cuda12-plugin", "jax-cuda12-pjrt", "numpy", "scipy", "classy", "cuda-toolkit", "cuda-bindings",
        "cuda-core", "cuda-pathfinder", "nvidia-cuda-nvcc-cu12", "nvidia-cuda-nvrtc-cu12",
        "nvidia-cuda-runtime-cu12", "nvidia-nvjitlink-cu12", "nvidia-cublas-cu12", "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12", "nvidia-cudnn-cu12", "pytest", "pytest-benchmark")
META["versions"] = {p: version(p) for p in PKGS}
META["python"] = sys.version
META["platform"] = platform.platform()
META["host"] = platform.node()
META["date_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
META["mode"] = MODE
META["argv"] = sys.argv
META["env"] = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "NUMBA_CACHE_DIR", "TMPDIR", "LD_PRELOAD",
                                             "XLA_PYTHON_CLIENT_PREALLOCATE", "NUMBA_NUM_THREADS", "DISCO_DESC",
                                             "DISCO_THERMO_WEIGHT", "DISCO_CUDA_MAX_REGISTERS")}
try:
    head = open(os.path.join(os.getcwd(), ".git", "HEAD")).read().strip()
    if head.startswith("ref:"):
        head = open(os.path.join(os.getcwd(), ".git", head.split()[1])).read().strip()
    META["discoeb_commit"] = head
except Exception as e:  # noqa: BLE001
    META["discoeb_commit"] = f"unreadable: {e!r}"
META["discoeb_file"] = discoeb.__file__
META["modax_file"] = modax.__file__
try:
    dev = jax.devices()[0]
    META["jax_device"] = {"repr": repr(dev), "platform": dev.platform, "device_kind": getattr(dev, "device_kind", None)}
except Exception as e:  # noqa: BLE001
    META["jax_device"] = repr(e)
try:
    from numba_cuda_mlir import cuda as _cuda

    d = _cuda.get_current_device()
    name = d.name.decode() if isinstance(d.name, bytes) else str(d.name)
    META["gpu"] = {"model": name, "compute_capability": list(d.compute_capability)}
except Exception as e:  # noqa: BLE001
    META["gpu"] = {"model": (META.get("jax_device") or {}).get("device_kind") if isinstance(META.get("jax_device"), dict) else None,
                   "numba_query_error": repr(e)}
try:
    import numba_enzyme.cuda as _nec

    META["workarounds_active"] = {
        "cc6x_patch_loaded": "cc6x_patch" in sys.modules,
        "numba_enzyme_validate_cc_patched": getattr(_nec._validate_compute_capability, "__module__", "") == "cc6x_patch",
        "jax_ensure_compile_time_eval_is_stock": jax.ensure_compile_time_eval.__module__.startswith("jax"),
    }
except Exception as e:  # noqa: BLE001
    META["workarounds_active"] = repr(e)
META["solver_presets"] = {k: getattr(P, k) for k in ("PERTURB_RTOL", "PERTURB_ATOL", "PERTURB_FIRST_STEP", "PERTURB_MAX_STEPS",
                                                     "TAU_START", "MATTER_THERMO_WEIGHT", "TRAJECTORIES_PER_BLOCK",
                                                     "CUDA_MAX_REGISTERS", "N_BACKGROUND_GRID")}
META["solver_presets"]["THERMO_ERROR_WEIGHT"] = discoeb.THERMO_ERROR_WEIGHT
META["DIFFERENTIABLE_PARAMETERS"] = list(P.DIFFERENTIABLE_PARAMETERS)
META["DIRECTED_NAMES"] = list(P.DIRECTED_NAMES)
log(json.dumps({k: META[k] for k in ("versions", "discoeb_commit", "gpu", "jax_device", "env")}, indent=1))

# ---------------------------------------------------------------- inputs
K = np.geomspace(1e-4, 1.0, 128)
assert np.array_equal(K, TP.MATTER_POWER_K)
C1 = TP._perturbed_cosmologies(1)
C2 = TP._perturbed_cosmologies(2)
C128 = TP._perturbed_cosmologies(128)
COSMO_KEYS = list(TP.REFERENCE_COSMOLOGY.keys())


def cosmo_table(cs):
    return f64([[c[k] for k in COSMO_KEYS] for c in cs])


A["k"] = f64(K)
A["cosmo_keys"] = np.array(COSMO_KEYS)
A["c1_params"] = cosmo_table(C1)
A["c2_params"] = cosmo_table(C2)
A["c128_params"] = cosmo_table(C128)
META["notes"]["cosmologies"] = (
    "c1 = _perturbed_cosmologies(1) = [REFERENCE_COSMOLOGY]; c2 = _perturbed_cosmologies(2) (phase -1, +1); "
    "c128 = _perturbed_cosmologies(128) (phase = linspace(-1,1,128)). Deterministic, no seed. "
    "cX_params columns are cosmo_keys. c128[0] == c2[0] != c1[0].")
META["notes"]["trajectory_order"] = "modax rows are k-major: row = k_idx * n_cosmo + cosmology_idx, k ascending"
save()

# ---------------------------------------------------------------- P(k) + timings
solve_pk = P.solve_matter_power_spectrum_batch


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


@section("pk")
def _pk():
    tm = META["timings_s"]
    pk1, tm["pk_c1_first_call"] = timed(lambda: solve_pk(K, C1))
    log(f"pk c1 first call {tm['pk_c1_first_call']:.2f} s")
    A["pk_c1"] = f64(pk1)
    save()
    warm = []
    for _ in range(3):
        pk1b, t = timed(lambda: solve_pk(K, C1))
        warm.append(t)
    tm["pk_c1_warm_calls"] = warm
    tm["pk_c1_warm"] = warm[0]
    META["notes"]["pk_c1_repeat_bitwise_equal"] = bool(np.array_equal(pk1, pk1b))
    log(f"pk c1 warm {warm}")

    pk128, tm["pk_c128_first_call"] = timed(lambda: solve_pk(K, C128))
    log(f"pk c128 first call {tm['pk_c128_first_call']:.2f} s")
    A["pk_c128"] = f64(pk128)
    save()
    warm = []
    for _ in range(3):
        pk128b, t = timed(lambda: solve_pk(K, C128))
        warm.append(t)
    tm["pk_c128_warm_calls"] = warm
    tm["pk_c128_warm"] = warm[0]
    META["notes"]["pk_c128_repeat_bitwise_equal"] = bool(np.array_equal(pk128, pk128b))
    log(f"pk c128 warm {warm}")
    META["notes"]["timings"] = (
        "pk_*_first_call: first call in this process (kernel compile or load from NUMBA_CACHE_DIR, JAX tracing, "
        "bridge load). pk_*_warm: the second call with the same inputs (pk_*_warm_calls: three repeats; DISCO-EB's "
        "lru caches hold the packed batch). Whether the numba disk cache was empty is in meta['numba_cache_at_start'].")
    for name, pk in (("c1", A["pk_c1"]), ("c128", A["pk_c128"])):
        log(name, "P(k) shape", pk.shape, "finite", bool(np.isfinite(pk).all()), "min", pk.min(), "max", pk.max())


try:
    cache_dir = os.environ.get("NUMBA_CACHE_DIR")
    META["numba_cache_at_start"] = {"dir": cache_dir, "n_files": len(os.listdir(cache_dir)) if cache_dir and os.path.isdir(cache_dir) else 0}
except Exception as e:  # noqa: BLE001
    META["numba_cache_at_start"] = repr(e)

if MODE == "cold":
    save()
    log("cold mode: done", json.dumps(META["timings_s"]))
    sys.exit(0)


# ---------------------------------------------------------------- modax return_stats
@section("stats")
def _stats():
    captured = []
    orig = modax.solve

    def with_stats(*a, **kw):
        out = orig(*a, return_stats=True, **kw)
        captured.append(out[-1])
        return out[0] if len(out) == 2 else out[:-1]

    modax.solve = with_stats  # P._launch does `from modax import solve` at call time
    try:
        for tag, cs, ref in (("c1", C1, "pk_c1"), ("c128", C128, "pk_c128")):
            captured.clear()
            pk = solve_pk(K, cs)
            (stats,) = captured
            for field in ("accepted_steps", "rejected_steps", "loop_steps"):
                A[f"stats_{tag}_{field}"] = np.asarray(stats[field]).astype(np.int64)
            META["notes"][f"stats_{tag}_pk_bitwise_equal_to_plain_solve"] = bool(np.array_equal(f64(pk), A[ref])) if ref in A else None
            acc = A[f"stats_{tag}_accepted_steps"]
            log(tag, "stats shape", acc.shape, "accepted mean", acc.mean(), "min", acc.min(), "max", acc.max(),
                "rejected mean", A[f"stats_{tag}_rejected_steps"].mean(), "loop mean", A[f"stats_{tag}_loop_steps"].mean())
    finally:
        modax.solve = orig
    META["notes"]["stats"] = (
        "modax.solve wrapped from this script with return_stats=True (no DISCO-EB edit). One entry per trajectory, "
        "row = k_idx * n_cosmo + cosmology_idx with k ascending (K is already ascending). For the split joint "
        "kernel modax returns one set of counters per trajectory; which block they count is modax's definition.")


# ---------------------------------------------------------------- perturbation history
@section("history")
def _history():
    k16 = K[::8]
    A["hist_k"] = f64(k16)
    hist, k_sorted, prepared = P.solve_perturbation_history_batch(k16, C1)
    A["hist_c1_default"] = f64(hist)  # (n_k, n_cosmo, n_save, nvar)
    A["hist_c1_default_k_sorted"] = f64(k_sorted)
    A["hist_c1_tau0"] = f64(prepared.tau0)
    A["hist_c1_lna_start"] = f64(prepared.lna_start)
    A["hist_c1_z_re"] = f64(prepared.z_re)
    # default tau_save=None: _packed uses t_span = (TAU_START, max tau0)
    A["hist_c1_default_tau_save"] = f64([P.TAU_START, float(np.max(prepared.tau0))])
    index = prepared.model.index
    names = [n for n, _ in sorted(index.items(), key=lambda kv: kv[1])]
    A["hist_state_names"] = np.array(names)
    A["hist_state_index"] = np.array([index[n] for n in names], dtype=np.int64)
    META["notes"]["history"] = (
        "hist_c1_default: solve_perturbation_history_batch(K[::8], c1) with default arguments; its default save "
        "times are only the two endpoints (TAU_START, tau0), saved as hist_c1_default_tau_save. Shape "
        "(n_k, n_cosmo, n_save, nvar), k ascending. hist_c1_tau49: the same call with "
        "tau_save = concatenate(([TAU_START], SOLVE_EB_TAU)) (49 times, as test_solve_eb_keeps_what_the_history_holds "
        "builds it), saved as hist_c1_tau49_tau_save; saves past tau0 hold the final state. "
        "hist_state_names[i] is the state at index hist_state_index[i] (model.index; nvar = %d)." % prepared.model.nvar)
    log("hist default", A["hist_c1_default"].shape, "nvar", prepared.model.nvar, "tau0", prepared.tau0)
    save()
    tau49 = np.concatenate(([P.TAU_START], TP.SOLVE_EB_TAU))
    hist49, _, _ = P.solve_perturbation_history_batch(k16, C1, tau_save=tau49)
    A["hist_c1_tau49"] = f64(hist49)
    A["hist_c1_tau49_tau_save"] = f64(tau49)
    log("hist tau49", A["hist_c1_tau49"].shape, "finite", bool(np.isfinite(A["hist_c1_tau49"]).all()))


# ---------------------------------------------------------------- solve_eb as the test calls it
EB_NAMES = ("R", "clxc", "clxb", "theta_2", "n_15", "sigma", "adotoa")


@section("solve_eb")
def _solve_eb():
    A["eb_k"] = f64(TP.SOLVE_EB_K)
    A["eb_tau"] = f64(TP.SOLVE_EB_TAU)
    A["eb_names"] = np.array(EB_NAMES)
    sol = P.solve_eb(C2, TP.SOLVE_EB_K, TP.SOLVE_EB_TAU, keep=EB_NAMES)
    A["eb_c2_values"] = f64(sol.values)  # (n_cosmo, n_names, n_k, n_tau), caller's (descending) k order
    A["eb_c2_tau0"] = f64(sol.tau0)
    log("solve_eb c2", A["eb_c2_values"].shape, "tau0", sol.tau0)
    save()
    # the comparison solves of the same test
    tau49 = np.concatenate(([P.TAU_START], TP.SOLVE_EB_TAU))
    hist, k_sorted, prepared = P.solve_perturbation_history_batch(TP.SOLVE_EB_K, C2, tau_save=tau49)
    A["eb_c2_history"] = f64(hist)  # (n_k asc, n_cosmo, 49, nvar)
    A["eb_c2_history_k_sorted"] = f64(k_sorted)
    A["eb_c2_history_tau_save"] = f64(tau49)
    pk = P.solve_matter_power_spectrum_batch(TP.SOLVE_EB_K, C2, thermo_weight=discoeb.THERMO_ERROR_WEIGHT)
    A["eb_c2_pk_batch_thermo_weight1"] = f64(pk)
    # the test's last assertion, recomputed: P(k) from solve_eb's contrasts today against the batch P(k)
    rel = []
    for ci, cosmology in enumerate(prepared.cosmologies):
        grhoc, grhob = P.density_coefficients(cosmology)[2:4]
        clxc = A["eb_c2_values"][ci, EB_NAMES.index("clxc"), :, -1]
        clxb = A["eb_c2_values"][ci, EB_NAMES.index("clxb"), :, -1]
        delta_m = (grhoc * clxc + grhob * clxb) / (grhoc + grhob)
        k = np.asarray(TP.SOLVE_EB_K)
        pk_eb = 2.0 * np.pi**2 / k**3 * cosmology.A_s * (k / cosmology.k_pivot) ** (cosmology.n_s - 1.0) * delta_m**2
        rel.append(pk_eb / pk[ci] - 1.0)
    A["eb_c2_pk_from_solve_eb_rel_to_batch"] = f64(rel)
    log("max |pk(solve_eb)/pk(batch) - 1| =", np.abs(rel).max(), "(test gate rtol 1e-9)")
    order = np.argsort(TP.SOLVE_EB_K)
    kept = A["eb_c2_values"][:, :, order, :]
    h = A["eb_c2_history"][:, :, 1:, :]
    worst = {}
    for name in ("clxc", "clxb", "theta_2", "n_15"):
        j = EB_NAMES.index(name)
        exp = np.stack([h[:, ci, :, prepared.model.index[name]] for ci in range(2)])
        worst[name] = {"max_abs": float(np.abs(kept[:, j] - exp).max()), "bitwise_equal": bool(np.array_equal(kept[:, j], exp))}
    META["notes"]["solve_eb_vs_history"] = worst
    log("solve_eb vs history", worst)
    META["notes"]["solve_eb"] = (
        "eb_c2_values = solve_eb(c2, SOLVE_EB_K, SOLVE_EB_TAU, keep=eb_names).values with SOLVE_EB_K = K[::16][::-1] "
        "(descending) and SOLVE_EB_TAU = geomspace(1, 2e4, 48), exactly as test_solve_eb_keeps_what_the_history_holds. "
        "Shape (n_cosmo, n_names, n_k, n_tau) in the caller's k order.")


# ---------------------------------------------------------------- density check (host numpy vs JAX-side)
@section("densities")
def _densities():
    out = {}
    for tag, c in (("c1_0", C1[0]), ("c128_0", C128[0])):
        host = f64(P.density_coefficients(discoeb.as_cosmology(c)))
        fixed, theta = P._split_cosmologies(P._as_batch(c))
        # solve_eb.inputs_of for one cosmology: jnp.stack([_input_vector(fixed[0], jnp.asarray(theta)[0])])
        theta_j = jnp.asarray(theta, dtype=jnp.float64)
        vec = f64(jnp.stack([P._input_vector(fixed[0], theta_j[0])]))[0]
        # the test's own form (test_solve_eb_derivatives)
        theta_t = jnp.asarray([c[key] for key in P.DIFFERENTIABLE_PARAMETERS])
        fixed_t = {k: v for k, v in c.items() if k not in P.DIFFERENTIABLE_PARAMETERS}
        vec_t = f64(P._input_vector(fixed_t, theta_t))
        vec_d = f64(P.directed_inputs(c))[0]
        vec_np = f64(P._input_vector(fixed[0], np.asarray(theta, dtype=np.float64)[0], xp=np))
        A[f"dens_{tag}_host"] = host
        A[f"dens_{tag}_jax_solve_eb"] = vec
        A[f"dens_{tag}_jax_test_form"] = vec_t
        A[f"dens_{tag}_jax_directed_inputs"] = vec_d
        A[f"dens_{tag}_numpy_input_vector"] = vec_np
        u = ulps(host, vec[:5])
        A[f"dens_{tag}_ulp_jax_minus_host"] = u
        out[tag] = {
            "names": list(P.DIRECTED_NAMES[:5]),
            "host_hex": [float(x).hex() for x in host],
            "jax_hex": [float(x).hex() for x in vec[:5]],
            "ulp_jax_minus_host": u.tolist(),
            "bitwise_equal": bool(np.array_equal(host, vec[:5])),
            "test_form_equals_solve_eb_form": bool(np.array_equal(vec, vec_t)),
            "directed_inputs_equals_solve_eb_form": bool(np.array_equal(vec, vec_d)),
            "numpy_input_vector_equals_host": bool(np.array_equal(vec_np[:5], host)),
            "thermal_inputs_equal_plain": bool(np.array_equal(vec[5:], f64([c["YHe"], c["Tcmb"], c["H0"], c["Omegab"]]))),
            "jax_array_device": str(getattr(P._input_vector(fixed[0], theta_j[0]), "device", None)),
        }
        try:  # the same JAX arithmetic on the CPU backend, to separate XLA-GPU from XLA-CPU rounding
            with jax.default_device(jax.devices("cpu")[0]):
                vec_cpu = f64(jnp.stack([P._input_vector(fixed[0], jnp.asarray(theta, dtype=jnp.float64)[0])]))[0]
            A[f"dens_{tag}_jax_cpu_device"] = vec_cpu
            out[tag]["ulp_jaxcpu_minus_host"] = ulps(host, vec_cpu[:5]).tolist()
            out[tag]["ulp_jaxcpu_minus_jaxgpu"] = ulps(vec[:5], vec_cpu[:5]).tolist()
        except Exception as e:  # noqa: BLE001
            out[tag]["jax_cpu_device_error"] = repr(e)
        log(tag, json.dumps(out[tag], indent=1))
    # the whole batch of 128: host against solve_eb's batched (column-wise) JAX path
    host128 = f64([P.density_coefficients(discoeb.as_cosmology(c)) for c in C128])
    fixed, theta = P._split_cosmologies(tuple(C128))
    theta_j = jnp.asarray(theta, dtype=jnp.float64)
    keys = set(fixed[0])
    columns = {k: jnp.asarray([f[k] for f in fixed], dtype=jnp.float64) for k in keys}
    batch = f64(P._input_vector(columns, [theta_j[:, j] for j in range(theta_j.shape[1])]).T)
    single = f64(jnp.stack([P._input_vector(fixed[i], theta_j[i]) for i in range(len(fixed))]))
    A["dens_c128_host"] = host128
    A["dens_c128_jax_batched"] = batch
    A["dens_c128_jax_one_by_one"] = single
    u = ulps(host128, batch[:, :5])
    A["dens_c128_ulp_jax_batched_minus_host"] = u
    out["c128_batch"] = {
        "n_cosmologies_bitwise_equal_all5": int(np.all(u == 0, axis=1).sum()),
        "max_abs_ulp_per_density": np.abs(u).max(axis=0).tolist(),
        "n_differing_per_density": (u != 0).sum(axis=0).tolist(),
        "batched_equals_one_by_one": bool(np.array_equal(batch, single)),
    }
    log("c128", json.dumps(out["c128_batch"]))
    META["density_check"] = out
    META["notes"]["densities"] = (
        "dens_*_host = np.asarray(discoeb.density_coefficients(discoeb.as_cosmology(c))) (numpy on the host); "
        "dens_*_jax_solve_eb = the 9 DIRECTED_NAMES inputs exactly as solve_eb.inputs_of forms them for one cosmology "
        "(perturbations._input_vector(fixed, jnp theta), eager JAX on the default device); first five are the densities. "
        "ulp = int64 bit-pattern difference jax - host.")


# ---------------------------------------------------------------- thermal history
@section("thermo")
def _thermo():
    import test_thermo as TT

    A["thermo_names"] = np.array(TT.THERMAL_KEEP)
    A["thermo_k"] = f64(TT.THERMAL_K)
    tau, tau0, hist = TT.kernel_thermal_history(C1)
    A["thermo_c1_tau"] = f64(tau)
    A["thermo_c1_tau0"] = f64(tau0)
    A["thermo_c1"] = f64([hist[n][0] for n in TT.THERMAL_KEEP])  # (n_names, n_tau)
    log("thermo c1", A["thermo_c1"].shape)
    save()
    tau, tau0, hist = TT.kernel_thermal_history(C128)
    A["thermo_c128_tau"] = f64(tau)
    A["thermo_c128_tau0"] = f64(tau0)
    A["thermo_c128_idx0"] = f64([hist[n][0] for n in TT.THERMAL_KEEP])
    A["thermo_c128_idx81"] = f64([hist[n][81] for n in TT.THERMAL_KEEP])
    A["thermo_c128_all"] = f64([hist[n] for n in TT.THERMAL_KEEP])  # (n_names, 128, n_tau)
    save()
    # cosmology 81 solved alone on the batch's tau grid: does a trajectory depend on its batch?
    sol = discoeb.solve_eb([C128[81]], TT.THERMAL_K, tau, keep=TT.THERMAL_KEEP)
    alone = f64([np.asarray(sol[n])[0, 0, :] for n in TT.THERMAL_KEEP])
    A["thermo_c128_idx81_solved_alone"] = alone
    META["notes"]["thermo_idx81_alone_bitwise_equal_to_batch_row"] = bool(np.array_equal(alone, A["thermo_c128_idx81"]))
    META["notes"]["thermo_idx81_alone_max_rel_diff"] = float(
        np.max(np.abs(alone - A["thermo_c128_idx81"]) / np.maximum(np.abs(A["thermo_c128_idx81"]), 1e-300)))
    META["notes"]["thermo"] = (
        "tests/test_thermo.py::kernel_thermal_history(cosmologies): solve_eb(cosmologies, k=[1e-3], "
        "tau=geomspace(TAU_START, max tau0, 2000), keep=thermo_names). thermo_c1 from c1 (the reference cosmology); "
        "thermo_c128_idx0 / idx81 are rows 0 and 81 of the c128 batch solve (thermo_c128_all holds all 128), on "
        "thermo_c128_tau; thermo_c128_idx81_solved_alone is solve_eb([c128[81]]) on that same tau grid. "
        "Rows of each array follow thermo_names; values past a cosmology's own tau0 hold its state today.")
    log("thermo c128 done; idx81 alone == batch row:", META["notes"]["thermo_idx81_alone_bitwise_equal_to_batch_row"],
        META["notes"]["thermo_idx81_alone_max_rel_diff"])


# ---------------------------------------------------------------- derivatives: matter power spectrum
@section("deriv_pk")
def _deriv_pk():
    cosmology = C1[0]
    densities = jnp.asarray(np.asarray(P.density_coefficients(P._as_cosmology(cosmology)), dtype=np.float64))
    A["dpk_densities"] = f64(densities)
    A["dpk_density_names"] = np.array(TP.DENSITY_NAMES)

    def total_power(d):
        return jnp.sum(P.matter_power_spectrum_jax(K, cosmology, d))

    vg = jax.value_and_grad(total_power)
    (value, grad), t_first = timed(lambda: tuple(np.asarray(x) for x in jax.block_until_ready(vg(densities))))
    (value2, grad2), t_warm = timed(lambda: tuple(np.asarray(x) for x in jax.block_until_ready(vg(densities))))
    META["timings_s"]["dpk_value_and_grad_first_call"] = t_first
    META["timings_s"]["dpk_value_and_grad_warm"] = t_warm
    A["dpk_value"] = f64(value)
    A["dpk_grad"] = f64(grad)
    META["notes"]["dpk_repeat_bitwise_equal"] = bool(np.array_equal(grad, grad2) and np.array_equal(value, value2))
    log("d(sum P)/d densities", grad, "value", value, f"first {t_first:.1f} s warm {t_warm:.2f} s")
    save()
    A["dpk_pk_jax"] = f64(P.matter_power_spectrum_jax(K, cosmology, densities))
    A["dpk_primal_total"] = f64(total_power(densities))
    save()
    # the test's central differences (FD_STEP relative), its noisy side
    plus, minus, steps = [], [], []
    for i in range(5):
        step = TP.FD_STEP * abs(float(densities[i]))
        steps.append(step)
        plus.append(float(total_power(densities.at[i].add(step))))
        minus.append(float(total_power(densities.at[i].add(-step))))
    A["dpk_fd_step"] = f64(steps)
    A["dpk_fd_plus"] = f64(plus)
    A["dpk_fd_minus"] = f64(minus)
    A["dpk_fd"] = (A["dpk_fd_plus"] - A["dpk_fd_minus"]) / (2.0 * A["dpk_fd_step"])
    rel = np.abs(A["dpk_grad"] - A["dpk_fd"]) / np.abs(A["dpk_fd"])
    log("grad vs central difference, relative:", rel, "(test gate 5e-3)")
    META["notes"]["deriv_pk"] = (
        "As test_matter_power_spectrum_derivatives: cosmology c1[0], dpk_densities = host density_coefficients, "
        "dpk_value / dpk_grad = jax.value_and_grad(sum(matter_power_spectrum_jax(K, cosmology, d)))(densities). "
        "dpk_pk_jax is that P(k) itself; dpk_fd* are the test's central differences at relative step FD_STEP = 1e-2.")


# ---------------------------------------------------------------- derivatives: solve_eb
@section("deriv_eb")
def _deriv_eb():
    cosmology = C1[0]
    theta = jnp.asarray([cosmology[key] for key in P.DIFFERENTIABLE_PARAMETERS])
    fixed = {k: v for k, v in cosmology.items() if k not in P.DIFFERENTIABLE_PARAMETERS}
    densities = P._input_vector(fixed, theta)[None, :]
    tau = np.geomspace(10.0, 2.0e4, 12)
    names = ("R", "clxc", "theta_2")
    A["deb_theta"] = f64(theta)
    A["deb_theta_names"] = np.array(P.DIFFERENTIABLE_PARAMETERS)
    A["deb_inputs"] = f64(densities)  # (1, 9), DIRECTED_NAMES
    A["deb_input_names"] = np.array(P.DIRECTED_NAMES)
    A["deb_k"] = f64(TP.SOLVE_EB_K)
    A["deb_tau"] = f64(tau)
    A["deb_names"] = np.array(names)

    plain = P.solve_eb(cosmology, TP.SOLVE_EB_K, tau, keep=names)
    A["deb_plain_values"] = f64(plain.values)
    A["deb_tau0"] = f64(plain.tau0)
    prepared = P._prepare_solve((P._as_cosmology(cosmology),))
    tau_grid = np.concatenate(([P.TAU_START], tau))
    kept = P._kept_values(prepared, TP.SOLVE_EB_K, tau_grid, names, ())
    base = np.asarray(kept(densities))
    A["deb_base"] = f64(base)
    META["notes"]["deb_plain_equals_base_bitwise"] = bool(np.array_equal(A["deb_plain_values"], A["deb_base"]))
    scale = np.abs(base).max(axis=(2, 3), keepdims=True)
    A["deb_scale"] = f64(scale)
    save()

    (primal, _), t = timed(lambda: jax.block_until_ready(jax.jvp(kept, (densities,), (jnp.zeros_like(densities),))))
    META["timings_s"]["deb_jvp_first_call"] = t
    A["deb_jvp_primal"] = f64(primal)
    log(f"jvp primal first call {t:.1f} s; max |primal - base| / scale.max = {np.abs(A['deb_jvp_primal'] - base).max() / scale.max():.3g}")
    save()

    jac_full, t = timed(lambda: np.asarray(jax.jacfwd(kept)(densities)))
    META["timings_s"]["deb_jacfwd_inputs"] = t
    jac = jac_full[..., 0, :]
    A["deb_jac_inputs"] = f64(jac)  # (1, 3, n_k, n_tau, 9)
    log("jacfwd(kept)(inputs)", jac.shape, f"{t:.1f} s", "finite", bool(np.isfinite(jac).all()))
    save()

    weights = 1.0 / scale

    def total(d):
        return jnp.sum(kept(d) * weights)

    grad, t = timed(lambda: np.asarray(jax.grad(total)(densities))[0])
    META["timings_s"]["deb_grad_inputs"] = t
    A["deb_grad_inputs"] = f64(grad)
    summed = np.einsum("cqkt,cqktj->j", np.broadcast_to(weights, base.shape), jac)
    A["deb_grad_inputs_from_jac"] = f64(summed)
    log("grad vs contracted jacfwd rel:", np.abs(grad / summed - 1.0).max())
    save()

    def of_parameters(th):
        param = {**fixed, **dict(zip(P.DIFFERENTIABLE_PARAMETERS, th))}
        return P.solve_eb(param, TP.SOLVE_EB_K, tau, keep=names).values

    jac_theta, t = timed(lambda: np.asarray(jax.jacfwd(of_parameters)(theta)))
    META["timings_s"]["deb_jacfwd_theta"] = t
    A["deb_jac_theta"] = f64(jac_theta)  # (1, 3, n_k, n_tau, 5)
    d_densities = np.asarray(jax.jacfwd(lambda th: P._input_vector(fixed, th))(theta))
    A["deb_dinputs_dtheta"] = f64(d_densities)  # (9, 5)
    chain = np.einsum("cqktj,jp->cqktp", jac, d_densities)
    log("jac_theta vs chain rule: max abs diff / max|jac_theta| =", np.abs(jac_theta - chain).max() / np.abs(jac_theta).max())
    save()
    grad_theta, t = timed(lambda: np.asarray(jax.grad(lambda th: jnp.sum(of_parameters(th) * weights))(theta)))
    META["timings_s"]["deb_grad_theta"] = t
    A["deb_grad_theta"] = f64(grad_theta)
    log("grad_theta", grad_theta)
    META["notes"]["deriv_eb"] = (
        "As test_solve_eb_derivatives: cosmology c1[0], deb_k = SOLVE_EB_K (descending), deb_tau = geomspace(10, 2e4, 12), "
        "names (R, clxc, theta_2). deb_inputs = _input_vector(fixed, theta)[None] (JAX-side, 9 DIRECTED_NAMES). "
        "deb_plain_values = solve_eb(...).values; deb_base = _kept_values(...)(inputs); deb_jvp_primal = primal of the "
        "joint (fp64-LU sensitivity) launch; deb_jac_inputs = jacfwd(kept)(inputs)[..., 0, :] (1,3,n_k,n_tau,9); "
        "deb_grad_inputs = grad of sum(kept * 1/scale); deb_jac_theta = jacfwd of solve_eb values wrt "
        "DIFFERENTIABLE_PARAMETERS (1,3,n_k,n_tau,5); deb_grad_theta = grad of the weighted sum wrt theta; "
        "deb_dinputs_dtheta = jacfwd(_input_vector) (9,5). Derivatives hold tau0 and z_re fixed. "
        "deb_fd_jac_inputs (if present) = the test's converged central differences (rtol=atol=1e-6, max_steps=200000, "
        "relative step 1e-3) of kept wrt each of the 9 inputs.")
    save()

    kept_converged = P._kept_values(prepared, TP.SOLVE_EB_K, tau_grid, names, (), rtol=1.0e-6, atol=1.0e-6, max_steps=200_000)
    A["deb_base_converged"] = f64(kept_converged(densities))
    fd = np.empty_like(jac)
    t0 = time.perf_counter()
    noise = {"R": 5.0e-6, "clxc": 5.0e-6, "theta_2": 1.0e-4}
    report = {}
    for j, dn in enumerate(P.DIRECTED_NAMES):
        step = 1.0e-3 * float(densities[0, j])
        unit = jnp.zeros_like(densities).at[0, j].set(1.0)
        fd[..., j] = (np.asarray(kept_converged(densities + step * unit)) - np.asarray(kept_converged(densities - step * unit))) / (2.0 * step)
        err = np.abs(jac[..., j] - fd[..., j]) * step / scale
        report[dn] = {name: float(err[0, q].max()) for q, name in enumerate(names)}
    META["timings_s"]["deb_fd_converged"] = time.perf_counter() - t0
    A["deb_fd_jac_inputs"] = f64(fd)
    META["notes"]["deb_jac_vs_fd_err_over_scale"] = {"gates": noise, "measured": report}
    log("jac vs converged FD (err*step/scale):", json.dumps(report))


save()
log("DONE. arrays:")
for name in sorted(A):
    log(f"  {name:45s} {A[name].dtype!s:8s} {A[name].shape}")
log("timings:", json.dumps(META["timings_s"], indent=1))
log("errors:", list(META["errors"]))
log("wrote", OUT, os.path.getsize(OUT), "bytes")
