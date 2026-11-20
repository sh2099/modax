"""Compare two records of ``make_gpu_reference_21.py``: the CPU target's against a GPU's.

    python compare_with_gpu_21.py OURS.npz THEIRS.npz [OUT.json]

Needs only numpy: no modax, no DISCO-EB. ``THEIRS`` is the reference (the
GPU's), and every scale is taken from it.

Per array: whether the shapes match, and ``max`` and ``median`` of
``|ours - theirs| / scale``. The same numbers divided by the solve's
``rtol`` say how many tolerances apart the two are. The scale is a largest
magnitude in the reference, so that an element passing through zero is not
measured against itself:

* for kept quantities, ``(cosmology, name, k, a)`` (the histories and the
  reionization arrays), that of the quantity over every mode and scale
  factor. A solve holds each quantity to its tolerance on the scale the
  quantity has, not on the scale of one mode's curve: ``theta_b`` at the
  lowest mode peaks at 4e-11 of its largest value and ``E_2`` at 2e-5, and
  against their own curves the two records' rounding-level differences
  there read as 0.06 and 0.14. The per-curve numbers are reported beside the
  judged ones (``max (curve scale)``), so a curve that is wrong where its
  quantity is small still shows;
* for P(k), one curve; for a derivative, one column (one parameter over k).

Thresholds, in units of ``rtol`` (1e-4, DISCO-EB's default for both):

* values (P(k), histories, the reionization depth): max 10, median 1. Two
  solves of the same equations on different step sequences (a GPU warp
  advances an upstream block together, a CPU thread alone) differ by a few
  local tolerances, accumulated; P(k), quadratic in the contrasts, doubles
  that. DISCO-EB 2.0.1 on modax 0.3 agreed with a P100 to 2.7 (max) and 0.03
  (median) of them over 128 x 128 P(k) values.
* derivatives (jacfwd, jvp tangents, z_re): max 50, median 5. The joint
  launch runs with ``sens_error_control=False``, so nothing holds the
  sensitivities to the tolerance, and RECFAST's switches make them
  depend on the step sequence more than the values do; central differences
  of converged solves only pin these tangents to 2.5e-3 to 5.6e-3 of their
  scale (cpu_harness/results/discoeb_2.1.json).
* inputs must be bitwise identical, or nothing else means anything.
* step counts: the mean accepted steps per trajectory within 10 percent.

Self-checks within each record are printed too: ``vmap`` over three
cosmologies against the three one at a time, values and jacfwd, and
``split_derivatives`` against one thread. On the CPU target the vmapped
rows after the first are garbage: modax's ``prepare_inputs`` passes a 2-D
``(n, n_vars)`` ``error_weights`` through unchecked, solve_eb builds it for
one cosmology's 64 trajectories, and a vmap folds three cosmologies' 192
into one launch, so the kernel reads weights past the array's end. That is
upstream modax's code (unchanged on this branch); the GPU record says
whether the device reads the same garbage.

A flagged array is not a failure in itself: the summary says which and by
how much, for a person to judge.
"""

import json
import sys

import numpy as np

RTOL = 1.0e-4
VALUE_GATE = (10.0, 1.0)
DERIVATIVE_GATE = (50.0, 5.0)
STEP_GATE = 0.10
INPUTS = (
    "k",
    "a_grid",
    "input_keys",
    "lcdm32_inputs",
    "lcdm32_omega_c",
    "hist_k",
    "deriv_theta0",
    "vmap_jacfwd_theta",
)
LABELS = ("hist_keep", "hist_mnu_keep", "reio_keep", "deriv_names")
DERIVATIVES = (
    "jacfwd_pk",
    "jacfwd_pk_split",
    "jvp_pk_tangent",
    "vmap_jacfwd_pk",
    "loop_jacfwd_pk",
    "reio_jvp_tangent",
)
SELF_CHECKS = (
    ("vmap_jacfwd_pk", "loop_jacfwd_pk"),
    ("vmap_pk", "loop_pk"),
    ("jacfwd_pk_split", "jacfwd_pk"),
)


KEPT = ("hist_planck", "hist_mnu", "reio_values", "reio_jvp_value", "reio_jvp_tangent")
"""Arrays laid out (cosmology, name, k, a): judged against each quantity's scale over k and a."""


def series_scale(theirs, name):
    """The magnitude each element is measured against: the max over the axis that runs along one series."""
    t = np.abs(theirs)
    if name.startswith("hist") or name.startswith("reio"):
        axis = -1  # (cosmology, name, k, a): along a
    elif name.startswith("jacfwd") or name.endswith("_jacfwd_pk"):
        axis = -2  # (..., k, parameter): along k, one column at a time
    else:
        axis = -1  # P(k) curves, tangents along k
    scale = t.max(axis=axis, keepdims=True)
    return np.where(scale > 0.0, scale, 1.0)


def compare(ours, theirs):
    out = {}
    names = sorted((set(ours.files) | set(theirs.files)) - {"meta"})
    for name in names:
        if name not in ours.files or name not in theirs.files:
            out[name] = {
                "status": "MISSING in "
                + ("ours" if name not in ours.files else "theirs")
            }
            continue
        a, b = ours[name], theirs[name]
        if a.shape != b.shape:
            out[name] = {
                "status": "SHAPE",
                "ours": list(a.shape),
                "theirs": list(b.shape),
            }
            continue
        if (
            name in INPUTS
            or name in LABELS
            or name.endswith(("_param_names", "_param_values"))
        ):
            out[name] = {
                "status": "ok" if a.tobytes() == b.tobytes() else "INPUTS DIFFER",
                "kind": "input",
            }
            continue
        if name.startswith("stats_"):
            ma, mb = float(a.mean()), float(b.mean())
            entry = {
                "kind": "steps",
                "mean ours": ma,
                "mean theirs": mb,
                "max |diff|": int(np.abs(a - b).max()),
            }
            if name.endswith("accepted_steps"):
                entry["status"] = "ok" if abs(ma / mb - 1.0) <= STEP_GATE else "FLAG"
            else:
                entry["status"] = "info"
            out[name] = entry
            continue
        a, b = a.astype(np.float64), b.astype(np.float64)
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        rel = np.abs(a - b) / series_scale(b, name)
        curve = None
        if name in KEPT:
            curve = rel
            scale = np.abs(b).max(axis=(-2, -1), keepdims=True)
            rel = np.abs(a - b) / np.where(scale > 0.0, scale, 1.0)
        kind = "derivative" if name in DERIVATIVES else "value"
        gate = DERIVATIVE_GATE if kind == "derivative" else VALUE_GATE
        entry = {
            "kind": kind,
            "shape": list(a.shape),
            "max": float(np.nanmax(rel)),
            "median": float(np.nanmedian(rel)),
            "max / rtol": float(np.nanmax(rel) / RTOL),
            "median / rtol": float(np.nanmedian(rel) / RTOL),
            "at": [int(i) for i in np.unravel_index(np.nanargmax(rel), rel.shape)],
            "finite": finite,
        }
        if curve is not None:
            entry["scale"] = "quantity over k and a"
            entry["max (curve scale)"] = float(np.nanmax(curve))
            entry["median (curve scale)"] = float(np.nanmedian(curve))
            entry["at (curve scale)"] = [
                int(i) for i in np.unravel_index(np.nanargmax(curve), curve.shape)
            ]
        ok = (
            finite
            and entry["max / rtol"] <= gate[0]
            and entry["median / rtol"] <= gate[1]
        )
        entry["status"] = "ok" if ok else "FLAG"
        out[name] = entry
    return out


def main():
    ours, theirs = np.load(sys.argv[1]), np.load(sys.argv[2])
    meta = {
        side: json.loads(str(f["meta"]))
        for side, f in (("ours", ours), ("theirs", theirs))
    }
    result = {
        "files": sys.argv[1:3],
        "meta": {
            side: {
                k: m.get(k)
                for k in (
                    "target",
                    "gpu",
                    "cpu",
                    "jax_device",
                    "versions",
                    "discoeb_git",
                    "host",
                    "errors",
                )
            }
            for side, m in meta.items()
        },
        "thresholds (in rtol)": {
            "rtol": RTOL,
            "values max, median": VALUE_GATE,
            "derivatives max, median": DERIVATIVE_GATE,
            "accepted steps": STEP_GATE,
        },
        "arrays": compare(ours, theirs),
    }
    print(
        "self-checks within each record (max |x - y| / scale): vmap against one at a time, split against one thread"
    )
    result["self-checks"] = {}
    for side, f in (("ours", ours), ("theirs", theirs)):
        for x, y in SELF_CHECKS:
            if x in f.files and y in f.files:
                rel = float(np.nanmax(np.abs(f[x] - f[y]) / series_scale(f[y], y)))
                result["self-checks"][f"{side}: {x} vs {y}"] = rel
                print(f"  {side:6s} {x:16s} vs {y:16s} {rel:.2e}")
    for side, m in meta.items():
        if m.get("errors"):
            print(f"{side}: failed sections {sorted(m['errors'])}")
    print(
        f"{'array':34s} {'status':14s} {'max':>10s} {'median':>10s} {'max/rtol':>9s} {'med/rtol':>9s}  note"
    )
    for name, e in result["arrays"].items():
        if "max" in e:
            print(
                f"{name:34s} {e['status']:14s} {e['max']:10.2e} {e['median']:10.2e} {e['max / rtol']:9.2f} {e['median / rtol']:9.3f}  {e['kind']} at {e['at']}"
            )
            if "max (curve scale)" in e:
                print(
                    f"{'':34s} {'(per curve)':14s} {e['max (curve scale)']:10.2e} {e['median (curve scale)']:10.2e}"
                    f" {e['max (curve scale)'] / RTOL:9.2f} {e['median (curve scale)'] / RTOL:9.3f}  at {e['at (curve scale)']}"
                )
        elif e.get("kind") == "steps":
            print(
                f"{name:34s} {e['status']:14s} mean {e['mean ours']:.1f} against {e['mean theirs']:.1f}, max |diff| {e['max |diff|']}"
            )
        else:
            print(f"{name:34s} {e['status']}")
    bad = {
        n: e["status"]
        for n, e in result["arrays"].items()
        if e["status"] not in ("ok", "info")
    }
    result["summary"] = {
        "flagged": bad,
        "n_arrays": len(result["arrays"]),
        "pass": not bad,
    }
    print(
        f"\nSUMMARY: {len(result['arrays'])} arrays, {len(bad)} flagged"
        + "".join(f"\n  {n}: {s}" for n, s in bad.items())
    )
    if len(sys.argv) > 3:
        with open(sys.argv[3], "w") as f:
            json.dump(result, f, indent=1)
        print(f"wrote {sys.argv[3]}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
