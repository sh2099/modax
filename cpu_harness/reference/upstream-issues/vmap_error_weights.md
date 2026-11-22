# Per-trajectory error weights are read past their end under `jax.vmap`

Found 2026-11-20 with DISCO-EB 2.1, in modax v0.5.0 (`11991ee`); the
cpu-target branch leaves this code as it is. Beside this file:
`vmap_error_weights.py`, the minimal reproducer (weak on a GPU: rows off by
1e-11 to 1e-9); `vmap_error_weights_loud.py`, the one to show; 
`vmap_error_weights_evidence.py`, the observation, intervention and
dose-response below; `vmap_error_weights_discoeb.py`, the same through
DISCO-EB; `runs/`, their outputs.

## What happens

`modax.solve(..., error_weights=w)` takes `w` as `(n_vars,)` or as
`(N, n_vars)`, one row per trajectory. Give it the 2-D form and map the
solve with `jax.vmap` (or `jax.vmap(jax.jacfwd(...))`, the batch form
DISCO-EB's docs recommend): the mapped solves are folded into one launch of
`axis_size * N` trajectories, but `w` still has `N` rows. Every trajectory
after the first `N` gets its weights from memory past the end of the
array. Its step control is then arbitrary, and so are its results:

- the reproducer (a two-state cascade, `N = 1`, three mapped parameter
  sets): rows 2 and 3 wrong by 3.8e-5 and 5.0e-5 of the scale in
  `jacfwd`, 1.8e-6 and 1.2e-5 in the values, and by 4e-10 to 1 in other
  runs; with no weights or with `(n_vars,)` weights every row agrees
  bitwise;
- DISCO-EB 2.1 (`solve_eb` always passes `(n_traj, n_vars)` weights),
  `vmap(jacfwd)` of P(k) over three cosmologies: rows 2 and 3 off by up to
  1.0 of the scale on the CPU target and by 1.5e60 on an RTX A400 with
  stock modax 0.5.0 (`cpu_harness/results/gpu_comparison_2.1.json`,
  "self-checks"). Plain `jax.vmap` of `solve_eb` raises "vmap over an
  already-ensembled y0 is not supported" before getting there.

Nothing raises, on either target: the kernel indexes `weights[i, j]` for
every trajectory `i` without a bounds check.

## Where (v0.5.0)

- `modax/_numba_common.py:199-213`, `build_error_weights`: a 2-D array
  "passes through as it is", with no check that it has `n` rows.
- `modax/_jax_common.py:147`, `prepare_inputs`: calls it with the launch's
  `n`, which after a vmap is `axis_size * N`.
- `modax/_jax_common.py:377-407`, the directional solver's `_solve_vmap`
  (`make_directional_solver`), and `:245-266`, `make_custom_vmap_solver`'s:
  they flatten `y0`, `params` and the tangents to `axis_size * N` rows
  (`_ensemble_rows`, `_broadcast_for_vmap`) but the weights are not an
  argument of the mapped function. They travel in `settings`, closed over
  by `run_for` (`modax/_solve.py:225-241`), which repeats them only for
  `max_directions_per_thread` replicas (`copies`), not for mapped
  trajectories.

## Evidence (2026-11-21)

Run on stock modax-solvers 0.5.0 (the installed wheel, no source edits) on
calx069's RTX A400 in DISCO-EB 2.1's stock environment, and on this
branch's CPU target on a Mac as support. Outputs in `runs/`.

**1. The mismatch, observed at launch** (`vmap_error_weights_evidence.py`,
`vmap_error_weights_discoeb.py`). A runtime wrap of the name
`build_error_weights` that `prepare_inputs` calls prints what it is given,
the launch's trajectory count `n`, and what it returns:

    loop, (N, n_vars) weights:   ((2, 3), 2, 3, (2, 3))
    vmap(jacfwd) over B=4:       ((2, 3), 8, 3, (2, 3))       <- 8 trajectories, 2 weight rows
    DISCO-EB loop:               ((8, 55), 8, (8, 55))
    DISCO-EB vmap(jacfwd) of 3:  ((8, 55), 24, (8, 55))       <- 24 trajectories, 8 rows

**2. Intervention.** The same mapped call, with only the returned weights
tiled to `n` rows in the order the vmap rule flattens trajectories
(`b * N + i`), against the same calls one at a time, max |diff| / scale per
mapped row:

| | as shipped | tiled at launch |
|---|---|---|
| Robertson, `(N, n_vars)` ones, GPU | 0, 1.1e-7, 3.1e-7, 8.5e-8 | 0, 0, 0, 0 (bitwise) |
| the discriminator: `(n_vars,)` weights `[1, 1e-3, 1e2]`, GPU | 0, 0, 0, 0 | |
| the same numbers as `(N, n_vars)`, GPU | 0, 3.4e-8, 3.8e-7, 2.8e-8 | 0, 0, 0, 0 |
| values, `jax.vmap(f)`, `(1, n_vars)`, GPU (absolute; values of order 1) | 0, 7.4e-4, 6.0e-7, 6.0e-7 | 0, 0, 0, 0 |
| DISCO-EB P(k) jacfwd, 3 cosmologies, GPU | 1.3e-7, **2.3e20, 3.4e24** | 1.3e-7, 9.2e-8, 4.7e-7 |
| DISCO-EB, CPU target | 6.4e-7, 1.9e51, 8.5e51 | 6.4e-7, 1.9e-16, 1.8e-16 |

Same numbers, different shape: the `(n_vars,)` form passes and the
`(N, n_vars)` form with the identical values fails. Given `n` rows, every
case agrees bitwise (the toy), or to 5e-7 for DISCO-EB on the GPU, where a
24-trajectory launch puts other modes in a warp than the 8-trajectory loop
does and its hierarchy kernel advances an upstream block per warp; that is
the level of the unmapped checks, rtol being 1e-4. The DISCO-EB reference
cosmology's own 6.4e-7 (CPU) and 1.3e-7 (GPU) appear tiled or not and follow
that cosmology to whichever position it is mapped at, so they are not the
weights; not traced further.

**3. Dose-response.** The tail trick (handing modax the first `N` rows of a
larger buffer of ours) is not sound: `prepare_inputs` copies the weights
into a JAX array (`jnp.asarray`), so what the kernel reads past the end is
XLA's memory, not ours, on either target. Instead the launch's weights are
padded to `n` rows with a chosen value `v` and the mapped rows `b >= 1`
compared with single launches given `v` explicitly. Bitwise equal for every
`v` on both targets, although `v` changes the answer of an honest launch by
0 (`v = 1`), 2.2e-10 (0.5), 4.9e-10 (0), and 0.23 (1e300, NaN: the step
controller gives up):

    pad v = 1.0     per row: 0  0  0  0        honest launch changed by 0.0
    pad v = 0.0     per row: 0  0  0  0        honest launch changed by 4.896e-10
    pad v = 1e300   per row: 0  0  0  0        honest launch changed by 0.2253
    pad v = nan     per row: 0  0  0  0        honest launch changed by 0.2253

So the kernel takes trajectory `b * N + i`'s weights from row `b * N + i` of
whatever it is handed, and its answer follows whatever is there. What the
memory past a short array holds is not controlled, and that is why the
shipped call is wrong by different amounts in different runs, and sometimes
not at all: of three CPU runs of the evidence script, two had the rows wrong
and one right.

**4. A loud reproducer** (`vmap_error_weights_loud.py`, no monkeypatch):
three Robertson systems side by side (9 states), 32 trajectories a call,
`vmap(jacfwd)` over 16 scalings, so 480 trajectories read their weights past
the end. Three GPU runs, `(N, n_vars)` ones:

    per row max |diff| / scale: 0  1.7e-07 ... 1.1e-07                (15 of 16 rows)
    mean accepted steps, mapped: 128  979 2065 4276 3448 ... 3802      (run 3: 981 for row 1)
    mean accepted steps, alone:  128  128  128  128  128 ... 130

and the `(n_vars,)` control agrees bitwise with the same step counts. On the
GPU the stray weights are large, so the controller takes up to 30 times the
steps and the answer is, if anything, more accurate; on the CPU target they
were tiny: one to six steps or none, rows wrong by 1e301 (one run) or
0.6-0.97 of the scale (another). Either way every mapped trajectory after the first call's
is solved under weights nobody gave it.

**5. DISCO-EB alone** (`vmap_error_weights_discoeb.py`, DISCO-EB's public
API plus the stated wrap): the table's DISCO-EB rows, 2.8 minutes on the
A400.

**Rivals ruled out.**
- (a) A tangent or direction layout fault under vmap in 0.5's scheme
  (`dp` in the parameter row, replicas, `copies`): with no weights or
  `(n_vars,)` weights, `vmap(jacfwd)` is bitwise the loop with
  `max_directions_per_thread` None, 1 and 2, on the GPU and the CPU target;
  so is the `(N, n_vars)` form once tiled.
- (b) Something else in the 2-D-weights path: the 2-D form with the
  passing 1-D form's exact values fails, and the same array tiled to `n`
  rows passes, bitwise; the only thing changed is the row count.
- (c) Something in solve_eb's own use (its hook, keep, grid): solve_eb's
  vmapped `jacfwd` agrees with the loop once only the weights are tiled.

**Not verified.** The GPU's stray values themselves (the device memory
after the weights buffer) were not read; that they are what the kernel used
is inferred from 2 and 3. The `copies > 1` replica path combined with a
vmap over 2-D weights (`run_for` repeats the rows for replicas) was tested
only with the tiling intervention, not as shipped.

## Why it goes unnoticed

The tests that vmap a solve pass no 2-D weights, and a 2-D array of the
wrong length is never rejected. DISCO-EB's own vmap test maps
`matter_power_spectrum`, not `solve_eb`'s derivative.

## Suggested fix

Two parts, either of which stops the silent read:

1. `build_error_weights` should raise unless a 2-D array has exactly `n`
   rows (or one, broadcast), as `prepare_inputs` already does for a
   mismatched `t_span`.
2. Under a vmap the weights have to follow the trajectories. The simplest
   form keeps them out of the closure: tile a 2-D `error_weights` by the
   mapped axis where the rule flattens `y0` and `params` (`np.tile(w,
   (axis_size, 1))`, matching `_ensemble_rows`' row order `b * N + i`),
   in both `_solve_vmap` rules. Mapping the weights themselves (a
   per-cosmology weight) would need them as a traced argument of the
   custom_vmap'd function.

Not fixed on cpu-target-0.5, which is kept behaving as upstream does.

## A fix, tested (2026-11-22)

`vmap_error_weights_fix.diff`, against v0.5.0: `build_error_weights` repeats
a 2-D array whose row count divides the launch's `n` until it has `n` rows
(`np.tile`, so trajectory `b * N + i` reads row `i`, the order the vmap
rules flatten in) and raises for any other row count. Not applied to
cpu-target-0.5. It is the fix at the point the weights are built; tiling in
the two vmap rules and raising here on any mismatch would be stricter (a
wrong row count that divides `n` is tiled, not refused).

On calx069's RTX A400, a copy of upstream v0.5.0 with the patch, no
monkeypatching (`runs/calx069_fix_gpu.txt`):

- `vmap_error_weights_loud.py`: all 16 mapped rows bitwise equal to the
  calls one at a time, the same step counts, exit 0;
- `vmap_error_weights_three_ways.py` (DISCO-EB 2.1.0, three cosmologies):
  `vmap(jacfwd)` against the loop 7.7e-8, 0, 5.9e-6, where the stock wheel
  gives 7.7e-8, 2.3e20, 3.4e24.

On the CPU target with the patch, 128 cosmologies x 128 modes x 5
parameters: `vmap(jacfwd)` within 7.8e-5 of the loop
(`../stack-comparison/RESULTS.md`).

## Without a fix: a batch in one launch

`vmap_error_weights_three_ways.py` also takes the jacobians with no vmap:
the batch goes to `solve_eb` as a list, each cosmology's parameters plus one
shared shift, and `jax.jacfwd` is taken in the shift. The cosmologies are
independent, so that is each one's own jacobian, in one launch whose weights
have the right rows. On the stock wheel on the A400 it agrees with the loop
to 1.4e-7 and takes 0.82 s against the loop's 2.16 s.

modax's own suite on a copy of cpu-target-0.5 with the patch (CPU target,
Enzyme): 260 passed, 5 skipped, 1 failed, the same as without it (the
failure is the step-sequence test that fails on every stack). Not run on a
GPU.

Reported upstream on 2026-11-22: https://github.com/lawrenceberry/modax/issues/2
