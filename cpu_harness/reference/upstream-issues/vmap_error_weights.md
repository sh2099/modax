# Per-trajectory error weights are read past their end under `jax.vmap`

Found 2026-11-20 with DISCO-EB 2.1, in modax v0.5.0 (`11991ee`); the
cpu-target branch leaves this code as it is. `vmap_error_weights.py` beside
this file reproduces it with modax alone.

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
