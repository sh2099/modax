# CPU stack C against stack D (Apple M4 Pro, 2026-11-22)

C: DISCO-EB 2.0.1 (`6c48e2d`) on cpu-target-0.3 (`8206330`), `stack_c.py`
through `modax-0.3/cpu_harness/disco`. D: DISCO-EB 2.1.0 (`6385553`) on a
scratch copy of cpu-target-0.5 (`538a8a7`) with
`../upstream-issues/vmap_error_weights_fix.diff` applied, `stack_d.py`
through `disco --modax 0.5` with `MODAX_05` naming the copy. Enzyme
derivatives on both. The cosmologies are each test suite's
`_perturbed_cosmologies(n)`, the modes its `MATTER_POWER_K` (128). One run
each; warm is the best of three. Numbers in `stack_c.json`, `stack_d.json`.

| P(k) | C first call | C warm | D first call | D warm |
|---|---|---|---|---|
| 1 cosmology | 18.7 s | 0.013 s | 16.9 s | 0.014 s |
| 128 | 3.8 s | 0.93 s | 4.0 s | 0.92 s |
| 1024 | 10.6 s | 7.42 s | 10.8 s | 7.31 s |

| P(k), D against C | median rel. | 99.9th pct | largest | over 1e-3 |
|---|---|---|---|---|
| 1 x 128 | 2.8e-5 | 1.5e-4 | 1.5e-4 | 0 |
| 128 x 128 | 2.9e-5 | 1.6e-4 | 3.7e-4 | 0 |
| 1024 x 128 | 2.9e-5 | 1.9e-4 | 4.4e-4 | 0 |

D is 2.3e-5 above C on average; the largest differences are at k = 0.01 to
0.1 /Mpc. Both solve at rtol = atol = 1e-4. Not isolated: taken to be
2.1.0's own changes to the solve (it integrates in the scale factor).

Derivatives, one cosmology, five parameters: C (gradient of summed P(k) in
its five densities) 40.9 s first, 0.23 s warm; D (`jacfwd` of P(k) in
omega_c, omega_b, n_s, A_s, N_eff) 28.4 s first, 0.080 s warm. The
parameters differ, so the values were not compared.

Batched jacobians on D with the fix, five parameters, 128 modes:

| cosmologies | loop | `vmap(jacfwd)` | shared shift | vmap vs loop | shift vs loop |
|---|---|---|---|---|---|
| 8 | 0.65 s | 0.53 s | 0.55 s | 2.7e-5 | 2.1e-5 |
| 128 | 10.6 s | 8.5 s | 8.6 s | 7.8e-5 | 7.8e-5 |

Differences are the largest over the batch, relative to each parameter's
scale. "Shared shift" is `jacfwd` in one shift added to every cosmology of a
batch given to `solve_eb`, one launch with no vmap; it needs no fix. Not
explained: why these differences are larger than the 1e-7 of the 8-mode
test in `../upstream-issues/runs/calx069_fix_gpu.txt`.
