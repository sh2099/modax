"""Rodas5P on the CPU: modax's kernel compiled with plain numba, one trajectory per loop iteration.

A transcription of ``_make_kernel`` in ``modax/rodas5P.py`` with three
substitutions and nothing else changed in the stepping arithmetic:

==============================  ==============================================
CUDA original                   here
==============================  ==============================================
``cuda.jit(device=True)``       ``numba.njit``
``cuda.local.array(n, dtype)``  ``np.empty(n, dtype)``
``blockIdx * tpb + threadIdx``  the loop variable of a ``prange`` over trajectories
==============================  ==============================================

One thing differs in substance. The CUDA kernel takes ``df/dy`` and ``df/dt``
from Enzyme, which has no macOS build; here they are forward differences,
one perturbed right-hand side per colour group of the sparsity pattern
(``modax._sparsity.colour_sparsity``, the same colouring the Enzyme sweeps
use) and one more perturbed in ``t``. The matrix is the dense row-major one
with the dense partial-pivoting LU of ``modax.rodas5P.dense_lu_solver``; the
pattern only reduces the number of right-hand-side evaluations.

Not here: save hooks, sensitivities, the compiled sparse LU, the JAX bridge.

The tableau, the controller constants and the dense-output coefficients are
copied from ``modax/rodas5P.py`` and ``modax/_numba_common.py`` rather than
imported, since importing those modules needs ``numba_cuda_mlir``;
``tests/test_solver.py`` checks the copy against the source text.
"""

from __future__ import annotations

import math

import numba
import numpy as np
from numba import njit, prange
from numba.core.registry import CPUDispatcher

# fmt: off
# Rodas5P W-transformed coefficients (Steinebach 2023, BIT 63:27), as modax/rodas5P.py.
GAMMA = 0.21193756319429014

A21 = 3.0
A31 = 2.849394379747939
A32 = 0.45842242204463923
A41 = -6.954028509809101
A42 = 2.489845061869568
A43 = -10.358996098473584
A51 = 2.8029986275628964
A52 = 0.5072464736228206
A53 = -0.3988312541770524
A54 = -0.04721187230404641
A61 = -7.502846399306121
A62 = 2.561846144803919
A63 = -11.627539656261098
A64 = -0.18268767659942256
A65 = 0.030198172008377946

C21 = -14.155112264123755
C31 = -17.97296035885952
C32 = -2.859693295451294
C41 = 147.12150275711716
C42 = -1.41221402718213
C43 = 71.68940251302358
C51 = 165.43517024871676
C52 = -0.4592823456491126
C53 = 42.90938336958603
C54 = -5.961986721573306
C61 = 24.854864614690072
C62 = -3.0009227002832186
C63 = 47.4931110020768
C64 = 5.5814197821558125
C65 = -0.6610691825249471
C71 = 30.91273214028599
C72 = -3.1208243349937974
C73 = 77.79954646070892
C74 = 34.28646028294783
C75 = -19.097331116725623
C76 = -28.087943162872662
C81 = 37.80277123390563
C82 = -3.2571969029072276
C83 = 112.26918849496327
C84 = 66.9347231244047
C85 = -40.06618937091002
C86 = -54.66780262877968
C87 = -9.48861652309627

C2 = 0.6358126895828704
C3 = 0.4095798393397535
C4 = 0.9769306725060716
C5 = 0.4288403609558664

D1 = GAMMA
D2 = -0.42387512638858027
D3 = -0.3384627126235924
D4 =  1.8046452872882734
D5 =  2.325825639765069

# Dense-output rows (the h1, h2, h3 of the kernel's continuous extension).
H1 = (25.948786856663858, -2.5579724845846235, 10.433815404888879, -2.3679251022685204,
      0.524948541321073, 1.1241088310450404, 0.4272876194431874, -0.17202221070155493)
H2 = (-9.91568850695171, -0.9689944594115154, 3.0438037242978453, -24.495224566215796,
      20.176138334709044, 15.98066361424651, -6.789040303419874, -6.710236069923372)
H3 = (11.419903575922262, 2.8879645146136994, 72.92137995996029, 80.12511834622643,
      -52.072871366152654, -59.78993625266729, -0.15582684282751913, 4.883087185713722)
# fmt: on

ORDER = 6
FACTOR_MIN = 0.2
FACTOR_MAX = 6.0
SAFETY = 0.9

# Row s of C holds the s entries multiplying k_0 .. k_{s-1}; row s of A the
# s + 1 entries of the update after stage s (stages 0..4 only: 5..7 accumulate).
_C = np.zeros((8, 8))
for _s, _row in enumerate(
    (
        (C21,),
        (C31, C32),
        (C41, C42, C43),
        (C51, C52, C53, C54),
        (C61, C62, C63, C64, C65),
        (C71, C72, C73, C74, C75, C76),
        (C81, C82, C83, C84, C85, C86, C87),
    ),
    start=1,
):
    _C[_s, : len(_row)] = _row
_A = np.zeros((5, 5))
for _s, _row in enumerate(
    (
        (A21,),
        (A31, A32),
        (A41, A42, A43),
        (A51, A52, A53, A54),
        (A61, A62, A63, A64, A65),
    )
):
    _A[_s, : len(_row)] = _row
_D = np.array([D1, D2, D3, D4, D5, 0.0, 0.0, 0.0])
_CT = np.array([0.0, C2, C3, C4, C5, 1.0, 1.0, 1.0])
_H1, _H2, _H3 = np.array(H1), np.array(H2), np.array(H3)

SQRT_EPS = math.sqrt(np.finfo(np.float64).eps)


def controller_exponents(pcoeff: float, icoeff: float, dcoeff: float, order: int):
    """``(e1, e2, e3)`` of Soderlind's controller, as ``PIDGains.exponents`` in modax."""
    q = -1.0 / order
    return (
        q * (icoeff + pcoeff + dcoeff),
        -q * (pcoeff + 2.0 * dcoeff),
        q * dcoeff,
    )


def _colour_groups(n_vars: int, pattern):
    """The normalised pattern as flat arrays grouped by colour, for the finite-difference sweeps.

    Returns ``(n_colours, col_start, col_idx, ent_start, ent_row, ent_col)``:
    colour ``g`` perturbs the columns ``col_idx[col_start[g]:col_start[g+1]]``
    together and recovers the entries ``(ent_row, ent_col)`` in
    ``ent_start[g]:ent_start[g+1]``. With no pattern every column is its own
    colour and every row an entry.
    """
    if pattern is None:
        colour = list(range(n_vars))
        pattern = tuple(tuple(range(n_vars)) for _ in range(n_vars))
    else:
        from modax._sparsity import colour_sparsity

        colour = list(colour_sparsity(pattern).colour)
    n_colours = max(colour) + 1
    cols = [[] for _ in range(n_colours)]
    for c, g in enumerate(colour):
        cols[g].append(c)
    entries = [[] for _ in range(n_colours)]
    for row, row_cols in enumerate(pattern):
        for c in row_cols:
            entries[colour[c]].append((row, c))
    col_start = np.cumsum([0] + [len(c) for c in cols]).astype(np.int64)
    col_idx = np.array([c for group in cols for c in group], dtype=np.int64)
    ent_start = np.cumsum([0] + [len(e) for e in entries]).astype(np.int64)
    ent_row = np.array([r for group in entries for r, _ in group], dtype=np.int64)
    ent_col = np.array([c for group in entries for _, c in group], dtype=np.int64)
    return n_colours, col_start, col_idx, ent_start, ent_row, ent_col


def as_cpu_function(fn):
    """``fn`` compiled with ``njit`` unless it already is."""
    if isinstance(fn, CPUDispatcher):
        return fn
    return njit(fn)


_SOLVERS: dict = {}


def make_solver(
    ode_fn,
    n_vars: int,
    n_params: int,
    *,
    sparsity=None,
    lu_precision: str = "fp64",
    tf_index: int = -1,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
):
    """Build the Rodas5P solver of ``ode_fn(y, t, p) -> tuple``.

    Returns an object with ``solve(y0, t_span, params, ...)``, ``jacobian(y, t,
    p, ...)`` (the finite-difference ``df/dy`` and ``df/dt`` the stepper uses)
    and ``rhs(y, t, p)``. ``sparsity`` is whatever
    ``modax._sparsity.normalize_sparsity`` accepts; ``lu_precision`` is
    ``"fp64"`` or ``"fp32"`` as in modax (the CUDA forward solve defaults to
    fp32); ``tf_index`` names the parameter column holding each trajectory's own
    end time; the gains are modax's ``pcoeff``, ``icoeff``, ``dcoeff``.
    """
    if lu_precision not in ("fp32", "fp64"):
        raise ValueError(f"lu_precision must be 'fp32' or 'fp64', got {lu_precision!r}")
    key_pattern = None
    if sparsity is not None:
        from modax._sparsity import normalize_sparsity

        key_pattern = normalize_sparsity(sparsity, n_vars)
    key = (
        ode_fn,
        n_vars,
        n_params,
        key_pattern,
        lu_precision,
        tf_index,
        pcoeff,
        icoeff,
        dcoeff,
    )
    if key in _SOLVERS:
        return _SOLVERS[key]

    f = as_cpu_function(ode_fn)
    n = int(n_vars)
    lu_dtype = np.float32 if lu_precision == "fp32" else np.float64
    n_colours, col_start, col_idx, ent_start, ent_row, ent_col = _colour_groups(
        n, key_pattern
    )
    e1, e2, e3 = controller_exponents(pcoeff, icoeff, dcoeff, ORDER)
    C, A, D, CT, HH1, HH2, HH3 = _C, _A, _D, _CT, _H1, _H2, _H3

    @njit
    def ode_write(y_row, t, p_row, out):
        values = f(y_row, t, p_row)
        for j in range(n):
            out[j] = values[j]

    @njit
    def factorize_local(lu, ipiv):
        for i in range(n):
            max_val = abs(lu[i * n + i])
            pivot = i
            for k in range(i + 1, n):
                val = abs(lu[k * n + i])
                if val > max_val:
                    max_val = val
                    pivot = k
            ipiv[i] = pivot
            if pivot != i:
                for j in range(n):
                    tmp = lu[i * n + j]
                    lu[i * n + j] = lu[pivot * n + j]
                    lu[pivot * n + j] = tmp
            piv = lu[i * n + i]
            if piv != 0.0:
                inv_piv = 1.0 / piv
                for k in range(i + 1, n):
                    factor = lu[k * n + i] * inv_piv
                    lu[k * n + i] = factor
                    for j in range(i + 1, n):
                        lu[k * n + j] -= factor * lu[i * n + j]

    @njit
    def solve_local(lu, ipiv, rhs):
        for i in range(n):
            pivot = ipiv[i]
            if pivot != i:
                tmp = rhs[i]
                rhs[i] = rhs[pivot]
                rhs[pivot] = tmp
            acc = rhs[i]
            for j in range(i):
                acc -= lu[i * n + j] * rhs[j]
            rhs[i] = acc
        for i in range(n - 1, -1, -1):
            acc = rhs[i]
            for j in range(i + 1, n):
                acc -= lu[i * n + j] * rhs[j]
            rhs[i] = acc / lu[i * n + i]

    @njit
    def write_negated_jacobian(
        y, t, p_row, lu, dT, f0, y_pert, f_pert, fd_floor, t_scale
    ):
        """``-df/dy`` into ``lu`` (dense row-major) and ``df/dt`` into ``dT``, by forward differences.

        ``f0`` must hold ``f(y, t, p)``. Column ``c`` is perturbed by
        ``sqrt(eps) * max(|y_c|, fd_floor)``; the columns of a colour group
        share no row, so one evaluation yields the whole group.
        """
        for i in range(n * n):
            lu[i] = 0.0
        for j in range(n):
            y_pert[j] = y[j]
        for g in range(n_colours):
            for a in range(col_start[g], col_start[g + 1]):
                c = col_idx[a]
                y_pert[c] = y[c] + SQRT_EPS * max(abs(y[c]), fd_floor)
            ode_write(y_pert, t, p_row, f_pert)
            for a in range(ent_start[g], ent_start[g + 1]):
                r = ent_row[a]
                c = ent_col[a]
                # the step actually taken, after rounding into y_pert
                h = y_pert[c] - y[c]
                lu[r * n + c] = -(f_pert[r] - f0[r]) / h
            for a in range(col_start[g], col_start[g + 1]):
                c = col_idx[a]
                y_pert[c] = y[c]
        ht = SQRT_EPS * max(abs(t), t_scale)
        t_pert = t + ht
        ht = t_pert - t
        ode_write(y, t_pert, p_row, f_pert)
        for j in range(n):
            dT[j] = (f_pert[j] - f0[j]) / ht

    @njit
    def next_step(err_norm, err_prev, err_prev2, accept):
        if math.isnan(err_norm) or err_norm > 1e18:
            safe_err = 1e18
        elif err_norm == 0.0:
            safe_err = 1e-18
        else:
            safe_err = err_norm
        factor = SAFETY * safe_err**e1 * err_prev**e2 * err_prev2**e3
        if accept:
            err_prev2 = err_prev
            err_prev = safe_err
        if factor < FACTOR_MIN:
            factor = FACTOR_MIN
        elif factor > FACTOR_MAX:
            factor = FACTOR_MAX
        return factor, err_prev, err_prev2

    @njit
    def solve_one(
        i, y0, times, params, dt0, rtol, atol, max_steps, weights, fd_floor,
        hist, accepted_out, rejected_out, loop_out,
    ):  # fmt: skip
        n_save = times.shape[0]
        tf = times[n_save - 1]
        dt_init = dt0 if dt0 > 0.0 else (tf - times[0]) * 1e-6

        lu_buf = np.empty(n * n, lu_dtype)
        rhs_buf = np.empty(n, lu_dtype)
        ipiv_buf = np.empty(n, np.int32)
        y = np.empty(n, np.float64)
        u = np.empty(n, np.float64)
        work = np.empty(n, np.float64)
        dT = np.empty(n, np.float64)
        k_stages = np.empty((8, n), lu_dtype)
        f0 = np.empty(n, np.float64)
        y_pert = np.empty(n, np.float64)
        f_pert = np.empty(n, np.float64)

        for j in range(n):
            y[j] = y0[i, j]
            hist[i, 0, j] = y[j]

        t = times[0]
        save_idx = 1
        dt = dt_init
        n_steps = 0
        accepted = 0
        rejected = 0
        err_prev = 1.0
        err_prev2 = 1.0

        tf_local = tf
        if tf_index >= 0:
            tf_local = params[i, tf_index]
        p_row = params[i]

        while save_idx < n_save and t < tf_local and n_steps < max_steps:
            dt_use = dt
            dt_use = min(dt_use, tf_local - t)
            dt_use = max(dt_use, 1e-30)
            inv_dt = 1.0 / dt_use
            t_end = t + dt_use

            dtgamma_inv = 1.0 / (dt_use * GAMMA)
            ode_write(y, t, p_row, f0)
            write_negated_jacobian(
                y, t, p_row, lu_buf, dT, f0, y_pert, f_pert, fd_floor, dt_use
            )
            for d in range(n):
                lu_buf[d * n + d] += lu_dtype(dtgamma_inv)
            factorize_local(lu_buf, ipiv_buf)

            # The eight stages: right-hand side from the tableau row, one solve
            # against the step's factorisation, then the update of u.
            for s in range(8):
                if s == 0:
                    for j in range(n):
                        work[j] = f0[j]
                elif s < 5:
                    ode_write(u, t + CT[s] * dt_use, p_row, work)
                else:
                    ode_write(u, t_end, p_row, work)
                d = D[s]
                for j in range(n):
                    if s == 0:
                        value = work[j] + dt_use * d * dT[j]
                    else:
                        acc = C[s, 0] * k_stages[0, j]
                        for k in range(1, s):
                            acc += C[s, k] * k_stages[k, j]
                        if d != 0.0:
                            value = work[j] + dt_use * d * dT[j] + acc * inv_dt
                        else:
                            value = work[j] + acc * inv_dt
                    rhs_buf[j] = lu_dtype(value)
                solve_local(lu_buf, ipiv_buf, rhs_buf)
                for j in range(n):
                    k_stages[s, j] = rhs_buf[j]
                if s < 5:
                    for j in range(n):
                        acc = A[s, 0] * k_stages[0, j]
                        for k in range(1, s + 1):
                            acc += A[s, k] * k_stages[k, j]
                        u[j] = y[j] + acc
                elif s < 7:
                    for j in range(n):
                        u[j] += k_stages[s, j]

            err_local = 0.0
            for j in range(n):
                y_new_j = u[j] + k_stages[7, j]
                scale = atol + rtol * max(math.fabs(y[j]), math.fabs(y_new_j))
                r = weights[i, j] * k_stages[7, j] / scale
                err_local += r * r
            err_norm = math.sqrt(err_local / n)
            accept = err_norm <= 1.0 and not math.isnan(err_norm)

            factor, err_prev, err_prev2 = next_step(
                err_norm, err_prev, err_prev2, accept
            )
            dt = dt_use * factor

            if accept:
                t_old = t
                t_new = t_old + dt_use
                while save_idx < n_save and times[save_idx] <= t_new + 1e-12 * max(
                    1.0, math.fabs(times[save_idx])
                ):
                    theta = (times[save_idx] - t_old) / dt_use
                    theta1 = 1.0 - theta
                    for j in range(n):
                        h1 = HH1[0] * k_stages[0, j]
                        h2 = HH2[0] * k_stages[0, j]
                        h3 = HH3[0] * k_stages[0, j]
                        for k in range(1, 8):
                            h1 += HH1[k] * k_stages[k, j]
                            h2 += HH2[k] * k_stages[k, j]
                            h3 += HH3[k] * k_stages[k, j]
                        y_new_j = u[j] + k_stages[7, j]
                        hist[i, save_idx, j] = theta1 * y[j] + theta * (
                            y_new_j + theta1 * (h1 + theta * (h2 + theta * h3))
                        )
                    save_idx += 1
                for j in range(n):
                    y[j] = u[j] + k_stages[7, j]
                t += dt_use
                accepted += 1
            else:
                rejected += 1
            n_steps += 1

        while save_idx < n_save:
            for j in range(n):
                hist[i, save_idx, j] = y[j]
            save_idx += 1
        accepted_out[i] = accepted
        rejected_out[i] = rejected
        loop_out[i] = n_steps

    @njit
    def run_serial(
        y0, times, params, dt0, rtol, atol, max_steps, weights, fd_floor,
        hist, accepted_out, rejected_out, loop_out,
    ):  # fmt: skip
        for i in range(y0.shape[0]):
            solve_one(
                i, y0, times, params, dt0, rtol, atol, max_steps, weights, fd_floor,
                hist, accepted_out, rejected_out, loop_out,
            )  # fmt: skip

    @njit(parallel=True)
    def run_parallel(
        y0, times, params, dt0, rtol, atol, max_steps, weights, fd_floor,
        hist, accepted_out, rejected_out, loop_out,
    ):  # fmt: skip
        for i in prange(y0.shape[0]):
            solve_one(
                i, y0, times, params, dt0, rtol, atol, max_steps, weights, fd_floor,
                hist, accepted_out, rejected_out, loop_out,
            )  # fmt: skip

    @njit
    def jacobian_at(y, t, p_row, fd_floor, t_scale):
        lu = np.empty(n * n, np.float64)
        dT = np.empty(n, np.float64)
        f0 = np.empty(n, np.float64)
        y_pert = np.empty(n, np.float64)
        f_pert = np.empty(n, np.float64)
        ode_write(y, t, p_row, f0)
        write_negated_jacobian(
            y, t, p_row, lu, dT, f0, y_pert, f_pert, fd_floor, t_scale
        )
        return -lu.reshape(n, n), dT, f0

    solver = Rodas5PCPU(
        n, int(n_params), n_colours, run_serial, run_parallel, jacobian_at, f
    )
    _SOLVERS[key] = solver
    return solver


class Rodas5PCPU:
    """A compiled CPU Rodas5P solver; see :func:`make_solver`."""

    def __init__(
        self, n_vars, n_params, n_colours, run_serial, run_parallel, jacobian_at, f
    ):
        self.n_vars, self.n_params, self.n_colours = n_vars, n_params, n_colours
        self._run_serial, self._run_parallel = run_serial, run_parallel
        self._jacobian_at, self._f = jacobian_at, f

    def rhs(self, y, t, p):
        return np.array(
            self._f(np.asarray(y, np.float64), float(t), np.asarray(p, np.float64))
        )

    def jacobian(self, y, t, p, *, fd_floor: float = 1.0, t_scale: float = 0.0):
        """``(df/dy, df/dt, f)`` at one point, by the stepper's finite differences."""
        return self._jacobian_at(
            np.ascontiguousarray(y, np.float64),
            float(t),
            np.ascontiguousarray(p, np.float64),
            float(fd_floor),
            float(t_scale),
        )

    def solve(
        self,
        y0,
        t_span,
        params,
        *,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        first_step: float | None = None,
        max_steps: int = 100_000,
        error_weights=None,
        fd_floor: float | None = None,
        parallel: bool = True,
        threads: int | None = None,
    ):
        """Integrate every row of ``y0``; returns ``(hist, stats)``.

        ``hist`` is ``(n, len(t_span), n_vars)`` as modax's; ``stats`` holds the
        per-trajectory ``accepted_steps``, ``rejected_steps`` and
        ``loop_steps``. ``fd_floor`` is the absolute scale below which a state
        component is perturbed as if it had that size (default ``atol / rtol``).
        ``threads`` caps the parallel loop, which hands out one trajectory at a
        time since their step counts differ widely.
        """
        y0 = np.ascontiguousarray(np.atleast_2d(np.asarray(y0, np.float64)))
        params = np.ascontiguousarray(np.atleast_2d(np.asarray(params, np.float64)))
        times = np.ascontiguousarray(np.asarray(t_span, np.float64))
        n_traj = max(y0.shape[0], params.shape[0])
        if y0.shape[0] != n_traj:
            y0 = np.ascontiguousarray(np.broadcast_to(y0, (n_traj, self.n_vars)))
        if params.shape[0] != n_traj:
            params = np.ascontiguousarray(
                np.broadcast_to(params, (n_traj, params.shape[1]))
            )
        if error_weights is None:
            weights = np.ones((n_traj, self.n_vars))
        else:
            weights = np.ascontiguousarray(
                np.broadcast_to(
                    np.asarray(error_weights, np.float64), (n_traj, self.n_vars)
                )
            )
        if fd_floor is None:
            fd_floor = atol / rtol if rtol > 0.0 else 1.0
        dt0 = 0.0 if first_step is None else float(first_step)
        hist = np.empty((n_traj, len(times), self.n_vars))
        accepted = np.zeros(n_traj, np.int32)
        rejected = np.zeros(n_traj, np.int32)
        loops = np.zeros(n_traj, np.int32)
        args = (
            y0, times, params, dt0, float(rtol), float(atol), int(max_steps), weights,
            float(fd_floor), hist, accepted, rejected, loops,
        )  # fmt: skip
        if parallel and n_traj > 1:
            previous = numba.get_num_threads()
            if threads is not None:
                numba.set_num_threads(int(threads))
            try:
                with numba.parallel_chunksize(1):
                    self._run_parallel(*args)
            finally:
                numba.set_num_threads(previous)
        else:
            self._run_serial(*args)
        stats = {
            "accepted_steps": accepted,
            "rejected_steps": rejected,
            "loop_steps": loops,
        }
        return hist, stats
