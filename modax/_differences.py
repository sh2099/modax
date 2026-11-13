"""Directional derivatives by finite differences, for the target without Enzyme.

On CUDA every derivative of a user's device function is an Enzyme forward
sweep (``numba_enzyme.jvp``). The CPU target has no Enzyme, so the two things
:mod:`modax._sensitivity` builds from it, the directional derivative
``Df(x)[u]`` and the second-order ``D2f(x)[u, v] + Df(x)[w]``, are built here
from evaluations of the function itself, behind the same calling convention,
and everything above them runs unchanged. (The Jacobian has a writer of its
own, ``make_difference_jacobian`` in ``modax/_rodas5P.py``, because it can
size each column's step separately and a directional derivative cannot.)

A difference needs a step, and a step needs a scale. An Enzyme sweep takes a
direction of any size; here a direction ``u = (dy, dt, dp)`` is split, which
the derivative's linearity allows, into the part along the state and time and
the part along the parameters, and each part is differenced with a step sized
for its own variables:

* the state-and-time part moves no state by more than a fraction ``delta`` of
  its size, and time by no more than ``delta`` of its own. A state far below
  the others, or below the size at which the solve's absolute tolerance takes
  over from the relative one, is counted as that large, and time near zero as
  large as the solve's first step. The last two come from the launch, in the
  two columns it appends to the parameter row
  (:py:obj:`ensemble_driver <modax._target.ensemble_driver>`), so ``p`` here
  is always that row or a copy that keeps them;
* the parameter part moves no parameter by more than ``delta |p_j|``. A
  parameter that is exactly zero has no size of its own and is counted as the
  smallest nonzero one the direction touches: small enough not to disturb
  whatever the parameter multiplies, at the price of a difference that may be
  limited by rounding and not by curvature.

Differences are central, so the truncation error is second order in the step
and exact for a function quadratic along the direction. ``delta`` is
``eps**(1/3)`` for a first derivative, which balances that against the
rounding of ``f``: about ``4e-11`` of the function's own size over the
direction's. A second derivative is a four-point mixed difference with both
steps at ``eps**(1/4)``, or three points when the two directions are the same
one, good to about ``sqrt(eps)`` on the same footing: a term much smaller
than the function is noisy.

What this cannot do is know the scale of a variable from the variable alone.
A solve whose state or parameters carry sizes these rules misjudge gets
derivatives of lower accuracy with no warning. On CUDA none of this applies.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from modax._numba_common import as_cuda_device
from modax._target import DeviceFunction, cuda, types

_EPS = float(np.finfo(np.float64).eps)
FIRST_ORDER_STEP = _EPS ** (1.0 / 3.0)
"""Relative step of a central first difference."""
SECOND_ORDER_STEP = _EPS**0.25
"""Relative step of a second difference."""
SMALL_COMPONENT = 1.0e-3
"""A state below this fraction of the largest is stepped as if it were this large (first order)."""


_PARAMETER_GROUPS: dict[tuple[object, int], tuple[int, ...]] = {}


def set_parameter_groups(fn: object, n_params: int, groups: tuple[int, ...]) -> None:
    """Declare which of ``fn``'s parameters are sized together.

    ``groups[j]`` is a label, and parameters with the same label are counted
    as large as the largest of them. The hierarchy uses it for the six
    Hermite coefficients of one upstream quantity: the quantity's value at
    either end of a step passes through zero while its other coefficients do
    not, and a step sized by that value alone would drown the difference in
    rounding. Declared before the function's derivatives are built.
    """
    _PARAMETER_GROUPS[(fn, n_params)] = tuple(int(g) for g in groups)


def parameter_groups(fn: object, n_params: int) -> tuple[int, ...]:
    """The labels :func:`set_parameter_groups` gave, or every parameter its own."""
    return _PARAMETER_GROUPS.get((fn, n_params), tuple(range(n_params)))


def step_rules(
    n_vars: int, n_params: int, groups: tuple[int, ...] | None = None
) -> tuple[DeviceFunction, DeviceFunction]:
    """``flow_step(y, t, dy, dt, floor, time_scale, delta)`` and ``param_step(p, dp, delta)``.

    The largest step along a direction that keeps every variable within
    ``delta`` of its own size, or zero for a direction that is zero. A
    parameter's size is the largest magnitude in its group (``groups``), and
    where that is zero the smallest nonzero size the direction touches.
    """
    labels = tuple(range(n_params)) if groups is None else tuple(groups)
    members: dict[int, list[int]] = {}
    for j, label in enumerate(labels):
        members.setdefault(label, []).append(j)
    order = [members[labels[j]] for j in range(n_params)]
    start = np.cumsum([0, *(len(m) for m in order)]).astype(np.int32)
    member = np.array([i for m in order for i in m], dtype=np.int32)
    if n_params == 0:
        member = np.zeros(1, dtype=np.int32)

    @cuda.jit(device=True)
    def flow_step(y, t, dy, dt, floor, time_scale, delta):
        steepest = abs(dt) / max(abs(t), time_scale)
        for j in range(n_vars):
            if dy[j] != 0.0:
                steepest = max(steepest, abs(dy[j]) / max(abs(y[j]), floor))
        return delta / steepest if steepest > 0.0 else 0.0

    @cuda.jit(device=True)
    def size_of(p, j):
        size = 0.0
        for a in range(start[j], start[j + 1]):
            size = max(size, abs(p[member[a]]))
        return size

    @cuda.jit(device=True)
    def param_step(p, dp, delta):
        smallest = 0.0
        for j in range(n_params):
            if dp[j] != 0.0:
                size = size_of(p, j)
                if size > 0.0 and (smallest == 0.0 or size < smallest):
                    smallest = size
        if smallest == 0.0:
            smallest = 1.0
        steepest = 0.0
        for j in range(n_params):
            if dp[j] != 0.0:
                size = size_of(p, j)
                steepest = max(
                    steepest, abs(dp[j]) / (size if size > 0.0 else smallest)
                )
        if steepest > 0.0 and steepest < np.inf:
            return delta / steepest
        return 0.0

    return flow_step, param_step


def _largest(n_vars: int) -> DeviceFunction:
    @cuda.jit(device=True)
    def largest(y):
        size = 0.0
        for j in range(n_vars):
            size = max(size, abs(y[j]))
        return size

    return largest


def difference_tangent(
    fn: Callable[..., tuple[float, ...]] | DeviceFunction,
    n_out: int,
    n_vars: int,
    n_params: int,
) -> DeviceFunction:
    """``tangent(out, y, t, p, dy, dt, dp)``: ``Df(y, t, p)[(dy, dt, dp)]`` into ``out``.

    The convention of an Enzyme forward sweep with one direction set. ``fn``
    returns ``n_out`` values and reads the first ``n_vars`` entries of ``y``
    and the first ``n_params`` of ``p``, either of which may be longer (a
    caller passes its whole augmented state). The cost is two evaluations of
    ``fn`` for the state-and-time part and two for the parameter part; a part
    whose direction is zero costs none.
    """
    fn_device = as_cuda_device(fn)
    flow_step, param_step = step_rules(n_vars, n_params, parameter_groups(fn, n_params))
    largest_of = _largest(n_vars)
    delta = FIRST_ORDER_STEP
    # a caller's row may carry columns past the function's own
    p_size = n_params + 2

    @cuda.jit(device=True)
    def tangent(out, y, t, p, dy, dt, dp):
        y_moved = cuda.local.array(n_vars, types.float64)
        p_moved = cuda.local.array(p_size, types.float64)
        for j in range(n_vars):
            y_moved[j] = y[j]
        for j in range(n_params):
            p_moved[j] = p[j]
        for j in range(n_out):
            out[j] = 0.0

        floor = max(SMALL_COMPONENT * largest_of(y), p[n_params])
        h = flow_step(y, t, dy, dt, floor, p[n_params + 1], delta)
        if h > 0.0:
            for j in range(n_vars):
                y_moved[j] = y[j] + h * dy[j]
            ahead = fn_device(y_moved, t + h * dt, p_moved)
            for j in range(n_vars):
                y_moved[j] = y[j] - h * dy[j]
            behind = fn_device(y_moved, t - h * dt, p_moved)
            for j in range(n_vars):
                y_moved[j] = y[j]
            for j in range(n_out):
                out[j] += (ahead[j] - behind[j]) * (0.5 / h)

        h = param_step(p, dp, delta)
        if h > 0.0:
            for j in range(n_params):
                p_moved[j] = p[j] + h * dp[j]
            ahead = fn_device(y_moved, t, p_moved)
            for j in range(n_params):
                p_moved[j] = p[j] - h * dp[j]
            behind = fn_device(y_moved, t, p_moved)
            for j in range(n_out):
                out[j] += (ahead[j] - behind[j]) * (0.5 / h)

    return tangent


def difference_second_tangent(
    fn: Callable[..., tuple[float, ...]] | DeviceFunction,
    n_out: int,
    n_vars: int,
    n_params: int,
) -> DeviceFunction:
    """``second(out, y, t, p, *u, *v, *w)``: ``D2f(y, t, p)[u, v] + Df(y, t, p)[w]`` into ``out``.

    The convention of Enzyme's forward-over-forward sweep, each of ``u``, ``v``
    and ``w`` a ``(dy, dt, dp)`` triple. ``v``'s parameter part is ignored: no
    caller in this package uses it (``v`` is a stage increment ``(k, 0, 0)``,
    the time direction ``(0, 1, 0)`` or the trajectory's own direction
    ``(y', 1, 0)``).

    ``u`` is split as :func:`difference_tangent` splits a direction, and each
    part paired with ``v`` is the four-point mixed difference
    ``(f(+a, +b) - f(+a, -b) - f(-a, +b) + f(-a, -b)) / 4ab``. Where ``u`` and
    ``v`` are the same state-and-time direction (the second derivative of a
    quantity along the trajectory) it is the three-point
    ``(f(+b) - 2 f(0) + f(-b)) / b**2``. Steps are sized as the first
    difference sizes them but at ``eps**(1/4)`` and with every state counted
    as large as the largest: the finer rule shrinks both steps wherever a
    state passes near zero, and this difference divides the rounding of ``f``
    by their product. ``Df[w]`` is one first difference more.
    """
    fn_device = as_cuda_device(fn)
    first = difference_tangent(fn, n_out, n_vars, n_params)
    flow_step, param_step = step_rules(n_vars, n_params, parameter_groups(fn, n_params))
    largest_of = _largest(n_vars)
    delta = SECOND_ORDER_STEP
    p_size = n_params + 2

    @cuda.jit(device=True)
    def corner(out, weight, y, t, p, a, u_y, u_t, a_p, u_p, b, v_y, v_t):
        """Add ``weight * f`` at the point moved ``a`` along ``u``'s flow part, ``a_p`` along its parameters and ``b`` along ``v``."""
        y_moved = cuda.local.array(n_vars, types.float64)
        p_moved = cuda.local.array(p_size, types.float64)
        for j in range(n_vars):
            y_moved[j] = y[j] + a * u_y[j] + b * v_y[j]
        for j in range(n_params):
            p_moved[j] = p[j] + a_p * u_p[j]
        values = fn_device(y_moved, t + a * u_t + b * v_t, p_moved)
        for j in range(n_out):
            out[j] += weight * values[j]

    @cuda.jit(device=True)
    def second(out, y, t, p, u_y, u_t, u_p, v_y, v_t, v_p, w_y, w_t, w_p):
        floor = max(largest_of(y), p[n_params])
        time_scale = p[n_params + 1]
        for j in range(n_out):
            out[j] = 0.0
        b = flow_step(y, t, v_y, v_t, floor, time_scale, delta)
        if b > 0.0:
            a = flow_step(y, t, u_y, u_t, floor, time_scale, delta)
            same = u_t == v_t
            for j in range(n_vars):
                same = same and u_y[j] == v_y[j]
            if a > 0.0 and same:
                weight = 1.0 / (b * b)
                corner(out, weight, y, t, p, 0.0, u_y, u_t, 0.0, u_p, b, v_y, v_t)
                corner(
                    out, -2.0 * weight, y, t, p, 0.0, u_y, u_t, 0.0, u_p, 0.0, v_y, v_t
                )
                corner(out, weight, y, t, p, 0.0, u_y, u_t, 0.0, u_p, -b, v_y, v_t)
            elif a > 0.0:
                weight = 0.25 / (a * b)
                corner(out, weight, y, t, p, a, u_y, u_t, 0.0, u_p, b, v_y, v_t)
                corner(out, -weight, y, t, p, a, u_y, u_t, 0.0, u_p, -b, v_y, v_t)
                corner(out, -weight, y, t, p, -a, u_y, u_t, 0.0, u_p, b, v_y, v_t)
                corner(out, weight, y, t, p, -a, u_y, u_t, 0.0, u_p, -b, v_y, v_t)
            a_p = param_step(p, u_p, delta)
            if a_p > 0.0:
                weight = 0.25 / (a_p * b)
                corner(out, weight, y, t, p, 0.0, u_y, u_t, a_p, u_p, b, v_y, v_t)
                corner(out, -weight, y, t, p, 0.0, u_y, u_t, a_p, u_p, -b, v_y, v_t)
                corner(out, -weight, y, t, p, 0.0, u_y, u_t, -a_p, u_p, b, v_y, v_t)
                corner(out, weight, y, t, p, 0.0, u_y, u_t, -a_p, u_p, -b, v_y, v_t)

        varies = w_t != 0.0
        for j in range(n_vars):
            varies = varies or w_y[j] != 0.0
        for j in range(n_params):
            varies = varies or w_p[j] != 0.0
        if varies:
            linear = cuda.local.array(n_out, types.float64)
            first(linear, y, t, p, w_y, w_t, w_p)
            for j in range(n_out):
                out[j] += linear[j]

    return second
