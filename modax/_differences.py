"""Directional derivatives by finite differences, for the target without Enzyme.

On CUDA every derivative of a user's device function is an Enzyme forward
sweep (``numba_enzyme.jvp``). The CPU target has no Enzyme, so the two things
[`modax._sensitivity`][] builds from it -- the directional derivative
``Df(x)[u]`` and the bilinear ``D2f(x)[u, v]`` -- are built here from
evaluations of the function itself, behind the same calling convention, and
the sensitivity machinery above them runs unchanged.

A difference needs a step, and a step needs a scale. An Enzyme sweep takes a
direction of any size; here a direction ``u = (dy, dt, dp)`` is split into its
state, parameter and time parts, which the derivative's linearity allows, and
each part is differenced with a step sized for its own variables:

* the state part along ``dy`` moves no component by more than a fraction
  ``delta`` of its size, with a component smaller than a thousandth of the
  largest counted as that large;
* each parameter the direction touches is stepped by ``delta |p_j|``. A
  parameter that is exactly zero has no size of its own and is stepped by
  ``delta`` times the smallest nonzero magnitude in the row: small enough not
  to disturb whatever the parameter multiplies, at the price of a difference
  that may be limited by rounding rather than by curvature;
* time is stepped by ``delta max(|t|, 1)``.

Differences are central, so the truncation error is second order in the step
and exact for a function quadratic along the direction. ``delta`` is
``eps**(1/3)`` for a first derivative, which balances that against the
rounding of ``f``: about ``4e-11`` of the function's own size over the
direction's. The second derivative is a four-point mixed difference with both
steps at ``eps**(1/4)``, good to about ``sqrt(eps)`` on the same footing, so
a term much smaller than the function is noisy; it only ever enters the
Jacobian of the joint system, where an error costs the method order and not
the answer.

What this cannot do is know the scale of a variable from the variable alone.
A solve whose state or parameters carry sizes these rules misjudge (a
parameter of zero whose natural step is far from the row's smallest entry) gets
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
"""Relative outer step of a difference of first differences."""
SMALL_COMPONENT = 1.0e-3
"""A state component below this fraction of the largest is stepped as if it were this large."""


def difference_tangent(
    fn: Callable[..., tuple[float, ...]] | DeviceFunction,
    n_out: int,
    n_vars: int,
    n_params: int,
) -> DeviceFunction:
    """``tangent(out, y, t, p, dy, dt, dp)``: ``Df(y, t, p)[(dy, dt, dp)]`` into ``out``.

    The convention of an Enzyme forward sweep with one direction set. ``fn``
    returns ``n_out`` values and reads the first ``n_vars`` entries of ``y``
    and the first ``n_params`` of ``p``, which may be longer (a caller passes
    its whole augmented state). The cost is two evaluations of ``fn`` for the
    state part, two for each parameter the direction touches and two for
    time; a part whose direction is zero costs none.
    """
    fn_device = as_cuda_device(fn)
    delta = FIRST_ORDER_STEP

    @cuda.jit(device=True)
    def tangent(out, y, t, p, dy, dt, dp):
        y_moved = cuda.local.array(n_vars, types.float64)
        p_moved = cuda.local.array(n_params, types.float64)
        largest = 0.0
        for j in range(n_vars):
            y_moved[j] = y[j]
            largest = max(largest, abs(y[j]))
        for j in range(n_params):
            p_moved[j] = p[j]
        for j in range(n_out):
            out[j] = 0.0

        # State part: one central difference along dy.
        floor = SMALL_COMPONENT * largest if largest > 0.0 else 1.0
        steepest = 0.0
        for j in range(n_vars):
            if dy[j] != 0.0:
                steepest = max(steepest, abs(dy[j]) / max(abs(y[j]), floor))
        if steepest > 0.0:
            h = delta / steepest
            for j in range(n_vars):
                y_moved[j] = y[j] + h * dy[j]
            ahead = fn_device(y_moved, t, p_moved)
            for j in range(n_vars):
                y_moved[j] = y[j] - h * dy[j]
            behind = fn_device(y_moved, t, p_moved)
            for j in range(n_vars):
                y_moved[j] = y[j]
            inverse = 0.5 / h
            for j in range(n_out):
                out[j] += (ahead[j] - behind[j]) * inverse

        # Parameter part: one central difference for each parameter touched.
        smallest = 0.0
        for j in range(n_params):
            size = abs(p[j])
            if size > 0.0 and (smallest == 0.0 or size < smallest):
                smallest = size
        if smallest == 0.0:
            smallest = 1.0
        for k in range(n_params):
            if dp[k] != 0.0:
                size = abs(p[k])
                h = delta * (size if size > 0.0 else smallest)
                p_moved[k] = p[k] + h
                ahead = fn_device(y_moved, t, p_moved)
                p_moved[k] = p[k] - h
                behind = fn_device(y_moved, t, p_moved)
                p_moved[k] = p[k]
                scale = dp[k] * 0.5 / h
                for j in range(n_out):
                    out[j] += (ahead[j] - behind[j]) * scale

        # Time part.
        if dt != 0.0:
            h = delta * max(abs(t), 1.0)
            ahead = fn_device(y_moved, t + h, p_moved)
            behind = fn_device(y_moved, t - h, p_moved)
            scale = dt * 0.5 / h
            for j in range(n_out):
                out[j] += (ahead[j] - behind[j]) * scale

    return tangent


def difference_second_tangent(
    fn: Callable[..., tuple[float, ...]] | DeviceFunction,
    n_vars: int,
    n_params: int,
) -> DeviceFunction:
    """``second(out, y, t, p, *u, *v, *w)``: ``D2f(y, t, p)[u, v]`` into ``out``.

    The convention of Enzyme's forward-over-forward sweep, each of ``u``, ``v``
    and ``w`` a ``(dy, dt, dp)`` triple. ``w``, the variation of the inner
    direction, is ignored: every caller passes zero for it, which is the plain
    bilinear form computed here. So are the time part of ``u`` and the
    parameter part of ``v``, which no caller in this package uses: ``u`` is a
    sensitivity's ``(S_k, 0, e_k)`` and ``v`` a stage increment ``(k, 0, 0)``
    or the time direction ``(0, 1, 0)``.

    Each pairing of a part of ``u`` with a part of ``v`` is the four-point
    mixed difference ``(f(+a, +b) - f(+a, -b) - f(-a, +b) + f(-a, -b)) / 4ab``,
    with both steps sized as [`difference_tangent`][modax._differences.difference_tangent]
    sizes them but at ``eps**(1/4)`` and with every state component counted as
    large as the largest: four evaluations of ``fn`` a pairing.
    The result is good to roughly ``sqrt(eps)`` of ``|f|`` over the two steps,
    so a term that is small beside the function's own size comes out noisy.
    """
    fn_device = as_cuda_device(fn)
    delta = SECOND_ORDER_STEP

    @cuda.jit(device=True)
    def corner(out, weight, y, t, p, u_y, a_y, k, a_p, v_y, b_y, b_t):
        """Add ``weight * f`` at the point moved by ``a`` along ``u`` and ``b`` along ``v``."""
        y_moved = cuda.local.array(n_vars, types.float64)
        p_moved = cuda.local.array(n_params, types.float64)
        for j in range(n_vars):
            y_moved[j] = y[j] + a_y * u_y[j] + b_y * v_y[j]
        for j in range(n_params):
            p_moved[j] = p[j]
        if k >= 0:
            p_moved[k] += a_p
        values = fn_device(y_moved, t + b_t, p_moved)
        for j in range(n_vars):
            out[j] += weight * values[j]

    @cuda.jit(device=True)
    def pairing(out, scale, y, t, p, u_y, a_y, k, a_p, v_y, b_y, b_t):
        """Add ``scale`` times the four-point difference for one part of ``u`` and one of ``v``."""
        corner(out, scale, y, t, p, u_y, a_y, k, a_p, v_y, b_y, b_t)
        corner(out, -scale, y, t, p, u_y, a_y, k, a_p, v_y, -b_y, -b_t)
        corner(out, -scale, y, t, p, u_y, -a_y, k, -a_p, v_y, b_y, b_t)
        corner(out, scale, y, t, p, u_y, -a_y, k, -a_p, v_y, -b_y, -b_t)

    @cuda.jit(device=True)
    def second(out, y, t, p, u_y, u_t, u_p, v_y, v_t, v_p, w_y, w_t, w_p):
        largest = 0.0
        for j in range(n_vars):
            largest = max(largest, abs(y[j]))
            out[j] = 0.0
        # Every component is sized as the largest. The first difference's finer
        # rule (SMALL_COMPONENT) shrinks both steps wherever a component passes
        # near zero, and this difference divides the rounding of f by their
        # product: on a damped oscillator it left the sensitivities a hundred
        # times less accurate than the state (1e-6 against 8e-9 with this).
        floor = largest if largest > 0.0 else 1.0
        smallest = 0.0
        for j in range(n_params):
            size = abs(p[j])
            if size > 0.0 and (smallest == 0.0 or size < smallest):
                smallest = size
        if smallest == 0.0:
            smallest = 1.0

        # The steps of the two state parts, zero where the part is absent.
        steepest_u = 0.0
        steepest_v = 0.0
        for j in range(n_vars):
            size = max(abs(y[j]), floor)
            if u_y[j] != 0.0:
                steepest_u = max(steepest_u, abs(u_y[j]) / size)
            if v_y[j] != 0.0:
                steepest_v = max(steepest_v, abs(v_y[j]) / size)
        a_y = delta / steepest_u if steepest_u > 0.0 else 0.0
        b_y = delta / steepest_v if steepest_v > 0.0 else 0.0
        b_t = delta * max(abs(t), 1.0) if v_t != 0.0 else 0.0

        # v's state part, then its time part, each against every part of u.
        for part in range(2):
            if part == 0:
                step_y, step_t, step = b_y, 0.0, b_y
                weight_v = 1.0
            else:
                step_y, step_t, step = 0.0, b_t, b_t
                weight_v = v_t
            if step == 0.0:
                continue
            if a_y > 0.0:
                scale = weight_v * 0.25 / (a_y * step)
                pairing(out, scale, y, t, p, u_y, a_y, -1, 0.0, v_y, step_y, step_t)
            for k in range(n_params):
                if u_p[k] != 0.0:
                    size = abs(p[k])
                    a_p = delta * (size if size > 0.0 else smallest)
                    scale = weight_v * u_p[k] * 0.25 / (a_p * step)
                    pairing(out, scale, y, t, p, u_y, 0.0, k, a_p, v_y, step_y, step_t)

    return second
