"""The CPU target's finite-difference Jacobian as a host-callable function, for the checks."""

from __future__ import annotations

import numpy as np


def difference_jacobian(ode_fn, n_vars: int, sparsity=None):
    """``jacobian(y, t, p, rtol=1e-4, atol=1e-4, dt=1.0) -> (df/dy, df/dt)`` as the solver forms them.

    Built from modax's own writer (``modax.rodas5P._make_difference_jacobian``)
    in the layout the kernel uses: the dense grid without a pattern, the sparse
    factorisation's packed slots with one. Entries outside the pattern come
    back zero.
    """
    from modax._numba_common import make_cuda_local_vector_writer
    from modax._sparse_direct import sparse_direct_solver_for
    from modax._sparsity import dense_jacobian, normalize_sparsity
    from modax._target import cuda
    from modax.rodas5P import _make_difference_jacobian

    if sparsity is None:
        pattern, structure = None, dense_jacobian(n_vars)
        entries = [(r, c) for r in range(n_vars) for c in range(n_vars)]
    else:
        pattern = normalize_sparsity(sparsity, n_vars)
        structure = sparse_direct_solver_for(pattern, "amd").compressed
        entries = [(r, c) for r, cols in enumerate(pattern) for c in cols]
    size = structure.size

    @cuda.jit(device=True)
    def clear_matrix(lu):
        for i in range(size):
            lu[i] = 0.0

    writer = _make_difference_jacobian(
        ode_write=make_cuda_local_vector_writer(ode_fn, n_vars),
        structure=structure,
        pattern=pattern,
        clear_matrix=clear_matrix,
        lu_dtype=np.float64,
    )
    rows = np.array([r for r, _ in entries])
    cols = np.array([c for _, c in entries])
    slots = np.array([structure.slot(r, c) for r, c in entries])

    def jacobian(y, t, p, rtol=1e-4, atol=1e-4, dt=1.0):
        lu, dT = np.zeros(size), np.empty(n_vars)
        writer(
            np.ascontiguousarray(y, np.float64),
            float(t),
            np.ascontiguousarray(p, np.float64),
            lu,
            dT,
            float(rtol),
            float(atol),
            float(dt),
        )
        dense = np.zeros((n_vars, n_vars))
        dense[rows, cols] = -lu[slots]
        return dense, dT

    jacobian.n_colours = structure.n_colours
    return jacobian
