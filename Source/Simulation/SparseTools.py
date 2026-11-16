"""Sparse products on several threads.

scipy multiplies two sparse matrices on one thread. The rows of a product A @ B are independent, and scipy releases the
interpreter lock while it computes them, so the rows of A go in blocks to a pool of threads and the pieces are stacked: the
same matrix, entry by entry. Measured on the tangent of the recovered stress (Q @ dsigma/dx, 369 k rows, 18 M entries of
the product): 0.53 s on one thread, 0.26 s on twelve; memory bandwidth and the stacking limit it to about half.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp

THREADS = max(1, min(12, os.cpu_count() or 1))
MIN_ENTRIES = 2_000_000          # below this many entries of A the threads cost more than they save

_pool = None


def matmat(A, B):

    """A @ B for sparse matrices, with the rows of A in blocks on several threads when A is large."""

    A = A.tocsr()
    if THREADS == 1 or A.nnz < MIN_ENTRIES or A.shape[0] < 2 * THREADS:
        return A @ B

    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(THREADS)

    B = B.tocsr()
    # blocks of rows with about the same number of entries
    bounds = np.searchsorted(A.indptr, np.linspace(0, A.nnz, THREADS + 1)).astype(int)
    bounds[0], bounds[-1] = 0, A.shape[0]
    bounds = np.unique(bounds)

    parts = list(_pool.map(lambda k: A[bounds[k]:bounds[k + 1]] @ B, range(len(bounds) - 1)))

    return sp.vstack(parts, format='csr')
