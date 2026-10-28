import numpy as np
import scipy.sparse.linalg as spla

from Source.enums import DirectSolvers


import pypardiso


def _same_sparse(a, b):

    """True if two sparse matrices have the same structure and values."""

    if b is None or a.shape != b.shape or a.nnz != b.nnz:
        return False

    return (np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)
            and np.array_equal(a.data, b.data))


class FactorizedSolver:

    """Direct solver that keeps the factorization of its last matrix.

    solve(A, b) factorizes A only when it differs from the previous matrix (same structure and values),
    so a transient linear problem, repeated solves with new right-hand sides, or a parameter sweep that
    leaves the matrix unchanged cost one factorization. `n_factorizations` counts them.
    """

    def __init__(self, kind=DirectSolvers.PARDISO):

        if kind not in (DirectSolvers.PARDISO, DirectSolvers.SuperLU):
            raise NotImplementedError(f"Solver directo no implementado: {kind}")

        self.kind = kind
        self.n_factorizations = 0
        self._A = None
        self._lu = None
        self._pardiso = pypardiso.PyPardisoSolver() if kind == DirectSolvers.PARDISO else None

    def solve(self, A, b):

        A = A.tocsr() if self.kind == DirectSolvers.PARDISO else A.tocsc()

        if A is not self._A and not _same_sparse(A, self._A):

            self.n_factorizations += 1
            self._A = A

            if self.kind == DirectSolvers.PARDISO:
                self._pardiso.factorize(A)
            else:
                self._lu = spla.splu(A)

        if self.kind == DirectSolvers.PARDISO:
            return self._pardiso.solve(A, b)

        return self._lu.solve(b)


class DirectSolver:

    def __init__(self, A, b, x0, options=None):

        self.A = A
        self.b = b

        self.x0 = x0

        self.options = options

    def solve(self):

        if self.options['Solver'] == DirectSolvers.PARDISO:
            return self.PARDISO()
        elif self.options['Solver'] == DirectSolvers.SuperLU:
            return self.SuperLU()

        raise NotImplementedError(f"Solver directo no implementado: {self.options['Solver']}")

    def PARDISO(self):
        # print('Solver: PARDISO (Parallel Direct Sparse Solver)')

        x = pypardiso.spsolve(self.A, self.b)

        return x

    def SuperLU(self):
        # print('Solver: SuperLU (Supernodal LU)')

        x = spla.spsolve(self.A, self.b)
        return x
