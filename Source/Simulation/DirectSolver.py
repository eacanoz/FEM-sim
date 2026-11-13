import numpy as np
import scipy.sparse.linalg as spla

from Source.enums import DirectSolvers


import pypardiso
from pypardiso.pardiso_wrapper import PyPardisoError

from Source.logger_config import sim_logger

# Checked PARDISO solves. PARDISO perturbs tiny pivots and, for nonsymmetric matrices, reorders by default with a maximum weighted
# matching (iparm 13); on 3D Taylor-Hood saddle points that left relative residuals of 1e-6 to 1e3 (a wide die of hex27, 28 perturbed
# pivots) where SuperLU gives 1e-15 and the same solver without the matching 1e-13. Above REFINE_ABOVE the solution is refined with the
# factorization (cheap); only if it stays above BROKEN_ABOVE is the matrix refactorized without the matching, kept if that is good.
# BROKEN_ABOVE is loose on purpose: ill-conditioned systems (viscoelastic, 1e-7 with PARDISO and SuperLU alike) must not refactorize.
REFINE_ABOVE = 1e-10
BROKEN_ABOVE = 1e-6
_NO_MATCHING = {1: 1, 2: 2, 8: 2, 10: 13, 11: 0, 13: 0}     # iparm (1-based): user values, nested dissection, no scaling/matching


def _relative_residual(A, x, b):
    norm = np.linalg.norm(b)
    return np.linalg.norm(b - A @ x) / norm if norm > 0 else np.linalg.norm(A @ x)


def _refined(solver, A, b, x):

    """Up to 3 steps of iterative refinement with the factorization of `solver`; returns the best (x, residual)."""

    best, residual = x, _relative_residual(A, x, b)
    for _ in range(3):
        if not residual > REFINE_ABOVE:
            break
        x = best + solver.solve(A, b - A @ best)
        r = _relative_residual(A, x, b)
        if not r < residual:
            break
        best, residual = x, r
    return best, residual


def no_matching_pardiso():

    """A PARDISO solver without scaling and weighted matching (nonsymmetric, nested dissection)."""

    solver = pypardiso.PyPardisoSolver()
    for k, v in _NO_MATCHING.items():
        solver.set_iparm(k, v)
    return solver


def checked_pardiso(A, b, solver, retry=None, factorized=True):

    """Solves A x = b with the PARDISO `solver` (already factorized if `factorized`), checking the residual: refinement, then a
    factorization without the matching (in `retry`, a no_matching_pardiso() made if None), used only if it gives a good residual.
    Returns (x, mode, retry) with mode 'default' or 'retry'; the caller keeps `retry` (and frees it) to reuse that factorization."""

    A = A.tocsr()
    if not factorized:
        solver.factorize(A)
    x, residual = _refined(solver, A, b, solver.solve(A, b))
    if not residual > BROKEN_ABOVE:
        return x, 'default', retry

    retry = retry if retry is not None else no_matching_pardiso()
    retry.factorize(A)
    y, second = _refined(retry, A, b, retry.solve(A, b))
    # The retry is used if it is good, or if the PARDISO answer is worse than x = 0 (relative residual above 1: a planar contraction of
    # hex27 gave 1.5e3 there, and 2e-6 without the matching)
    if not second > BROKEN_ABOVE or (residual > 1.0 and second < residual):
        sim_logger.info(f"PARDISO: residual {residual:.1e}; refactorized without the weighted matching ({second:.1e}).")
        return y, 'retry', retry

    # Neither factorization is good: the matrix is close to singular (a viscoelastic Newton step at a hard Weissenberg number left 3e-2
    # with PARDISO, 4e-4 without the matching and 1e-2 with SuperLU). Keep the PARDISO answer, as before these checks: replacing it changes
    # the Newton path of problems validated with it.
    sim_logger.warning(f"PARDISO: residual {residual:.1e} ({second:.1e} without matching): matrix close to singular, kept as is.")
    return x, 'default', retry


def _same_pattern(a, b):

    """True if two sparse matrices of the same format have the same structure (their values may differ)."""

    if b is None or a.format != b.format or a.shape != b.shape or a.nnz != b.nnz:
        return False

    return np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)


def _same_sparse(a, b):

    """True if two sparse matrices have the same structure and values."""

    if b is None or a.shape != b.shape or a.nnz != b.nnz:
        return False

    return (np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)
            and np.array_equal(a.data, b.data))


def _numeric_factorization(solver, A):

    """Numeric factorization of A (PARDISO phase 22) with the analysis the solver already has, of a matrix with the same
    structure: the reordering and the symbolic factorization are not repeated. Mirrors PyPardisoSolver.factorize."""

    solver._check_A(A)
    solver.factorized_A = solver._hash_csr_matrix(A) if A.nnz > solver.size_limit_storage else A.copy()
    solver.set_phase(22)
    solver._call_pardiso(A, np.zeros((A.shape[0], 1)))


class FactorizedSolver:

    """Direct solver that keeps the factorization of its last matrix.

    solve(A, b) factorizes A only when it differs from the previous matrix (same structure and values),
    so a transient linear problem, repeated solves with new right-hand sides, or a parameter sweep that
    leaves the matrix unchanged cost one factorization. `n_factorizations` counts them.

    PARDISO, new values on the same structure (the tangent of every Newton iteration, a new time step): only the numeric
    factorization is repeated, with the analysis (reordering, symbolic factorization) of the first matrix. The analysis is half
    of a factorization on equal-order 2D meshes and most of it on Taylor-Hood ones (measured: 1.1 of 2.0 s with 187 k unknowns
    of Q4, 3.8 of 4.4 s of Q9). `n_analyses` counts the analyses. The weighted matching of that analysis comes from the values
    of its own matrix, so the first solve with each reused analysis is checked: a residual above REFINE_ABOVE after refinement
    repeats the whole factorization, and if that is clearly better (10 times) the solver stops reusing; if it is not, the
    matrix is ill-conditioned by itself (viscoelastic) and the reuse stays, repeating the analysis only above BROKEN_ABOVE.
    reuse_analysis = False restores a full factorization per matrix.
    """

    reuse_analysis = True

    def __init__(self, kind=DirectSolvers.PARDISO):

        if kind not in (DirectSolvers.PARDISO, DirectSolvers.SuperLU):
            raise NotImplementedError(f"Solver directo no implementado: {kind}")

        self.kind = kind
        self.n_factorizations = 0
        self.n_analyses = 0
        self._A = None
        self._lu = None
        self._pardiso = pypardiso.PyPardisoSolver() if kind == DirectSolvers.PARDISO else None
        self._retry = None           # PARDISO without the weighted matching, made only if a solve needs it (checked_pardiso)
        self._mode = 'default'       # which factorization of the current matrix is trusted: 'default' or 'retry'
        self._reused = False         # the current factorization uses the analysis of an earlier matrix
        self._reuse_trusted = False  # a full factorization did not improve on a reused analysis: residuals up to BROKEN_ABOVE pass

    def _factorize_pardiso(self, A, previous):

        self._reused = False
        if self.reuse_analysis and self.n_analyses and _same_pattern(A, previous):
            try:
                _numeric_factorization(self._pardiso, A)
                self._reused = True
            except PyPardisoError:                # a pivot the old reordering cannot take: start again
                pass

        if not self._reused:
            self._pardiso.factorize(A)            # the same solver object: PARDISO reuses (and releases) its own memory
            self.n_analyses += 1

    def _solve_reused(self, A, b):

        """Solution with a reused analysis if its residual is good; otherwise None, with A factorized again in full."""

        x, residual = _refined(self._pardiso, A, b, self._pardiso.solve(A, b))
        if not residual > REFINE_ABOVE or (self._reuse_trusted and not residual > BROKEN_ABOVE):
            return x

        self._pardiso.factorize(A)
        self.n_analyses += 1
        self._reused = False
        full = _relative_residual(A, self._pardiso.solve(A, b), b)
        if full < 0.1 * residual and not self._reuse_trusted:
            sim_logger.info(f"PARDISO: residual {residual:.1e} with a reused analysis, {full:.1e} with its own: "
                            f"the analysis is not reused.")
            self.reuse_analysis = False
        else:
            self._reuse_trusted = True

        return None

    def solve(self, A, b):

        A = A.tocsr() if self.kind == DirectSolvers.PARDISO else A.tocsc()
        if not A.has_sorted_indices:
            A.sort_indices()                      # PARDISO sorts them anyway; unsorted, an equal matrix would look new

        if A is not self._A and not _same_sparse(A, self._A):

            self.n_factorizations += 1
            previous, self._A = self._A, A

            if self.kind == DirectSolvers.PARDISO:
                self._factorize_pardiso(A, previous)
                self._mode = 'default'
                if self._retry is not None:       # a new matrix starts again with the default reordering
                    self._retry.free_memory(everything=True)
                    self._retry = None
            else:
                self._lu = spla.splu(A)

        if self.kind == DirectSolvers.PARDISO:
            if self._mode == 'retry':
                return _refined(self._retry, A, b, self._retry.solve(A, b))[0]
            if self._reused:
                x = self._solve_reused(A, b)
                if x is not None:
                    return x
            x, self._mode, self._retry = checked_pardiso(A, b, self._pardiso, self._retry)
            return x

        return self._lu.solve(b)

    def __del__(self):
        for solver in (getattr(self, '_pardiso', None), getattr(self, '_retry', None)):
            if solver is not None:
                try:
                    solver.free_memory(everything=True)
                except Exception:
                    pass


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

        solver = pypardiso.PyPardisoSolver()
        try:
            x, _, retry = checked_pardiso(self.A, self.b, solver, factorized=False)
            if retry is not None:
                retry.free_memory(everything=True)
        finally:
            solver.free_memory(everything=True)

        return x

    def SuperLU(self):
        # print('Solver: SuperLU (Supernodal LU)')

        x = spla.spsolve(self.A, self.b)
        return x
