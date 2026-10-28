import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from Source.enums import IterativeSolvers, Preconditioners
from Source.logger_config import sim_logger


def build_preconditioner(A, kind):

    """Preconditioner M ~ A^-1 (a LinearOperator) for the matrix A, or None.

    iLU: incomplete LU (any matrix). Jacobi: inverse of the diagonal. AMG: smoothed-aggregation algebraic
    multigrid from pyamg, one V-cycle per application; it is the one that scales (iterations almost
    independent of the mesh) for diffusion-dominated symmetric problems. Needs the pyamg package.
    """

    if kind is None:
        return None

    if kind == Preconditioners.iLU:
        ilu = spla.spilu(A.tocsc())
        return spla.LinearOperator(A.shape, ilu.solve)

    if kind == Preconditioners.Jacobi:
        inverse = 1.0 / A.diagonal()
        return spla.LinearOperator(A.shape, lambda r: inverse * r)

    if kind == Preconditioners.AMG:
        try:
            import pyamg
        except ImportError as error:
            raise ImportError("El precondicionador AMG necesita el paquete pyamg (pip install pyamg).") from error

        return pyamg.smoothed_aggregation_solver(A.tocsr()).aspreconditioner(cycle='V')

    raise ValueError(f"Precondicionador no implementado: {kind}")


class IterativeSolver:

    """Krylov solver (CG, BiCGSTAB, GMRES) with an optional preconditioner.

    options: 'Solver', 'Preconditioner', 'Tolerance' (relative residual, default 1e-10) and 'Max Iterations'.
    `preconditioner` can be passed already built (to reuse it over several solves); `iterations` and `converged`
    report the last solve.
    """

    def __init__(self, A, b, x0, options=None, preconditioner=None):

        self.A = A
        self.b = b

        self.x0 = x0

        self.options = options
        self.preconditioner = preconditioner

        self.iterations = 0
        self.converged = True

    def solve(self):

        self.iterations = 0

        name = self.options['Solver']
        method = {IterativeSolvers.BiCGSTAB: spla.bicgstab, IterativeSolvers.GMRES: spla.gmres,
                  IterativeSolvers.CG: spla.cg}.get(name)

        if method is None:
            raise ValueError(f"Iterative solver not implemented: {name}")

        M = self.preconditioner
        if M is None:
            M = build_preconditioner(self.A, self.options.get('Preconditioner'))

        def count(_):
            self.iterations += 1

        kwargs = {'x0': self.x0, 'M': M, 'rtol': self.options.get('Tolerance', 1e-10), 'atol': 0.0,
                  'maxiter': self.options.get('Max Iterations')}

        if method is spla.gmres:
            kwargs['callback'] = count
            kwargs['callback_type'] = 'pr_norm'
        else:
            kwargs['callback'] = count

        x, info = method(self.A, self.b, **kwargs)

        self.converged = info == 0
        if not self.converged:
            sim_logger.warning(f"{name.value} did not converge (info={info}, {self.iterations} iterations).")

        return x
