"""Cost of the 3D viscoelastic duct (Giesekus, DEVSS-G; 19 unknowns per node) with PARDISO and with GMRES + the block preconditioner
(the b and G fields go through an ILU of their own block): Newton steps, Krylov iterations, time and memory.
Run from the project root:  python Test/manual/viscoelastic_3d_cost.py [ny nz ...]  (pairs of cells of the section)
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from heat_3d_scaling import PeakMemory
import test_3d_viscoelastic as t
from Source.enums import StudyType, ProblemType, SolverType, DirectSolvers, IterativeSolvers, Preconditioners


def run(ny, nz, solver, alpha=0.3):
    options = (dict(Method=SolverType.direct, Solver=DirectSolvers.PARDISO) if solver == 'direct' else
               dict(Method=SolverType.iterative, Solver=IterativeSolvers.GMRES, prec=Preconditioners.SaddlePoint, tol=1e-9,
                    maxiter=3000))
    original = t.Model.solverConfiguration

    def configure(self, **kw):
        return original(self, **{**kw, **options})

    t.Model.solverConfiguration = configure
    try:
        with PeakMemory() as memory:
            start = time.perf_counter()
            model, mesh, u = t.duct(alpha, n=(2, ny, nz), L=1.0)
            seconds = time.perf_counter() - start
    finally:
        t.Model.solverConfiguration = original
    U = np.asarray(model.sol['u'])
    print(f"section {ny} x {nz} {solver:6s}: {model.layout.ndof} unknowns, Newton {model.newton_iterations} converged "
          f"{model.nonlinear_converged}, Krylov {model.solver_iterations} (unconverged {model.solver_unconverged}), {seconds:.1f} s, "
          f"peak +{(memory.peak - memory.base) / 2 ** 30:.2f} GB, u_max {U[:, 0].max():.6f}", flush=True)
    return U


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [4, 3, 8, 6]
    for ny, nz in zip(sizes[::2], sizes[1::2]):
        a = run(ny, nz, 'direct')
        b = run(ny, nz, 'block')
        print(f"   block - direct: {np.max(np.abs(a - b)) / np.max(np.abs(a)):.1e}", flush=True)
