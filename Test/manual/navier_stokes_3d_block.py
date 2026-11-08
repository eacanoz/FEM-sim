"""Steady Navier-Stokes in 3D (Ethier-Steinman field with the body force nu d^2 u) with PARDISO and with GMRES + the block
preconditioner: Newton steps, Krylov iterations, time and memory, at several viscosities and meshes (docs/PLAN_3D.md, phase 2).
Run from the project root:  python Test/manual/navier_stokes_3d_block.py [element] [n] [nu ...]
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from heat_3d_scaling import PeakMemory
import test_3d_flow as t
from Source.enums import StudyType, ProblemType, SolverType, DirectSolvers, IterativeSolvers, Preconditioners


def run(name, n, nu, solver, supg=False):
    with PeakMemory() as memory:
        start = time.perf_counter()
        mesh = t.box(name, n, (2.0, 2.0, 2.0), (-1.0, -1.0, -1.0))
        model = t.Model(name='es', dim=3, mesh=mesh, mat=t.material('X', miu=nu, rho=1.0), psc=t.ff)
        flow = model.physics
        flow.setInertia(True)
        if supg:
            flow.setStabilization('SUPG')
        flow.setBodyForce(lambda x, y, z: tuple(np.moveaxis(nu * t.ES_D ** 2 * t.es_velocity(x, y, z), -1, 0)))
        flow.addBC_VelocityEdges(mesh.boundary_faces(), lambda x, y, z: tuple(t.es_velocity(x, y, z)))
        flow.setPressureReference((0.0, 0.0, 0.0), float(t.es_pressure(0.0, 0.0, 0.0)))
        if solver == 'direct':
            options = dict(Method=SolverType.direct, Solver=DirectSolvers.PARDISO)
        else:
            options = dict(Method=SolverType.iterative, Solver=IterativeSolvers.GMRES, prec=Preconditioners.SaddlePoint, tol=1e-8,
                           maxiter=3000)
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, **options)
        model.solve()
        seconds = time.perf_counter() - start
    error = t.l2(mesh, model.sol['u'], t.es_velocity)
    krylov = model.solver_iterations
    print(f"{name} n = {n} nu = {nu} {'SUPG ' if supg else ''}{solver:6s}: {model.layout.ndof} unknowns, Newton "
          f"{model.newton_iterations} converged {model.nonlinear_converged}, Krylov {krylov} (unconverged {model.solver_unconverged}), "
          f"{seconds:.1f} s, peak +{(memory.peak - memory.base) / 2 ** 30:.2f} GB, u L2 {error:.2e}", flush=True)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else 'hex27'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    nus = [float(a) for a in sys.argv[3:]] or [0.1, 0.01]
    for nu in nus:
        for solver in ('direct', 'block'):
            run(name, n, nu, solver, supg=nu < 0.05)
