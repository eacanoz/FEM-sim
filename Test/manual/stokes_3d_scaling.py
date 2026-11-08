"""Time and memory of 3D Stokes (rectangular duct, Taylor-Hood or PSPG) with PARDISO and with GMRES + the block preconditioner.

Measures what limits the 3D flow runs (docs/PLAN_3D.md, phase 2). Run from the project root:
    python Test/manual/stokes_3d_scaling.py [element] [n1 n2 ...] [--solver direct|block|both]
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from heat_3d_scaling import PeakMemory, KIND
from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.enums import StudyType, ProblemType, SolverType, DirectSolvers, IterativeSolvers, Preconditioners


def run(name, n, solver, a=1.0, b=0.6, L=2.0, G=2.0, mu=0.5):
    with PeakMemory() as memory:
        t0 = time.perf_counter()
        mesh = Mesh()
        mesh.generate3DMesh((L, a, b), (2 * n, n, max(1, round(n * b / a))), *KIND[name])
        walls = mesh.boundary_faces(lambda x, y, z: min(y, a - y, z, b - z) < 1e-9)
        inlet = mesh.boundary_faces(lambda x, y, z: x < 1e-9)
        outlet = mesh.boundary_faces(lambda x, y, z: x > L - 1e-9)
        t1 = time.perf_counter()
        model = Model(name='duct', dim=3, mesh=mesh, mat=material('X', miu=mu), psc=ff)
        flow = model.physics
        flow.addBC_NoSlipEdges(walls)
        flow.addBC_OutletEdges(inlet, G * L)
        flow.addBC_OutletEdges(outlet, 0.0)
        if solver == 'direct':
            options = dict(Method=SolverType.direct, Solver=DirectSolvers.PARDISO)
        else:
            options = dict(Method=SolverType.iterative, Solver=IterativeSolvers.GMRES, prec=Preconditioners.SaddlePoint, tol=1e-8,
                           maxiter=2000)
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear, **options)
        model.solve()
        t2 = time.perf_counter()

    n_dof = model.layout.ndof
    Q = -flow.flowRate(inlet)
    iterations = sum(model.solver_iterations)
    print(f"{name:6s} n = {n:3d} {solver:6s}: {mesh.getNoE():8d} elements {n_dof:9d} unknowns | mesh {t1 - t0:6.1f} s, "
          f"model + solve {t2 - t1:7.1f} s ({iterations} it) | peak +{(memory.peak - memory.base) / 2 ** 30:5.2f} GB | Q {Q:.6f}",
          flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    solver = 'both'
    for a in sys.argv[1:]:
        if a.startswith('--solver='):
            solver = a.split('=', 1)[1]
    name = args[0] if args else 'hex27'
    sizes = [int(a) for a in args[1:]] or [4, 6, 8]
    for n in sizes:
        for s in (('direct', 'block') if solver == 'both' else (solver,)):
            run(name, n, s)
