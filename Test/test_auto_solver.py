"""Automatic linear-solver selection (Method auto): the choice by size and kind, the fallback to the direct solver, and
that the result is that of the direct solver.

Run from the project root:  python Test/test_auto_solver.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
from Source.Simulation import SolverSelection as selection
from Source.enums import ElementType, SolverType, DirectSolvers, IterativeSolvers, Preconditioners


def poisson(n=20, convection=False):
    mesh = Mesh()
    mesh.generate2DMesh((1.0, 1.0), (n, n), ElementType.quadrilateral)
    model = Model(name='poisson', dim=2, mesh=mesh, mat=material('X', k=1.0 if not convection else 0.01, rho=1.0, Cp=1.0), psc=ht)
    model.physics.Convection = convection
    if convection:
        model.physics.setVelocity((1.0, 0.5))
        model.physics.setStabilization('SUPG')
    model.physics.F_const = 1.0
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(), 0.0)
    model.physics.initField('T', 0.0)
    return model


def solve(model, **options):
    model.solverConfiguration(**options)
    model.solve()
    return np.asarray(model.sol['T'])


def test_choice_by_size_and_kind():
    small, large = selection.SMALL_SYSTEM - 1, selection.ITERATIVE_SCALAR
    assert selection.choose_options(small, True, False)['Solver'] == DirectSolvers.SuperLU
    assert selection.choose_options(selection.SMALL_SYSTEM, True, False)['Solver'] == DirectSolvers.PARDISO
    assert selection.choose_options(large, True, False)['Solver'] == IterativeSolvers.CG
    assert selection.choose_options(large, True, False)['prec'] == Preconditioners.AMG
    assert selection.choose_options(large, False, False)['Solver'] == IterativeSolvers.BiCGSTAB
    assert selection.choose_options(small, True, True)['Solver'] == DirectSolvers.PARDISO
    assert selection.choose_options(selection.ITERATIVE_FLOW, True, True)['prec'] == Preconditioners.SaddlePoint


def test_auto_matches_the_direct_solution_and_accepts_a_string():
    reference = solve(poisson(), Method=SolverType.direct, Solver=DirectSolvers.PARDISO)

    model = poisson()
    T = solve(model, Method='auto')
    assert model.solverOptions['Method'] == SolverType.auto
    assert np.max(np.abs(T - reference)) < 1e-10
    assert model.solver_iterations == []                      # a small system goes to a direct solver


def test_auto_uses_the_iterative_solver_above_the_threshold(monkeypatch=None):
    reference = solve(poisson(), Method=SolverType.direct, Solver=DirectSolvers.PARDISO)

    saved = selection.ITERATIVE_SCALAR
    selection.ITERATIVE_SCALAR = 100
    try:
        model = poisson()
        T = solve(model, Method=SolverType.auto, tol=1e-12)
        assert model.solver_iterations and model.solver_unconverged == 0
        assert np.max(np.abs(T - reference)) < 1e-8

        # not symmetric: BiCGSTAB + iLU
        direct = solve(poisson(convection=True), Method=SolverType.direct, Solver=DirectSolvers.PARDISO)
        model = poisson(convection=True)
        T = solve(model, Method=SolverType.auto, tol=1e-12)
        assert model.solver_iterations
        assert np.max(np.abs(T - direct)) < 1e-7
    finally:
        selection.ITERATIVE_SCALAR = saved


def test_auto_falls_back_to_the_direct_solver_when_the_iterations_run_out():
    reference = solve(poisson(), Method=SolverType.direct, Solver=DirectSolvers.PARDISO)

    saved = selection.ITERATIVE_SCALAR
    selection.ITERATIVE_SCALAR = 100
    try:
        model = poisson()
        T = solve(model, Method=SolverType.auto, tol=1e-14, maxiter=1)
        assert model.solver_unconverged == 1
        assert np.max(np.abs(T - reference)) < 1e-10          # the answer is the direct one, not the unconverged one
    finally:
        selection.ITERATIVE_SCALAR = saved


def test_gmres_maxiter_counts_iterations_not_restart_cycles():
    from Source.Simulation.IterativeSolver import IterativeSolver
    import scipy.sparse as sp

    n = 400
    A = (sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)) * n ** 2).tocsr()
    solver = IterativeSolver(A, np.ones(n), None, {'Solver': IterativeSolvers.GMRES, 'Tolerance': 1e-12, 'Max Iterations': 40})
    solver.solve()
    assert not solver.converged and solver.iterations <= 40, solver.iterations


if __name__ == "__main__":
    tests = [test_choice_by_size_and_kind, test_auto_matches_the_direct_solution_and_accepts_a_string,
             test_auto_uses_the_iterative_solver_above_the_threshold,
             test_auto_falls_back_to_the_direct_solver_when_the_iterations_run_out,
             test_gmres_maxiter_counts_iterations_not_restart_cycles]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
