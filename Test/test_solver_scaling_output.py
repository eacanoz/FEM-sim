"""Solver scaling (AMG, iterative solvers on the reduced system, factorization reuse) and ParaView output.

Run from the project root:  python Test/test_solver_scaling_output.py
"""

import os
import sys
import tempfile
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
from Source.enums import (ElementType, ShapeFunctionType, StudyType, ProblemType, SolverType, DirectSolvers,
                          IterativeSolvers, Preconditioners)


def poisson(n, element=ElementType.quadrilateral):

    """-lap T = 1 on the unit square, T = 0 on the boundary."""

    mesh = Mesh()
    mesh.generate2DMesh((1.0, 1.0), (n, n), element)
    model = Model(name='poisson', dim=2, mesh=mesh, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
    model.physics.Convection = False
    model.physics.F_const = 1.0
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(), 0.0)
    model.physics.initField('T', 0.0)
    return model


def solve_with(model, method=SolverType.direct, solver=DirectSolvers.PARDISO, prec=Preconditioners.AMG,
               kind=ProblemType.linear, **kwargs):
    model.solverConfiguration(Type=kind, Method=method, Solver=solver, prec=prec, **kwargs)
    model.solve()
    return np.asarray(model.sol['T'])


# ---------------------------------------------------------------------------------------------
# Iterative solvers and AMG
# ---------------------------------------------------------------------------------------------

def test_cg_with_amg_matches_the_direct_solution():
    reference = solve_with(poisson(40))

    for prec in (Preconditioners.AMG, Preconditioners.iLU, Preconditioners.Jacobi):
        model = poisson(40)
        T = solve_with(model, SolverType.iterative, IterativeSolvers.CG, prec)
        assert np.max(np.abs(T - reference)) < 1e-8, (prec, np.max(np.abs(T - reference)))
        assert len(model.solver_iterations) == 1 and model.solver_iterations[0] > 0


def test_amg_iterations_do_not_grow_with_the_mesh():
    amg, plain = [], []
    for n in (20, 40, 80):
        model = poisson(n)
        solve_with(model, SolverType.iterative, IterativeSolvers.CG, Preconditioners.AMG)
        amg.append(model.solver_iterations[0])

        model = poisson(n)
        solve_with(model, SolverType.iterative, IterativeSolvers.CG, None)
        plain.append(model.solver_iterations[0])

    assert amg[-1] <= amg[0] + 4, amg                  # nearly mesh independent
    assert plain[-1] > 2.5 * plain[0], plain            # without it, iterations grow with 1/h
    assert amg[-1] < plain[-1] / 4, (amg, plain)


def test_iterative_newton_with_amg_matches_direct_newton():

    """Nonlinear k(T): every Newton step solved with GMRES + AMG on the reduced system."""

    def build():
        model = poisson(16)
        model.physics.setConductivityLaw(lambda T: 1.0 + 0.5 * T ** 2 + 0.0 * jnp.sum(T))
        return model

    direct = solve_with(build(), kind=ProblemType.nonlinear)
    model = build()
    T = solve_with(model, SolverType.iterative, IterativeSolvers.GMRES, Preconditioners.AMG, kind=ProblemType.nonlinear)

    assert np.max(np.abs(T - direct)) < 1e-7, np.max(np.abs(T - direct))
    assert len(model.solver_iterations) >= 2


def test_iterative_respects_dirichlet_values_and_tolerance():
    mesh = Mesh()
    mesh.Generate_Mesh(1, 1.0, 50, ElementType.line)
    mesh.setElementShapeFunction(ShapeFunctionType.linear)
    model = Model(name='1d', dim=1, mesh=mesh, mat=material('X', k=2.0, rho=1, Cp=1), psc=ht)
    model.physics.Convection = False
    model.physics.addBC_Temperature(0, 10.0)
    model.physics.addBC_Temperature(50, 70.0)
    model.physics.initField('T', 0.0)

    T = solve_with(model, SolverType.iterative, IterativeSolvers.CG, Preconditioners.AMG, tol=1e-12)
    assert np.allclose(T, np.linspace(10.0, 70.0, 51), atol=1e-8)


def test_amg_needs_pyamg_message_and_unknown_preconditioner():
    from Source.Simulation.IterativeSolver import build_preconditioner
    import scipy.sparse as sp

    A = sp.identity(4, format='csr')
    try:
        build_preconditioner(A, 'nope')
    except ValueError as error:
        assert 'nope' in str(error)
    else:
        raise AssertionError("expected ValueError for an unknown preconditioner")


# ---------------------------------------------------------------------------------------------
# Factorization reuse
# ---------------------------------------------------------------------------------------------

def transient_model():
    model = poisson(12)
    model.physics.initField('T', lambda x, y: 100.0 * x * (1 - x) * y * (1 - y))
    return model


def test_linear_transient_factorizes_once_and_solvers_agree():
    results = {}
    for label, method, solver, prec in (('pardiso', SolverType.direct, DirectSolvers.PARDISO, None),
                                         ('superlu', SolverType.direct, DirectSolvers.SuperLU, None),
                                         ('cg-amg', SolverType.iterative, IterativeSolvers.CG, Preconditioners.AMG)):
        model = transient_model()
        model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.linear, Method=method, Solver=solver,
                                  prec=prec, timeDisc=1.0, timeStep=0.01, totalTime=0.3, tol=1e-12)
        model.solve()
        results[label] = np.asarray(model.sol['T'])

        if method == SolverType.direct:
            assert model._factorizer.n_factorizations == 1, (label, model._factorizer.n_factorizations)

    assert np.allclose(results['pardiso'], results['superlu'], atol=1e-9)
    assert np.allclose(results['pardiso'], results['cg-amg'], atol=1e-7)


def test_repeated_solves_reuse_the_factorization_until_the_matrix_changes():
    model = poisson(20)
    model.solverConfiguration(Type=ProblemType.linear, Method=SolverType.direct, Solver=DirectSolvers.PARDISO)
    model.solve()
    first = np.asarray(model.sol['T']).copy()
    assert model._factorizer.n_factorizations == 1

    model.solve()                                                    # same A and b
    assert model._factorizer.n_factorizations == 1
    assert np.allclose(np.asarray(model.sol['T']), first)

    model.physics.F_const = 3.0                                      # only the right-hand side changes
    model.solve()
    assert model._factorizer.n_factorizations == 1
    assert np.allclose(np.asarray(model.sol['T']), 3.0 * first, atol=1e-10)

    model.physics.K_const = 2.0                                      # the matrix changes
    model.solve()
    assert model._factorizer.n_factorizations == 2
    assert np.allclose(np.asarray(model.sol['T']), 1.5 * first, atol=1e-10)


# ---------------------------------------------------------------------------------------------
# VTK output
# ---------------------------------------------------------------------------------------------

def read_vtu(path):
    root = ET.parse(path).getroot()
    piece = root.find('UnstructuredGrid/Piece')

    def array(element):
        values = np.array(element.text.split(), dtype=float)
        n = int(element.get('NumberOfComponents'))
        return values.reshape(-1, n) if n > 1 else values

    data = {a.get('Name'): array(a) for a in piece.find('PointData')}
    points = array(piece.find('Points/DataArray'))
    cells = {a.get('Name'): array(a) for a in piece.find('Cells')}
    return int(piece.get('NumberOfPoints')), int(piece.get('NumberOfCells')), points, cells, data


def test_vtu_of_a_steady_2d_solution_round_trips():
    for element, cell_type, nne in ((ElementType.quadrilateral, 9, 4), (ElementType.triangle, 5, 3)):
        model = poisson(6, element)
        solve_with(model)

        with tempfile.TemporaryDirectory() as folder:
            files = model.exportVTK(os.path.join(folder, 'out', 'plate'))
            assert [os.path.basename(f) for f in files] == ['plate.vtu']
            n_points, n_cells, points, cells, data = read_vtu(files[0])

        assert n_points == 49 and n_cells == (36 if nne == 4 else 72)
        assert np.allclose(points[:, :2], [n.coor for n in model.mesh.NL]) and np.all(points[:, 2] == 0)
        assert np.array_equal(cells['connectivity'].reshape(-1, nne), [e.getNodesId() for e in model.mesh.EL])
        assert np.array_equal(cells['offsets'], np.arange(1, n_cells + 1) * nne) and np.all(cells['types'] == cell_type)
        assert np.allclose(data['T'], model.sol['T'], rtol=1e-8, atol=1e-12)


def test_transient_export_writes_a_time_series():
    model = transient_model()
    model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.linear, timeDisc=1.0, timeStep=0.02, totalTime=0.2)
    model.solve()

    with tempfile.TemporaryDirectory() as folder:
        files = model.exportVTK(os.path.join(folder, 'run'), every=4)
        names = [os.path.basename(f) for f in files]
        assert names == ['run_0000.vtu', 'run_0001.vtu', 'run_0002.vtu', 'run_0003.vtu', 'run.pvd'], names   # steps 0, 4, 8, 10

        entries = [(float(d.get('timestep')), d.get('file')) for d in ET.parse(files[-1]).getroot().iter('DataSet')]
        assert [e[1] for e in entries] == names[:-1]
        assert np.allclose([e[0] for e in entries], model.timeVector[[0, 4, 8, 10]])

        last = read_vtu(files[-2])[4]['T']
        assert np.allclose(last, np.asarray(model.sol['T'])[:, -1], rtol=1e-8, atol=1e-12)


def test_vtu_1d_and_errors():
    mesh = Mesh()
    mesh.Generate_Mesh(1, 2.0, 4, ElementType.line)
    mesh.setElementShapeFunction(ShapeFunctionType.linear)
    model = Model(name='1d', dim=1, mesh=mesh, mat=material('X', k=1, rho=1, Cp=1), psc=ht)

    try:
        model.exportVTK('x')
    except RuntimeError as error:
        assert 'solve' in str(error)
    else:
        raise AssertionError("expected RuntimeError before solving")

    model.physics.Convection = False
    model.physics.addBC_Temperature(0, 0.0)
    model.physics.addBC_Temperature(4, 8.0)
    model.physics.initField('T', 0.0)
    model.solverConfiguration()
    model.solve()

    with tempfile.TemporaryDirectory() as folder:
        files = model.exportVTK(os.path.join(folder, 'line'))
        n_points, n_cells, points, cells, data = read_vtu(files[0])
        assert n_points == 5 and n_cells == 4 and np.all(cells['types'] == 3)
        assert np.allclose(points[:, 0], np.linspace(0, 2, 5)) and np.all(points[:, 1:] == 0)
        assert np.allclose(data['T'], np.linspace(0, 8, 5))

        try:
            model.exportVTK(os.path.join(folder, 'bad'), fields=['nope'])
        except KeyError as error:
            assert 'nope' in str(error)
        else:
            raise AssertionError("expected KeyError for an unknown field")


if __name__ == "__main__":
    tests = [test_cg_with_amg_matches_the_direct_solution, test_amg_iterations_do_not_grow_with_the_mesh,
             test_iterative_newton_with_amg_matches_direct_newton, test_iterative_respects_dirichlet_values_and_tolerance,
             test_amg_needs_pyamg_message_and_unknown_preconditioner,
             test_linear_transient_factorizes_once_and_solvers_agree,
             test_repeated_solves_reuse_the_factorization_until_the_matrix_changes,
             test_vtu_of_a_steady_2d_solution_round_trips, test_transient_export_writes_a_time_series,
             test_vtu_1d_and_errors]
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
