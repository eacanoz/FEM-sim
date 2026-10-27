"""Checks for the nonlinear (Newton) and transient (theta-method) solvers.

Run from the project root:  python Test/test_solvers.py
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
from Source.enums import ElementType, ShapeFunctionType, StudyType, ProblemType, SolverType, DirectSolvers


def radiation_model(T_init):

    """Same setup as Simulation.py: Dirichlet inlet, radiation at the outlet, convection on."""

    mesh = Mesh()
    mesh.Generate_Mesh(1, 2.0, 10, ElementType.line)
    mesh.setElementShapeFunction(ShapeFunctionType.linear)

    model = Model(name='rad', dim=1, mesh=mesh, mat=material('X', k=1, rho=1, Cp=1), psc=ht)
    model.physics.Convection = True
    model.physics.initField('T', T_init)
    model.physics.addBC_Temperature(id=0, T=60)
    model.physics.addBC_Radiation(id=10, epsilon=0.7, T_ext=26)

    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear,
                              Method=SolverType.direct, Solver=DirectSolvers.PARDISO)
    return model


def test_newton_radiation_reference():
    """Outlet temperature of the Simulation.py case (value from the element-based Newton solver)."""

    model = radiation_model(60.0)
    model.solve()

    T = np.array(model.sol['T'])
    assert abs(T[-1] - 33.066) < 2e-3, T[-1]


def test_newton_line_search_from_bad_guess():
    """A poor initial guess (300 C) must converge to the same solution."""

    good = radiation_model(60.0)
    good.solve()

    bad = radiation_model(300.0)
    bad.solve()

    assert np.allclose(good.sol['T'], bad.sol['T'], atol=1e-6), np.max(np.abs(good.sol['T'] - bad.sol['T']))


def decay_model(theta, dt, n_elements=40, total_time=0.1):

    """1D transient, Dirichlet 0 at both ends, initial sin(pi x). Exact: exp(-pi^2 t) sin(pi x)."""

    mesh = Mesh()
    mesh.Generate_Mesh(1, 1.0, n_elements, ElementType.line)
    mesh.setElementShapeFunction(ShapeFunctionType.linear)

    model = Model(name='decay', dim=1, mesh=mesh, mat=material('X', k=1, rho=1, Cp=1), psc=ht)
    model.physics.Convection = False

    x = np.array(mesh.getXCoor())
    model.physics.var['T'].updateField(list(np.sin(np.pi * x)))

    model.physics.addBC_Temperature(id=0, T=0.0)
    model.physics.addBC_Temperature(id=n_elements, T=0.0)

    model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.linear,
                              Method=SolverType.direct, Solver=DirectSolvers.PARDISO,
                              timeDisc=theta, timeStep=dt, totalTime=total_time)
    model.solve()

    x_nodes = np.array(mesh.getXCoor())
    exact = np.exp(-np.pi ** 2 * total_time) * np.sin(np.pi * x_nodes)

    return np.array(model.sol['T'])[:, -1], exact


def test_transient_implicit_euler_decay():
    T, exact = decay_model(theta=1.0, dt=0.001)
    assert np.max(np.abs(T - exact)) < 1e-2, np.max(np.abs(T - exact))


def test_transient_crank_nicolson_decay():
    T, exact = decay_model(theta=0.5, dt=0.01)
    assert np.max(np.abs(T - exact)) < 1e-2, np.max(np.abs(T - exact))


if __name__ == "__main__":
    tests = [test_newton_radiation_reference, test_newton_line_search_from_bad_guess,
             test_transient_implicit_euler_decay, test_transient_crank_nicolson_decay]
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
