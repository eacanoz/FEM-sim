"""Analytic checks for the 1D heat-conduction solver.

Run from the project root:  python Test/test_analytic.py
(or with pytest). Each check compares the FEM result with a closed-form solution.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # no windows while testing

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
from Source.enums import ElementType, ShapeFunctionType, StudyType, ProblemType, SolverType, DirectSolvers, IterativeSolvers, Preconditioners

SIGMA = 5.6704e-8
KELVIN = 273.15


def build_bar(NoE=10, L=2.0, k=1.0):
    mesh = Mesh()
    mesh.Generate_Mesh(1, L, NoE, ElementType.line)
    mesh.setElementShapeFunction(ShapeFunctionType.linear)
    model = Model(name='bar', dim=1, mesh=mesh, mat=material('X', k=k, rho=1, Cp=1), psc=ht)
    model.physics.Convection = False
    model.physics.initField('T', 40.0)
    return model, mesh


def test_linear_profile_dirichlet():
    """Two Dirichlet ends (60 and 20 C): exact solution is a straight line."""
    model, _ = build_bar()
    model.physics.addBC_Temperature(id=0, T=60)
    model.physics.addBC_Temperature(id=10, T=20)
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear,
                              Method=SolverType.direct, Solver=DirectSolvers.PARDISO)
    model.solve()

    exact = np.linspace(60, 20, 11)
    assert np.allclose(model.sol['T'], exact, atol=1e-8), model.sol['T']


def test_linear_profile_iterative_cg():
    """Same problem with CG + iLU (matrix is SPD after BC enforcement)."""
    model, _ = build_bar()
    model.physics.addBC_Temperature(id=0, T=60)
    model.physics.addBC_Temperature(id=10, T=20)
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear,
                              Method=SolverType.iterative, Solver=IterativeSolvers.CG,
                              prec=Preconditioners.iLU)
    model.solve()

    exact = np.linspace(60, 20, 11)
    assert np.allclose(model.sol['T'], exact, atol=1e-6), model.sol['T']


def test_newton_convection_exact():
    """Dirichlet T0 at x=0, convection (h, T_inf) at x=L.

    No source term, so T(x) is linear and the flux balance at x=L gives
    T_L = (k*T0 + h*L*T_inf) / (k + h*L).
    """
    k, L, h, T0, T_inf = 1.0, 2.0, 5.0, 80.0, 20.0
    model, _ = build_bar(NoE=10, L=L, k=k)
    model.physics.addBC_Temperature(id=0, T=T0)
    model.physics.addBC_Convection(id=10, h_c=h, T_ext=T_inf)
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear,
                              Method=SolverType.direct, Solver=DirectSolvers.PARDISO)
    model.solve()

    T_L = (k * T0 + h * L * T_inf) / (k + h * L)
    exact = np.linspace(T0, T_L, 11)
    assert np.allclose(model.sol['T'], exact, atol=1e-8), (model.sol['T'], exact)


def test_radiation_coefficient_uses_kelvin():
    """h(T)*(T - T_ext) must equal eps*sigma*(Ts^4 - Text^4) with absolute temperatures."""
    eps, T_ext, Ts = 0.7, 26.0, 60.0
    model, _ = build_bar()
    model.physics.addBC_Radiation(id=10, epsilon=eps, T_ext=T_ext)

    h = model.mesh.NL[10].BC['T']['h']
    flux_model = float(h(Ts)) * (Ts - T_ext)
    flux_exact = eps * SIGMA * ((Ts + KELVIN) ** 4 - (T_ext + KELVIN) ** 4)

    assert np.isclose(flux_model, flux_exact, rtol=1e-10), (flux_model, flux_exact)


def test_solve_without_solver_configuration():
    """solve() must set a default configuration instead of crashing."""
    model, _ = build_bar()
    model.physics.addBC_Temperature(id=0, T=60)
    model.physics.addBC_Temperature(id=10, T=20)
    model.solve()

    assert np.allclose(model.sol['T'], np.linspace(60, 20, 11), atol=1e-8)


if __name__ == "__main__":
    tests = [test_linear_profile_dirichlet, test_linear_profile_iterative_cg,
             test_newton_convection_exact, test_radiation_coefficient_uses_kelvin,
             test_solve_without_solver_configuration]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # report and keep going
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
