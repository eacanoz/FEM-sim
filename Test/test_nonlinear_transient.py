"""2D nonlinear (radiation) and transient heat conduction.

Run from the project root:  python Test/test_nonlinear_transient.py
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
from Source.Simulation.Discretization import Discretization
from Source.enums import ElementType, ShapeFunctionType, StudyType, ProblemType, SolverType, DirectSolvers


def model_2d(nx, ny, Lx=1.0, Ly=1.0, k=1.0, rho_cp=1.0):

    mesh = Mesh()
    mesh.generate2DMesh((Lx, Ly), (nx, ny), ElementType.quadrilateral)

    model = Model(name='2d', dim=2, mesh=mesh, mat=material('X', k=k, rho=rho_cp, Cp=1.0), psc=ht)
    model.physics.Convection = False

    return model, mesh


def steady(model, problem=ProblemType.nonlinear):
    model.solverConfiguration(Study=StudyType.steady_state, Type=problem,
                              Method=SolverType.direct, Solver=DirectSolvers.PARDISO)


def transient(model, theta, dt, total_time):
    model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.linear,
                              Method=SolverType.direct, Solver=DirectSolvers.PARDISO,
                              timeDisc=theta, timeStep=dt, totalTime=total_time)


def mode(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


# ---------------------------------------------------------------------------------------------
# 2D nonlinear
# ---------------------------------------------------------------------------------------------

def radiation_2d_and_1d(nx=8, ny=3, L=2.0):

    """Dirichlet on the left, radiation on the right, insulated top and bottom: the solution does not
    depend on y, so the 2D result must equal the 1D result on the same x nodes."""

    model2, mesh2 = model_2d(nx, ny, Lx=L, Ly=1.0, k=1.0)
    model2.physics.initField('T', 60.0)
    model2.physics.addBC_TemperatureEdges(mesh2.boundary_edges(lambda x, y: x < 1e-12), 60.0)
    model2.physics.addBC_RadiationEdges(mesh2.boundary_edges(lambda x, y: x > L - 1e-12), 0.7, 26.0)
    steady(model2)
    model2.solve()

    mesh1 = Mesh()
    mesh1.Generate_Mesh(1, L, nx, ElementType.line)
    mesh1.setElementShapeFunction(ShapeFunctionType.linear)
    model1 = Model(name='1d', dim=1, mesh=mesh1, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
    model1.physics.Convection = False
    model1.physics.initField('T', 60.0)
    model1.physics.addBC_Temperature(id=0, T=60.0)
    model1.physics.addBC_Radiation(id=nx, epsilon=0.7, T_ext=26.0)
    steady(model1)
    model1.solve()

    return model2, model1, nx


def test_radiation_2d_equals_1d():
    model2, model1, nx = radiation_2d_and_1d()

    T2 = np.asarray(model2.sol['T']).reshape(-1, nx + 1)   # rows are y levels
    T1 = np.asarray(model1.sol['T'])

    assert np.allclose(T2, T1[None, :], atol=1e-8), np.max(np.abs(T2 - T1[None, :]))
    assert T1[-1] < 60.0 - 1.0   # radiation really cools the end


def test_radiation_2d_energy_balance():
    """At the steady state the heat entering through the left edge equals the heat radiated on the right."""

    nx, ny, L = 8, 3, 2.0
    model, mesh = model_2d(nx, ny, Lx=L, Ly=1.0, k=1.0)
    model.physics.initField('T', 60.0)
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x < 1e-12), 60.0)
    right = mesh.boundary_edges(lambda x, y: x > L - 1e-12)
    model.physics.addBC_RadiationEdges(right, 0.7, 26.0)
    steady(model)
    model.solve()

    T = np.asarray(model.sol['T'])
    disc = Discretization(mesh)

    # Radiated power on the right edge: integral of h(T)(T - T_ext) with the facet quadrature
    facets = disc.facets(right)
    Tq = np.einsum('qk,fk->fq', facets.N, T[facets.nodes])
    Tk, Text = Tq + 273.15, 26.0 + 273.15
    radiated = np.sum(facets.detJw * 0.7 * 5.6704e-8 * (Tk ** 4 - Text ** 4))

    # Heat entering through the left side: reaction of the Dirichlet nodes in the conduction operator
    left_nodes = [n.id for n in mesh.NL if n.coor[0] < 1e-12]
    K = model._system()._A.toarray()
    entering = (K @ T)[left_nodes].sum()

    assert np.isclose(entering, radiated, rtol=1e-6), (entering, radiated)


def test_jacobian_2d_radiation_matches_finite_differences():
    model, mesh = model_2d(4, 3, Lx=2.0, Ly=1.0, k=1.5)
    model.physics.initField('T', 50.0)
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x < 1e-12), 80.0)
    model.physics.addBC_RadiationEdges(mesh.boundary_edges(lambda x, y: x > 2.0 - 1e-12), 0.8, 20.0)
    model.physics.addBC_ConvectionEdges(mesh.boundary_edges(lambda x, y: y > 1.0 - 1e-12), lambda T: 5.0 + 0.1 * T, 20.0)
    steady(model)
    system = model._system()

    x = 30.0 + 40.0 * np.random.default_rng(2).random(model.layout.ndof)
    J, R = system.residual_and_tangent(x)

    eps = 1e-6
    J_fd = np.zeros(J.shape)
    for j in range(len(x)):
        xp = x.copy()
        xp[j] += eps
        J_fd[:, j] = (system.residual_and_tangent(xp)[1] - R) / eps

    assert np.allclose(J.toarray(), J_fd, rtol=1e-5, atol=1e-5), np.max(np.abs(J.toarray() - J_fd))


def test_newton_2d_from_bad_initial_guess():
    sols = []
    for T0 in (60.0, 400.0):
        model, mesh = model_2d(6, 4, Lx=2.0, Ly=1.0)
        model.physics.initField('T', T0)
        model.physics.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x < 1e-12), 60.0)
        model.physics.addBC_RadiationEdges(mesh.boundary_edges(lambda x, y: x > 2.0 - 1e-12), 0.7, 26.0)
        steady(model)
        model.solve()
        sols.append(np.asarray(model.sol['T']))

    assert np.allclose(sols[0], sols[1], atol=1e-6)


# ---------------------------------------------------------------------------------------------
# 2D transient
# ---------------------------------------------------------------------------------------------

def decay_error(theta, dt, n=16, total_time=0.05, alpha=1.0):

    """Dirichlet 0, initial sin(pi x) sin(pi y); exact: exp(-2 alpha pi^2 t) times the mode."""

    model, mesh = model_2d(n, n, k=alpha, rho_cp=1.0)
    model.physics.initField('T', mode)
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(), 0.0)
    transient(model, theta, dt, total_time)
    model.solve()

    xy = np.array([node.coor for node in mesh.NL])
    exact = np.exp(-2 * alpha * np.pi ** 2 * total_time) * mode(xy[:, 0], xy[:, 1])

    return np.max(np.abs(np.asarray(model.sol['T'])[:, -1] - exact)), model


def test_transient_2d_decay_matches_analytic():
    err_cn, model = decay_error(theta=0.5, dt=0.0025)
    err_euler, _ = decay_error(theta=1.0, dt=0.0025)

    T = np.asarray(model.sol['T'])
    assert T.shape == (model.mesh.getNoN(), 21)
    assert err_cn < 2e-3, err_cn
    assert err_euler < 1e-2, err_euler
    assert err_cn < err_euler


def test_transient_2d_time_convergence_orders():
    """Self-convergence in time on one mesh: Euler converges with order 1 and Crank-Nicolson with order 2."""

    def final_state(theta, dt):
        return np.asarray(decay_error(theta, dt, n=8, total_time=0.04)[1].sol['T'])[:, -1]

    reference = final_state(0.5, 0.04 / 128)

    for theta, expected in ((1.0, 1.0), (0.5, 2.0)):
        errors = [np.max(np.abs(final_state(theta, 0.04 / n) - reference)) for n in (4, 8, 16)]
        rates = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
        print(f"   theta={theta}: errors {['%.2e' % e for e in errors]} rates {['%.2f' % r for r in rates]}")
        assert all(r > expected - 0.15 for r in rates), (theta, rates)


def test_transient_2d_conserves_energy():
    """Insulated plate with a flux leaving through one edge: sum(M dT) = -q * length * time, for any theta."""

    n, q, total_time = 6, 4.0, 0.5
    for theta in (1.0, 0.5):
        model, mesh = model_2d(n, n, Lx=1.0, Ly=0.5, k=2.0, rho_cp=3.0)
        model.physics.initField('T', lambda x, y: 100.0 + 10.0 * x * y)
        model.physics.addBC_HeatFluxEdges(mesh.boundary_edges(lambda x, y: x > 1.0 - 1e-12), q)
        transient(model, theta, 0.05, total_time)
        T0 = np.asarray(model.physics.var['T'].values).copy()
        model.solve()

        M = model._system().mass()
        T_end = np.asarray(model.sol['T'])[:, -1]
        change = np.sum(M @ (T_end - T0))

        assert np.isclose(change, -q * 0.5 * total_time, rtol=1e-9), (theta, change)


def test_transient_radiation_reaches_the_steady_state():
    """Long-time limit of the nonlinear transient problem equals the Newton steady solution."""

    def build():
        model, mesh = model_2d(8, 4, Lx=1.0, Ly=0.5, k=2.0, rho_cp=2.0)
        model.physics.initField('T', 200.0)
        model.physics.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x < 1e-12), 60.0)
        model.physics.addBC_RadiationEdges(mesh.boundary_edges(lambda x, y: x > 1.0 - 1e-12), 0.8, 26.0)
        model.physics.addBC_ConvectionEdges(mesh.boundary_edges(lambda x, y: y < 1e-12), 8.0, 26.0)
        return model

    stationary = build()
    steady(stationary)
    stationary.solve()

    # Crank-Nicolson is not L-stable: with a discontinuous initial state (200 against 60 on the
    # Dirichlet edge) its stiffest modes ring for a long time at large steps, so it needs a smaller dt.
    for theta, dt in ((1.0, 0.05), (0.5, 0.02)):
        model = build()
        transient(model, theta, dt, 6.0)
        model.solve()

        T_final = np.asarray(model.sol['T'])[:, -1]
        assert np.allclose(T_final, stationary.sol['T'], atol=1e-4), (theta, np.max(np.abs(T_final - stationary.sol['T'])))

        # Monotone cooling of the hottest point
        hottest = np.asarray(model.sol['T']).max(axis=0)
        assert np.all(np.diff(hottest) <= 1e-9)


def test_init_field_with_function_and_postprocess_step():
    model, mesh = model_2d(4, 4)
    model.physics.initField('T', lambda x, y: x + 2 * y)
    xy = np.array([node.coor for node in mesh.NL])
    assert np.allclose(model.physics.var['T'].values, xy[:, 0] + 2 * xy[:, 1])

    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(), 0.0)
    transient(model, 1.0, 0.1, 0.3)
    model.solve()

    fig, ax = matplotlib.pyplot.subplots()
    model.postProcess2D(step=1, ax=ax, vmin=0.0, vmax=3.0)
    matplotlib.pyplot.close(fig)


if __name__ == "__main__":
    tests = [test_radiation_2d_equals_1d, test_radiation_2d_energy_balance,
             test_jacobian_2d_radiation_matches_finite_differences, test_newton_2d_from_bad_initial_guess,
             test_transient_2d_decay_matches_analytic, test_transient_2d_time_convergence_orders,
             test_transient_2d_conserves_energy, test_transient_radiation_reaches_the_steady_state,
             test_init_field_with_function_and_postprocess_step]
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
