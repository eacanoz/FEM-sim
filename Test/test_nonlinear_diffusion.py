"""Temperature-dependent conductivity k(T): nonlinear diffusion in 1D and 2D, steady and transient.

Run from the project root:  python Test/test_nonlinear_diffusion.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
from Source.Simulation.Discretization import Discretization
from Source.enums import ElementType, ShapeFunctionType, StudyType, ProblemType, SolverType, DirectSolvers

K0, BETA = 2.0, 0.01            # k(T) = K0 (1 + BETA T)
k_linear_in_T = lambda T: K0 * (1.0 + BETA * T)


def configure(model, study=StudyType.steady_state, **time):
    model.solverConfiguration(Study=study, Type=ProblemType.nonlinear, Method=SolverType.direct,
                              Solver=DirectSolvers.PARDISO, **time)


def model_1d(n, L=1.0, k_law=k_linear_in_T):
    mesh = Mesh()
    mesh.Generate_Mesh(1, L, n, ElementType.line)
    mesh.setElementShapeFunction(ShapeFunctionType.linear)
    model = Model(name='k(T)', dim=1, mesh=mesh, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
    model.physics.Convection = False
    model.physics.setConductivityLaw(k_law)
    return model, mesh


def model_2d(nx, ny, Lx=1.0, Ly=1.0, k_law=k_linear_in_T, rho_cp=1.0):
    mesh = Mesh()
    mesh.generate2DMesh((Lx, Ly), (nx, ny), ElementType.quadrilateral)
    model = Model(name='k(T)', dim=2, mesh=mesh, mat=material('X', k=1.0, rho=rho_cp, Cp=1.0), psc=ht)
    model.physics.Convection = False
    model.physics.setConductivityLaw(k_law)
    return model, mesh


def kirchhoff(x, L, T1, T2):

    """Exact steady solution for k = K0 (1 + BETA T): T + BETA T^2 / 2 is linear in x."""

    psi = lambda T: T + BETA * T ** 2 / 2
    p = psi(T1) + (psi(T2) - psi(T1)) * x / L

    return (-1.0 + np.sqrt(1.0 + 2.0 * BETA * p)) / BETA


# ---------------------------------------------------------------------------------------------
# Steady state
# ---------------------------------------------------------------------------------------------

def solve_1d(n, T1=20.0, T2=300.0, T0=100.0):
    model, mesh = model_1d(n)
    model.physics.initField('T', T0)
    model.physics.addBC_Temperature(0, T1)
    model.physics.addBC_Temperature(n, T2)
    configure(model)
    model.solve()
    return model, np.array(mesh.getXCoor())


def test_kirchhoff_exact_solution():

    """With k linear in T and no source, 1D P1 elements with 2-point Gauss are nodally exact:
    k(T_h) dT_h/dx is integrated exactly, so the nodal values match the Kirchhoff solution to roundoff."""

    for n in (4, 8, 16, 32):
        model, x = solve_1d(n)
        error = np.max(np.abs(np.asarray(model.sol['T']) - kirchhoff(x, 1.0, 20.0, 300.0)))
        assert error < 1e-9, (n, error)

    # The profile is not linear (k grows with T), but it is monotone between the boundary values
    T = np.asarray(model.sol['T'])
    assert np.all(np.diff(T) > 0) and abs(T[0] - 20.0) < 1e-12 and abs(T[-1] - 300.0) < 1e-12
    assert np.max(np.abs(T - np.linspace(20.0, 300.0, len(T)))) > 5.0


def test_result_does_not_depend_on_the_initial_guess():
    sols = [np.asarray(solve_1d(16, T0=T0)[0].sol['T']) for T0 in (20.0, 300.0, 800.0)]

    assert np.allclose(sols[0], sols[1], atol=1e-8) and np.allclose(sols[0], sols[2], atol=1e-8)


def test_2d_with_y_independent_data_equals_1d():
    nx, ny = 8, 3
    model2, mesh2 = model_2d(nx, ny)
    model2.physics.initField('T', 100.0)
    model2.physics.addBC_TemperatureEdges(mesh2.boundary_edges(lambda x, y: x < 1e-12), 20.0)
    model2.physics.addBC_TemperatureEdges(mesh2.boundary_edges(lambda x, y: x > 1 - 1e-12), 300.0)
    configure(model2)
    model2.solve()

    model1, _ = solve_1d(nx)

    T2 = np.asarray(model2.sol['T']).reshape(-1, nx + 1)
    assert np.allclose(T2, np.asarray(model1.sol['T'])[None, :], atol=1e-8)


def l2_error_manufactured(n):

    """-div((1 + T^2) grad T) = S with T = sin(pi x) sin(pi y), T = 0 on the boundary."""

    model, mesh = model_2d(n, n, k_law=lambda T: 1.0 + T ** 2)

    exact = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)

    def source(x, y):
        T = exact(x, y)
        grad2 = np.pi ** 2 * (np.cos(np.pi * x) ** 2 * np.sin(np.pi * y) ** 2
                              + np.sin(np.pi * x) ** 2 * np.cos(np.pi * y) ** 2)
        return (1.0 + T ** 2) * 2 * np.pi ** 2 * T - 2.0 * T * grad2

    model.physics.F_const = source
    model.physics.initField('T', 0.2)
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(), 0.0)
    configure(model)
    model.solve()

    disc = Discretization(mesh)
    T_nodes = np.asarray(model.sol['T'])
    T_gauss = np.einsum('gi,ei->eg', disc.N, T_nodes[disc.conn])

    return np.sqrt(np.sum(disc.detJw * (T_gauss - exact(disc.x_gp[..., 0], disc.x_gp[..., 1])) ** 2))


def test_2d_manufactured_nonlinear_diffusion_order_two():
    errors = [l2_error_manufactured(n) for n in (4, 8, 16)]
    rates = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]

    print("   L2 errors:", ["%.3e" % e for e in errors], "rates:", ["%.2f" % r for r in rates])
    assert all(r > 1.9 for r in rates), rates


def test_jacobian_with_k_of_T_convection_and_radiation_matches_finite_differences():
    model, mesh = model_2d(4, 3, Lx=2.0, Ly=1.0, k_law=lambda T: 1.0 + 0.02 * T + 1e-4 * T ** 2)
    model.physics.Convection = True
    model.physics.vel = [0.4, 0.2]
    model.physics.initField('T', 50.0)
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x < 1e-12), 80.0)
    model.physics.addBC_RadiationEdges(mesh.boundary_edges(lambda x, y: x > 2.0 - 1e-12), 0.8, 20.0)
    configure(model)
    system = model._system()

    x = 30.0 + 40.0 * np.random.default_rng(3).random(model.layout.ndof)
    J, R = system.residual_and_tangent(x)

    eps = 1e-6
    J_fd = np.zeros(J.shape)
    for j in range(len(x)):
        xp = x.copy()
        xp[j] += eps
        J_fd[:, j] = (system.residual_and_tangent(xp)[1] - R) / eps

    assert np.allclose(J.toarray(), J_fd, rtol=1e-5, atol=1e-5), np.max(np.abs(J.toarray() - J_fd))


def test_constant_law_reproduces_the_constant_conductivity_matrix():
    for n in (6,):
        law_model, mesh = model_1d(n, k_law=lambda T: 2.5 + 0.0 * T)
        const_model, _ = model_1d(n)
        const_model.physics.setConductivityLaw(None)
        const_model.physics.K_const = 2.5

        for model in (law_model, const_model):
            model.physics.initField('T', 40.0)
            model.physics.addBC_Temperature(0, 60.0)
            model.physics.addBC_Convection(n, 5.0, 20.0)
            configure(model)

        A_law, b_law = law_model.assembleGlobalSystem()
        A_const, b_const = const_model.assembleGlobalSystem()

        assert np.allclose(A_law.toarray(), A_const.toarray(), atol=1e-10)
        assert np.allclose(b_law, b_const, atol=1e-10)


# ---------------------------------------------------------------------------------------------
# Transient
# ---------------------------------------------------------------------------------------------

def test_transient_conserves_energy_with_k_of_T():
    """Nonlinear conduction is still conservative: sum(M dT) = -q * length * time for any theta."""

    q, total_time = 3.0, 0.4
    for theta in (1.0, 0.5):
        model, mesh = model_2d(6, 4, Lx=1.0, Ly=0.5, rho_cp=2.0)
        model.physics.initField('T', lambda x, y: 100.0 + 50.0 * x * y)
        model.physics.addBC_HeatFluxEdges(mesh.boundary_edges(lambda x, y: x > 1.0 - 1e-12), q)
        configure(model, study=StudyType.transient, timeDisc=theta, timeStep=0.02, totalTime=total_time)
        T0 = np.asarray(model.physics.var['T'].values).copy()
        model.solve()

        change = np.sum(model._system().mass() @ (np.asarray(model.sol['T'])[:, -1] - T0))
        assert np.isclose(change, -q * 0.5 * total_time, rtol=1e-8), (theta, change)


def test_transient_tends_to_the_nonlinear_steady_state():
    n = 16
    steady, _ = solve_1d(n)

    model, mesh = model_1d(n)
    model.physics.initField('T', 100.0)
    model.physics.addBC_Temperature(0, 20.0)
    model.physics.addBC_Temperature(n, 300.0)
    configure(model, study=StudyType.transient, timeDisc=1.0, timeStep=0.01, totalTime=2.0)
    model.solve()

    assert np.allclose(np.asarray(model.sol['T'])[:, -1], steady.sol['T'], atol=1e-6)


if __name__ == "__main__":
    tests = [test_kirchhoff_exact_solution, test_result_does_not_depend_on_the_initial_guess,
             test_2d_with_y_independent_data_equals_1d, test_2d_manufactured_nonlinear_diffusion_order_two,
             test_jacobian_with_k_of_T_convection_and_radiation_matches_finite_differences,
             test_constant_law_reproduces_the_constant_conductivity_matrix,
             test_transient_conserves_energy_with_k_of_T, test_transient_tends_to_the_nonlinear_steady_state]
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
