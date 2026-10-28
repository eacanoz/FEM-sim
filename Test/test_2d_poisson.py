"""Checks for the 2D steady heat conduction (Q4 elements).

Run from the project root:  python Test/test_2d_poisson.py
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
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers


def make_model(Lx, Ly, nx, ny, k=1.0):

    mesh = Mesh()
    mesh.generate2DMesh((Lx, Ly), (nx, ny), ElementType.quadrilateral)

    model = Model(name='2d', dim=2, mesh=mesh, mat=material('X', k=k, rho=1, Cp=1), psc=ht)
    model.physics.Convection = False

    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear,
                              Method=SolverType.direct, Solver=DirectSolvers.PARDISO)
    return model, mesh


def nodal_xy(mesh):
    return np.array([node.coor[:2] for node in mesh.NL])


def test_mesh_topology():
    """3x2 elements on [0, 2] x [0, 1]: 12 nodes, 6 elements, 10 boundary edges."""

    _, mesh = make_model(2.0, 1.0, 3, 2)

    assert mesh.getNoN() == 12 and mesh.getNoE() == 6
    assert len(mesh.boundary_edges(lambda x, y: True)) == 2 * (3 + 2)

    # Counter-clockwise: the signed area of every element is positive
    for element in mesh.EL:
        xy = np.array([node.coor[:2] for node in element.nodes])
        x, y = xy[:, 0], xy[:, 1]
        area2 = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        assert area2 > 0


def test_patch_dirichlet_linear_field():
    """T = 2x + 3y + 1 is reproduced exactly when it is imposed on the whole boundary."""

    model, mesh = make_model(2.0, 1.0, 4, 3)

    f = lambda x, y: 2 * x + 3 * y + 1
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: True), f)
    model.solve()

    exact = f(*nodal_xy(mesh).T)
    assert np.max(np.abs(model.sol['T'] - exact)) < 1e-10


def test_neumann_edge_patch():
    """T = 2x + 1 with Dirichlet on left, top and bottom and a prescribed flux on the right edge.

    k = 1, so the heat entering through x = Lx is -k dT/dx = -2 W/m2.
    """

    model, mesh = make_model(2.0, 1.0, 4, 3)

    f = lambda x, y: 2 * x + 1
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x < 2.0 - 1e-12), f)
    model.physics.addBC_HeatFluxEdges(mesh.boundary_edges(lambda x, y: x > 2.0 - 1e-12), -2.0)
    model.solve()

    exact = f(*nodal_xy(mesh).T)
    assert np.max(np.abs(model.sol['T'] - exact)) < 1e-10


def test_newton_edge_patch():
    """Same field with convection on the right edge: heat leaving through x = Lx is h (T - T_ext) = -k dT/dx = -2 with T = 5 at x = 2 and h = 4, so T_ext = 5.5."""

    model, mesh = make_model(2.0, 1.0, 4, 3)

    f = lambda x, y: 2 * x + 1
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x < 2.0 - 1e-12), f)
    model.physics.addBC_ConvectionEdges(mesh.boundary_edges(lambda x, y: x > 2.0 - 1e-12), 4.0, 5.5)
    model.solve()

    exact = f(*nodal_xy(mesh).T)
    assert np.max(np.abs(model.sol['T'] - exact)) < 1e-10


def l2_error_poisson(n):

    """Solves -lap(T) = 2 pi^2 sin(pi x) sin(pi y) on the unit square with T = 0 on the boundary."""

    model, mesh = make_model(1.0, 1.0, n, n)

    exact = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    model.physics.F_const = lambda x, y: 2 * np.pi ** 2 * exact(x, y)
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: True), 0.0)
    model.solve()

    geo = Discretization(mesh)
    T_nodes = np.asarray(model.sol["T"])

    T_gp = np.einsum('gi,ei->eg', geo.N, T_nodes[geo.conn])
    err2 = np.sum(geo.detJw * (T_gp - exact(geo.x_gp[..., 0], geo.x_gp[..., 1])) ** 2)

    return np.sqrt(err2)


def test_poisson_manufactured_order_two():
    """Q4 converges with order 2 in the L2 norm."""

    sizes = [4, 8, 16, 32]
    errors = [l2_error_poisson(n) for n in sizes]
    rates = [np.log(errors[i] / errors[i + 1]) / np.log(2) for i in range(len(sizes) - 1)]

    print("   L2 errors:", ["%.3e" % e for e in errors], "rates:", ["%.2f" % r for r in rates])

    assert all(r > 1.9 for r in rates[1:]), rates


def test_convection_2d_matches_1d():
    """Convection-diffusion with a flow along x and Dirichlet left/right does not depend on y.

    The Q4 result must equal the 1D result on the same x nodes.
    """

    from Source.enums import ShapeFunctionType

    nx = 8

    model2d, mesh2d = make_model(1.0, 1.0, nx, 3, k=0.5)
    model2d.physics.Convection = True
    model2d.physics.vel = [2.0, 0.0]
    model2d.physics.addBC_TemperatureEdges(mesh2d.boundary_edges(lambda x, y: x < 1e-12), 100.0)
    model2d.physics.addBC_TemperatureEdges(mesh2d.boundary_edges(lambda x, y: x > 1 - 1e-12), 20.0)
    model2d.solve()

    mesh1d = Mesh()
    mesh1d.Generate_Mesh(1, 1.0, nx, ElementType.line)
    mesh1d.setElementShapeFunction(ShapeFunctionType.linear)
    model1d = Model(name='1d', dim=1, mesh=mesh1d, mat=material('X', k=0.5, rho=1, Cp=1), psc=ht)
    model1d.physics.Convection = True
    model1d.physics.vel = 2.0
    model1d.physics.addBC_Temperature(id=0, T=100.0)
    model1d.physics.addBC_Temperature(id=nx, T=20.0)
    model1d.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear,
                                Method=SolverType.direct, Solver=DirectSolvers.PARDISO)
    model1d.solve()

    T2 = np.asarray(model2d.sol['T']).reshape(4, nx + 1)   # rows are y levels
    T1 = np.asarray(model1d.sol['T'])

    assert np.allclose(T2, T1[None, :], atol=1e-9), np.max(np.abs(T2 - T1[None, :]))


if __name__ == "__main__":
    tests = [test_mesh_topology, test_patch_dirichlet_linear_field, test_neumann_edge_patch,
             test_newton_edge_patch, test_poisson_manufactured_order_two,
             test_convection_2d_matches_1d]
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
