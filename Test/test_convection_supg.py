"""Shared convection velocity and SUPG stabilization of the convection-diffusion operator.

Run from the project root:  python Test/test_convection_supg.py
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
from Source.Physics.MassTransfer import mt
from Source.Simulation.Discretization import Discretization
import Source.Simulation.Operators as ops
from Source.enums import ElementType, ShapeFunctionType, StudyType, ProblemType, SolverType, DirectSolvers


def configure(model, study=StudyType.steady_state, **time):
    model.solverConfiguration(Study=study, Type=ProblemType.linear, Method=SolverType.direct,
                              Solver=DirectSolvers.PARDISO, **time)


def line_mesh(n):
    mesh = Mesh()
    mesh.Generate_Mesh(1, 1.0, n, ElementType.line)
    mesh.setElementShapeFunction(ShapeFunctionType.linear)
    return mesh


def ends(mesh, left, right, field='T'):
    for node in mesh.NL:
        x = node.coor[0]
        if x < 1e-12:
            node.BC[field] = {'type': 'Dirichlet', 'value': left}
        elif x > 1 - 1e-12:
            node.BC[field] = {'type': 'Dirichlet', 'value': right}


def boundary_layer_1d(n, Pe, stab):

    """Steady c v u' = k u'' with u(0) = 0, u(1) = 1; Pe = c v L / k. Returns (model.sol, exact)."""

    mesh = line_mesh(n)
    model = Model(name='cd', dim=1, mesh=mesh, mat=material('X', k=1.0 / Pe, rho=1.0, Cp=1.0), psc=ht)
    model.physics.setVelocity(1.0)
    model.physics.setStabilization(stab)
    ends(mesh, 0.0, 1.0)
    model.physics.initField('T', 0.0)
    configure(model)
    model.solve()

    x = np.array([node.coor[0] for node in mesh.NL])
    return np.asarray(model.sol['T']), np.expm1(Pe * x) / np.expm1(Pe)


def test_supg_is_nodally_exact_in_1d_at_high_peclet():
    for n, Pe in ((10, 50.0), (10, 500.0), (20, 20.0)):
        u, exact = boundary_layer_1d(n, Pe, 'SUPG')
        assert np.max(np.abs(u - exact)) < 1e-8, (n, Pe, np.max(np.abs(u - exact)))


def test_galerkin_oscillates_and_supg_does_not():
    galerkin, exact = boundary_layer_1d(10, 50.0, None)
    supg, _ = boundary_layer_1d(10, 50.0, 'SUPG')

    assert galerkin.min() < -0.05 or np.any(np.diff(galerkin) < -1e-3)   # wiggles
    assert np.all(np.diff(supg) >= -1e-12) and supg.min() >= -1e-12 and supg.max() <= 1 + 1e-12


def test_supg_does_nothing_when_diffusion_dominates():
    galerkin, _ = boundary_layer_1d(10, 0.01, None)
    supg, _ = boundary_layer_1d(10, 0.01, 'SUPG')

    assert np.max(np.abs(galerkin - supg)) < 1e-4


def test_tau_limits_and_reversed_velocity():
    mesh = Mesh()
    mesh.generate2DMesh((1.0, 1.0), (4, 4), ElementType.quadrilateral)
    disc = Discretization(mesh)

    h = 0.25
    tau_adv = ops.supg_tau(disc, 1.0, (2.0, 0.0), k=None)
    assert np.allclose(tau_adv, h / (2 * 2.0))                       # pure advection: h / (2 |a|)
    assert np.allclose(ops.supg_tau(disc, 1.0, (-2.0, 0.0), None), tau_adv)   # sign does not matter
    assert np.allclose(ops.supg_tau(disc, 1.0, (0.0, 0.0), None), 0.0)
    assert np.all(ops.supg_tau(disc, 1.0, (2.0, 0.0), k=1e6) < 1e-6)  # diffusion dominates: tau -> 0

    # Reversed flow gives the mirrored solution
    mesh2 = line_mesh(10)
    model = Model(name='r', dim=1, mesh=mesh2, mat=material('X', k=1 / 50, rho=1.0, Cp=1.0), psc=ht)
    model.physics.setVelocity(-1.0)
    model.physics.setStabilization('SUPG')
    ends(mesh2, 1.0, 0.0)
    model.physics.initField('T', 0.0)
    configure(model)
    model.solve()
    _, exact = boundary_layer_1d(10, 50.0, 'SUPG')
    assert np.max(np.abs(np.asarray(model.sol['T']) - exact[::-1])) < 1e-8


def test_2d_flow_along_x_equals_1d():
    nx, ny, Pe = 10, 3, 50.0
    mesh = Mesh()
    mesh.generate2DMesh((1.0, 0.3), (nx, ny), ElementType.quadrilateral)
    model = Model(name='cd2', dim=2, mesh=mesh, mat=material('X', k=1.0 / Pe, rho=1.0, Cp=1.0), psc=ht)
    model.physics.setVelocity([1.0, 0.0])
    model.physics.setStabilization('SUPG')
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x < 1e-12), 0.0)
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x > 1 - 1e-12), 1.0)
    model.physics.initField('T', 0.0)
    configure(model)
    model.solve()

    u1, _ = boundary_layer_1d(nx, Pe, 'SUPG')
    T = np.asarray(model.sol['T']).reshape(-1, nx + 1)
    assert np.allclose(T, u1[None, :], atol=1e-8)


def manufactured_error(n, stab):

    """c v . grad u - k lap u = f with u = sin(pi x) sin(pi y), v = (1, 0.5), k small but resolved."""

    k, v = 0.02, (1.0, 0.5)
    u = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    f = lambda x, y: (v[0] * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
                      + v[1] * np.pi * np.sin(np.pi * x) * np.cos(np.pi * y) + 2 * k * np.pi ** 2 * u(x, y))

    mesh = Mesh()
    mesh.generate2DMesh((1.0, 1.0), (n, n), ElementType.quadrilateral)
    model = Model(name='mms', dim=2, mesh=mesh, mat=material('X', k=k, rho=1.0, Cp=1.0), psc=ht)
    model.physics.setVelocity(list(v))
    model.physics.setStabilization(stab)
    model.physics.F_const = f
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: True), 0.0)
    model.physics.initField('T', 0.0)
    configure(model)
    model.solve()

    xy = np.array([node.coor for node in mesh.NL])
    return np.max(np.abs(np.asarray(model.sol['T']) - u(xy[:, 0], xy[:, 1])))


def test_supg_is_consistent_second_order_with_a_source():
    errors = [manufactured_error(n, 'SUPG') for n in (8, 16, 32)]

    assert errors[0] / errors[1] > 3.0 and errors[1] / errors[2] > 3.0, errors
    assert errors[2] < 5e-4, errors


def test_shared_velocity_makes_heat_and_species_follow_the_same_flow():

    """With rho Cp = 1, k = D and the same conditions, T (in the model units) and the species coincide."""

    Pe = 40.0
    mesh = line_mesh(10)
    model = Model(name='shared', dim=1, mesh=mesh, mat=material('X', k=1.0 / Pe, rho=1.0, Cp=1.0), psc=[ht, mt])
    model.physics.mass.setChemSpecies('A', 'A')
    model.physics.mass.setDiffusivity('A', 1.0 / Pe)
    model.physics.setVelocity(1.0)             # one call for both parts
    model.physics.setStabilization('SUPG')

    assert model.physics.heat.Convection and model.physics.mass.Convection
    assert model.physics.heat.Stab == 'SUPG' and model.physics.mass.Stab == 'SUPG'

    ends(mesh, 0.0, 1.0, 'T')
    ends(mesh, 0.0, 1.0, 'A')
    model.physics.initField('T', 0.0)
    model.physics.initField('A', 0.0)
    configure(model)
    model.solve()

    _, exact = boundary_layer_1d(10, Pe, 'SUPG')
    assert np.allclose(np.asarray(model.sol['T']), exact, atol=1e-8)
    assert np.allclose(np.asarray(model.sol['A']), exact, atol=1e-8)


def test_transient_tends_to_the_stabilized_steady_state():
    Pe = 50.0
    mesh = line_mesh(10)
    model = Model(name='tr', dim=1, mesh=mesh, mat=material('X', k=1.0 / Pe, rho=1.0, Cp=1.0), psc=ht)
    model.physics.setVelocity(1.0)
    model.physics.setStabilization('SUPG')
    ends(mesh, 0.0, 1.0)
    model.physics.initField('T', 0.0)
    configure(model, StudyType.transient, timeDisc=1.0, timeStep=0.05, totalTime=6.0)
    model.solve()

    _, exact = boundary_layer_1d(10, Pe, 'SUPG')
    assert np.max(np.abs(np.asarray(model.sol['T'])[:, -1] - exact)) < 1e-6


def test_errors():
    model = Model(name='e', dim=1, mesh=line_mesh(4), mat=material('X', k=1, rho=1, Cp=1), psc=ht)

    try:
        model.physics.setStabilization('GLS')
    except ValueError as error:
        assert 'GLS' in str(error)
    else:
        raise AssertionError("expected ValueError for an unknown stabilization")

    mesh = Mesh()
    mesh.generate2DMesh((1.0, 1.0), (2, 2), ElementType.quadrilateral)
    try:
        ops.convection(Discretization(mesh), 1.0, 1.0)
    except ValueError as error:
        assert '2D' in str(error)
    else:
        raise AssertionError("expected ValueError for a scalar velocity in 2D")


if __name__ == "__main__":
    tests = [test_supg_is_nodally_exact_in_1d_at_high_peclet, test_galerkin_oscillates_and_supg_does_not,
             test_supg_does_nothing_when_diffusion_dominates, test_tau_limits_and_reversed_velocity,
             test_2d_flow_along_x_equals_1d, test_supg_is_consistent_second_order_with_a_source,
             test_shared_velocity_makes_heat_and_species_follow_the_same_flow,
             test_transient_tends_to_the_stabilized_steady_state, test_errors]
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
