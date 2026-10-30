"""Stage 17, priority 1: robustness of the nonlinear solver (field scaling, pseudo-transient continuation, relaxation).

Run from the project root:  python Test/test_nonlinear_globalization.py
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
from Source.Physics.FluidFlow import ff
from Source.Physics.HeatTransfer import ht
from Source.Physics.Rheology import Bingham, CrossWLF
from Source.Physics.Turbulence import sa
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers

# Ghia, Ghia & Shin (1982): u on the vertical centreline of the lid-driven cavity
GHIA = {1000.0: {0.9531: 0.46604, 0.5: -0.06080, 0.1719: -0.38289, 0.1016: -0.29730},
        5000.0: {0.9531: 0.46036, 0.5: -0.03039, 0.1719: -0.33050, 0.1016: -0.40435}}


def configure(model, **options):
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, Method=SolverType.direct,
                              Solver=DirectSolvers.PARDISO, **options)


def rect_mesh(size, div):
    mesh = Mesh()
    mesh.generate2DMesh(size, div, ElementType.quadrilateral)
    return mesh


def xy_of(mesh):
    return np.array([n.coor[:2] for n in mesh.NL], dtype=float)


def cavity(Re, n, **options):

    """Lid-driven cavity started from rest, without continuation in Re."""

    mesh = rect_mesh((1.0, 1.0), (n, n))
    model = Model(name='cavity', dim=2, mesh=mesh, mat=material('X', miu=1.0 / Re, rho=1.0), psc=ff)
    model.physics.setInertia(True)
    model.physics.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), (1.0, 0.0))
    model.physics.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or x < 1e-9 or x > 1 - 1e-9))
    configure(model, **options)
    model.solve()
    return model, mesh


def error_to_ghia(model, mesh, Re):
    xy = xy_of(mesh)
    line = np.abs(xy[:, 0] - 0.5) < 1e-9
    order = np.argsort(xy[line, 1])
    return max(abs(np.interp(y, xy[line, 1][order], model.sol['u'][line, 0][order]) - value)
               for y, value in GHIA[Re].items())


def bingham_channel(m, **options):

    """Pressure-driven Bingham flow between plates, from rest: no Newtonian solution to start from."""

    mesh = rect_mesh((0.5, 1.0), (2, 40))
    model = Model(name='bingham', dim=2, mesh=mesh, mat=material('X', miu=1.0), psc=ff)
    model.physics.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x < 1e-9), 2.0)
    model.physics.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x > 0.5 - 1e-9), 0.0)
    model.physics.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or y > 1 - 1e-9))
    model.physics.setViscosityLaw(Bingham(1.0, 0.5, m=m))
    configure(model, **options)
    model.solve()

    s = np.abs(xy_of(mesh)[:, 1] - 0.5)
    y0 = 0.125
    exact = np.where(s < y0, 2 * (0.25 - y0 ** 2) - 0.5 * (0.5 - y0), 2 * (0.25 - s ** 2) - 0.5 * (0.5 - s))
    return model, np.max(np.abs(model.sol['u'][:, 0] - exact)) / exact.max()


def cold_melt(**options):

    """Couette flow of a Cross-WLF melt started at the cold wall temperature, where the viscosity is about 1e8 times
    that of the hot wall."""

    mesh = rect_mesh((0.5, 1.0), (2, 24))
    model = Model(name='melt', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.0, k=1.0, Cp=1.0), psc=[ff, ht])
    fluid, heat = model.physics.fluid, model.physics.heat
    fluid.setViscosityLaw(CrossWLF(n=0.3, tau_star=3e4, D1=1e13, D2=373.15, A1=30.0, A2=51.6))
    fluid.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9))
    fluid.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), (1.0, 0.0))
    fluid.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x < 1e-9))
    fluid.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x > 0.5 - 1e-9))
    heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: y < 1e-9), 120.0)
    heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), 250.0)
    model.physics.initField('T', 120.0)
    configure(model, **options)
    model.solve()
    return model, xy_of(mesh)


def sa_channel_from_uniform(**options):

    """Spalart-Allmaras channel at Re_tau = 395 from a uniform state, instead of the law of the wall."""

    re_tau, beta = 395.0, 2.5
    mesh = rect_mesh((0.1, 1.0), (2, 48))
    for node in mesh.NL:
        x, e = node.coor
        node.coor = (x, 1.0 + np.tanh(beta * (e - 1.0)) / np.tanh(beta))

    model = Model(name='channel', dim=2, mesh=mesh, mat=material('X', miu=1.0 / re_tau, rho=1.0), psc=[ff, sa])
    fluid, turbulence = model.physics.fluid, model.physics.turbulence
    fluid.setInertia(True)
    fluid.setBodyForce((1.0, 0.0))
    wall = mesh.boundary_edges(lambda x, y: y < 1e-9)
    fluid.addBC_NoSlipEdges(wall)
    fluid.addBC_SymmetryEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9))
    fluid.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x < 1e-9))
    fluid.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x > 0.1 - 1e-9))
    turbulence.setWalls(wall)
    turbulence.coupleFlow()

    n = mesh.getNoN()
    fluid.var['u'].updateField(np.c_[10.0 * np.ones(n), np.zeros(n)])
    turbulence.var['nut'].updateField(np.full(n, 1e-2))
    configure(model, **options)
    model.solve()
    return model


# ----- tests -----

def test_field_scales_and_relaxation_weights():

    mesh = rect_mesh((1.0, 1.0), (2, 2))
    model = Model(name='m', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.0, k=1.0, Cp=1.0), psc=[ff, ht])
    model.physics.fluid.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9))
    model.physics.heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: y < 1e-9), 300.0)
    configure(model)
    system = model._system()
    layout = system.layout

    x = np.zeros(layout.ndof)
    x[layout.dofs('u').ravel()[0]] = -5.0
    scales = system.field_scales(x)
    assert np.all(scales[layout.dofs('u').ravel()] == 5.0)          # largest value of the state
    assert np.all(scales[layout.dofs('T').ravel()] == 300.0)        # a Dirichlet value counts
    assert np.all(scales[layout.dofs('p').ravel()] == 1.0)          # all zero

    weights = system.relaxation_weights({'T': 0.5})
    assert np.all(weights[layout.dofs('T').ravel()] == 0.5) and np.all(weights[layout.dofs('u').ravel()] == 1.0)
    assert np.all(system.relaxation_weights(None) == 1.0)

    try:
        system.relaxation_weights({'omega': 0.5})
    except ValueError:
        pass
    else:
        raise AssertionError("a relaxation of an unknown field must fail")

    try:
        model.solverConfiguration(globalization='magic')
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown globalization must fail")


def test_all_strategies_reach_the_same_solution():

    """The pseudo-time and the relaxation change the path, not the steady solution."""

    newton, _ = cavity(100.0, 24)
    pseudo, _ = cavity(100.0, 24, globalization='pseudo-transient')
    relaxed, _ = cavity(100.0, 24, relaxation={'u': 0.7})

    for model in (newton, pseudo, relaxed):
        assert model.nonlinear_converged

    for model in (pseudo, relaxed):
        assert np.max(np.abs(model.sol['u'] - newton.sol['u'])) < 1e-7
        assert np.max(np.abs(model.sol['p'] - newton.sol['p'])) < 1e-6

    assert relaxed.newton_iterations > newton.newton_iterations      # relaxing loses the quadratic convergence
    print(f"   iterations: Newton {newton.newton_iterations}, pseudo-transient {pseudo.newton_iterations}, "
          f"relaxed {relaxed.newton_iterations}")


def test_cavity_at_high_reynolds_from_rest():

    """Re = 5000 from rest: the Newton line search stalls here (over 2 minutes without converging), and the
    pseudo-transient solver needs no continuation in Re."""

    Re = 5000.0
    model, mesh = cavity(Re, 64, globalization='pseudo-transient')
    err = error_to_ghia(model, mesh, Re)
    print(f"   Re = {Re:.0f} from rest: {model.newton_iterations} iterations, largest error to Ghia {err:.3f}")
    assert model.nonlinear_converged
    assert model.newton_iterations < 60
    assert err < 0.08


def test_bingham_without_a_newtonian_start():

    newton, err_newton = bingham_channel(2000.0)
    pseudo, err_pseudo = bingham_channel(2000.0, globalization='pseudo-transient')
    print(f"   iterations: Newton {newton.newton_iterations}, pseudo-transient {pseudo.newton_iterations}; "
          f"relative errors {err_newton:.1e}, {err_pseudo:.1e}")
    assert pseudo.nonlinear_converged
    assert err_pseudo < 1e-3
    assert pseudo.newton_iterations < newton.newton_iterations


def test_cold_melt_with_cross_wlf():

    """Boundary values enter before the first step: imposing a hot wall during it changes the viscosity by orders of
    magnitude, and every step is rejected whatever the CFL."""

    model, xy = cold_melt(globalization='pseudo-transient')
    print(f"   Cross-WLF from the cold wall temperature: {model.newton_iterations} iterations")
    assert model.nonlinear_converged
    assert np.max(np.abs(model.sol['T'] - (120.0 + 130.0 * xy[:, 1]))) < 1e-6


def test_field_scaling_makes_newton_converge_from_a_uniform_state():

    """Spalart-Allmaras from a uniform state: with the row scaling alone Newton stalled at U_c+ = 8 after 200 iterations,
    because the residual of one field decided the line search for all of them."""

    for options in ({}, {'globalization': 'pseudo-transient'}):
        model = sa_channel_from_uniform(**options)
        centreline = model.sol['u'][:, 0].max()
        print(f"   {options or 'Newton'}: {model.newton_iterations} iterations, U_c+ = {centreline:.2f} (DNS 20.1)")
        assert model.nonlinear_converged
        assert abs(centreline - 20.1) < 0.03 * 20.1


if __name__ == "__main__":

    tests = [test_field_scales_and_relaxation_weights, test_all_strategies_reach_the_same_solution,
             test_cavity_at_high_reynolds_from_rest, test_bingham_without_a_newtonian_start,
             test_cold_melt_with_cross_wlf, test_field_scaling_makes_newton_converge_from_a_uniform_state]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
