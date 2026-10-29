"""Stage 14 of the fluids plan: transient incompressible flow with Newton inside every step, BDF2 and consistent
stabilized masses; boundary forces from the reactions.

Run from the project root:  python Test/test_transient_flow.py
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
from Source.Physics.Rheology import CarreauYasuda
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers

NU = 0.1


def configure(model, study=StudyType.steady_state, kind=ProblemType.nonlinear, **time):
    model.solverConfiguration(Study=study, Type=kind, Method=SolverType.direct, Solver=DirectSolvers.PARDISO, **time)


def rect_mesh(size, div, element=ElementType.quadrilateral):
    mesh = Mesh()
    mesh.generate2DMesh(size, div, element)
    return mesh


def xy_of(mesh):
    return np.array([n.coor[:2] for n in mesh.NL], dtype=float)


def edges(mesh, selector):
    return mesh.boundary_edges(selector)


# ----- Taylor-Green vortex -----

def taylor_green_exact(x, y, t):

    """u = (sin x cos y, -cos x sin y) F, p = (cos 2x + cos 2y) F^2 / 4, F = exp(-2 nu t) (rho = 1): an exact
    Navier-Stokes solution. On [0, pi]^2 the walls have zero normal velocity and zero shear stress: slip walls."""

    F = np.exp(-2 * NU * t)
    return (np.c_[np.sin(x) * np.cos(y) * F, -np.cos(x) * np.sin(y) * F],
            0.25 * (np.cos(2 * x) + np.cos(2 * y)) * F ** 2)


def taylor_green(n, dt, T, scheme='BDF2', theta=1.0):

    mesh = rect_mesh((np.pi, np.pi), (n, n))
    model = Model(name='taylor-green', dim=2, mesh=mesh, mat=material('X', miu=NU, rho=1.0), psc=ff)
    model.physics.setInertia(True)
    for selector in (lambda x, y: x < 1e-9, lambda x, y: x > np.pi - 1e-9,
                     lambda x, y: y < 1e-9, lambda x, y: y > np.pi - 1e-9):
        model.physics.addBC_SymmetryEdges(edges(mesh, selector))
    model.physics.setPressureReference((np.pi / 2, np.pi / 2), -0.5)          # value of the exact p at t = 0
    model.physics.initField('u', lambda x, y: tuple(taylor_green_exact(np.array([x]), np.array([y]), 0.0)[0][0]))
    configure(model, StudyType.transient, timeDisc=theta, timeStep=dt, totalTime=T, timeScheme=scheme)
    model.solve()

    return model, xy_of(mesh)


def test_taylor_green_vortex():

    """Accuracy against the exact solution, and the temporal error of each scheme against a reference with a much
    smaller step on the same mesh (the spatial error cancels): implicit Euler order 1, BDF2 order 2. Crank-Nicolson
    is stable and by far the most accurate at these steps, but shows a small first-order term asymptotically."""

    model, xy = taylor_green(24, 0.05, 1.0)
    ue, pe = taylor_green_exact(xy[:, 0], xy[:, 1], 1.0)
    dp = model.sol['p'][:, -1] - pe
    err_u, err_p = np.max(np.abs(model.sol['u'][..., -1] - ue)), np.max(np.abs(dp - dp.mean()))
    print(f"   24x24, BDF2 dt = 0.05: |u - exact| {err_u:.1e}, |p - exact| {err_p:.1e}, "
          f"Newton per step {np.mean(model.step_iterations):.1f}")
    assert err_u < 1e-3 and err_p < 2e-3 and max(model.step_iterations) <= 4

    n, T = 10, 0.8
    reference = taylor_green(n, 0.005, T)[0].sol['u'][..., -1]
    errors = {}
    for scheme, theta in (('theta', 1.0), ('theta', 0.5), ('BDF2', 1.0)):
        e = [np.max(np.abs(taylor_green(n, dt, T, scheme, theta)[0].sol['u'][..., -1] - reference))
             for dt in (0.2, 0.1, 0.05)]
        errors[(scheme, theta)] = e
        rates = np.log2(np.array(e[:-1]) / np.array(e[1:]))
        print(f"   {scheme} theta = {theta}: temporal errors {['%.1e' % v for v in e]} rates {['%.2f' % r for r in rates]}")

    rate = lambda e: np.log2(e[-2] / e[-1])
    assert abs(rate(errors[('theta', 1.0)]) - 1.0) < 0.2
    assert abs(rate(errors[('BDF2', 1.0)]) - 2.0) < 0.2
    assert all(c < 0.5 * b for c, b in zip(errors[('theta', 0.5)], errors[('BDF2', 1.0)]))


# ----- start-up flow -----

def startup(theta, dt, t_end, scheme='theta', H=1.0, mu=1.0, rho=1.0, G=1.0):

    """Body-force driven start-up between plates, with inertia (zero here, but the nonlinear path is used)."""

    mesh = rect_mesh((0.5, H), (2, 24))
    model = Model(name='startup', dim=2, mesh=mesh, mat=material('X', miu=mu, rho=rho), psc=ff)
    model.physics.setInertia(True)
    model.physics.setBodyForce((G, 0.0))
    model.physics.addBC_NoSlipEdges(edges(mesh, lambda x, y: y < 1e-9 or y > H - 1e-9))
    model.physics.addBC_OutletEdges(edges(mesh, lambda x, y: x < 1e-9))
    model.physics.addBC_OutletEdges(edges(mesh, lambda x, y: x > 0.5 - 1e-9))
    model.physics.initField('u', 0.0)
    configure(model, StudyType.transient, timeDisc=theta, timeStep=dt, totalTime=t_end, timeScheme=scheme)
    model.solve()

    y = xy_of(mesh)[:, 1]
    series = sum(4 * G * H ** 2 / (mu * (k * np.pi) ** 3) * np.sin(k * np.pi * y / H)
                 * np.exp(-(k * np.pi) ** 2 * mu / rho * t_end / H ** 2) for k in range(1, 400, 2))
    exact = G / (2 * mu) * y * (H - y) - series

    return model, np.max(np.abs(model.sol['u'][:, 0, -1] - exact)) / np.max(exact)


def test_startup_flow_and_a_smooth_pressure():

    """Against the Fourier series. The exact pressure is zero; the computed one is a small spatial PSPG error of the
    start-up (large curvature next to the walls) that decays monotonically and is the same for every scheme: with
    Crank-Nicolson it no longer oscillates from step to step (algebraic rows and unknowns fully implicit)."""

    pressures = {}
    for theta, scheme in ((0.5, 'theta'), (1.0, 'BDF2')):
        model, err = startup(theta, 0.0025, 0.05, scheme)
        P = model.sol['p']
        worst = np.argmax(np.abs(P[:, -1]))
        steps = np.diff(P[worst, 1:])
        print(f"   {scheme} theta = {theta}: relative velocity error {err:.1e}, max |p| {np.max(np.abs(P)):.1e}, "
              f"final {np.max(np.abs(P[:, -1])):.1e}")
        assert err < 3e-3 and np.max(np.abs(P[:, -1])) < 3e-3
        assert np.all(steps > 0) or np.all(steps < 0)                       # monotone: no oscillation
        pressures[scheme] = P[:, -1]

    assert np.max(np.abs(pressures['theta'] - pressures['BDF2'])) < 5e-4


def test_transient_reaches_the_steady_state():

    """Lid-driven cavity at Re = 10 with a shear-thinning Carreau law: BDF2 with large steps ends on the steady
    Newton solution (the consistent stabilized masses do not shift the steady state)."""

    def cavity():
        mesh = rect_mesh((1.0, 1.0), (12, 12), ElementType.triangle)
        model = Model(name='cavity', dim=2, mesh=mesh, mat=material('X', miu=0.1, rho=1.0), psc=ff)
        model.physics.setInertia(True)
        model.physics.setViscosityLaw(CarreauYasuda(mu0=0.1, lam=1.0, n=0.5))
        model.physics.addBC_VelocityEdges(edges(mesh, lambda x, y: y > 1 - 1e-9), (1.0, 0.0))
        model.physics.addBC_NoSlipEdges(edges(mesh, lambda x, y: y < 1e-9 or x < 1e-9 or x > 1 - 1e-9))
        model.physics.initField('u', 0.0)
        return model

    steady = cavity()
    configure(steady)
    steady.solve()

    transient = cavity()
    configure(transient, StudyType.transient, timeStep=1.0, totalTime=60.0, timeScheme='BDF2')
    transient.solve()

    du = np.max(np.abs(transient.sol['u'][..., -1] - steady.sol['u']))
    dp = np.max(np.abs(transient.sol['p'][:, -1] - steady.sol['p']))
    print(f"   transient end vs steady: |du| {du:.1e}, |dp| {dp:.1e}; Newton per step {np.mean(transient.step_iterations):.1f}")
    assert du < 1e-6 and dp < 1e-5


# ----- tangent with the time derivative -----

def test_tangent_with_the_time_derivative():

    """Flow (Carreau, inertia, PSPG/SUPG with rho du/dt at the new time level) and heat convected by u (SUPG with
    rho Cp dT/dt), with a time context as the transient solver sets it, against finite differences (tau frozen)."""

    for element in (ElementType.quadrilateral, ElementType.triangle):
        mesh = rect_mesh((1.0, 1.0), (2, 2), element)
        model = Model(name='tangent', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.3, Cp=1.5, k=0.3),
                      psc=[ff, ht])
        fluid = model.physics.fluid
        fluid.setViscosityLaw(CarreauYasuda(mu0=2.0, lam=0.7, n=0.4))
        fluid.setInertia(True)
        fluid.setViscousHeating(True)
        model.physics.setVelocity('u')
        model.physics.setStabilization('SUPG')
        configure(model, StudyType.transient, timeStep=0.1, totalTime=1.0)

        layout = model.layout
        rng = np.random.default_rng(6)
        x = rng.normal(scale=0.5, size=layout.ndof)
        system = model._system()
        system.time = {'alpha': 15.0, 'beta': rng.normal(size=layout.ndof), 'theta': 1.0, 'x_old': np.zeros(layout.ndof),
                       'dt': 0.1}
        for term in system._terms:
            term.tau_state = x.copy()

        J, _ = system._assemble(x)
        FD = np.zeros((layout.ndof, layout.ndof))
        for k in range(layout.ndof):
            e = np.zeros(layout.ndof)
            e[k] = 1e-6
            FD[:, k] = (system._assemble(x + e)[1] - system._assemble(x - e)[1]) / 2e-6

        err = np.max(np.abs(J.toarray() - FD)) / np.max(np.abs(FD))
        assert err < 1e-7, (element, err)


def test_transient_heat_carried_by_the_computed_flow():

    """Plug flow solved by ff carries a transient temperature exactly as ht with the same prescribed velocity
    (SUPG with the time derivative inside the term = linear SUPG mass)."""

    U, k, L = 1.0, 0.02, 1.0
    mesh = rect_mesh((L, 0.2), (20, 2), ElementType.triangle)
    inlet, outlet = edges(mesh, lambda x, y: x < 1e-9), edges(mesh, lambda x, y: x > L - 1e-9)
    time = dict(timeDisc=1.0, timeStep=0.05, totalTime=0.5)

    coupled = Model(name='coupled', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.0, Cp=1.0, k=k), psc=[ff, ht])
    fluid, heat = coupled.physics.fluid, coupled.physics.heat
    fluid.addBC_VelocityEdges(inlet, (U, 0.0))
    fluid.addBC_SymmetryEdges(edges(mesh, lambda x, y: y < 1e-9))
    fluid.addBC_SymmetryEdges(edges(mesh, lambda x, y: y > 0.2 - 1e-9))
    fluid.addBC_OutletEdges(outlet)
    coupled.physics.setVelocity('u')
    coupled.physics.setStabilization('SUPG')
    heat.addBC_TemperatureEdges(inlet, 1.0)
    coupled.physics.initField('u', (U, 0.0))
    coupled.physics.initField('T', 0.0)
    configure(coupled, StudyType.transient, **time)
    coupled.solve()

    alone = Model(name='alone', dim=2, mesh=mesh, mat=material('X', rho=1.0, Cp=1.0, k=k), psc=ht)
    alone.physics.setVelocity((U, 0.0))
    alone.physics.setStabilization('SUPG')
    alone.physics.addBC_TemperatureEdges(inlet, 1.0)
    alone.physics.initField('T', 0.0)
    configure(alone, StudyType.transient, kind=ProblemType.linear, **time)
    alone.solve()

    assert np.max(np.abs(coupled.sol['T'] - alone.sol['T'])) < 1e-8, np.max(np.abs(coupled.sol['T'] - alone.sol['T']))


# ----- forces -----

def test_boundary_forces_from_the_reactions():

    """Momentum balance of a pressure-driven channel: the walls carry the pressure force dp * H (per unit depth);
    a pipe carries dp * pi R^2. A Couette lid feels -mu U / H per unit length."""

    H, L, dp = 1.0, 2.0, 3.0
    mesh = rect_mesh((L, H), (8, 6), ElementType.triangle)
    model = Model(name='channel', dim=2, mesh=mesh, mat=material('X', miu=0.7), psc=ff)
    walls = edges(mesh, lambda x, y: y < 1e-9 or y > H - 1e-9)
    model.physics.addBC_OutletEdges(edges(mesh, lambda x, y: x < 1e-9), dp)
    model.physics.addBC_OutletEdges(edges(mesh, lambda x, y: x > L - 1e-9), 0.0)
    model.physics.addBC_NoSlipEdges(walls)
    configure(model, kind=ProblemType.linear)
    model.solve()
    F = model.physics.boundaryForce(walls)
    assert np.isclose(F[0], dp * H, rtol=1e-10) and abs(F[1]) < 1e-3 * dp * L, F      # T3 mesh not symmetric in y

    mu, U = 0.7, 2.0
    mesh = rect_mesh((L, H), (4, 6))
    model = Model(name='couette', dim=2, mesh=mesh, mat=material('X', miu=mu), psc=ff)
    lid = edges(mesh, lambda x, y: y > H - 1e-9)
    model.physics.addBC_NoSlipEdges(edges(mesh, lambda x, y: y < 1e-9))
    model.physics.addBC_VelocityEdges(lid, (U, 0.0))
    model.physics.addBC_OutletEdges(edges(mesh, lambda x, y: x < 1e-9))
    model.physics.addBC_OutletEdges(edges(mesh, lambda x, y: x > L - 1e-9))
    configure(model, kind=ProblemType.linear)
    model.solve()
    F = model.physics.boundaryForce(lid)
    assert np.isclose(F[0], -mu * U / H * L, rtol=1e-8), F

    R = 0.5
    mesh = rect_mesh((R, L), (5, 8))
    model = Model(name='pipe', dim=2, mesh=mesh, mat=material('X', miu=0.7), psc=ff, axisymmetric=True)
    wall = edges(mesh, lambda x, y: x > R - 1e-9)
    model.physics.addBC_OutletEdges(edges(mesh, lambda x, y: y < 1e-9), dp)
    model.physics.addBC_OutletEdges(edges(mesh, lambda x, y: y > L - 1e-9), 0.0)
    model.physics.addBC_SymmetryEdges(edges(mesh, lambda x, y: x < 1e-9))
    model.physics.addBC_NoSlipEdges(wall)
    configure(model, kind=ProblemType.linear)
    model.solve()
    assert np.isclose(model.physics.boundaryForce(wall)[1], dp * np.pi * R ** 2, rtol=1e-10)


def test_scheme_errors():
    mesh = rect_mesh((1.0, 1.0), (2, 2))
    model = Model(name='err', dim=2, mesh=mesh, mat=material('X', miu=1.0), psc=ff)
    try:
        model.solverConfiguration(Study=StudyType.transient, timeScheme='RK4')
    except ValueError as error:
        assert 'BDF2' in str(error)
    else:
        raise AssertionError("expected an error for an unknown time scheme")


if __name__ == "__main__":
    tests = [test_taylor_green_vortex, test_startup_flow_and_a_smooth_pressure, test_transient_reaches_the_steady_state,
             test_tangent_with_the_time_derivative, test_transient_heat_carried_by_the_computed_flow,
             test_boundary_forces_from_the_reactions, test_scheme_errors]
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
