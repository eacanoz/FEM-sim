"""Stage 10 of the fluids plan: creeping (Stokes) flow with equal-order elements and PSPG.

Run from the project root:  python Test/test_stokes.py
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
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers


def configure(model, study=StudyType.steady_state, **time):
    model.solverConfiguration(Study=study, Type=ProblemType.linear, Method=SolverType.direct,
                              Solver=DirectSolvers.PARDISO, **time)


def rect_mesh(size, div, element=ElementType.quadrilateral):
    mesh = Mesh()
    mesh.generate2DMesh(size, div, element)
    return mesh


def xy_of(mesh):
    return np.array([n.coor[:2] for n in mesh.NL], dtype=float)


def edges(mesh, selector):
    return mesh.boundary_edges(selector)


def poiseuille(n, element=ElementType.quadrilateral, L=4.0, H=1.0, Umax=1.0, mu=2.0, projection=True):

    """Channel with a parabolic inlet, no-slip walls and a fully developed outlet at p = 0."""

    mesh = rect_mesh((L, H), (4 * n, n), element)
    model = Model(name='poiseuille', dim=2, mesh=mesh, mat=material('X', miu=mu), psc=ff)
    model.physics.viscous_projection = projection

    inlet = edges(mesh, lambda x, y: x < 1e-9)
    outlet = edges(mesh, lambda x, y: x > L - 1e-9)
    model.physics.addBC_VelocityEdges(inlet, lambda x, y: (4 * Umax * y * (H - y) / H ** 2, 0.0))
    model.physics.addBC_OutletEdges(outlet, 0.0)
    model.physics.addBC_NoSlipEdges(edges(mesh, lambda x, y: y < 1e-9 or y > H - 1e-9))
    configure(model)
    model.solve()

    xy = xy_of(mesh)
    u_exact = np.c_[4 * Umax * xy[:, 1] * (H - xy[:, 1]) / H ** 2, np.zeros(len(xy))]
    drop = 8 * mu * Umax / H ** 2 * L

    return model, mesh, np.max(np.abs(model.sol['u'] - u_exact)) / Umax, model.sol['p'][xy[:, 0] < 1e-9].mean() / drop - 1.0


# ----- exact and analytical solutions -----

def test_linear_flow_with_body_force_is_exact():

    """u = (y, x) (divergence free, no viscous force) and p = 2x + 3y with f = grad p: exact on distorted meshes."""

    for element in (ElementType.quadrilateral, ElementType.triangle):
        mesh = rect_mesh((1.0, 1.0), (6, 6), element)
        rng = np.random.default_rng(7)
        for node in mesh.NL:
            x, y = node.coor
            if 1e-9 < x < 1 - 1e-9 and 1e-9 < y < 1 - 1e-9:
                node.coor = (x + 0.03 * rng.uniform(-1, 1), y + 0.03 * rng.uniform(-1, 1))

        model = Model(name='patch', dim=2, mesh=mesh, mat=material('X', miu=1.7), psc=ff)
        model.physics.setBodyForce((2.0, 3.0))
        model.physics.addBC_VelocityEdges(edges(mesh, None), lambda x, y: (y, x))
        model.physics.setPressureReference((0.0, 0.0), value=0.0)
        configure(model)
        model.solve()

        xy = xy_of(mesh)
        assert np.max(np.abs(model.sol['u'] - np.c_[xy[:, 1], xy[:, 0]])) < 1e-10
        assert np.max(np.abs(model.sol['p'] - (2 * xy[:, 0] + 3 * xy[:, 1]))) < 1e-9


def test_poiseuille_is_nodally_exact_and_conserves_the_flow_rate():

    """With the recovered viscous stress in PSPG the Q4 equations hold for the exact parabola and linear pressure:
    velocity and pressure drop are exact at the nodes. On T3 both converge at second order. Without the recovered
    stress (plain PSPG) the pressure drop is only O(h^1.5)."""

    for element in (ElementType.quadrilateral, ElementType.triangle):
        errors, drops = [], []
        for n in (4, 8, 16):
            model, mesh, err_u, err_drop = poiseuille(n, element)
            errors.append(err_u)
            drops.append(abs(err_drop))

            q_in = model.physics.flowRate(edges(mesh, lambda x, y: x < 1e-9))
            q_out = model.physics.flowRate(edges(mesh, lambda x, y: x > 4.0 - 1e-9))
            assert abs(q_in + q_out) < 1e-12, (q_in, q_out)          # integral of div u = 0 exactly

        if element == ElementType.quadrilateral:
            assert max(errors) < 1e-11 and max(drops) < 1e-11, (errors, drops)
        else:
            rates_u = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
            rates_p = np.log2(np.array(drops[:-1]) / np.array(drops[1:]))
            print(f"   triangle: velocity {['%.1e' % e for e in errors]} rates {['%.2f' % r for r in rates_u]}; "
                  f"pressure drop {['%.1e' % e for e in drops]} rates {['%.2f' % r for r in rates_p]}")
            assert rates_u[-1] > 1.8 and rates_p[-1] > 1.8 and drops[-1] < 1e-3

        errors, drops = [], []
        for n in (4, 8, 16):
            _, _, err_u, err_drop = poiseuille(n, element, projection=False)
            errors.append(err_u)
            drops.append(abs(err_drop))
        rates_u = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
        rates_p = np.log2(np.array(drops[:-1]) / np.array(drops[1:]))
        print(f"   {element.name}, plain PSPG: velocity rates {['%.2f' % r for r in rates_u]}; "
              f"pressure drop {['%.1e' % e for e in drops]} rates {['%.2f' % r for r in rates_p]}")
        assert rates_u[-1] > 1.8 and rates_p[-1] > 1.2


def test_viscous_projection_removes_the_pspg_pressure_error():

    """Without the projected viscous stress, PSPG on linear elements loses consistency: the pressure drop is first order."""

    _, _, _, with_projection = poiseuille(8)
    _, _, _, without = poiseuille(8, projection=False)
    print(f"   pressure drop error with projection {abs(with_projection):.1e}, without {abs(without):.1e}")
    assert abs(with_projection) < 0.25 * abs(without)


def exact_stokes(mu):

    """Exact Stokes flow without body force: psi = x e^x sin(y) is biharmonic (x times a harmonic function).

    u = (psi_y, -psi_x), p = 2 mu e^x cos(y) and the traction sigma . n on x = 1 (n = (1, 0)).
    """

    U = lambda x, y: np.stack(np.broadcast_arrays(x * np.exp(x) * np.cos(y), -(1 + x) * np.exp(x) * np.sin(y)), axis=-1)
    P = lambda x, y: 2 * mu * np.exp(x) * np.cos(y)

    def traction(x, y):
        ux = (1 + x) * np.exp(x) * np.cos(y)
        uy = -x * np.exp(x) * np.sin(y)
        vx = -(2 + x) * np.exp(x) * np.sin(y)
        return (-P(x, y) + 2 * mu * ux, mu * (uy + vx))

    return U, P, traction


def test_exact_stokes_flow_converges():

    """Velocity on three sides, exact traction on the fourth. Equal-order PSPG: velocity O(h^2), pressure O(h) in the
    maximum norm and O(h^1.5) in the mean square, the known rates of stabilized P1/Q1 elements."""

    mu = 0.7
    U, P, traction = exact_stokes(mu)

    for element in (ElementType.quadrilateral, ElementType.triangle):
        eu, ep, rms = [], [], []
        for n in (8, 16, 32):
            mesh = rect_mesh((1.0, 1.0), (n, n), element)
            model = Model(name='exact', dim=2, mesh=mesh, mat=material('X', miu=mu), psc=ff)
            model.physics.addBC_VelocityEdges(edges(mesh, lambda x, y: x < 1 - 1e-9), lambda x, y: tuple(U(x, y)))
            model.physics.addBC_TractionEdges(edges(mesh, lambda x, y: x > 1 - 1e-9), traction)
            configure(model)
            model.solve()

            xy = xy_of(mesh)
            eu.append(np.max(np.abs(model.sol['u'] - U(xy[:, 0], xy[:, 1]))))
            dp = model.sol['p'] - P(xy[:, 0], xy[:, 1])
            ep.append(np.max(np.abs(dp)))
            rms.append(np.sqrt(np.mean(dp ** 2)))

        rate = lambda e: np.log2(e[-2] / e[-1])
        print(f"   {element.name}: u {['%.1e' % e for e in eu]} rate {rate(eu):.2f}; p max {['%.1e' % e for e in ep]} "
              f"rate {rate(ep):.2f}; p rms rate {rate(rms):.2f}")
        assert rate(eu) > 1.8 and rate(ep) > 0.75 and rate(rms) > 1.3


def test_lid_driven_cavity():

    """Stokes cavity: primary vortex psi = -0.1001 near y = 0.764 (Shankar & Deshpande 2000), mirror symmetry
    u(x, y) = (u_x, -u_y)(1 - x, y) and no net flow across the vertical centerline."""

    n = 64
    mesh = rect_mesh((1.0, 1.0), (n, n))
    model = Model(name='cavity', dim=2, mesh=mesh, mat=material('X', miu=1.0), psc=ff)
    model.physics.addBC_VelocityEdges(edges(mesh, lambda x, y: y > 1 - 1e-9), (1.0, 0.0))
    model.physics.addBC_NoSlipEdges(edges(mesh, lambda x, y: y < 1e-9 or x < 1e-9 or x > 1 - 1e-9))  # corners: wall
    configure(model)
    model.solve()

    xy, u = xy_of(mesh), model.sol['u']

    line = np.abs(xy[:, 0] - 0.5) < 1e-9
    order = np.argsort(xy[line, 1])
    y, ux = xy[line, 1][order], u[line, 0][order]
    psi = np.concatenate([[0.0], np.cumsum(0.5 * (ux[1:] + ux[:-1]) * np.diff(y))])

    k = np.argmin(psi)
    print(f"   psi_min = {psi[k]:.5f} at y = {y[k]:.3f}; net flow across x = 0.5: {psi[-1]:.1e}")
    assert abs(psi[k] + 0.1001) < 0.0005 and abs(y[k] - 0.764) < 0.02
    assert abs(psi[-1]) < 1e-5

    index = {(round(a, 9), round(b, 9)): i for i, (a, b) in enumerate(xy)}
    mirror = np.array([index[(round(1 - a, 9), round(b, 9))] for a, b in xy])
    assert np.max(np.abs(u[:, 0] - u[mirror, 0])) < 1e-12 and np.max(np.abs(u[:, 1] + u[mirror, 1])) < 1e-12


def test_symmetry_plane_reproduces_the_full_channel():

    L, H, n = 2.0, 1.0, 8
    results = {}

    for half in (False, True):
        height = H / 2 if half else H
        mesh = rect_mesh((L, height), (2 * n, n // 2 if half else n))
        model = Model(name='sym', dim=2, mesh=mesh, mat=material('X', miu=1.0), psc=ff)
        model.physics.viscous_projection = False    # the lumped projection is one-sided on the plane; see below
        model.physics.addBC_VelocityEdges(edges(mesh, lambda x, y: x < 1e-9), lambda x, y: (y * (H - y), 0.0))
        model.physics.addBC_OutletEdges(edges(mesh, lambda x, y: x > L - 1e-9))
        model.physics.addBC_NoSlipEdges(edges(mesh, lambda x, y: y < 1e-9))
        if half:
            model.physics.addBC_SymmetryEdges(edges(mesh, lambda x, y: y > height - 1e-9))
        else:
            model.physics.addBC_NoSlipEdges(edges(mesh, lambda x, y: y > H - 1e-9))
        configure(model)
        model.solve()
        xy = xy_of(mesh)
        keep = xy[:, 1] <= H / 2 + 1e-9
        results[half] = (xy[keep], model.sol['u'][keep], model.sol['p'][keep])

    (xf, uf, pf), (xh, uh, ph) = results[False], results[True]
    assert np.allclose(xf, xh)
    assert np.max(np.abs(uf - uh)) < 1e-12 and np.max(np.abs(pf - ph)) < 1e-10


def startup_model(theta, dt, t_end, H=1.0, mu=1.0, rho=1.0, G=1.0):

    mesh = rect_mesh((0.5, H), (2, 24))
    model = Model(name='startup', dim=2, mesh=mesh, mat=material('X', miu=mu, rho=rho), psc=ff)
    model.physics.setBodyForce((G, 0.0))
    model.physics.addBC_NoSlipEdges(edges(mesh, lambda x, y: y < 1e-9 or y > H - 1e-9))
    model.physics.addBC_OutletEdges(edges(mesh, lambda x, y: x < 1e-9))
    model.physics.addBC_OutletEdges(edges(mesh, lambda x, y: x > 0.5 - 1e-9))
    model.physics.initField('u', 0.0)
    configure(model, StudyType.transient, timeDisc=theta, timeStep=dt, totalTime=t_end)
    model.solve()

    return model, xy_of(mesh)


def test_startup_flow_between_plates():

    """Body-force driven start-up from rest, rho du/dt = mu u'' + G, against the Fourier series solution, and its
    steady limit. The pressure is an algebraic unknown: Crank-Nicolson leaves an oscillating O(h^2 / dt) pressure
    (the velocity stays second order); implicit Euler damps it, and the steady state has p = 0 exactly."""

    H, mu, rho, G, t_end = 1.0, 1.0, 1.0, 1.0, 0.05

    model, xy = startup_model(0.5, 0.0025, t_end)
    y = xy[:, 1]
    series = sum(4 * G * H ** 2 / (mu * (k * np.pi) ** 3) * np.sin(k * np.pi * y / H)
                 * np.exp(-(k * np.pi) ** 2 * mu / rho * t_end / H ** 2) for k in range(1, 400, 2))
    exact = G / (2 * mu) * y * (H - y) - series

    U = model.sol['u']
    assert U.shape == (len(xy), 2, 21)
    err = np.max(np.abs(U[:, 0, -1] - exact)) / np.max(exact)
    print(f"   Crank-Nicolson at t = {t_end}: relative velocity error {err:.1e}, "
          f"max |p| {np.max(np.abs(model.sol['p'][:, -1])):.1e}")
    assert err < 5e-3 and np.max(np.abs(U[:, 1, -1])) < 1e-10 and np.max(np.abs(model.sol['p'][:, -1])) < 0.03

    model, xy = startup_model(1.0, 0.05, 3.0)
    p = np.abs(model.sol['p']).max(axis=0)
    assert np.all(np.diff(p[1:]) <= 1e-14)                                    # decays monotonically
    assert np.max(np.abs(model.sol['u'][:, 0, -1] - G / (2 * mu) * xy[:, 1] * (H - xy[:, 1]))) < 1e-6
    assert p[-1] < 1e-6


def test_errors():

    mesh = rect_mesh((1.0, 1.0), (2, 2), ElementType.triangle)
    model = Model(name='err', dim=2, mesh=mesh, mat=material('X', miu=1.0), psc=ff)

    for call, text in ((lambda: model.physics.setStabilization('GLS'), 'PSPG'),
                       (lambda: model.physics.setVelocity((1.0, 0.0)), 'incognita'),
                       (lambda: model.physics.addBC_Velocity(0, (1.0,)), 'componentes'),
                       (lambda: model.physics.addBC_SymmetryEdges([(0, 4)]), 'alineadas'),
                       (lambda: model.physics.addBC_SymmetryEdges(edges(mesh, None)), 'horizontales')):
        try:
            call()
        except (ValueError, AttributeError) as error:
            assert text in str(error), str(error)
        else:
            raise AssertionError(f"expected an error containing {text!r}")


if __name__ == "__main__":
    tests = [test_linear_flow_with_body_force_is_exact, test_poiseuille_is_nodally_exact_and_conserves_the_flow_rate,
             test_viscous_projection_removes_the_pspg_pressure_error, test_exact_stokes_flow_converges,
             test_lid_driven_cavity, test_symmetry_plane_reproduces_the_full_channel, test_startup_flow_between_plates,
             test_errors]
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
