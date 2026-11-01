"""Stage 17: wall slip of polymer melts and the Phan-Thien-Tanner model.

Run from the project root:  python Test/test_slip_ptt.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.Rheology import NavierSlip, PowerSlip, PowerLaw
from Source.Physics.Viscoelastic import ve, conformation_rate
from Source.enums import ElementType, StudyType, ProblemType

ELEMENTS = ((ElementType.quadrilateral, 2), (ElementType.triangle, 2), (ElementType.quadrilateral, 1), (ElementType.triangle, 1))


def rect(size, div, element=ElementType.quadrilateral, order=2):
    mesh = Mesh()
    mesh.generate2DMesh(size, div, element, order=order)
    return mesh


def xy_of(mesh):
    return np.array([n.coor[:2] for n in mesh.NL], dtype=float)


def steady(model, **options):
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, **options)
    model.solve()


def channel(law, G=2.0, mu=0.5, element=ElementType.quadrilateral, order=2, ny=6):

    """Half channel between the axis y = 0 and a slipping wall y = 1: tau_w = G."""

    mesh = rect((1.0, 1.0), (3, ny), element, order)
    model = Model(name='slip', dim=2, mesh=mesh, mat=material('X', miu=mu), psc=ff)
    flow = model.physics
    flow.addBC_SymmetryEdges(mesh.boundary_edges(lambda x, y: y < 1e-9))
    flow.addBC_SlipLawEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), law)
    flow.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x < 1e-9), G)
    flow.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x > 1 - 1e-9), 0.0)
    return model, mesh


# ----- wall slip -----

def test_navier_slip_is_exact():

    """Poiseuille with slip: u = G (1 - y^2) / (2 mu) + G / beta, quadratic. Exact with Taylor-Hood, nodally close with equal order."""

    G, mu, beta = 2.0, 0.5, 4.0
    for element, order in ELEMENTS:
        model, mesh = channel(NavierSlip(beta), G, mu, element, order, ny=6 if order == 2 else 24)
        steady(model)
        y = xy_of(mesh)[:, 1]
        exact = G * (1 - y ** 2) / (2 * mu) + G / beta
        err = np.max(np.abs(model.sol['u'][:, 0] - exact))
        assert model.nonlinear_converged and err < (1e-10 if order == 2 else 2e-2), (element, order, err)
        assert np.max(np.abs(model.sol['u'][:, 1])) < (1e-10 if order == 2 else 1e-2)


def test_no_slip_and_free_slip_limits():

    """Beta large: the wall sticks. Beta small: plug flow, with u_s = tau_w / beta."""

    G, mu = 2.0, 0.5
    model, mesh = channel(NavierSlip(1e8), G, mu)
    steady(model)
    y = xy_of(mesh)[:, 1]
    assert np.max(np.abs(model.sol['u'][:, 0] - G * (1 - y ** 2) / (2 * mu))) < 1e-7

    model, mesh = channel(NavierSlip(1e-3), G, mu)
    steady(model)
    u = model.sol['u'][:, 0]
    assert abs(u[np.argmax(xy_of(mesh)[:, 1])] - G / 1e-3) < 1e-6 and (u.max() - u.min()) / u.max() < 2e-3


def test_power_slip():

    """tau = k u_s^m at the wall: u_s = (tau_w / k)^(1 / m), any m."""

    G, mu = 2.0, 0.5
    for k, m in ((4.0, 0.5), (1.5, 2.0), (3.0, 0.3)):
        model, mesh = channel(PowerSlip(k, m), G, mu)
        steady(model, globalization='pseudo-transient')
        y = xy_of(mesh)[:, 1]
        exact = G * (1 - y ** 2) / (2 * mu) + (G / k) ** (1.0 / m)
        assert model.nonlinear_converged
        assert np.max(np.abs(model.sol['u'][:, 0] - exact)) < 1e-7, (k, m)


def test_slip_threshold():

    """Below tau_c the wall sticks (slip ~ u_c small); above it, tau_w - tau_c drives the slip, tau = tau_c + k u_s^m."""

    mu, tau_c, k, m = 0.5, 1.0, 4.0, 1.0
    law = PowerSlip(k, m, tau_c=tau_c, u_c=1e-5)
    for G, expected in ((0.5, 0.0), (2.0, (2.0 - tau_c) / k)):
        model, mesh = channel(law, G, mu)
        steady(model, globalization='pseudo-transient')
        y = xy_of(mesh)[:, 1]
        slip = model.sol['u'][np.argmax(y), 0]
        assert model.nonlinear_converged, G
        assert abs(slip - expected) < 1e-3 * max(expected, 0.1), (G, slip, expected)


def test_mooney_analysis():

    """The classical test of slip in capillaries: for two radii at the same wall stress, gamma_a = gamma_true + 4 u_s / R. A power law fluid
    with Navier slip, axisymmetric. Mooney recovers the slip velocity that was given."""

    K, n, beta, tau_w = 2.0, 0.5, 40.0, 3.0

    def apparent(R):
        L = 2.0 * R
        mesh = rect((R, L), (8, 8))
        model = Model(name='capillary', dim=2, mesh=mesh, mat=material('X', miu=1.0), psc=ff, axisymmetric=True)
        flow = model.physics
        flow.addBC_SymmetryEdges(mesh.boundary_edges(lambda x, y: x < 1e-12))
        flow.addBC_SlipLawEdges(mesh.boundary_edges(lambda x, y: x > R * (1 - 1e-9)), NavierSlip(beta))
        outlet = mesh.boundary_edges(lambda x, y: y > L * (1 - 1e-9))
        flow.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: y < 1e-12 * L), 2.0 * tau_w * L / R)
        flow.addBC_OutletEdges(outlet, 0.0)
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear)
        model.solve()
        flow.setViscosityLaw(PowerLaw(K, n, gamma_min=1e-6))
        steady(model, globalization='pseudo-transient')
        assert model.nonlinear_converged
        return 4.0 * flow.flowRate(outlet) / (np.pi * R ** 3)

    R1, R2 = 0.5, 1.0
    a1, a2 = apparent(R1), apparent(R2)
    measured = (a1 - a2) / (4.0 / R1 - 4.0 / R2)
    print(f"   slip velocity from Mooney {measured:.5f}, given {tau_w / beta:.5f}")
    assert abs(measured / (tau_w / beta) - 1) < 2e-3

    # And the true shear rate of the power law at that stress
    true = a2 - 4.0 * measured / R2
    wall = (tau_w / K) ** (1.0 / n)
    assert abs(true / (4.0 * n / (3.0 * n + 1.0) * wall) - 1) < 5e-3


def test_slip_tangent():

    for element, order in ELEMENTS:
        mesh = rect((1.0, 1.0), (2, 2), element, order)
        model = Model(name='tangent', dim=2, mesh=mesh, mat=material('X', miu=0.5, rho=1.0), psc=ff)
        flow = model.physics
        flow.setInertia(True)
        flow.addBC_SlipLawEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), PowerSlip(2.0, 0.5, tau_c=0.5, u_c=0.1, u_min=1e-3))
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
        system = model._system()
        n = system.layout.ndof
        x = np.random.default_rng(2).normal(scale=0.3, size=n)
        for term in system._terms:
            if hasattr(term, 'tau_state'):
                term.tau_state = x.copy()
        J, _ = system._assemble(x)
        FD = np.zeros((n, n))
        for k in range(n):
            e = np.zeros(n)
            e[k] = 1e-6
            FD[:, k] = (system._assemble(x + e)[1] - system._assemble(x - e)[1]) / 2e-6
        assert np.max(np.abs(J.toarray() - FD)) / np.max(np.abs(FD)) < 1e-7, (element, order)


def test_viscoelastic_channel_with_slip():

    """Oldroyd-B: the velocity is the Newtonian one, with the slip of the wall stress, and the stress depends on the shear rate alone."""

    G, mu_s, mu_p, lam, beta = 2.0, 0.5, 0.5, 0.5, 4.0
    mesh = rect((1.0, 1.0), (4, 16))
    model = Model(name='slip', dim=2, mesh=mesh, mat=material('X', miu=mu_s), psc=[ff, ve])
    flow, polymer = model.physics.fluid, model.physics.viscoelastic
    polymer.setModes([(mu_p, lam, 0.0)])
    flow.addBC_SymmetryEdges(mesh.boundary_edges(lambda x, y: y < 1e-9))
    flow.addBC_SlipLawEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), NavierSlip(beta))
    left, right = mesh.boundary_edges(lambda x, y: x < 1e-9), mesh.boundary_edges(lambda x, y: x > 1 - 1e-9)
    flow.addBC_OutletEdges(left, G)
    flow.addBC_OutletEdges(right, 0.0)
    polymer.addBC_OutletEdges(left)
    polymer.addBC_OutletEdges(right)
    polymer.addBC_InflowEdges(left, lambda x, y: [[0.0, -G * y / (mu_s + mu_p)], [0.0, 0.0]])
    steady(model, globalization='pseudo-transient')
    y = xy_of(mesh)[:, 1]
    assert model.nonlinear_converged
    assert np.max(np.abs(model.sol['u'][:, 0] - (G * (1 - y ** 2) / (2 * (mu_s + mu_p)) + G / beta))) < 3e-3
    assert np.max(np.abs(model.sol['polymer_stress'][:, 0] - 2 * lam * mu_p * (G * y / (mu_s + mu_p)) ** 2)) < 2e-2


# ----- PTT -----

def reference(L, lam, eps, kind, T):

    """c of a homogeneous plane flow with solve_ivp, on the equation of c and without code of the project."""

    def rate(t, c):
        C = np.array([[c[0], c[1]], [c[1], c[2]]])
        E = C - np.eye(2)
        trace = E[0, 0] + E[1, 1]
        factor = 1.0 + eps * trace if kind == 'ptt-linear' else np.exp(eps * trace)
        d = L @ C + C @ L.T - factor * E / lam
        return [d[0, 0], d[0, 1], d[1, 1]]

    return solve_ivp(rate, (0, T), [1.0, 0.0, 1.0], method='LSODA', rtol=1e-11, atol=1e-13, dense_output=True)


def test_ptt_rate_of_b_is_the_rate_of_the_conformation():

    rng = np.random.default_rng(1)
    for kind in ('ptt-linear', 'ptt-exp'):
        for _ in range(8):
            A = rng.normal(size=(2, 2))
            C = A @ A.T + np.eye(2)
            w, v = np.linalg.eigh(C)
            B = v @ np.diag(np.sqrt(w)) @ v.T
            L = rng.normal(size=(2, 2))
            lam, eps = rng.uniform(0.1, 2.0), rng.uniform(0.01, 0.3)
            r = np.asarray(conformation_rate(jnp.array([B[0, 0], B[0, 1], B[1, 1]]), jnp.asarray(L), 0.0, lam, eps, kind))
            rate = np.array([[r[0], r[1]], [r[1], r[2]]])
            E = C - np.eye(2)
            factor = 1 + eps * np.trace(E) if kind == 'ptt-linear' else np.exp(eps * np.trace(E))
            assert np.allclose(rate @ B + B @ rate, L @ C + C @ L.T - factor * E / lam, atol=1e-11)

    # Hoop: tr E takes the out-of-plane component, and its equation has the same factor
    b = jnp.array([1.2, 0.1, 0.9, 1.3])
    h, lam, eps = 0.4, 0.8, 0.2
    ct = 1.3 ** 2
    trace = 1.2 ** 2 + 0.1 ** 2 + 0.1 ** 2 + 0.9 ** 2 - 2.0 + ct - 1.0
    r = float(conformation_rate(b, jnp.zeros((2, 2)), h, lam, eps, 'ptt-linear')[3])
    assert np.isclose(2 * 1.3 * r, 2 * h * ct - (1 + eps * trace) * (ct - 1) / lam)


def test_ptt_viscometric_and_extensional_steady_states():

    """Shear: Y tau_xy = mu gamma and N1 = 2 lam tau_xy^2 / mu. Planar extension is bounded for PTT, unlike Oldroyd-B."""

    mu, lam, eps = 1.0, 1.0, 0.25
    mesh = rect((1.0, 1.0), (1, 1))
    model = Model(name='ptt', dim=2, mesh=mesh, mat=material('X', miu=0.0), psc=[ff, ve])
    polymer = model.physics.viscoelastic

    for kind in ('ptt-linear', 'ptt-exp'):
        polymer.setModes([(mu, lam, eps)], kind)
        law = polymer.inelasticEquivalent(0.0)
        for rate in (0.01, 1.0, 30.0, 300.0):
            tau = polymer.stress(0, polymer.developed([[0.0, rate], [0.0, 0.0]])[None])[0]
            shear, first = tau[1], tau[0] - tau[2]
            assert np.isclose(first, 2 * lam * shear ** 2 / mu, rtol=1e-6), (kind, rate)
            assert np.isclose(float(law.viscosity(rate)), shear / rate, rtol=1e-6), (kind, rate, shear / rate)
        assert np.isclose(float(law(0.0, 0.0)), mu)
        assert np.isfinite(float(jax.grad(lambda g: law(g, 0.0))(0.0))) and np.isfinite(float(jax.grad(lambda g: law(g, 0.0))(1e-30)))

        # Extension above the coil-stretch rate 1 / (2 lam): Oldroyd-B would grow without bound
        rate = 5.0 / lam
        L = np.array([[rate, 0.0], [0.0, -rate]])
        tau = polymer.stress(0, polymer.developed(L)[None])[0]
        ref = reference(L, lam, eps, kind, 100.0 * lam).sol(100.0 * lam)
        assert np.isclose(tau[0], mu / lam * (ref[0] - 1.0), rtol=1e-5), kind


def test_ptt_homogeneous_flows():

    """The solver against solve_ivp, second order in time."""

    T = 3.0
    for kind in ('ptt-linear', 'ptt-exp'):
        for L in (np.array([[0.0, 1.5], [0.0, 0.0]]), np.array([[0.6, 0.0], [0.0, -0.6]])):
            ref = reference(L, 1.0, 0.25, kind, T)
            errors = []
            for steps in (40, 80):
                mesh = rect((1.0, 1.0), (2, 2))
                model = Model(name='h', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.0), psc=[ff, ve])
                flow, polymer = model.physics.fluid, model.physics.viscoelastic
                polymer.setModes([(1.0, 1.0, 0.25)], kind)
                xy = xy_of(mesh)
                for k, node in enumerate(mesh.NL):
                    node.BC['u'] = flow.setDirichletBC(list(L @ xy[k]))
                for k in {n for e in mesh.EL for n in e.getNodesId()[:4]}:
                    mesh.NL[k].BC['p'] = flow.setDirichletBC(0.0)
                flow.var['u'].updateField(xy @ L.T)
                model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, timeStep=T / steps, totalTime=T, timeScheme='BDF2')
                model.solve()
                b = model.sol['b0'][..., -1]
                c = np.stack([b[:, 0] ** 2 + b[:, 1] ** 2, b[:, 1] * (b[:, 0] + b[:, 2]), b[:, 1] ** 2 + b[:, 2] ** 2], axis=1)
                errors.append(np.max(np.abs(c - ref.sol(T))))
            assert np.log2(errors[0] / errors[1]) > 1.8 and errors[1] < 2e-3, (kind, errors)


def test_ptt_tangent_and_errors():

    for kind, axisymmetric in (('ptt-linear', False), ('ptt-exp', False), ('ptt-linear', True)):
        mesh = rect((1.0, 1.0), (2, 2))
        for node in mesh.NL:
            node.coor = (node.coor[0] + 0.5, node.coor[1])
        model = Model(name='tangent', dim=2, mesh=mesh, mat=material('X', miu=0.5), psc=[ff, ve], axisymmetric=axisymmetric)
        polymer = model.physics.viscoelastic
        polymer.setModes([(1.0, 0.7, 0.2)], kind)
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
        system = model._system()
        layout = system.layout
        rng = np.random.default_rng(3)
        x = rng.normal(scale=0.1, size=layout.ndof)
        d = layout.dofs('b0')
        for c in ([0, 2, 3] if axisymmetric else [0, 2]):
            x[d[:, c]] += 1.3
        for term in system._terms:
            if hasattr(term, 'tau_state'):
                term.tau_state = x.copy()
        J, _ = system.residual_and_tangent(x)
        FD = np.zeros((layout.ndof, layout.ndof))
        for k in range(layout.ndof):
            e = np.zeros(layout.ndof)
            e[k] = 1e-6
            FD[:, k] = (system.residual(x + e) - system.residual(x - e)) / 2e-6
        assert np.max(np.abs(J.toarray() - FD)) / np.max(np.abs(FD)) < 1e-7, (kind, axisymmetric)

    try:
        polymer.setModes([(1.0, 1.0, 0.1)], 'oldroyd')
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown model must fail")


if __name__ == "__main__":
    tests = [test_navier_slip_is_exact, test_no_slip_and_free_slip_limits, test_power_slip, test_slip_threshold, test_mooney_analysis,
             test_slip_tangent, test_viscoelastic_channel_with_slip, test_ptt_rate_of_b_is_the_rate_of_the_conformation,
             test_ptt_viscometric_and_extensional_steady_states, test_ptt_homogeneous_flows, test_ptt_tangent_and_errors]
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
