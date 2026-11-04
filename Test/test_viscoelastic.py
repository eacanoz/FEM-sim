"""Stage 17, priority 6: viscoelastic flow with the square root of the conformation, on Taylor-Hood elements.

Run from the project root:  python Test/test_viscoelastic.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import gmsh
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.Viscoelastic import ve, conformation_rate
from Source.enums import ElementType, StudyType, ProblemType

ELEMENTS = (ElementType.quadrilateral, ElementType.triangle)

# Drag of Oldroyd-B past a confined cylinder, beta = 0.59, channel of half height 2R (Hulsen et al. 2005, as tabulated in
# arXiv:2308.09394). The other columns of that table differ by less than 0.03 %.
HULSEN = {0.1: 130.363, 0.2: 126.626, 0.3: 123.193, 0.4: 120.596, 0.5: 118.836, 0.6: 117.775, 0.7: 117.315}


def mesh2(size, div, element=ElementType.quadrilateral):
    mesh = Mesh()
    mesh.generate2DMesh(size, div, element, order=2)
    return mesh


def xy_of(mesh):
    return np.array([n.coor[:2] for n in mesh.NL], dtype=float)


def steady(model, **options):
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, **options)
    model.solve()


def conformation(b):

    """c = b b from the nodal b, (xx, xy, yy)."""

    return np.stack([b[:, 0] ** 2 + b[:, 1] ** 2, b[:, 1] * (b[:, 0] + b[:, 2]), b[:, 1] ** 2 + b[:, 2] ** 2], axis=1)


def reference(L, lam, alpha, T):

    """The conformation of a homogeneous flow with solve_ivp, on the equation of c and without code of the project."""

    def rate(t, c):
        C = np.array([[c[0], c[1]], [c[1], c[2]]])
        E = C - np.eye(2)
        d = L @ C + C @ L.T - E / lam - alpha / lam * E @ E
        return [d[0, 0], d[0, 1], d[1, 1]]

    return solve_ivp(rate, (0, T), [1.0, 0.0, 1.0], method='LSODA', rtol=1e-11, atol=1e-13, dense_output=True)


def couette(element, lam, alpha=0.0, n=4, mu_s=0.5, mu_p=0.5):

    mesh = mesh2((1.0, 1.0), (n, n), element)
    model = Model(name='couette', dim=2, mesh=mesh, mat=material('X', miu=mu_s, rho=1.0), psc=[ff, ve])
    flow, polymer = model.physics.fluid, model.physics.viscoelastic
    polymer.setModes([(mu_p, lam, alpha)])
    flow.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9))
    flow.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), (1.0, 0.0))
    left, right = mesh.boundary_edges(lambda x, y: x < 1e-9), mesh.boundary_edges(lambda x, y: x > 1 - 1e-9)
    flow.addBC_OutletEdges(left)
    flow.addBC_OutletEdges(right)
    polymer.addBC_OutletEdges(left)
    polymer.addBC_OutletEdges(right)
    polymer.addBC_InflowEdges(left, lambda x, y: [[0.0, 1.0], [0.0, 0.0]])
    return model, mesh


def channel(element, ny, lam, mu_s=0.5, mu_p=0.5, G=2.0, devss=0.0):

    """Pressure-driven flow between plates, with the developed polymer stress at the inlet."""

    mesh = mesh2((1.0, 1.0), (max(2, ny // 2), ny), element)
    model = Model(name='channel', dim=2, mesh=mesh, mat=material('X', miu=mu_s), psc=[ff, ve])
    flow, polymer = model.physics.fluid, model.physics.viscoelastic
    polymer.setModes([(mu_p, lam, 0.0)])
    if devss:
        polymer.setDEVSS(devss)
    gamma = lambda y: G / (mu_s + mu_p) * (0.5 - y)
    flow.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or y > 1 - 1e-9))
    left, right = mesh.boundary_edges(lambda x, y: x < 1e-9), mesh.boundary_edges(lambda x, y: x > 1 - 1e-9)
    flow.addBC_OutletEdges(left, G)
    flow.addBC_OutletEdges(right, 0.0)
    polymer.addBC_OutletEdges(left)
    polymer.addBC_OutletEdges(right)
    polymer.addBC_InflowEdges(left, lambda x, y: [[0.0, gamma(y)], [0.0, 0.0]])
    return model, mesh, gamma


def homogeneous(L, lam, alpha, steps, T):

    """The velocity imposed everywhere: only b is unknown. The pressure is fixed too, or its rows would have no unknown."""

    mesh = mesh2((1.0, 1.0), (2, 2))
    model = Model(name='homogeneous', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.0), psc=[ff, ve])
    flow, polymer = model.physics.fluid, model.physics.viscoelastic
    polymer.setModes([(1.0, lam, alpha)])
    xy = xy_of(mesh)
    for k, node in enumerate(mesh.NL):
        node.BC['u'] = flow.setDirichletBC(list(L @ xy[k]))
    for k in {n for element in mesh.EL for n in element.getNodesId()[:4]}:
        mesh.NL[k].BC['p'] = flow.setDirichletBC(0.0)
    flow.var['u'].updateField(xy @ L.T)
    model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, timeStep=T / steps, totalTime=T, timeScheme='BDF2')
    model.solve()
    return model


def tangent_error(model, hoop=None):

    system = model._system()
    layout = system.layout
    rng = np.random.default_rng(3)
    x = rng.normal(scale=0.2, size=layout.ndof)
    for name in layout.names:
        if name.startswith('b'):
            d = layout.dofs(name)
            x[d[:, 0]] += 1.5
            x[d[:, 2]] += 1.5
    if hoop is not None:
        x[hoop] += 1.5
    for term in system._terms:
        if hasattr(term, 'tau_state'):
            term.tau_state = x.copy()

    J, _ = system.residual_and_tangent(x)
    FD = np.zeros((layout.ndof, layout.ndof))
    for k in range(layout.ndof):
        e = np.zeros(layout.ndof)
        e[k] = 1e-6
        FD[:, k] = (system.residual(x + e) - system.residual(x - e)) / 2e-6
    return np.max(np.abs(J.toarray() - FD)) / np.max(np.abs(FD))


# ----- the equation of b -----

def test_rate_of_b_is_the_rate_of_the_conformation():

    """b' b + b b' must equal the right-hand side of the equation of c, for any b, L and alpha."""

    rng = np.random.default_rng(1)
    for _ in range(10):
        A = rng.normal(size=(2, 2))
        C = A @ A.T + np.eye(2)
        values, vectors = np.linalg.eigh(C)
        B = vectors @ np.diag(np.sqrt(values)) @ vectors.T
        L = rng.normal(size=(2, 2))
        lam, alpha = rng.uniform(0.1, 2.0), rng.uniform(0.0, 0.5)
        r = np.asarray(conformation_rate(jnp.array([B[0, 0], B[0, 1], B[1, 1]]), jnp.asarray(L), 0.0, lam, alpha))
        rate = np.array([[r[0], r[1]], [r[1], r[2]]])
        E = C - np.eye(2)
        assert np.allclose(rate @ B + B @ rate, L @ C + C @ L.T - E / lam - alpha / lam * E @ E, atol=1e-12)

    # The hoop component, c_t' = 2 h c_t - (c_t - 1) / lam - (alpha / lam) (c_t - 1)^2
    b = jnp.array([1.3, 0.2, 0.9, 1.4])
    h, lam, alpha = 0.7, 0.8, 0.25
    rate = float(conformation_rate(b, jnp.zeros((2, 2)), h, lam, alpha)[3])
    c = 1.4 ** 2
    assert np.isclose(2 * 1.4 * rate, 2 * h * c - (c - 1) / lam - alpha / lam * (c - 1) ** 2)


def test_tangent():

    for element in ELEMENTS:
        for devss in (0.0, 0.5):
            mesh = mesh2((1.0, 1.0), (2, 2), element)
            model = Model(name='tangent', dim=2, mesh=mesh, mat=material('X', miu=0.5, rho=1.0), psc=[ff, ve])
            model.physics.fluid.setInertia(True)
            polymer = model.physics.viscoelastic
            polymer.setModes([(1.0, 0.7, 0.2)])
            if devss:
                polymer.setDEVSS(devss)
            polymer.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x > 1 - 1e-9))
            model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
            err = tangent_error(model)
            assert err < 1e-7, (element, devss, err)


def test_tangent_in_a_time_step():

    """The SUPG of db/dt depends on the state through alpha, with the mass of the solver apart."""

    mesh = mesh2((1.0, 1.0), (2, 2))
    model = Model(name='tangent', dim=2, mesh=mesh, mat=material('X', miu=0.5, rho=1.0), psc=[ff, ve])
    model.physics.viscoelastic.setModes([(1.0, 0.7, 0.2)])
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
    system = model._system()
    n = system.layout.ndof
    old = np.random.default_rng(8).normal(scale=0.1, size=n)
    d = system.layout.dofs('b0')
    old[d[:, 0]] += 1.5
    old[d[:, 2]] += 1.5
    system.time = {'alpha': 20.0, 'beta': -20.0 * old, 'theta': 1.0, 'x_old': old, 'dt': 0.05}
    try:
        err = tangent_error(model)
    finally:
        system.time = None
    assert err < 1e-7, err

    system.time = {'alpha': 20.0, 'beta': -20.0 * old, 'theta': 0.5, 'x_old': old, 'dt': 0.05}
    try:
        system.residual(old)
    except ValueError:
        pass
    else:
        raise AssertionError("Crank-Nicolson must be refused: the SUPG of b needs the old level")
    finally:
        system.time = None


# ----- shear flows -----

def test_couette_is_exact():

    """Uniform shear: b is constant, so the discrete solution is exact, with the polymer stress out of the outlet traction."""

    for element in ELEMENTS:
        for Wi in (0.5, 2.0):
            model, mesh = couette(element, Wi)
            steady(model, globalization='pseudo-transient')
            s = model.sol['polymer_stress']
            y = xy_of(mesh)[:, 1]
            assert model.nonlinear_converged
            assert np.max(np.abs(model.sol['u'][:, 0] - y)) < 1e-8 and np.max(np.abs(model.sol['p'])) < 1e-8
            assert np.max(np.abs(s[:, 0] - 2 * Wi * 0.5)) < 1e-7 and np.max(np.abs(s[:, 1] - 0.5)) < 1e-7
            assert np.max(np.abs(s[:, 2])) < 1e-8


def test_a_spurious_root_is_rejected():

    """From rest at Wi = 2, without the check of positive definiteness, the solver stopped at an indefinite b with a wrong flow.
    Now it reaches the solution or says that it does not."""

    for element in ELEMENTS:
        for Wi in (2.0, 10.0):
            model, mesh = couette(element, Wi)
            steady(model, globalization='pseudo-transient')
            b = model.sol['b0']
            assert np.all(b[:, 0] * b[:, 2] - b[:, 1] ** 2 > 0.0) and np.all(b[:, 0] + b[:, 2] > 0.0)
            if model.nonlinear_converged:
                assert np.max(np.abs(model.sol['u'][:, 0] - xy_of(mesh)[:, 1])) < 1e-7


def test_continuation_in_weissenberg():

    """Solving from the previous Wi: Couette stays exact and the channel far beyond the reach from rest. Newton is quadratic
    here, once a pressure that is zero up to round-off does not count in the norm (System.field_scales). At Wi = 64 this Couette
    reports convergence with a stress 19 % off, though the exact solution solves the discrete system too (at Wi = 123 and 2.3 % with
    a factor of 1.3): a limit of the method that the flag of convergence does not show."""

    model, mesh = couette(ElementType.quadrilateral, 0.5)
    polymer = model.physics.viscoelastic
    Wi = 0.5
    for _ in range(7):
        steady(model)
        assert model.nonlinear_converged and model.newton_iterations < 25, (Wi, model.newton_iterations)
        assert np.max(np.abs(model.sol['polymer_stress'][:, 0] / (2 * Wi * 0.5) - 1)) < 1e-7
        polymer.setWeissenberg(2.0)
        Wi *= 2.0

    model, mesh, gamma = channel(ElementType.quadrilateral, 16, 0.25)
    polymer = model.physics.viscoelastic
    y = xy_of(mesh)[:, 1]
    lam = 0.25
    for _ in range(8):
        steady(model)
        assert model.nonlinear_converged, lam
        assert np.max(np.abs(model.sol['u'][:, 0] - y * (1 - y))) < 3e-3
        polymer.setWeissenberg(2.0)
        lam *= 2.0
    print(f"   channel: Wi at the wall {lam / 2.0 * gamma(0.0):.0f} with 16 elements")


def test_channel_converges_at_second_order():

    """b = sqrt(c) is no polynomial here, so this is not exact: second order in the velocity (the stress is linear) and in tau."""

    lam, mu_p = 0.5, 0.5
    for element in ELEMENTS:
        errors = []
        for ny in (8, 16, 32):
            model, mesh, gamma = channel(element, ny, lam)
            steady(model, globalization='pseudo-transient')
            xy = xy_of(mesh)
            y = xy[:, 1]
            s = model.sol['polymer_stress']
            errors.append((np.max(np.abs(model.sol['u'][:, 0] - y * (1 - y))), np.max(np.abs(model.sol['p'] - 2.0 * (1 - xy[:, 0]))),
                           np.max(np.abs(s[:, 0] - 2 * lam * mu_p * gamma(y) ** 2)), np.max(np.abs(s[:, 1] - mu_p * gamma(y)))))
            assert model.nonlinear_converged
        e = np.array(errors)
        rates = np.log2(e[-2] / e[-1])
        print(f"   {element.name}: u {e[-1, 0]:.1e}, p {e[-1, 1]:.1e}, tau_xx {e[-1, 2]:.1e}, tau_xy {e[-1, 3]:.1e}; rates {np.round(rates, 2)}")
        assert rates[0] > 1.9 and rates[1] > 1.8 and rates[2] > 1.7 and rates[3] > 1.5


def manufactured(element, n, alpha, lam=0.5, mu_s=0.5, mu_p=0.5):

    """Smooth u (divergence free, entering only through x = 0), p and an SPD b on the unit square, with the sources of the momentum
    equation and of the equation of b from automatic derivatives. Velocity everywhere on the boundary, b at the vertices of the inflow."""

    A = 0.3

    def exact(z):
        x, y = z
        u = jnp.array([1.0 + A * jnp.sin(jnp.pi * x) * jnp.sin(2 * jnp.pi * y), -A * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y) ** 2])
        p = jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y)
        b = jnp.array([1.3 + 0.2 * jnp.sin(2 * x + y), 0.2 * jnp.cos(x - 2 * y), 1.1 + 0.15 * jnp.sin(3 * x * y)])
        return u, p, b

    def stress(z):
        u, p, b = exact(z)
        L = jax.jacfwd(lambda q: exact(q)[0])(z)
        B = jnp.array([[b[0], b[1]], [b[1], b[2]]])
        return -p * jnp.eye(2) + mu_s * (L + L.T) + mu_p / lam * (B @ B - jnp.eye(2))

    def sources(z):
        u, _, b = exact(z)
        L = jax.jacfwd(lambda q: exact(q)[0])(z)
        grad_b = jax.jacfwd(lambda q: exact(q)[2])(z)
        f_b = grad_b @ u - conformation_rate(b, L, 0.0, lam, alpha)
        f_u = -jnp.einsum('abb->a', jax.jacfwd(stress)(z))
        return f_u, f_b

    batch = jax.jit(jax.vmap(sources))

    def at(x, y, which, k):
        z = jnp.stack([jnp.ravel(jnp.asarray(x, dtype=float)), jnp.ravel(jnp.asarray(y, dtype=float))], axis=1)
        return np.asarray(batch(z)[which][:, k]).reshape(np.shape(x))

    mesh = mesh2((1.0, 1.0), (n, n), element)
    model = Model(name='manufactured', dim=2, mesh=mesh, mat=material('X', miu=mu_s), psc=[ff, ve])
    flow, polymer = model.physics.fluid, model.physics.viscoelastic
    polymer.setModes([(mu_p, lam, alpha)])
    polymer.setDEVSS(mu_p)
    flow.setBodyForce(lambda x, y: (at(x, y, 0, 0), at(x, y, 0, 1)))
    polymer.setSource(lambda x, y: (at(x, y, 1, 0), at(x, y, 1, 1), at(x, y, 1, 2)))
    flow.addBC_VelocityEdges(mesh.boundary_edges(None), lambda x, y: tuple(float(v) for v in exact(jnp.array([x, y]))[0]))
    for k in sorted({int(e) for edge in mesh.boundary_edges(lambda x, y: x < 1e-9) for e in (edge[0], edge[-1])}):
        mesh.NL[k].BC['b0'] = polymer.setDirichletBC([float(v) for v in exact(jnp.array(mesh.NL[k].coor[:2]))[2]])
    steady(model)

    xy = xy_of(mesh)
    u_ex, p_ex, b_ex = (np.array([np.asarray(exact(jnp.array(z))[i]) for z in xy]) for i in range(3))
    vertices = np.unique(np.array([e.getNodesId()[:3 if element == ElementType.triangle else 4] for e in mesh.EL]).ravel())
    dp = model.sol['p'] - p_ex
    return (model, np.max(np.abs(model.sol['u'] - u_ex)), np.max(np.abs(dp[vertices] - dp[vertices].mean())),
            np.max(np.abs(model.sol['b0'][vertices] - b_ex[vertices])))


def test_manufactured_solution():

    """Order of the steady equations against a manufactured solution, Oldroyd-B and Giesekus, Wi of order 1. b is linear in each
    element, so second order is what can be asked of b and, through the stress, of u and p."""

    for element, alpha in ((ElementType.quadrilateral, 0.0), (ElementType.triangle, 0.0), (ElementType.quadrilateral, 0.3)):
        errors = []
        for n in (8, 16, 32):
            model, eu, ep, eb = manufactured(element, n, alpha)
            assert model.nonlinear_converged
            errors.append((eu, ep, eb))
        e = np.array(errors)
        rates = np.log2(e[:-1] / e[1:])
        print(f"   {element.name}, alpha {alpha}: u {e[:, 0]}, p {e[:, 1]}, b {e[:, 2]}; rates {np.round(rates[-1], 2)} "
              f"(before {np.round(rates[0], 2)})")
        assert np.all(rates[-1] > 1.8)


def test_giesekus_viscometric_functions():

    """Steady shear of Giesekus against its closed form (Bird, Armstrong & Hassager): viscosity and first normal stress."""

    mu, lam, alpha = 1.0, 1.0, 0.3
    mesh = mesh2((1.0, 1.0), (2, 2))
    model = Model(name='g', dim=2, mesh=mesh, mat=material('X', miu=0.0), psc=[ff, ve])
    polymer = model.physics.viscoelastic
    polymer.setModes([(mu, lam, alpha)])

    for rate in (0.1, 1.0, 10.0, 100.0):
        Wi = lam * rate
        chi = np.sqrt((np.sqrt(1 + 16 * alpha * (1 - alpha) * Wi ** 2) - 1) / (8 * alpha * (1 - alpha) * Wi ** 2))
        f = (1 - chi) / (1 + (1 - 2 * alpha) * chi)
        eta = mu * (1 - f) ** 2 / (1 + (1 - 2 * alpha) * f)
        psi1 = 2 * mu * lam * f * (1 - alpha * f) / (Wi ** 2 * alpha * (1 - f))

        tau = polymer.stress(0, polymer.developed([[0.0, rate], [0.0, 0.0]])[None])[0]
        assert np.isclose(tau[1] / rate, eta, rtol=1e-6), (rate, tau[1] / rate, eta)
        assert np.isclose((tau[0] - tau[2]) / rate ** 2, psi1, rtol=1e-6), (rate, (tau[0] - tau[2]) / rate ** 2, psi1)


# ----- transient -----

def test_homogeneous_flows_against_an_independent_integration():

    """Shear and planar extension, Oldroyd-B and Giesekus: second order in time, against solve_ivp on c."""

    T = 3.0
    for L in (np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.4, 0.0], [0.0, -0.4]])):
        for alpha in (0.0, 0.3):
            ref = reference(L, 1.0, alpha, T)
            errors = []
            for steps in (40, 80, 160):
                model = homogeneous(L, 1.0, alpha, steps, T)
                assert len(model.timeVector) == steps + 1
                c = conformation(model.sol['b0'][..., -1])
                errors.append(np.max(np.abs(c - ref.sol(T))))
            rate = np.log2(errors[-2] / errors[-1])
            assert rate > 1.9 and errors[-1] < 1e-3, (L, alpha, errors)


def test_start_up_of_couette():

    """Wall set in motion in a fluid at rest, with inertia: against a 1D method of lines on c, without code of the project."""

    mu_s, mu_p, lam, n = 0.5, 0.5, 1.0, 400
    y = np.linspace(0, 1, n + 1)
    dy = y[1] - y[0]

    def rate(t, z):
        u, cxy, cxx = z[:n + 1].copy(), z[n + 1:2 * n + 2], z[2 * n + 2:]
        u[0], u[-1] = 0.0, 1.0
        g = np.gradient(u, dy, edge_order=2)
        du = np.gradient(mu_s * g + mu_p / lam * cxy, dy, edge_order=2)
        du[0] = du[-1] = 0.0
        return np.concatenate([du, g - cxy / lam, 2 * g * cxy - (cxx - 1) / lam])

    z0 = np.concatenate([0 * y, 0 * y, 1 + 0 * y])
    z0[n] = 1.0
    ref = solve_ivp(rate, (0, 1.0), z0, method='BDF', rtol=1e-7, atol=1e-9, max_step=0.01).y[:n + 1, -1]
    ref[-1] = 1.0

    mesh = mesh2((0.2, 1.0), (1, 32))
    model = Model(name='start', dim=2, mesh=mesh, mat=material('X', miu=mu_s, rho=1.0), psc=[ff, ve])
    flow, polymer = model.physics.fluid, model.physics.viscoelastic
    polymer.setModes([(mu_p, lam, 0.0)])
    flow.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9))
    flow.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), (1.0, 0.0))
    left, right = mesh.boundary_edges(lambda x, y: x < 1e-9), mesh.boundary_edges(lambda x, y: x > 0.2 - 1e-9)
    flow.addBC_OutletEdges(left)
    flow.addBC_OutletEdges(right)
    polymer.addBC_OutletEdges(left)
    polymer.addBC_OutletEdges(right)
    model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, timeStep=0.01, totalTime=1.0, timeScheme='BDF2')
    model.solve()
    err = np.max(np.abs(model.sol['u'][:, 0, -1] - np.interp(xy_of(mesh)[:, 1], y, ref)))
    print(f"   velocity at t = 1 against the reference: {err:.1e}")
    assert len(model.timeVector) == 101 and err < 2e-4


# ----- several modes, axisymmetric -----

def pipe(element, modes, n=12, devss=0.0):

    """Pipe of radius 1 driven by a pressure difference 4 over a length 1: gamma_dot = 2 r, tau_w = 2."""

    mesh = mesh2((1.0, 1.0), (n, n), element)
    model = Model(name='pipe', dim=2, mesh=mesh, mat=material('X', miu=0.5), psc=[ff, ve], axisymmetric=True)
    flow, polymer = model.physics.fluid, model.physics.viscoelastic
    polymer.setModes(modes)
    if devss:
        polymer.setDEVSS(devss)
    mu = 0.5 + sum(m[0] for m in modes)
    inlet, outlet = mesh.boundary_edges(lambda x, y: y < 1e-9), mesh.boundary_edges(lambda x, y: y > 1 - 1e-9)
    flow.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: x > 1 - 1e-9))
    flow.addBC_SymmetryEdges(mesh.boundary_edges(lambda x, y: x < 1e-9))
    flow.addBC_OutletEdges(inlet, 4.0)
    flow.addBC_OutletEdges(outlet, 0.0)
    polymer.addBC_OutletEdges(inlet)
    polymer.addBC_OutletEdges(outlet)
    polymer.addBC_InflowEdges(inlet, lambda r, z: [[0.0, 0.0], [-2.0 * r / mu, 0.0]])
    return model, mesh, mu


def test_axisymmetric_tangent():

    mesh = mesh2((1.0, 1.0), (2, 2))
    for node in mesh.NL:
        node.coor = (node.coor[0] + 0.5, node.coor[1])
    model = Model(name='tangent', dim=2, mesh=mesh, mat=material('X', miu=0.5, rho=1.0), psc=[ff, ve], axisymmetric=True)
    polymer = model.physics.viscoelastic
    polymer.setModes([(1.0, 0.7, 0.2)])
    polymer.setDEVSS(0.5)
    polymer.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9))
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
    system = model._system()
    d = system.layout.dofs('b0')
    assert d.shape[1] == 4
    # tangent_error shifts b_rr and b_zz; the hoop component must be positive too
    err = tangent_error(model, hoop=d[:, 3])
    assert err < 1e-7, err


def test_pipe():

    """Oldroyd-B in a pipe: a parabolic velocity for any Wi, tau_zz = 2 lam mu_p gamma^2, tau_rz = mu_p gamma, no hoop stress."""

    lam, mu_p = 0.5, 0.5
    for element in ELEMENTS:
        errors = []
        for n in (8, 16):
            model, mesh, mu = pipe(element, [(mu_p, lam, 0.0)], n)
            steady(model, globalization='pseudo-transient')
            xy = xy_of(mesh)
            r = xy[:, 0]
            gamma = -2.0 * r / mu
            s = model.sol['polymer_stress']
            assert model.nonlinear_converged
            errors.append((np.max(np.abs(model.sol['u'][:, 1] - (1 - r ** 2) / mu)), np.max(np.abs(model.sol['p'] - 4.0 * (1 - xy[:, 1]))),
                           np.max(np.abs(s[:, 2] - 2 * lam * mu_p * gamma ** 2)), np.max(np.abs(s[:, 1] - mu_p * gamma)),
                           np.max(np.abs(s[:, 3]))))
        e = np.array(errors)
        rates = np.log2(e[0] / e[-1])
        print(f"   {element.name}: u {e[-1, 0]:.1e}, p {e[-1, 1]:.1e}, tau_zz {e[-1, 2]:.1e}, tau_rz {e[-1, 3]:.1e}, tau_tt {e[-1, 4]:.1e}; rates {np.round(rates, 2)}")
        assert e[-1, 0] < 1e-3 and e[-1, 2] < 2e-2 and rates[0] > 1.8 and e[-1, 4] < 1e-4 and rates[4] > 1.8


def test_two_identical_modes_are_one():

    """A mode split into two with the same relaxation time gives the same flow, and each carries its share of the stress."""

    one, mesh, _ = pipe(ElementType.quadrilateral, [(0.5, 0.5, 0.2)])
    two, _, _ = pipe(ElementType.quadrilateral, [(0.2, 0.5, 0.2), (0.3, 0.5, 0.2)])
    steady(one, globalization='pseudo-transient')
    steady(two, globalization='pseudo-transient')
    assert one.nonlinear_converged and two.nonlinear_converged
    assert np.max(np.abs(one.sol['u'] - two.sol['u'])) < 1e-8 and np.max(np.abs(one.sol['p'] - two.sol['p'])) < 1e-8
    assert np.max(np.abs(one.sol['polymer_stress'] - two.sol['polymer_stress'])) < 1e-7
    assert np.max(np.abs(two.sol['b0'] - two.sol['b1'])) < 1e-8


def test_spectrum_in_oscillatory_shear():

    """Small oscillatory shear: the stress of a spectrum against G' and G'' of the Maxwell modes."""

    modes = [(1.0, 0.1, 0.0), (0.5, 1.0, 0.3), (0.2, 5.0, 0.0)]
    w, amplitude, cycles, steps_per_cycle = 2.0, 1e-3, 8, 80
    period = 2 * np.pi / w
    T = cycles * period

    mesh = mesh2((1.0, 1.0), (1, 1))
    model = Model(name='saos', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.0), psc=[ff, ve])
    flow, polymer = model.physics.fluid, model.physics.viscoelastic
    polymer.setModes(modes)
    xy = xy_of(mesh)
    edges = mesh.boundary_edges(None)
    for k, node in enumerate(mesh.NL):
        node.BC['u'] = flow.setDirichletBC([xy[k, 1] * amplitude * w, 0.0])
    for k in {n for element in mesh.EL for n in element.getNodesId()[:4]}:
        mesh.NL[k].BC['p'] = flow.setDirichletBC(0.0)
    for node in mesh.NL:
        node.BC['u']['time'] = lambda t: np.cos(w * t)
    model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.linear, timeStep=period / steps_per_cycle, totalTime=T, timeScheme='BDF2')
    model.solve()

    t = np.asarray(model.timeVector)
    stress = sum(polymer.stress(m, model.sol[f'b{m}'][0, :, :].T)[:, 1] for m in range(len(modes)))
    last = t > T - period
    fit = np.linalg.lstsq(np.c_[np.sin(w * t[last]), np.cos(w * t[last])], stress[last], rcond=None)[0] / amplitude
    storage = sum(mu / lam * (w * lam) ** 2 / (1 + (w * lam) ** 2) for mu, lam, _ in modes)
    loss = sum(mu / lam * w * lam / (1 + (w * lam) ** 2) for mu, lam, _ in modes)
    print(f"   G' {fit[0]:.4f} (exact {storage:.4f}), G'' {fit[1]:.4f} (exact {loss:.4f})")
    # The longest mode (5 s) has not forgotten the start after 8 cycles: a transient of its own in tau, so 2 % at most
    assert abs(fit[0] / storage - 1) < 0.02 and abs(fit[1] / loss - 1) < 0.02


# ----- without solvent, and the benchmark -----

def test_channel_without_solvent():

    """DEVSS-G: no viscosity in the flow physics at all."""

    for element in ELEMENTS:
        model, mesh, gamma = channel(element, 16, 0.25, mu_s=0.0, mu_p=1.0, devss=1.0)
        polymer = model.physics.viscoelastic
        y = xy_of(mesh)[:, 1]
        lam = 0.25
        for _ in range(3):
            steady(model)
            s = model.sol['polymer_stress']
            assert model.nonlinear_converged
            assert np.max(np.abs(model.sol['u'][:, 0] - y * (1 - y))) < 3e-3
            assert np.max(np.abs(s[:, 1] - gamma(y))) < 0.05
            polymer.setWeissenberg(2.0)
            lam *= 2.0


def test_cylinder_in_a_channel():

    """Oldroyd-B, beta = 0.59: half channel 30R x 2R, Stokes, with continuation in Wi. On a coarse mesh."""

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        occ = gmsh.model.occ
        fluid, _ = occ.cut([(2, occ.addRectangle(0, 0, 0, 30, 2))], [(2, occ.addDisk(15, 0, 0, 1, 1))])
        occ.synchronize()
        groups = {'inlet': [], 'outlet': [], 'wall': [], 'axis': [], 'cylinder': []}
        for _, tag in gmsh.model.getBoundary(fluid, oriented=False):
            x, y, _ = occ.getCenterOfMass(1, tag)
            key = 'inlet' if abs(x) < 1e-9 else 'outlet' if abs(x - 30) < 1e-9 else 'wall' if abs(y - 2) < 1e-9 else 'axis' if abs(y) < 1e-9 else 'cylinder'
            groups[key].append(tag)
        for name, tags in groups.items():
            gmsh.model.addPhysicalGroup(1, tags, name=name)
        field = gmsh.model.mesh.field
        dist = field.add('Distance')
        field.setNumbers(dist, 'CurvesList', groups['cylinder'])
        field.setNumber(dist, 'Sampling', 200)
        threshold = field.add('Threshold')
        field.setNumber(threshold, 'InField', dist)
        field.setNumber(threshold, 'SizeMin', 0.1)
        field.setNumber(threshold, 'SizeMax', 0.5)
        field.setNumber(threshold, 'DistMin', 0.1)
        field.setNumber(threshold, 'DistMax', 2.5)
        field.setAsBackgroundMesh(threshold)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
        mesh = Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()

    mu_s, mu_p = 0.59, 0.41
    model = Model(name='cylinder', dim=2, mesh=mesh, mat=material('X', miu=mu_s), psc=[ff, ve])
    flow, polymer = model.physics.fluid, model.physics.viscoelastic
    polymer.setModes([(mu_p, 0.1, 0.0)])
    polymer.setDEVSS(mu_p)
    inlet, outlet, cylinder = mesh.boundary_edges('inlet'), mesh.boundary_edges('outlet'), mesh.boundary_edges('cylinder')
    profile = lambda x, y: (1.5 * (1 - (y / 2) ** 2), 0.0)
    flow.addBC_VelocityEdges(inlet, profile)
    flow.addBC_NoSlipEdges(mesh.boundary_edges('wall'))
    flow.addBC_NoSlipEdges(cylinder)
    flow.addBC_SymmetryEdges(mesh.boundary_edges('axis'))
    flow.addBC_OutletEdges(outlet, 0.0)
    polymer.addBC_OutletEdges(outlet)
    polymer.addBC_InflowEdges(inlet, lambda x, y: [[0.0, -0.75 * y], [0.0, 0.0]])
    flow.var['u'].updateField(np.array([profile(*n.coor[:2]) for n in mesh.NL]))

    for Wi in (0.1, 0.2, 0.3, 0.4):
        steady(model)
        drag = 2.0 * flow.boundaryForce(cylinder)[0] / (mu_s + mu_p)
        print(f"   Wi = {Wi}: {model.newton_iterations} iterations, drag {drag:.3f}, Hulsen {HULSEN[Wi]:.3f} ({drag / HULSEN[Wi] - 1:+.3%})")
        assert model.nonlinear_converged and abs(drag / HULSEN[Wi] - 1) < 0.002
        polymer.setWeissenberg((Wi + 0.1) / Wi)


def test_plane_contraction():

    """Oldroyd-B, beta = 1/9, 4:1 on a coarse mesh: at low De the entrance correction is near the Newtonian 0.374, and it falls
    with De (the published trend for this fluid). The numbers at high De are not converged in the mesh."""

    H, h, L = 4.0, 1.0, 20.0
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        geo = gmsh.model.geo
        points = [(-L, 0, 1.0), (L, 0, 1.0), (L, h, 1.0), (0, h, 0.0375), (0, H, 0.25), (-L, H, 1.0)]
        tags = [geo.addPoint(x, y, 0, size) for x, y, size in points]
        lines = [geo.addLine(tags[i], tags[(i + 1) % 6]) for i in range(6)]
        geo.addPlaneSurface([geo.addCurveLoop(lines)])
        geo.synchronize()
        for name, group in (('axis', lines[:1]), ('outlet', lines[1:2]), ('walls', lines[2:5]), ('inlet', lines[5:])):
            gmsh.model.addPhysicalGroup(1, group, name=name)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
        mesh = Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()

    mu_s, mu_p = 1.0 / 9.0, 8.0 / 9.0
    model = Model(name='contraction', dim=2, mesh=mesh, mat=material('X', miu=mu_s), psc=[ff, ve])
    flow, polymer = model.physics.fluid, model.physics.viscoelastic
    polymer.setModes([(mu_p, 0.1, 0.0)])
    polymer.setDEVSS(mu_p)
    inlet, outlet = mesh.boundary_edges('inlet'), mesh.boundary_edges('outlet')
    top = 1.5 * h / H
    flow.addBC_VelocityEdges(inlet, lambda x, y: (top * (1 - (y / H) ** 2), 0.0))
    flow.addBC_NoSlipEdges(mesh.boundary_edges('walls'))
    flow.addBC_SymmetryEdges(mesh.boundary_edges('axis'))
    flow.addBC_OutletEdges(outlet, 0.0)
    polymer.addBC_OutletEdges(outlet)
    polymer.addBC_InflowEdges(inlet, lambda x, y: [[0.0, -2.0 * top * y / H ** 2], [0.0, 0.0]])

    corrections = []
    de = 0.1
    for target in (0.1, 0.5, 1.0):
        polymer.setWeissenberg(target / de)
        de = target
        steady(model)
        assert model.nonlinear_converged, target
        q = flow.flowRate(outlet)
        dp = model.sol['p'][np.unique(np.ravel(inlet))].mean()
        corrections.append((dp - 3 * q / H ** 3 * L - 3 * q / h ** 3 * L) / (6 * q / h ** 2))
        assert abs(q - 1.0) < 1e-10
    print(f"   C at De = 0.1, 0.5, 1: {np.round(corrections, 3)}")
    assert 0.25 < corrections[0] < 0.37 and corrections[0] > corrections[1] > corrections[2]


def test_inelastic_equivalent_has_the_same_flow_curve():

    """The law against the steady shear stress of each mode, with Oldroyd-B and Giesekus together."""

    modes = [(1.0, 0.1, 0.3), (0.5, 1.0, 0.0), (0.2, 10.0, 0.15)]
    mesh = mesh2((1.0, 1.0), (1, 1))
    model = Model(name='curve', dim=2, mesh=mesh, mat=material('X', miu=0.3), psc=[ff, ve])
    polymer = model.physics.viscoelastic
    polymer.setModes(modes)
    law = polymer.inelasticEquivalent(0.3)
    for rate in (0.01, 1.0, 30.0, 1000.0):
        stress = sum(polymer.stress(m, polymer.developed([[0.0, rate], [0.0, 0.0]], mode=m)[None])[0, 1] for m in range(3))
        assert np.isclose(float(law.viscosity(rate)), 0.3 + stress / rate, rtol=1e-6), rate

    # At rest and at rates so low that 1 + s rounds to 1: the plateau, with a finite derivative for Newton
    for gamma2 in (0.0, 1e-30, 1e-12):
        assert np.isclose(float(law(gamma2, 0.0)), 0.3 + 1.7, rtol=1e-9), gamma2
        assert np.isfinite(float(jax.grad(lambda g: law(g, 0.0))(gamma2))), gamma2


def test_tangent_with_artificial_diffusion():

    """kappa is frozen in each iteration, like tau_s; it depends on the velocity through tau_state in this test."""

    mesh = mesh2((1.0, 1.0), (2, 2))
    model = Model(name='tangent', dim=2, mesh=mesh, mat=material('X', miu=0.5, rho=1.0), psc=[ff, ve])
    polymer = model.physics.viscoelastic
    polymer.setModes([(1.0, 0.7, 0.2)])
    polymer.setDiffusion(1.0, lambda x, y: np.exp(-x ** 2 - y ** 2))
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
    err = tangent_error(model)
    assert err < 1e-7, err


def test_diffusion_extends_the_reach_at_a_reentrant_corner():

    """Capillary with a long mode: without help it fails with b nearly singular next to the corner. With a small diffusion around the
    corner it goes further, and where both work it leaves the shear stress and the total pressure drop in place."""

    RB, LB, RC, LC = 4.75e-3, 8.0e-3, 0.5e-3, 16.0e-3

    def capillary(c, rates):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            geo = gmsh.model.geo
            points = [(0, 0, 1.4e-3), (RB, 0, 1.4e-3), (RB, LB, 1.4e-3), (RC, LB, 0.06e-3), (RC, LB + LC, 0.24e-3), (0, LB + LC, 0.24e-3), (0, LB, 0.24e-3)]
            tags = [geo.addPoint(x, y, 0, s) for x, y, s in points]
            lines = [geo.addLine(tags[i], tags[(i + 1) % 7]) for i in range(7)]
            geo.addPlaneSurface([geo.addCurveLoop(lines)])
            geo.synchronize()
            for name, group in (('inlet', lines[:1]), ('wall', lines[1:4]), ('outlet', lines[4:5]), ('axis', lines[5:])):
                gmsh.model.addPhysicalGroup(1, group, name=name)
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.setOrder(2)
            mesh = Mesh.from_gmsh_model()
        finally:
            gmsh.finalize()

        model = Model(name='capillary', dim=2, mesh=mesh, mat=material('X', miu=47.0), psc=[ff, ve], axisymmetric=True)
        flow, polymer = model.physics.fluid, model.physics.viscoelastic
        polymer.setModes([(945.0, 1.0, 0.3)])
        polymer.setDEVSS(945.0)
        if c:
            polymer.setDiffusion(c, lambda x, y: np.exp(-((x - RC) ** 2 + (y - LB) ** 2) / (0.3 * RC) ** 2))
        flow.addBC_SymmetryEdges(mesh.boundary_edges('axis'))
        flow.addBC_NoSlipEdges(mesh.boundary_edges('wall'))
        inlet, outlet = mesh.boundary_edges('inlet'), mesh.boundary_edges('outlet')
        flow.addBC_OutletEdges(outlet, 0.0)
        polymer.addBC_OutletEdges(outlet)
        state = {}

        def set_rate(rate):
            state['u'] = rate * RC ** 3 / (4 * RB ** 2)
            flow.addBC_VelocityEdges(inlet, lambda r, z: (0.0, 2 * state['u'] * (1 - (r / RB) ** 2)))

        set_rate(rates[0])
        polymer.addBC_InflowEdges(inlet, lambda r, z: [[0.0, 0.0], [-4 * state['u'] * r / RB ** 2, 0.0]])
        drops = {}
        nodes = np.unique(np.ravel(inlet))
        for rate in rates:
            set_rate(rate)
            polymer.setWeissenberg(1.0)
            steady(model)
            assert model.nonlinear_converged, (c, rate)
            drops[rate] = model.sol['p'][nodes].mean()
        return drops

    plain = capillary(0.0, (1.0, 9.0, 27.0))
    helped = capillary(1.0, (1.0, 9.0, 27.0, 60.0, 90.0, 135.0, 200.0))
    for rate in plain:
        assert abs(helped[rate] / plain[rate] - 1) < 0.01, rate


def test_energy_balance_falls_with_the_mesh_and_marks_a_wrong_solution():

    """An indicator of error, not a guarantee: see ve.energy_balance."""

    values = []
    for ny in (8, 16):
        model, _, _ = channel(ElementType.quadrilateral, ny, 0.5)
        steady(model)
        values.append(model.physics.viscoelastic.energy_balance())
    assert values[1] < 0.2 * values[0], values

    model, mesh = couette(ElementType.quadrilateral, 0.5)
    polymer = model.physics.viscoelastic
    steady(model)
    assert polymer.energy_balance() < 1e-8
    Wi = 0.5
    for _ in range(7):
        polymer.setWeissenberg(2.0)
        Wi *= 2.0
        steady(model)
    assert Wi == 64.0
    assert np.max(np.abs(model.sol['polymer_stress'][:, 0] / (2 * Wi * 0.5) - 1)) > 0.01     # the known wrong solution
    assert polymer.energy_balance() > 1e-6


def test_output_of_the_modes():

    """More than three components go to VTK one by one."""

    from Source.Post_processing.VTK import write_vtu
    import tempfile

    mesh = mesh2((1.0, 1.0), (1, 1))
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, 'modes.vtu')
        write_vtu(path, mesh, {'b0': np.ones((mesh.getNoN(), 4))})
        text = open(path, encoding='utf-8').read()
    assert all(f'Name="b0_{c}"' in text for c in range(4))


if __name__ == "__main__":
    tests = [test_rate_of_b_is_the_rate_of_the_conformation, test_tangent, test_tangent_in_a_time_step, test_couette_is_exact,
             test_a_spurious_root_is_rejected, test_continuation_in_weissenberg, test_channel_converges_at_second_order,
             test_manufactured_solution,
             test_giesekus_viscometric_functions, test_homogeneous_flows_against_an_independent_integration,
             test_start_up_of_couette, test_axisymmetric_tangent, test_pipe, test_two_identical_modes_are_one,
             test_spectrum_in_oscillatory_shear, test_channel_without_solvent, test_cylinder_in_a_channel,
             test_plane_contraction, test_inelastic_equivalent_has_the_same_flow_curve, test_tangent_with_artificial_diffusion,
             test_diffusion_extends_the_reach_at_a_reentrant_corner, test_energy_balance_falls_with_the_mesh_and_marks_a_wrong_solution,
             test_output_of_the_modes]
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
