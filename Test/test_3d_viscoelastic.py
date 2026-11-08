"""Checks for the viscoelastic flow in 3D (phase 4 of docs/PLAN_3D.md): the square root of the conformation with 6 components, DEVSS-G
with 9, on Taylor-Hood hex27 and tet10.

Run from the project root:  python Test/test_3d_viscoelastic.py
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

import numpy as np
import jax.numpy as jnp
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.HeatTransfer import ht
from Source.Physics.Viscoelastic import ve, conformation_rate, nodal_matrices
from Source.Physics.Rheology import wlf_shift
from Source.enums import ElementType, StudyType, ProblemType

KIND = {'hex27': ElementType.hexahedron, 'tet10': ElementType.tetrahedron}
SYM = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def box(name, n, L=(1.0, 1.0, 1.0)):
    mesh = Mesh()
    mesh.generate3DMesh(L, n if np.ndim(n) else (n, n, n), KIND[name], order=2)
    return mesh


def xyz_of(mesh):
    return np.array([node.coor for node in mesh.NL], dtype=float)


def vertices(mesh):
    n = 8 if mesh.element_type.startswith('hex') else 4
    return sorted({k for element in mesh.EL for k in element.getNodesId()[:n]})


def c_of(b):
    B = nodal_matrices(b)
    C = B @ B
    return np.stack([C[..., i, j] for i, j in SYM], axis=-1)


# ----- the equation of b -----

def test_rate_of_b_is_the_rate_of_the_conformation_in_3d():
    """For random positive definite b and velocity gradients L: b' b + b b' = L c + c L^T + f(c) for Giesekus, Oldroyd-B and both PTT
    (relative to the size of the right-hand side), and a plane flow embedded in 3D gives the rate of the 2D equation."""

    rng = np.random.default_rng(0)
    for kind, alpha in (('giesekus', 0.0), ('giesekus', 0.3), ('ptt-linear', 0.2), ('ptt-exp', 0.1)):
        worst, plane = 0.0, 0.0
        for _ in range(20):
            M = rng.normal(size=(3, 3))
            B = M @ M.T + 0.5 * np.eye(3)
            L = rng.normal(size=(3, 3))
            lam = 0.7
            b = np.array([B[i, j] for i, j in SYM])
            Rb = nodal_matrices(np.asarray(conformation_rate(jnp.asarray(b), jnp.asarray(L), 0.0, lam, alpha, kind)))
            C = B @ B
            E = C - np.eye(3)
            if kind == 'giesekus':
                f = -(E + alpha * E @ E) / lam
            else:
                f = -((1 + alpha * np.trace(E)) if kind == 'ptt-linear' else np.exp(alpha * np.trace(E))) * E / lam
            rhs = L @ C + C @ L.T + f
            worst = max(worst, np.max(np.abs(Rb @ B + B @ Rb - rhs)) / np.max(np.abs(rhs)))

            P = rng.normal(size=(2, 2))
            B2 = P @ P.T + 0.5 * np.eye(2)
            L2 = rng.normal(size=(2, 2))
            L3 = np.zeros((3, 3))
            L3[:2, :2] = L2
            r3 = np.asarray(conformation_rate(jnp.asarray([B2[0, 0], B2[0, 1], 0.0, B2[1, 1], 0.0, 1.0]), jnp.asarray(L3), 0.0, lam,
                                              alpha, kind))
            r2 = np.asarray(conformation_rate(jnp.asarray([B2[0, 0], B2[0, 1], B2[1, 1]]), jnp.asarray(L2), 0.0, lam, alpha, kind))
            plane = max(plane, np.max(np.abs(r3[[0, 1, 3]] - r2)) / max(np.max(np.abs(r2)), 1.0), np.max(np.abs(r3[[2, 4, 5]])))
        print(f"   {kind} alpha = {alpha}: conformation equation {worst:.1e}, plane embedding {plane:.1e}")
        assert worst < 1e-12 and plane < 1e-12, (kind, worst, plane)


def reference_3d(L, lam, alpha, T):

    """The conformation of a homogeneous 3D flow with solve_ivp on the equation of c, without code of the project."""

    def rate(t, c):
        C = np.array([[c[0], c[1], c[2]], [c[1], c[3], c[4]], [c[2], c[4], c[5]]])
        E = C - np.eye(3)
        d = L @ C + C @ L.T - E / lam - alpha / lam * E @ E
        return [d[i, j] for i, j in SYM]

    return solve_ivp(rate, (0, T), [1.0, 0.0, 0.0, 1.0, 0.0, 1.0], method='LSODA', rtol=1e-11, atol=1e-13, dense_output=True)


def homogeneous(name, L, lam, alpha, steps, T):

    """The velocity L x imposed at every node (and the pressure at the vertices): only b is unknown, one cell."""

    mesh = box(name, 1)
    model = Model(name='homogeneous', dim=3, mesh=mesh, mat=material('X', miu=1.0, rho=1.0), psc=[ff, ve])
    flow, polymer = model.physics.fluid, model.physics.viscoelastic
    polymer.setModes([(1.0, lam, alpha)])
    xyz = xyz_of(mesh)
    for k, node in enumerate(mesh.NL):
        node.BC['u'] = flow.setDirichletBC(list(L @ xyz[k]))
    for k in vertices(mesh):
        mesh.NL[k].BC['p'] = flow.setDirichletBC(0.0)
    flow.var['u'].updateField(xyz @ L.T)
    model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, timeStep=T / steps, totalTime=T, timeScheme='BDF2')
    model.solve()
    return model


def test_homogeneous_flows_against_an_independent_integration():
    """Shear, uniaxial extension and a general 3D gradient (shear in two planes plus extension), Oldroyd-B and Giesekus: the 6
    components of c from BDF2 against solve_ivp on the equation of c, second order in time; on tet10 too."""

    T = 2.0
    gradients = {'shear': np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                 'uniaxial': np.diag([0.4, -0.2, -0.2]),
                 'general': np.array([[0.2, 0.8, -0.3], [0.1, -0.3, 0.5], [0.0, 0.2, 0.1]])}
    for label, L in gradients.items():
        for alpha in (0.0, 0.3):
            ref = reference_3d(L, 1.0, alpha, T)
            errors = []
            for steps in (20, 40, 80):
                model = homogeneous('hex27', L, 1.0, alpha, steps, T)
                c = c_of(np.asarray(model.sol['b0'])[..., -1])
                errors.append(np.max(np.abs(c - ref.sol(T))))
            rate = np.log2(errors[-2] / errors[-1])
            print(f"   {label} alpha = {alpha}: errors {['%.1e' % e for e in errors]}, order {rate:.2f}")
            assert rate > 1.8 and errors[-1] < 2e-3, (label, alpha, errors)

    L = gradients['general']
    c = c_of(np.asarray(homogeneous('tet10', L, 1.0, 0.3, 80, T).sol['b0'])[..., -1])
    error = np.max(np.abs(c - reference_3d(L, 1.0, 0.3, T).sol(T)))
    print(f"   tet10 general alpha = 0.3: error {error:.1e}")
    assert error < 2e-3


def test_tangent_in_3d():
    """The Newton tangent of a 3D Giesekus flow with DEVSS-G, the WLF shift of the relaxation time, polymer heating and the outlet
    traction, in a BDF2 step, against central finite differences (tau of SUPG frozen at the state), on hex27 and tet10."""

    for name in ('hex27', 'tet10'):
        mesh = box(name, 1)
        model = Model(name='tangent', dim=3, mesh=mesh, mat=material('X', miu=0.3, rho=1.0, k=0.5, Cp=1.0), psc=[ff, ht, ve])
        flow, polymer = model.physics.fluid, model.physics.viscoelastic
        polymer.setModes([(0.7, 1.2, 0.3)])
        polymer.setDEVSS(0.7)
        polymer.setTemperatureShift(wlf_shift(8.86, 101.6, 200.0))
        polymer.setViscousHeating(True)
        flow.setInertia(True)
        model.physics.setVelocity('u')
        outlet = mesh.boundary_faces(lambda x, y, z: x > 1 - 1e-9)
        flow.addBC_OutletEdges(outlet, 0.0)
        polymer.addBC_OutletEdges(outlet)
        model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, timeStep=0.1, totalTime=0.1,
                                  timeScheme='BDF2')
        system = model._system()
        layout = system.layout
        rng = np.random.default_rng(5)
        x = rng.normal(scale=0.2, size=layout.ndof)
        x[layout.dofs('T')[:, 0]] = 210.0 + rng.normal(size=layout.n_nodes)
        b = layout.dofs('b0')
        x[b] = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0]) + 0.1 * rng.normal(size=b.shape)
        system.time = {'alpha': 15.0, 'beta': -15.0 * (x + 0.01 * rng.normal(size=layout.ndof)), 'theta': 1.0}
        for term in system._terms:
            if hasattr(term, 'tau_state'):
                term.tau_state = x.copy()
        J = system._assemble(x)[0].toarray()
        FD = np.zeros_like(J)
        for k in range(layout.ndof):
            e = np.zeros(layout.ndof)
            e[k] = 1e-6 * max(1.0, abs(x[k]))
            FD[:, k] = (system._assemble(x + e)[1] - system._assemble(x - e)[1]) / (2 * e[k])
        err = np.max(np.abs(J - FD) / (np.max(np.abs(FD), axis=0) + 1e-300))
        print(f"   {name}: {layout.ndof} unknowns, largest tangent error {err:.1e} (relative to each column)")
        assert err < 1e-5, err


# ----- 3D slab against the 2D channel -----

def slab_and_plane(build, nx, ny, L=1.0, H=1.0, W=None):

    """build(dim, mesh) solved on the 2D mesh and on the 3D slab (one cell in z, symmetric z faces): their largest differences.
    The slab is as thick as the square root of the area of a cell, so that the element size of the SUPG of b (the side of the
    square of the same area in 2D, of the cube of the same volume in 3D) is the same: otherwise both are right but differ by
    the O(h) of the stabilization (3e-4 in u with a slab of 0.5)."""

    if W is None:
        W = np.sqrt(L / nx * H / ny)

    solved = {}
    for dim in (2, 3):
        mesh = Mesh()
        if dim == 2:
            mesh.generate2DMesh((L, H), (nx, ny), ElementType.quadrilateral, order=2)
        else:
            mesh.generate3DMesh((L, H, W), (nx, ny, 1), ElementType.hexahedron, order=2)
        model = build(dim, mesh)
        if dim == 3:
            model.physics.fluid.addBC_SymmetryEdges(mesh.boundary_faces(lambda x, y, z: z < 1e-9 or z > W - 1e-9))
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
        model.solve()
        assert model.nonlinear_converged, dim
        solved[dim] = (model, mesh)
    (m2, mesh2), (m3, mesh3) = solved[2], solved[3]
    lookup = {tuple(np.round(n.coor[:2], 9)): k for k, n in enumerate(mesh2.NL)}
    index = np.array([lookup[tuple(np.round(n.coor[:2], 9))] for n in mesh3.NL])
    u2, u3 = np.asarray(m2.sol['u']), np.asarray(m3.sol['u'])
    c2, c3 = np.asarray(m2.physics.viscoelastic.stress(0)), np.asarray(m3.physics.viscoelastic.stress(0))
    du = max(np.max(np.abs(u3[:, :2] - u2[index])), np.max(np.abs(u3[:, 2]))) / np.max(np.abs(u2))
    # relative to the pressure or, where it vanishes (Couette), to the polymer stress
    dp = np.max(np.abs(np.asarray(m3.sol['p']) - np.asarray(m2.sol['p'])[index])) / max(np.max(np.abs(m2.sol['p'])),
                                                                                    np.max(np.abs(m2.physics.viscoelastic.stress(0))))
    plane = c3[:, [0, 1, 3]]                                         # xx, xy, yy of the 3D stress
    dt = max(np.max(np.abs(plane - c2[index])), np.max(np.abs(c3[:, [2, 4, 5]]))) / np.max(np.abs(c2))
    return du, dp, dt, solved


def test_couette_and_channel_match_2d():
    """Giesekus Couette (the plate y = 1 moving) and a Giesekus channel with a developed inflow, the outlet traction of the polymer and
    DEVSS-G: the 3D slab gives the velocity, pressure and polymer stress of the 2D solution."""

    def couette(dim, mesh):
        model = Model(name='couette', dim=dim, mesh=mesh, mat=material('X', miu=0.5, rho=1.0), psc=[ff, ve])
        flow, polymer = model.physics.fluid, model.physics.viscoelastic
        polymer.setModes([(0.5, 1.5, 0.3)])
        bottom = mesh.boundary_facets(lambda *c: c[1] < 1e-9)
        top = mesh.boundary_facets(lambda *c: c[1] > 1 - 1e-9)
        flow.addBC_NoSlipEdges(bottom)
        flow.addBC_VelocityEdges(top, [1.0] + [0.0] * (dim - 1))
        ends = mesh.boundary_facets(lambda *c: c[0] < 1e-9 or c[0] > 1 - 1e-9)
        left = mesh.boundary_facets(lambda *c: c[0] < 1e-9)
        flow.addBC_OutletEdges(mesh.boundary_facets(lambda *c: c[0] < 1e-9), 0.0)
        flow.addBC_OutletEdges(mesh.boundary_facets(lambda *c: c[0] > 1 - 1e-9), 0.0)
        polymer.addBC_OutletEdges(ends)
        grad = np.zeros((dim, dim))
        grad[0, 1] = 1.0
        polymer.addBC_InflowEdges(left, lambda *c: grad)
        return model

    H, Q = 1.0, 1.0

    def channel(dim, mesh):
        model = Model(name='channel', dim=dim, mesh=mesh, mat=material('X', miu=0.2, rho=1.0), psc=[ff, ve])
        flow, polymer = model.physics.fluid, model.physics.viscoelastic
        polymer.setModes([(0.8, 0.5, 0.2)])
        polymer.setDEVSS(0.8)
        flow.addBC_NoSlipEdges(mesh.boundary_facets(lambda *c: c[1] < 1e-9 or c[1] > H - 1e-9))
        inlet = mesh.boundary_facets(lambda *c: c[0] < 1e-9)
        flow.addBC_VelocityEdges(inlet, lambda *c: [6 * Q * c[1] * (H - c[1]) / H ** 3] + [0.0] * (dim - 1))

        def gradient(*c):
            g = np.zeros((dim, dim))
            g[0, 1] = 6 * Q * (H - 2 * c[1]) / H ** 3
            return g

        polymer.addBC_InflowEdges(inlet, gradient)
        outlet = mesh.boundary_facets(lambda *c: c[0] > 2 - 1e-9)
        flow.addBC_OutletEdges(outlet, 0.0)
        polymer.addBC_OutletEdges(outlet)
        return model

    for label, build, nx, ny, L in (('Couette', couette, 2, 4, 1.0), ('channel', channel, 4, 4, 2.0)):
        du, dp, dt, _ = slab_and_plane(build, nx, ny, L=L)
        print(f"   {label}: 3D - 2D, u {du:.1e}, p {dp:.1e}, polymer stress {dt:.1e}")
        assert du < 1e-8 and dp < 1e-7 and dt < 1e-8, (label, du, dp, dt)


def test_nonisothermal_diffusion_and_modes_match_2d():
    """More of the physics on the channel, 3D slab against 2D: a non-isothermal Giesekus melt (WLF shift of lam and mu, polymer and
    solvent heating, heat convected by the flow, cold walls), the artificial diffusion of b kept near the outlet corner, and two PTT
    modes (linear and exponential in two separate runs)."""

    H, Q, L = 1.0, 1.0, 2.0

    def channel(dim, mesh, modes, constitutive='giesekus', thermal=False, diffusion=None):
        psc = [ff, ht, ve] if thermal else [ff, ve]
        mat = material('X', miu=0.2, rho=1.0, k=0.05, Cp=1.0)
        model = Model(name='channel', dim=dim, mesh=mesh, mat=mat, psc=psc)
        flow, polymer = model.physics.fluid, model.physics.viscoelastic
        polymer.setModes(modes, constitutive)
        polymer.setDEVSS(sum(m[0] for m in modes))
        if thermal:
            polymer.setTemperatureShift(wlf_shift(8.86, 101.6, 200.0))
            polymer.setViscousHeating(True)
            flow.setViscousHeating(True)
            model.physics.setVelocity('u')
        if diffusion:
            polymer.setDiffusion(diffusion, lambda *c: np.exp(-((c[0] - L) ** 2 + (c[1] - H) ** 2) / 0.1))
        walls = mesh.boundary_facets(lambda *c: c[1] < 1e-9 or c[1] > H - 1e-9)
        flow.addBC_NoSlipEdges(walls)
        inlet = mesh.boundary_facets(lambda *c: c[0] < 1e-9)
        flow.addBC_VelocityEdges(inlet, lambda *c: [6 * Q * c[1] * (H - c[1]) / H ** 3] + [0.0] * (dim - 1))

        def gradient(*c):
            g = np.zeros((dim, dim))
            g[0, 1] = 6 * Q * (H - 2 * c[1]) / H ** 3
            return g

        if thermal:
            model.physics.heat.addBC_TemperatureEdges(inlet, 200.0)
            model.physics.heat.addBC_TemperatureEdges(walls, 195.0)
            model.physics.initField('T', 200.0)
        polymer.addBC_InflowEdges(inlet, gradient, temperature=200.0 if thermal else None)
        outlet = mesh.boundary_facets(lambda *c: c[0] > L - 1e-9)
        flow.addBC_OutletEdges(outlet, 0.0)
        polymer.addBC_OutletEdges(outlet)
        return model

    cases = (('non-isothermal Giesekus', dict(modes=[(0.8, 0.5, 0.2)], thermal=True)),
             ('artificial diffusion', dict(modes=[(0.8, 0.8, 0.2)], diffusion=0.5)),
             ('two PTT-linear modes', dict(modes=[(0.5, 0.2, 0.1), (0.3, 1.0, 0.25)], constitutive='ptt-linear')),
             ('two PTT-exp modes', dict(modes=[(0.5, 0.2, 0.1), (0.3, 1.0, 0.25)], constitutive='ptt-exp')))
    for label, options in cases:
        du, dp, dt, solved = slab_and_plane(lambda dim, mesh: channel(dim, mesh, **options), 4, 4, L=L)
        extra = ''
        if options.get('thermal'):
            (m2, mesh2), (m3, mesh3) = solved[2], solved[3]
            lookup = {tuple(np.round(n.coor[:2], 9)): k for k, n in enumerate(mesh2.NL)}
            index = np.array([lookup[tuple(np.round(n.coor[:2], 9))] for n in mesh3.NL])
            T2, T3 = np.asarray(m2.sol['T']), np.asarray(m3.sol['T'])
            dT = np.max(np.abs(T3 - T2[index])) / np.max(np.abs(T2 - 195.0))
            extra = f", T {dT:.1e} (heating to {T3.max():.3f})"
            assert dT < 1e-8 and T3.max() > 200.0, (dT, T3.max())
        print(f"   {label}: 3D - 2D, u {du:.1e}, p {dp:.1e}, polymer stress (mode 0) {dt:.1e}{extra}")
        assert du < 1e-8 and dp < 1e-7 and dt < 1e-8, (label, du, dp, dt)


# ----- a genuinely 3D flow -----

def duct_series(y, z, a, b, G, mu, terms=200):
    from test_3d_flow import duct_series as series
    return series(y, z, a, b, G, mu, terms)


def duct(alpha, n=(3, 4, 3), a=1.0, b=0.6, L=1.5, G=2.0, mu_s=0.3, mu_p=0.7, lam=0.5):

    """Pressure-driven rectangular duct (0 < y < a, 0 < z < b) with one mode; the inflow b is that of the Newtonian (mu_s + mu_p) duct
    flow, exact for Oldroyd-B."""

    mesh = Mesh()
    mesh.generate3DMesh((L, a, b), n, ElementType.hexahedron, order=2)
    model = Model(name='duct', dim=3, mesh=mesh, mat=material('X', miu=mu_s, rho=1.0), psc=[ff, ve])
    flow, polymer = model.physics.fluid, model.physics.viscoelastic
    polymer.setModes([(mu_p, lam, alpha)])
    polymer.setDEVSS(mu_p)
    flow.addBC_NoSlipEdges(mesh.boundary_faces(lambda x, y, z: min(y, a - y, z, b - z) < 1e-9))
    inlet = mesh.boundary_faces(lambda x, y, z: x < 1e-9)
    outlet = mesh.boundary_faces(lambda x, y, z: x > L - 1e-9)
    flow.addBC_OutletEdges(inlet, G * L)
    flow.addBC_OutletEdges(outlet, 0.0)
    polymer.addBC_OutletEdges(inlet)
    polymer.addBC_OutletEdges(outlet)
    h = 1e-6
    u = lambda y, z: duct_series(y, z, a, b, G, mu_s + mu_p)[0]

    def gradient(x, y, z):
        g = np.zeros((3, 3))
        g[0, 1] = (u(min(y + h, a), z) - u(max(y - h, 0.0), z)) / (min(y + h, a) - max(y - h, 0.0))
        g[0, 2] = (u(y, min(z + h, b)) - u(y, max(z - h, 0.0))) / (min(z + h, b) - max(z - h, 0.0))
        return g

    polymer.addBC_InflowEdges(inlet, gradient)
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
    model.solve()
    return model, mesh, u


def test_duct_oldroyd_b_is_newtonian_and_giesekus_has_a_secondary_flow():
    """Rectangular duct 1 x 0.6 (Wi about 0.3): with Oldroyd-B (N2 = 0) the steady flow is rectilinear, with the Newtonian velocity of
    mu_s + mu_p (the duct series); with Giesekus (N2 != 0) a secondary flow in the section appears. On two meshes of the section (4 x 3
    and 8 x 6 cells): the secondary flow of Oldroyd-B is discretization error and falls with order about 3 (measured 1.5e-3 -> 1.5e-4 of
    u_max), that of Giesekus stays (1.6e-3 -> 2.1e-3), 14 times larger on the fine mesh."""

    results = {}
    for n in ((2, 4, 3), (2, 8, 6)):
        for alpha in (0.0, 0.3):
            start = time.perf_counter()
            model, mesh, u = duct(alpha, n=n, L=1.0)
            assert model.nonlinear_converged
            xyz = xyz_of(mesh)
            U = np.asarray(model.sol['u'])
            middle = np.abs(xyz[:, 0] - 0.5) < 1e-9
            axial = np.max(np.abs(U[middle, 0] - u(xyz[middle, 1], xyz[middle, 2]))) / u(0.5, 0.3)
            secondary = np.max(np.abs(U[middle, 1:])) / np.max(U[:, 0])
            results[(n[1], alpha)] = (axial, secondary)
            print(f"   section {n[1]} x {n[2]}, alpha = {alpha}: axial velocity against the Newtonian series {axial:.1e}, secondary "
                  f"flow {secondary:.2e} of u_max ({model.layout.ndof} unknowns, {time.perf_counter() - start:.1f} s)")

    (a_c, s_c), (a_f, s_f) = results[(4, 0.0)], results[(8, 0.0)]
    assert a_f < a_c and a_f < 5e-3 and s_f < s_c / 4                 # Oldroyd-B: Newtonian and rectilinear in the limit
    g_c, g_f = results[(4, 0.3)][1], results[(8, 0.3)][1]
    assert abs(g_f - g_c) < 0.5 * g_f and g_f > 5 * s_f              # Giesekus: a secondary flow that converges


def test_block_preconditioner_with_viscoelastic_fields():
    """GMRES with the block preconditioner on the Giesekus duct (b and G through an ILU of their own block): the PARDISO solution in the
    same Newton steps (measured: 105-120 Krylov iterations here, 200-290 on the section of 8 x 6; Test/manual/viscoelastic_3d_cost.py)."""

    from Source.enums import SolverType, IterativeSolvers, Preconditioners

    solutions, steps = {}, {}
    original = Model.solverConfiguration
    for method in ('direct', 'block'):
        extra = {} if method == 'direct' else dict(Method=SolverType.iterative, Solver=IterativeSolvers.GMRES,
                                                   prec=Preconditioners.SaddlePoint, tol=1e-9, maxiter=2000)
        Model.solverConfiguration = lambda self, _extra=extra, **kw: original(self, **{**kw, **_extra})
        try:
            model, _, _ = duct(0.3, n=(2, 4, 3), L=1.0)
        finally:
            Model.solverConfiguration = original
        assert model.nonlinear_converged and model.solver_unconverged == 0
        solutions[method], steps[method] = np.asarray(model.sol['u']), model.newton_iterations
        if method == 'block':
            print(f"   Krylov iterations per Newton step: {model.solver_iterations}")
    assert steps['direct'] == steps['block']
    assert np.max(np.abs(solutions['block'] - solutions['direct'])) < 1e-8 * np.max(np.abs(solutions['direct']))


def main():
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    failures = 0

    for name, fn in tests:
        start = time.perf_counter()
        try:
            fn()
            print(f"PASS  {name} ({time.perf_counter() - start:.1f} s)")
        except Exception as exc:
            failures += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}: {exc!r}")

    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
