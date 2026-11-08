"""Checks for Stokes and Navier-Stokes in 3D (phase 2 of docs/PLAN_3D.md): Taylor-Hood P2/P1 (tet10) and Q2/Q1 (hex27), equal
order with PSPG (tet4, hex8).

Run from the project root:  python Test/test_3d_flow.py
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Simulation.Discretization import Discretization
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers

KIND = {'tet4': (ElementType.tetrahedron, 1), 'tet10': (ElementType.tetrahedron, 2),
        'hex8': (ElementType.hexahedron, 1), 'hex27': (ElementType.hexahedron, 2)}


def box(name, n, L=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    mesh = Mesh()
    mesh.generate3DMesh(L, n if np.ndim(n) else (n, n, n), *KIND[name], origin=origin)
    return mesh


def xyz_of(mesh):
    return np.array([node.coor for node in mesh.NL], dtype=float)


def steady(model, kind=ProblemType.linear, **options):
    model.solverConfiguration(Study=StudyType.steady_state, Type=kind, Method=SolverType.direct, Solver=DirectSolvers.PARDISO,
                              **options)
    model.solve()


def l2(mesh, nodal, exact, mean_free=False):

    """L2 error of a nodal field (n,) or (n, k) against exact(x, y, z) -> (..., ) or (..., k); mean_free removes the mean of both."""

    disc = Discretization(mesh)
    values = disc.interpolate(np.asarray(nodal, dtype=float))
    reference = exact(*np.moveaxis(disc.x_gp, -1, 0))
    if mean_free:
        volume = disc.detJw.sum()
        values = values - np.sum(disc.detJw * values) / volume
        reference = reference - np.sum(disc.detJw * reference) / volume
    diff = (values - reference) ** 2
    if diff.ndim == 3:
        diff = diff.sum(axis=-1)
    return np.sqrt(np.sum(disc.detJw * diff))


def rates_of(errors):
    return [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]


# ----- Stokes -----

def test_plane_poiseuille_is_exact():
    """Walls y = 0, 1, symmetry z = 0, 1, pressures G L and 0 at the ends: u = G / (2 mu) y (1 - y), p = G (L - x) belong to the
    Taylor-Hood spaces (exact, as in 2D) and are also reproduced by hex8 with PSPG; tet4 with PSPG is not exact (as T3)."""

    L, G, mu = 2.0, 3.0, 0.7
    for name in ('hex27', 'tet10', 'hex8', 'tet4'):
        mesh = box(name, (4, 3, 2), (L, 1.0, 1.0))
        model = Model(name='poiseuille', dim=3, mesh=mesh, mat=material('X', miu=mu), psc=ff)
        flow = model.physics
        assert flow.taylor_hood == (name in ('hex27', 'tet10'))
        flow.addBC_NoSlipEdges(mesh.boundary_faces(lambda x, y, z: y < 1e-9 or y > 1 - 1e-9))
        flow.addBC_SymmetryEdges(mesh.boundary_faces(lambda x, y, z: z < 1e-9 or z > 1 - 1e-9))
        flow.addBC_OutletEdges(mesh.boundary_faces(lambda x, y, z: x < 1e-9), G * L)
        flow.addBC_OutletEdges(mesh.boundary_faces(lambda x, y, z: x > L - 1e-9), 0.0)
        steady(model)

        xyz, u = xyz_of(mesh), np.asarray(model.sol['u'])
        errors = (np.max(np.abs(u[:, 0] - G / (2 * mu) * xyz[:, 1] * (1 - xyz[:, 1]))), np.max(np.abs(u[:, 1:])),
                  np.max(np.abs(model.sol['p'] - G * (L - xyz[:, 0]))))
        print(f"   {name}: u {errors[0]:.1e}, cross {errors[1]:.1e}, p {errors[2]:.1e}")
        if name == 'tet4':
            assert errors[0] < 2e-2
        else:
            assert max(errors) < 1e-10, (name, errors)


def duct_series(y, z, a, b, G, mu, terms=200):

    """Fully developed flow in the rectangular duct 0 < y < a, 0 < z < b under the pressure gradient -G (series in y):
    u = 4 G a^2 / (mu pi^3) sum over odd n of [1 - cosh(n pi (z - b/2) / a) / cosh(n pi b / (2a))] sin(n pi y / a) / n^3,
    Q = G a^3 b / (12 mu) [1 - 192 a / (pi^5 b) sum over odd n of tanh(n pi b / (2a)) / n^5].
    The part without cosh sums to the plane profile G y (a - y) / (2 mu); u is evaluated as that profile plus the harmonic cosh series,
    which converges exponentially inside the duct."""

    n = np.arange(1, 2 * terms, 2)
    y, z = np.asarray(y, dtype=float), np.asarray(z, dtype=float)
    shape = np.cosh(np.multiply.outer(z - b / 2, n * np.pi / a)) / np.cosh(n * np.pi * b / (2 * a))
    u = G / (2 * mu) * y * (a - y) - 4 * G * a ** 2 / (mu * np.pi ** 3) * np.sum(
        shape * np.sin(np.multiply.outer(y, n * np.pi / a)) / n ** 3, axis=-1)
    Q = G * a ** 3 * b / (12 * mu) * (1 - 192 * a / (np.pi ** 5 * b) * np.sum(np.tanh(n * np.pi * b / (2 * a)) / n ** 5))
    return u, Q


def test_duct_series_solves_the_equation():
    """The duct series (written here from its derivation) satisfies mu lap u = -G inside and u = 0 on the walls, and Q is the
    integral of u over the section."""

    a, b, G, mu, h = 1.0, 0.6, 2.0, 0.5, 1e-3
    rng = np.random.default_rng(0)
    y, z = rng.uniform(0.1, 0.9, 10) * a, rng.uniform(0.1, 0.9, 10) * b
    f = lambda y, z: duct_series(y, z, a, b, G, mu)[0]
    lap = (f(y + h, z) + f(y - h, z) + f(y, z + h) + f(y, z - h) - 4 * f(y, z)) / h ** 2
    assert np.allclose(mu * lap, -G, rtol=1e-4)
    edge = np.linspace(0, 1, 7)
    assert np.allclose(f(0 * edge, edge * b), 0, atol=1e-12) and np.allclose(f(a + 0 * edge, edge * b), 0, atol=1e-12)
    # on z = 0, b the profile and the series cancel through the Fourier identity: zero up to the truncation (200 terms)
    assert np.allclose(f(edge * a, 0 * edge), 0, atol=1e-5) and np.allclose(f(edge * a, b + 0 * edge), 0, atol=1e-5)
    from numpy.polynomial.legendre import leggauss
    t, w = leggauss(40)
    Y, Z = np.meshgrid((t + 1) * a / 2, (t + 1) * b / 2, indexing='ij')
    integral = np.einsum('i,j,ij->', w, w, f(Y, Z)) * a * b / 4
    assert np.isclose(integral, duct_series(0.0, 0.0, a, b, G, mu)[1], rtol=1e-6)


def duct_flow(name, n, a=1.0, b=0.6, L=1.0, G=2.0, mu=0.5):

    """Duct along x with the cross-section (0, a) x (0, b), no slip on its walls, pressures G L and 0 at the ends."""

    mesh = box(name, (max(1, n // 2), n, max(1, round(n * b / a))), (L, a, b))
    model = Model(name='duct', dim=3, mesh=mesh, mat=material('X', miu=mu), psc=ff)
    flow = model.physics
    flow.addBC_NoSlipEdges(mesh.boundary_faces(lambda x, y, z: min(y, a - y, z, b - z) < 1e-9))
    inlet = mesh.boundary_faces(lambda x, y, z: x < 1e-9)
    flow.addBC_OutletEdges(inlet, G * L)
    flow.addBC_OutletEdges(mesh.boundary_faces(lambda x, y, z: x > L - 1e-9), 0.0)
    steady(model)
    exact = lambda x, y, z: np.stack(np.broadcast_arrays(duct_series(y, z, a, b, G, mu)[0], 0 * x, 0 * x), axis=-1)
    Q = duct_series(0.0, 0.0, a, b, G, mu)[1]
    return l2(mesh, model.sol['u'], exact), abs(-flow.flowRate(inlet) - Q) / Q, model


def test_duct_against_the_fourier_series():
    """Rectangular duct (1 x 0.6): the velocity converges to the series with L2 order 3 (Taylor-Hood) and 2 (hex8 with PSPG), and
    the flow rate approaches the exact Q."""

    for name, sizes, expected in (('hex27', (2, 4, 8), 3.0), ('tet10', (2, 4, 8), 3.0), ('hex8', (4, 8, 16), 2.0)):
        start = time.perf_counter()
        results = [duct_flow(name, n)[:2] for n in sizes]
        errors, flows = [r[0] for r in results], [r[1] for r in results]
        rates = rates_of(errors)
        print(f"   {name}: u L2 {['%.2e' % e for e in errors]}, rates {['%.2f' % r for r in rates]}, "
              f"Q error {['%.1e' % q for q in flows]} ({time.perf_counter() - start:.1f} s)")
        assert rates[-1] > expected - 0.2, (name, rates)
        assert flows[-1] < (2e-3 if name != 'hex8' else 2e-2) and flows[-1] < flows[0], (name, flows)


def cylinder_mesh(R, L, size):

    """Pipe of radius R along x (0 < x < L) from gmsh, tet10 (curved), with groups inlet, outlet and wall."""

    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("pipe")
        gmsh.model.occ.addCylinder(0, 0, 0, L, 0, 0, R)
        gmsh.model.occ.synchronize()
        for name, test in (('inlet', lambda c: abs(c[0]) < 1e-9), ('outlet', lambda c: abs(c[0] - L) < 1e-9),
                           ('wall', lambda c: 1e-9 < c[0] < L - 1e-9)):
            tags = [s for _, s in gmsh.model.getEntities(2) if test(gmsh.model.occ.getCenterOfMass(2, s))]
            gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, tags), name)
        gmsh.model.addPhysicalGroup(3, [1])
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
        gmsh.model.mesh.setOrder(2)
        return Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()


def test_pipe_against_the_axisymmetric_model():
    """Pipe R = 0.5, L = 2 driven by the pressures G L and 0 on curved tet10: the flow rate and the velocity of every node match the
    axisymmetric model of the code (Hagen-Poiseuille, exact there: u(r) = G (R^2 - r^2) / (4 mu)), with errors that fall with h."""

    R, L, G, mu = 0.5, 2.0, 1.5, 0.8

    axi = Mesh()
    axi.generate2DMesh((R, L), (4, 8), ElementType.quadrilateral, order=2)
    model = Model(name='axi', dim=2, mesh=axi, mat=material('X', miu=mu), psc=ff, axisymmetric=True)
    model.physics.addBC_NoSlipEdges(axi.boundary_edges(lambda r, z: r > R - 1e-9))
    model.physics.addBC_SymmetryEdges(axi.boundary_edges(lambda r, z: r < 1e-9))
    inlet_axi = axi.boundary_edges(lambda r, z: z < 1e-9)
    model.physics.addBC_OutletEdges(inlet_axi, G * L)
    model.physics.addBC_OutletEdges(axi.boundary_edges(lambda r, z: z > L - 1e-9), 0.0)
    steady(model)
    Q_axi = -model.physics.flowRate(inlet_axi)                     # volumetric (2 pi included)
    rz = np.array([n.coor for n in axi.NL])
    profile = lambda r: G * (R ** 2 - r ** 2) / (4 * mu)
    assert np.isclose(Q_axi, np.pi * R ** 4 * G / (8 * mu), rtol=1e-10)   # Hagen-Poiseuille
    assert np.max(np.abs(np.asarray(model.sol['u'])[:, 1] - profile(rz[:, 0]))) < 1e-12

    errors = []
    for size in (0.25, 0.15):
        mesh = cylinder_mesh(R, L, size)
        model = Model(name='pipe', dim=3, mesh=mesh, mat=material('X', miu=mu), psc=ff)
        flow = model.physics
        flow.addBC_NoSlipEdges(mesh.boundary_faces('wall'))
        flow.addBC_OutletEdges(mesh.boundary_faces('inlet'), G * L)
        flow.addBC_OutletEdges(mesh.boundary_faces('outlet'), 0.0)
        steady(model)
        Q = -flow.flowRate(mesh.boundary_faces('inlet'))
        xyz = xyz_of(mesh)
        u = np.asarray(model.sol['u'])
        u_error = np.max(np.abs(u - np.stack([profile(np.hypot(xyz[:, 1], xyz[:, 2])), 0 * xyz[:, 0], 0 * xyz[:, 0]], -1)))
        errors.append((abs(Q - Q_axi) / Q_axi, u_error / profile(0.0)))
        print(f"   size {size}: {mesh.getNoE()} tet10, Q {Q:.6f} (axisymmetric {Q_axi:.6f}), max nodal u error "
              f"{errors[-1][1]:.1e} of u_max")

    assert errors[-1][0] < 2e-4 and errors[-1][1] < 5e-3, errors
    assert errors[-1][0] < errors[0][0] and errors[-1][1] < errors[0][1], errors


# ----- Navier-Stokes -----

ES_A, ES_D = np.pi / 4, np.pi / 2


def es_velocity(x, y, z):
    a, d = ES_A, ES_D
    return np.stack(np.broadcast_arrays(-a * (np.exp(a * x) * np.sin(a * y + d * z) + np.exp(a * z) * np.cos(a * x + d * y)),
                                        -a * (np.exp(a * y) * np.sin(a * z + d * x) + np.exp(a * x) * np.cos(a * y + d * z)),
                                        -a * (np.exp(a * z) * np.sin(a * x + d * y) + np.exp(a * y) * np.cos(a * z + d * x))),
                    axis=-1)


def es_pressure(x, y, z):
    a, d = ES_A, ES_D
    return -a * a / 2 * (np.exp(2 * a * x) + np.exp(2 * a * y) + np.exp(2 * a * z)
                         + 2 * np.sin(a * x + d * y) * np.cos(a * z + d * x) * np.exp(a * (y + z))
                         + 2 * np.sin(a * y + d * z) * np.cos(a * x + d * y) * np.exp(a * (z + x))
                         + 2 * np.sin(a * z + d * x) * np.cos(a * y + d * z) * np.exp(a * (x + y)))


def test_ethier_steinman_solution_is_exact():
    """The Ethier-Steinman (1994) field at t = 0, as written here: div u = 0, lap u = -d^2 u (Beltrami) and (u . grad) u = -grad p,
    checked by finite differences, so that the steady problem with the body force nu d^2 u has it as its exact solution."""

    rng = np.random.default_rng(0)
    X, h = rng.uniform(-1, 1, (20, 3)), 1e-4
    grad = lambda f: np.stack([(f(*(X + h * e).T) - f(*(X - h * e).T)) / (2 * h) for e in np.eye(3)], axis=-1)
    G = grad(es_velocity)
    lap = sum((es_velocity(*(X + h * e).T) - 2 * es_velocity(*X.T) + es_velocity(*(X - h * e).T)) / h ** 2 for e in np.eye(3))
    assert np.max(np.abs(np.trace(G, axis1=1, axis2=2))) < 1e-7
    assert np.max(np.abs(lap + ES_D ** 2 * es_velocity(*X.T))) < 1e-5
    assert np.max(np.abs(np.einsum('nb,nab->na', es_velocity(*X.T), G) + grad(es_pressure))) < 1e-7


def ethier_steinman(name, n, nu=0.1):

    """Steady Navier-Stokes in [-1, 1]^3 with rho = 1, the Ethier-Steinman velocity on the whole boundary and the body force
    nu d^2 u; the pressure is fixed at the node nearest to the origin. Returns the L2 errors of u and of p (mean-free)."""

    mesh = box(name, n, (2.0, 2.0, 2.0), (-1.0, -1.0, -1.0))
    model = Model(name='es', dim=3, mesh=mesh, mat=material('X', miu=nu, rho=1.0), psc=ff)
    flow = model.physics
    flow.setInertia(True)
    flow.setBodyForce(lambda x, y, z: tuple(np.moveaxis(nu * ES_D ** 2 * es_velocity(x, y, z), -1, 0)))
    flow.addBC_VelocityEdges(mesh.boundary_faces(), lambda x, y, z: tuple(es_velocity(x, y, z)))
    flow.setPressureReference((0.0, 0.0, 0.0), float(es_pressure(0.0, 0.0, 0.0)))
    steady(model, ProblemType.nonlinear)
    return (l2(mesh, model.sol['u'], es_velocity), l2(mesh, model.sol['p'], es_pressure, mean_free=True),
            mesh.getNoN(), model)


def test_ethier_steinman_navier_stokes_orders():
    """Steady Navier-Stokes with the exact Ethier-Steinman field (nu = 0.1, Re of the field about 10): Taylor-Hood gives u of order 3
    and p of order 2 in L2; hex8 with PSPG/SUPG/LSIC gives u of order 2."""

    for name, sizes, (ru, rp) in (('hex27', (2, 4, 8), (3.0, 2.0)), ('tet10', (2, 4, 8), (3.0, 2.0)), ('hex8', (4, 8, 16), (2.0, 1.0))):
        start = time.perf_counter()
        results = [ethier_steinman(name, n) for n in sizes]
        eu, ep = [r[0] for r in results], [r[1] for r in results]
        rates_u, rates_p = rates_of(eu), rates_of(ep)
        print(f"   {name}: u {['%.2e' % e for e in eu]} rates {['%.2f' % r for r in rates_u]}; p {['%.2e' % e for e in ep]} "
              f"rates {['%.2f' % r for r in rates_p]} ({time.perf_counter() - start:.1f} s, {results[-1][2]} nodes)")
        assert rates_u[-1] > ru - 0.25 and rates_p[-1] > rp - 0.3, (name, rates_u, rates_p)


def test_taylor_hood_supg_in_3d():
    """Ethier-Steinman at nu = 0.002 (convection dominated on these meshes): SUPG/PSPG of Taylor-Hood is consistent (at nu = 0.1 it
    converges with order 3, measured apart: tet10 u 3.5e-1, 1.4e-1, 1.6e-2 for n = 2, 4, 8, with a constant 3 times that of Galerkin
    on tetrahedra and close to it on hex27) and on the coarse hex27 mesh it lowers the velocity error several times; on tet10 Newton does not
    converge without it at nu = 0.002, and with it the pressure of the elements with every vertex on the wall stays bounded (SUPG
    alone gave an error of 10 there)."""

    def run(name, n, nu, supg):
        mesh = box(name, n, (2.0, 2.0, 2.0), (-1.0, -1.0, -1.0))
        model = Model(name='es', dim=3, mesh=mesh, mat=material('X', miu=nu, rho=1.0), psc=ff)
        flow = model.physics
        flow.setInertia(True)
        if supg:
            flow.setStabilization('SUPG')
        flow.setBodyForce(lambda x, y, z: tuple(np.moveaxis(nu * ES_D ** 2 * es_velocity(x, y, z), -1, 0)))
        flow.addBC_VelocityEdges(mesh.boundary_faces(), lambda x, y, z: tuple(es_velocity(x, y, z)))
        flow.setPressureReference((0.0, 0.0, 0.0), float(es_pressure(0.0, 0.0, 0.0)))
        steady(model, ProblemType.nonlinear)
        if not model.nonlinear_converged:
            return None
        xyz = xyz_of(mesh)
        dp = np.asarray(model.sol['p']) - es_pressure(*xyz.T)
        return (l2(mesh, model.sol['u'], es_velocity), l2(mesh, model.sol['p'], es_pressure, mean_free=True),
                np.max(np.abs(dp - np.median(dp))))

    # Galerkin on tet10 at nu = 0.002 is left out: Newton does not converge there (measured: ~100 s of iterations)
    e = {(name, nu, supg): run(name, n, nu, supg) for name, n in (('hex27', 4), ('tet10', 4))
         for nu in (0.1, 0.002) for supg in (False, True) if (name, nu, supg) != ('tet10', 0.002, False)}
    for key, value in e.items():
        label = f"   {key[0]} nu = {key[1]} {'SUPG/PSPG' if key[2] else 'Galerkin'}"
        print(label + (": Newton did not converge" if value is None else
                       f": u L2 {value[0]:.2e}, p L2 {value[1]:.2e}, p max {value[2]:.2e}"))

    assert all(e[(name, nu, True)] is not None for name in ('hex27', 'tet10') for nu in (0.1, 0.002))
    assert e[('hex27', 0.1, True)][0] < 1.5 * e[('hex27', 0.1, False)][0]
    assert e[('tet10', 0.1, True)][0] < 3.5 * e[('tet10', 0.1, False)][0]           # the PSPG term costs accuracy on tetrahedra
    assert e[('hex27', 0.002, True)][0] < e[('hex27', 0.002, False)][0] / 1.5
    assert e[('hex27', 0.002, True)][2] < 2 * e[('hex27', 0.002, False)][2]
    assert e[('tet10', 0.002, True)][2] < 2.0      # pressure range 6 (SUPG alone: 10 at the corners on 8^3; SUPG/PSPG there: 0.27)


def ethier_steinman_transient(name, n, dt, T, nu=0.1):

    """Unsteady Navier-Stokes in [-1, 1]^3 from the Ethier-Steinman field: u = u0 exp(-nu d^2 t), p = p0 exp(-2 nu d^2 t), no body
    force; the boundary velocity follows exp(-nu d^2 t) (setBCTime). BDF2. Returns (model, nodes)."""

    mesh = box(name, n, (2.0, 2.0, 2.0), (-1.0, -1.0, -1.0))
    model = Model(name='es-t', dim=3, mesh=mesh, mat=material('X', miu=nu, rho=1.0), psc=ff)
    flow = model.physics
    flow.setInertia(True)
    faces = mesh.boundary_faces()
    flow.addBC_VelocityEdges(faces, lambda x, y, z: tuple(es_velocity(x, y, z)))
    flow.setBCTime('u', faces, lambda t: np.exp(-nu * ES_D ** 2 * t))
    flow.setPressureReference((0.0, 0.0, 0.0), 0.0)
    flow.initField('u', lambda x, y, z: tuple(es_velocity(x, y, z)))
    model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, Method=SolverType.direct,
                              Solver=DirectSolvers.PARDISO, timeScheme='BDF2', timeStep=dt, totalTime=T)
    model.solve()
    return model, mesh


def test_ethier_steinman_transient_bdf2():
    """The unsteady Ethier-Steinman flow (nu = 0.1) with BDF2 on hex27: at t = 1 the error is that of the steady field on the same
    mesh (u L2 and the mean-free p), and the temporal error against a small-step reference on a coarse mesh falls with order 2."""

    nu, T = 0.1, 1.0
    model, mesh = ethier_steinman_transient('hex27', 4, 0.05, T, nu)
    decay = np.exp(-nu * ES_D ** 2 * T)
    e_u = l2(mesh, np.asarray(model.sol['u'])[..., -1], lambda x, y, z: decay * es_velocity(x, y, z))
    e_p = l2(mesh, np.asarray(model.sol['p'])[:, -1], lambda x, y, z: decay ** 2 * es_pressure(x, y, z), mean_free=True)
    scale_u = l2(mesh, 0 * np.asarray(model.sol['u'])[..., -1], lambda x, y, z: decay * es_velocity(x, y, z))
    print(f"   hex27 4^3, dt = 0.05, t = 1: u L2 {e_u:.2e} ({e_u / scale_u:.1e} relative), p L2 {e_p:.2e}, "
          f"Newton per step {np.mean(model.step_iterations):.1f} (max {max(model.step_iterations)}), "
          f"{model.tangent_assemblies} tangents")
    # the modified Newton keeps the tangent of the first steps (2 tangents for 20 steps, 4-7 iterations each); rebuilding it every
    # iteration gives 3 per step with 75 tangents
    assert e_u / scale_u < 1e-2 and e_p < 0.1 and max(model.step_iterations) <= 8

    reference = np.asarray(ethier_steinman_transient('hex27', 2, 0.0125, T, nu)[0].sol['u'])[..., -1]
    errors = [np.max(np.abs(np.asarray(ethier_steinman_transient('hex27', 2, dt, T, nu)[0].sol['u'])[..., -1] - reference))
              for dt in (0.2, 0.1, 0.05)]
    rates = rates_of(errors)
    print(f"   BDF2 temporal errors {['%.1e' % e for e in errors]}, rates {['%.2f' % r for r in rates]}")
    assert rates[-1] > 1.8, rates


def test_block_preconditioner_in_3d():
    """GMRES with the block preconditioner (AMG on the velocity with the 6 rigid modes of 3D, pressure mass for the Schur complement)
    solves the duct to the PARDISO solution in a number of iterations that hardly grows with the mesh."""

    from Source.enums import IterativeSolvers, Preconditioners

    a, b, L, G, mu = 1.0, 0.6, 2.0, 2.0, 0.5
    iterations = []
    for name in ('hex27', 'tet10'):
        for n in (4, 8):
            solutions = {}
            for method in ('direct', 'block'):
                mesh = box(name, (2 * n, n, round(n * b / a)), (L, a, b))
                model = Model(name='duct', dim=3, mesh=mesh, mat=material('X', miu=mu), psc=ff)
                flow = model.physics
                flow.addBC_NoSlipEdges(mesh.boundary_faces(lambda x, y, z: min(y, a - y, z, b - z) < 1e-9))
                flow.addBC_OutletEdges(mesh.boundary_faces(lambda x, y, z: x < 1e-9), G * L)
                flow.addBC_OutletEdges(mesh.boundary_faces(lambda x, y, z: x > L - 1e-9), 0.0)
                if method == 'direct':
                    steady(model)
                else:
                    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear, Method=SolverType.iterative,
                                              Solver=IterativeSolvers.GMRES, prec=Preconditioners.SaddlePoint, tol=1e-10,
                                              maxiter=1000)
                    model.solve()
                    assert model.solver_unconverged == 0
                    iterations.append(sum(model.solver_iterations))
                solutions[method] = (np.asarray(model.sol['u']), np.asarray(model.sol['p']))
            for k in range(2):
                scale = np.max(np.abs(solutions['direct'][k]))
                assert np.max(np.abs(solutions['block'][k] - solutions['direct'][k])) < 1e-6 * scale, (name, n, k)
    print(f"   GMRES iterations (hex27 n = 4, 8; tet10 n = 4, 8): {iterations}")
    assert iterations[1] < 1.5 * iterations[0] and iterations[3] < 1.5 * iterations[2], iterations


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
