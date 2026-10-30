"""Stage 17, priority 4: Taylor-Hood elements (Q2/Q1 and P2/P1) on quadratic meshes.

Run from the project root:  python Test/test_taylor_hood.py
"""

import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")

import gmsh
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Elements import ELEMENTS
from Source.Pre_processing.Mesh import Mesh
from Source.Post_processing.VTK import write_vtu
from Source.Simulation.Discretization import Discretization
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.HeatTransfer import ht
from Source.Physics.Rheology import CarreauYasuda, PowerLaw
from Source.enums import ElementType, StudyType, ProblemType

ELEMENT_TYPES = (ElementType.quadrilateral, ElementType.triangle)
GHIA = {100.0: {0.9531: 0.68717, 0.5: -0.20581, 0.1719: -0.10150, 0.1016: -0.06434},
        1000.0: {0.9531: 0.46604, 0.5: -0.06080, 0.1719: -0.38289, 0.1016: -0.29730}}


def mesh2(size, div, element, order=2, origin=(0.0, 0.0)):
    mesh = Mesh()
    mesh.generate2DMesh(size, div, element, order=order)
    for node in mesh.NL:
        node.coor = (node.coor[0] + origin[0], node.coor[1] + origin[1])
    return mesh


def xy_of(mesh):
    return np.array([n.coor[:2] for n in mesh.NL], dtype=float)


def steady(model, kind=ProblemType.nonlinear, **options):
    model.solverConfiguration(Study=StudyType.steady_state, Type=kind, **options)
    model.solve()


def pressure_violation(model):

    """Largest |p_m - sum w_v p_v|: the pressure must be linear in every element."""

    p = model.sol['p']
    disc = model.geometry
    return max(abs(p[c[node]] - sum(w * p[c[v]] for v, w in combination))
               for c in disc.conn for node, combination in disc.element.linear_subspace)


def kovasznay(n, element, Re=40.0):

    lam = Re / 2 - np.sqrt(Re ** 2 / 4 + 4 * np.pi ** 2)
    mu, k = 1.0 / Re, 2 * np.pi
    U = lambda x, y: np.stack(np.broadcast_arrays(1 - np.exp(lam * x) * np.cos(k * y),
                                                  lam / k * np.exp(lam * x) * np.sin(k * y)), axis=-1)
    P = lambda x, y: 0.5 * (1 - np.exp(2 * lam * x))

    def traction(x, y):
        e = np.exp(lam * x)
        ux, uy, vx = -lam * e * np.cos(k * y), k * e * np.sin(k * y), lam ** 2 / k * e * np.sin(k * y)
        return (-P(x, y) + 2 * mu * ux, mu * (uy + vx))

    mesh = mesh2((1.5, 2.0), (n, int(round(n * 4 / 3))), element, origin=(-0.5, -0.5))
    model = Model(name='kovasznay', dim=2, mesh=mesh, mat=material('X', miu=mu, rho=1.0), psc=ff)
    model.physics.setInertia(True)
    model.physics.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: x < 1 - 1e-9), lambda x, y: tuple(U(x, y)))
    model.physics.addBC_TractionEdges(mesh.boundary_edges(lambda x, y: x > 1 - 1e-9), traction)
    steady(model)
    xy = xy_of(mesh)
    return (np.max(np.abs(model.sol['u'] - U(xy[:, 0], xy[:, 1]))), np.max(np.abs(model.sol['p'] - P(xy[:, 0], xy[:, 1]))),
            model)


def cavity(n, Re, element, **options):

    mesh = mesh2((1.0, 1.0), (n, n), element)
    model = Model(name='cavity', dim=2, mesh=mesh, mat=material('X', miu=1.0 / Re, rho=1.0), psc=ff)
    model.physics.setInertia(True)
    model.physics.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), (1.0, 0.0))
    model.physics.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or x < 1e-9 or x > 1 - 1e-9))
    return model, mesh


def schafer_turek_mesh(lc_far, lc_cyl):

    """Channel with a cylinder (Schafer & Turek 1996) in curved quadratic triangles."""

    L, H = 2.2, 0.41
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        occ = gmsh.model.occ
        fluid, _ = occ.cut([(2, occ.addRectangle(0, 0, 0, L, H))], [(2, occ.addDisk(0.2, 0.2, 0, 0.05, 0.05))])
        occ.synchronize()
        groups = {'inlet': [], 'outlet': [], 'walls': [], 'cylinder': []}
        for _, tag in gmsh.model.getBoundary(fluid, oriented=False):
            x, y, _ = occ.getCenterOfMass(1, tag)
            key = 'inlet' if abs(x) < 1e-9 else 'outlet' if abs(x - L) < 1e-9 else 'walls' if min(y, H - y) < 1e-9 else 'cylinder'
            groups[key].append(tag)
        for name, tags in groups.items():
            gmsh.model.addPhysicalGroup(1, tags, name=name)
        field = gmsh.model.mesh.field
        dist = field.add('Distance')
        field.setNumbers(dist, 'CurvesList', groups['cylinder'])
        field.setNumber(dist, 'Sampling', 200)
        threshold = field.add('Threshold')
        field.setNumber(threshold, 'InField', dist)
        field.setNumber(threshold, 'SizeMin', lc_cyl)
        field.setNumber(threshold, 'SizeMax', lc_far)
        field.setNumber(threshold, 'DistMin', 0.0)
        field.setNumber(threshold, 'DistMax', 0.4)
        field.setAsBackgroundMesh(threshold)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
        return Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()


# ----- elements and meshes -----

def test_shape_functions_and_quadrature():

    for name in ('tri6', 'quad9'):
        element = ELEMENTS[name]
        N, dN = element.shape(element.nodes)
        assert np.allclose(N, np.eye(element.n_nodes), atol=1e-13)                  # Kronecker delta at the nodes

        rng = np.random.default_rng(1)
        points = rng.uniform(0.05, 0.4, size=(10, 2))
        N, dN = element.shape(points)
        assert np.allclose(N.sum(axis=1), 1.0) and np.allclose(dN.sum(axis=1), 0.0)

        h = 1e-6
        for a in range(2):
            e = np.zeros(2)
            e[a] = h
            fd = (element.shape(points + e)[0] - element.shape(points - e)[0]) / (2 * h)
            assert np.allclose(fd, dN[..., a], atol=1e-8)

        # The relations of the linear subspace reproduce any linear function of the reference coordinates
        for node, combination in element.linear_subspace:
            assert np.allclose(sum(w * element.nodes[v] for v, w in combination), element.nodes[node])

    # Degree of the 7-point rule on the triangle: integral of x^a y^b = a! b! / (a + b + 2)!
    from math import factorial
    points, weights = ELEMENTS['tri6'].quadrature(3)
    assert abs(weights.sum() - 0.5) < 1e-14
    for a in range(6):
        for b in range(6 - a):
            exact = factorial(a) * factorial(b) / factorial(a + b + 2)
            assert abs(np.sum(weights * points[:, 0] ** a * points[:, 1] ** b) - exact) < 1e-14, (a, b)


def test_quadratic_mesh_and_geometry():

    for element, name, nne, nel in ((ElementType.quadrilateral, 'quad9', 9, 6), (ElementType.triangle, 'tri6', 6, 12)):
        mesh = mesh2((3.0, 2.0), (3, 2), element)
        assert mesh.element_type == name and mesh.getNoN() == 7 * 5 and mesh.getNoE() == nel
        disc = Discretization(mesh)
        assert disc.conn.shape == (nel, nne) and abs(disc.detJw.sum() - 6.0) < 1e-12
        edges = mesh.boundary_edges(lambda x, y: y < 1e-9)
        assert len(edges) == 3 and all(len(edge) == 3 for edge in edges)
        facets = disc.facets(edges)
        assert abs(facets.detJw.sum() - 3.0) < 1e-12 and np.allclose(facets.normals, (0.0, -1.0))

    # L2 projection (consistent mass): a T6 has integral(N_vertex) = 0, a lumped one would divide by zero
    disc = Discretization(mesh2((1.0, 1.0), (3, 3), ElementType.triangle))
    assert np.allclose(disc.project(np.full(disc.detJw.shape, 2.5)), 2.5)
    assert np.allclose(disc.project(disc.x_gp), disc.node_xy)                       # it reproduces the quadratic space


def test_vtk_output_of_quadratic_cells():

    for element, cell in ((ElementType.quadrilateral, 28), (ElementType.triangle, 22)):
        mesh = mesh2((1.0, 1.0), (2, 2), element)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'field.vtu')
            write_vtu(path, mesh, {'p': xy_of(mesh)[:, 0]})
            assert os.path.getsize(path) > 0


# ----- exact solutions -----

def test_poiseuille_is_exact():

    """u quadratic and p linear belong to the spaces of Q2/Q1 and P2/P1: exact."""

    L, G, mu = 2.0, 3.0, 0.7
    for element in ELEMENT_TYPES:
        mesh = mesh2((L, 1.0), (4, 3), element)
        model = Model(name='poiseuille', dim=2, mesh=mesh, mat=material('X', miu=mu), psc=ff)
        flow = model.physics
        assert flow.taylor_hood and flow.Stab is None
        flow.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or y > 1 - 1e-9))
        flow.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x < 1e-9), G * L)
        flow.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x > L - 1e-9), 0.0)
        steady(model, ProblemType.linear)

        xy = xy_of(mesh)
        assert np.max(np.abs(model.sol['u'][:, 0] - G / (2 * mu) * xy[:, 1] * (1 - xy[:, 1]))) < 1e-12
        assert np.max(np.abs(model.sol['u'][:, 1])) < 1e-12
        assert np.max(np.abs(model.sol['p'] - G * (L - xy[:, 0]))) < 1e-11
        assert pressure_violation(model) < 1e-11


def test_hagen_poiseuille_also_on_the_axis():

    """The pressure on the axis was first order with PSPG. Taylor-Hood represents this flow exactly, axis included."""

    R, L, mu = 0.5, 2.0, 2.0
    for element in ELEMENT_TYPES:
        mesh = mesh2((R, L), (4, 16), element)
        model = Model(name='pipe', dim=2, mesh=mesh, mat=material('X', miu=mu), psc=ff, axisymmetric=True)
        flow = model.physics
        inlet = mesh.boundary_edges(lambda x, y: y < 1e-9)
        flow.addBC_VelocityEdges(inlet, lambda r, z: (0.0, 1 - (r / R) ** 2))
        flow.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: x > R - 1e-9))
        flow.addBC_SymmetryEdges(mesh.boundary_edges(lambda x, y: x < 1e-9))
        flow.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: y > L - 1e-9), 0.0)
        steady(model, ProblemType.linear)

        xy = xy_of(mesh)
        q = -flow.flowRate(inlet)
        grad = 8 * mu * q / (np.pi * R ** 4)
        assert abs(q - np.pi * R ** 2 / 2) < 1e-12
        assert np.max(np.abs(model.sol['p'] - grad * (L - xy[:, 1]))) / (grad * L) < 1e-12
        assert np.max(np.abs(model.sol['u'][:, 1] - (1 - (xy[:, 0] / R) ** 2))) < 1e-12


def test_kovasznay_orders():

    """Third order in u and second order in p, with the pressure linear in every element."""

    for element in ELEMENT_TYPES:
        results = [kovasznay(n, element) for n in (12, 24, 48)]
        eu, ep = np.array([r[0] for r in results]), np.array([r[1] for r in results])
        ru, rp = np.log2(eu[:-1] / eu[1:]), np.log2(ep[:-1] / ep[1:])
        print(f"   {element.name}: u {['%.1e' % e for e in eu]} rates {['%.2f' % r for r in ru]}; "
              f"p {['%.1e' % e for e in ep]} rates {['%.2f' % r for r in rp]}")
        assert ru[-1] > 2.6 and rp[-1] > 1.9
        assert pressure_violation(results[0][2]) < 1e-10


def test_lid_driven_cavity_against_ghia():

    for Re, n, tol in ((100.0, 16, 0.01), (1000.0, 32, 0.012)):
        for element in ELEMENT_TYPES:
            model, mesh = cavity(n, Re, element)
            steady(model, globalization='pseudo-transient')
            xy = xy_of(mesh)
            line = np.abs(xy[:, 0] - 0.5) < 1e-9
            order = np.argsort(xy[line, 1])
            err = max(abs(np.interp(y, xy[line, 1][order], model.sol['u'][line, 0][order]) - v) for y, v in GHIA[Re].items())
            print(f"   Re = {Re:.0f}, {n}x{n} {element.name}: {model.newton_iterations} iterations, largest error to Ghia {err:.4f}")
            assert model.nonlinear_converged and err < tol


def test_schafer_turek_on_curved_elements():

    """Steady benchmark 2D-1 (Re = 20): C_D, C_L and the pressure difference inside the published ranges."""

    u_max, mu = 0.3, 1e-3
    u_mean = 2.0 / 3.0 * u_max
    mesh = schafer_turek_mesh(0.025, 0.006)
    assert mesh.element_type == 'tri6'
    cylinder = mesh.boundary_edges('cylinder')

    # Midside nodes on the circle: the elements are curved
    xy = xy_of(mesh)
    radius = [np.hypot(*(xy[list(edge)] - (0.2, 0.2)).T) for edge in cylinder]
    assert np.allclose(radius, 0.05, atol=1e-10)

    model = Model(name='cylinder', dim=2, mesh=mesh, mat=material('X', miu=mu, rho=1.0), psc=ff)
    flow = model.physics
    flow.setInertia(True)
    flow.addBC_VelocityEdges(mesh.boundary_edges('inlet'), lambda x, y: (4 * u_max * y * (0.41 - y) / 0.41 ** 2, 0.0))
    flow.addBC_NoSlipEdges(mesh.boundary_edges('walls'))
    flow.addBC_NoSlipEdges(cylinder)
    flow.addBC_OutletEdges(mesh.boundary_edges('outlet'), 0.0, fully_developed=False)
    steady(model)

    fx, fy = flow.boundaryForce(cylinder)
    coefficient = 2.0 / (u_mean ** 2 * 0.1)

    def p_at(x0, y0):
        near = np.argsort(np.hypot(xy[:, 0] - x0, xy[:, 1] - y0))[:12]
        dx, dy = xy[near, 0] - x0, xy[near, 1] - y0
        return np.linalg.lstsq(np.c_[np.ones(12), dx, dy, dx ** 2, dy ** 2, dx * dy], model.sol['p'][near], rcond=None)[0][0]

    dp = p_at(0.15, 0.2) - p_at(0.25, 0.2)
    print(f"   {3 * len(xy)} unknowns: C_D {coefficient * fx:.4f}, C_L {coefficient * fy:.4f}, dp {dp:.4f}")
    assert 5.57 <= coefficient * fx <= 5.59
    assert 0.0104 <= coefficient * fy <= 0.0110
    assert 0.1172 <= dp <= 0.1176


def test_spurious_currents_with_buoyancy_on_a_graded_mesh():

    """Buoyancy that a linear pressure cannot balance: on a uniform mesh the moments of neighbour elements cancel and the
    fluid stays at rest, on a graded one small currents remain and vanish at third order. Taylor-Hood is not
    pressure-robust."""

    def stratified(n, element, graded):
        mesh = mesh2((1.0, 1.0), (n, n), element)
        if graded:
            for node in mesh.NL:
                node.coor = (node.coor[0], node.coor[1] ** 1.5)
        model = Model(name='stratified', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.0, Cp=1.0, k=1.0), psc=[ff, ht])
        model.physics.fluid.setInertia(True)
        model.physics.fluid.setBoussinesq((0.0, -1e4), beta=1.0, reference=0.5)
        model.physics.setVelocity('u')
        model.physics.fluid.addBC_NoSlipEdges(mesh.boundary_edges(None))
        model.physics.heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), 1.0)
        model.physics.heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: y < 1e-9), 0.0)
        steady(model)
        return np.max(np.abs(model.sol['u']))

    for element in ELEMENT_TYPES:
        assert stratified(10, element, graded=False) < 1e-10
        coarse, fine = stratified(10, element, True), stratified(20, element, True)
        print(f"   {element.name}, graded: max |u| {coarse:.1e}, {fine:.1e}, rate {np.log2(coarse / fine):.2f}")
        assert coarse > 1e-4 and np.log2(coarse / fine) > 2.5


# ----- polymer dies -----

def test_plane_contraction_entrance_loss():

    """Couette correction of the 4:1 contraction, creeping Newtonian flow. A coarse Taylor-Hood mesh beats a much finer
    equal-order one; the reference is the converged Taylor-Hood value (0.3740 with 402 531 unknowns, 0.3754 and falling
    with 397 545 of equal order)."""

    H, h, length, mu, q = 4.0e-3, 1.0e-3, 16.0e-3, 1000.0, 1.0e-6

    def correction(lc, order):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            geo = gmsh.model.geo
            points = [(-length, 0, lc), (length, 0, lc), (length, h, lc), (0, h, 0.15 * lc), (0, H, lc), (-length, H, lc)]
            tags = [geo.addPoint(x, y, 0, size) for x, y, size in points]
            lines = [geo.addLine(tags[i], tags[(i + 1) % 6]) for i in range(6)]
            geo.addPlaneSurface([geo.addCurveLoop(lines)])
            geo.synchronize()
            for name, group in (('axis', lines[:1]), ('outlet', lines[1:2]), ('walls', lines[2:5]), ('inlet', lines[5:])):
                gmsh.model.addPhysicalGroup(1, group, name=name)
            gmsh.model.mesh.generate(2)
            if order == 2:
                gmsh.model.mesh.setOrder(2)
            mesh = Mesh.from_gmsh_model()
        finally:
            gmsh.finalize()

        model = Model(name='contraction', dim=2, mesh=mesh, mat=material('X', miu=mu), psc=ff)
        flow = model.physics
        flow.addBC_OutletEdges(mesh.boundary_edges('outlet'), 0.0)
        flow.addBC_SymmetryEdges(mesh.boundary_edges('axis'))
        flow.addBC_VelocityEdges(mesh.boundary_edges('inlet'), lambda x, y: (1.5 * q / H * (1 - (y / H) ** 2), 0.0))
        flow.addBC_NoSlipEdges(mesh.boundary_edges('walls'))
        steady(model, ProblemType.linear)

        rate = flow.flowRate(mesh.boundary_edges('outlet'))
        dp = model.sol['p'][np.unique(np.ravel(mesh.boundary_edges('inlet')))].mean()
        loss = dp - 3 * mu * rate / H ** 3 * length - 3 * mu * rate / h ** 3 * length
        return loss / (2 * 3 * mu * rate / h ** 2), 3 * mesh.getNoN()

    reference = 0.3740
    coarse, n_coarse = correction(h / 2, 2)
    fine, n_fine = correction(h / 8, 1)
    print(f"   C: Taylor-Hood {coarse:.4f} with {n_coarse} unknowns, equal order {fine:.4f} with {n_fine}")
    assert n_fine > 3 * n_coarse
    assert abs(coarse - reference) < 0.002 and abs(coarse - reference) < abs(fine - reference)


def test_newton_stops_at_the_round_off_floor():

    """A linear problem in the nonlinear solver: Taylor-Hood reaches a relative residual of about 1e-9 in one iteration
    and cannot go below it. It used to go on to the 200 iterations and report no convergence."""

    mesh = mesh2((20.0e-3, 1.0e-3), (20, 4), ElementType.quadrilateral)
    model = Model(name='slit', dim=2, mesh=mesh, mat=material('X', miu=1000.0, rho=750.0, Cp=2700.0, k=0.17), psc=[ff, ht])
    fluid, heat = model.physics.fluid, model.physics.heat
    inlet = mesh.boundary_edges(lambda x, y: x < 1e-12)
    wall = mesh.boundary_edges(lambda x, y: y > 1.0e-3 - 1e-12)
    fluid.addBC_VelocityEdges(inlet, lambda x, y: (0.15 * (1 - (y / 1.0e-3) ** 2), 0.0))
    fluid.addBC_SymmetryEdges(mesh.boundary_edges(lambda x, y: y < 1e-12))
    fluid.addBC_NoSlipEdges(wall)
    fluid.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x > 20.0e-3 - 1e-12), 0.0)
    heat.addBC_TemperatureEdges(inlet, 230.0)
    heat.addBC_TemperatureEdges(wall, 230.0)
    model.physics.setVelocity('u')
    model.physics.setStabilization('SUPG')
    model.physics.initField('T', 230.0)
    steady(model)
    print(f"   {model.newton_iterations} iterations")
    assert model.nonlinear_converged and model.newton_iterations < 15


# ----- the rest of the flow physics -----

def test_tangent_with_every_term():

    """Inertia, a viscosity law, viscous heating, buoyancy and convected heat, against finite differences."""

    for element in ELEMENT_TYPES:
        mesh = mesh2((1.0, 1.0), (2, 2), element)
        model = Model(name='tangent', dim=2, mesh=mesh, mat=material('X', miu=0.1, rho=1.0, k=1.0, Cp=1.0), psc=[ff, ht])
        fluid, heat = model.physics.fluid, model.physics.heat
        fluid.setInertia(True)
        fluid.setViscosityLaw(CarreauYasuda(mu0=1.0, lam=1.0, n=0.5))
        fluid.setViscousHeating(True)
        fluid.setBoussinesq((0.0, -9.8), 0.01, 0.5)
        heat.setVelocity('u')
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)

        system = model._system()
        n = system.layout.ndof
        x = np.random.default_rng(4).normal(scale=0.5, size=n)
        for term in system._terms:
            if hasattr(term, 'tau_state'):
                term.tau_state = x.copy()

        J, _ = system.residual_and_tangent(x)
        FD = np.zeros((n, n))
        for k in range(n):
            e = np.zeros(n)
            e[k] = 1e-6
            FD[:, k] = (system.residual(x + e) - system.residual(x - e)) / 2e-6
        err = np.max(np.abs(J.toarray() - FD)) / np.max(np.abs(FD))
        assert err < 1e-7, (element, err)


def test_power_law_channel():

    K, n, G, L = 2.0, 0.5, 4.0, 0.5
    for element, tol in ((ElementType.quadrilateral, 2e-4), (ElementType.triangle, 5e-4)):
        mesh = mesh2((L, 1.0), (2, 16), element)
        model = Model(name='channel', dim=2, mesh=mesh, mat=material('X', miu=1.0), psc=ff)
        flow = model.physics
        flow.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x < 1e-9), G * L)
        flow.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x > L - 1e-9), 0.0)
        flow.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or y > 1 - 1e-9))
        steady(model, ProblemType.linear)
        flow.setViscosityLaw(PowerLaw(K, n, gamma_min=1e-4))
        steady(model)

        xy = xy_of(mesh)
        s, m = np.abs(xy[:, 1] - 0.5), (n + 1) / n
        exact = n / (n + 1) * (G / K) ** (1 / n) * (0.5 ** m - s ** m)
        err = np.max(np.abs(model.sol['u'][:, 0] - exact)) / exact.max()
        assert model.nonlinear_converged and err < tol, (element, err)
        assert np.all(np.isfinite(model.sol['viscosity']))


def test_transient_reaches_the_steady_state():

    for element in ELEMENT_TYPES:
        model, _ = cavity(8, 100.0, element)
        steady(model)
        u, p = model.sol['u'].copy(), model.sol['p'].copy()

        model, _ = cavity(8, 100.0, element)
        model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, timeStep=0.5, totalTime=40.0,
                                  timeScheme='BDF2')
        model.solve()
        assert np.max(np.abs(model.sol['u'][..., -1] - u)) < 1e-6
        assert np.max(np.abs(model.sol['p'][..., -1] - p)) < 1e-5


def test_pressure_reference_and_errors():

    model, mesh = cavity(3, 10.0, ElementType.quadrilateral)
    flow = model.physics

    try:
        flow.setStabilization('PSPG')
    except ValueError:
        pass
    else:
        raise AssertionError("PSPG must be refused with Taylor-Hood")

    midside = model.geometry.conn[0, 4]
    try:
        flow.setPressureReference(node=int(midside))
    except ValueError:
        pass
    else:
        raise AssertionError("the pressure cannot be fixed at a midside node")

    xy = xy_of(mesh)
    flow.setPressureReference(point=xy[midside], value=2.0)                       # goes to a vertex
    assert flow.pressure_reference[0] in model.geometry.conn[:, :4] and flow.pressure_reference[1] == 2.0
    steady(model)
    assert abs(model.sol['p'][flow.pressure_reference[0]] - 2.0) < 1e-12


def test_equal_order_meshes_stay_as_they_were():

    mesh = mesh2((1.0, 1.0), (2, 2), ElementType.quadrilateral, order=1)
    model = Model(name='q4', dim=2, mesh=mesh, mat=material('X', miu=1.0), psc=ff)
    assert mesh.element_type == 'quad4' and not model.physics.taylor_hood and model.physics.Stab == 'PSPG'
    assert model.geometry.order == 2


if __name__ == "__main__":
    tests = [test_shape_functions_and_quadrature, test_quadratic_mesh_and_geometry, test_vtk_output_of_quadratic_cells,
             test_poiseuille_is_exact, test_hagen_poiseuille_also_on_the_axis, test_kovasznay_orders,
             test_lid_driven_cavity_against_ghia, test_schafer_turek_on_curved_elements,
             test_spurious_currents_with_buoyancy_on_a_graded_mesh, test_plane_contraction_entrance_loss,
             test_newton_stops_at_the_round_off_floor, test_tangent_with_every_term, test_power_law_channel,
             test_transient_reaches_the_steady_state, test_pressure_reference_and_errors,
             test_equal_order_meshes_stay_as_they_were]
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
