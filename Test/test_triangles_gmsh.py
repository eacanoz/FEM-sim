"""Linear triangles (T3) and meshes imported from gmsh: non-rectangular geometry.

Run from the project root:  python Test/test_triangles_gmsh.py
"""

import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.Pre_processing.Elements import ELEMENTS
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
from Source.Physics.MassTransfer import mt
from Source.Simulation.Discretization import Discretization
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers


def configure(model, kind=ProblemType.linear):
    model.solverConfiguration(Study=StudyType.steady_state, Type=kind, Method=SolverType.direct,
                              Solver=DirectSolvers.PARDISO)


def triangle_mesh(nx, ny, Lx=1.0, Ly=1.0):
    mesh = Mesh()
    mesh.generate2DMesh((Lx, Ly), (nx, ny), ElementType.triangle)
    return mesh


def nodes_xy(mesh):
    return np.array([node.coor for node in mesh.NL])


# ---------------------------------------------------------------------------------------------
# Reference element and geometry
# ---------------------------------------------------------------------------------------------

def test_tri3_reference_element():
    tri = ELEMENTS['tri3']
    points, weights = tri.quadrature(2)
    N, dN = tri.shape(points)

    assert np.allclose(N.sum(axis=1), 1.0) and np.allclose(dN.sum(axis=1), 0.0)
    assert np.allclose(N @ tri.nodes, points)                        # nodes reproduce the coordinates
    assert abs(weights.sum() - 0.5) < 1e-15                          # area of the reference triangle

    # The 3-point rule integrates degree-2 polynomials exactly: int xi^2 = 1/12, int xi eta = 1/24
    assert abs(np.sum(weights * points[:, 0] ** 2) - 1 / 12) < 1e-15
    assert abs(np.sum(weights * points[:, 0] * points[:, 1]) - 1 / 24) < 1e-15

    one, w1 = tri.quadrature(1)
    assert abs(w1.sum() - 0.5) < 1e-15 and np.allclose(one, 1 / 3)


def test_structured_triangle_mesh_geometry():
    mesh = triangle_mesh(5, 4, Lx=2.0, Ly=1.5)
    disc = Discretization(mesh)

    assert mesh.element_type == 'tri3' and disc.conn.shape == (40, 3)
    assert abs(disc.detJw.sum() - 3.0) < 1e-12

    # Positive orientation of every element
    xy = disc.node_xy[disc.conn]
    area2 = (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1])
    assert np.all(area2 > 0)

    # Boundary: perimeter, outward normals (closed-surface identities)
    fs = disc.facets(mesh.boundary_edges())
    assert abs(fs.detJw.sum() - 7.0) < 1e-12
    assert np.allclose(np.einsum('fq,fqa->a', fs.detJw, fs.normals), 0.0, atol=1e-12)
    assert abs(np.einsum('fq,fqa,fqa->', fs.detJw, fs.x, fs.normals) - 2 * 3.0) < 1e-12


# ---------------------------------------------------------------------------------------------
# Solutions on triangles
# ---------------------------------------------------------------------------------------------

def test_patch_test_linear_field_with_dirichlet_and_flux():

    """T = 2x + 3y is reproduced exactly (conductivity 2): Dirichlet on three sides, flux -k dT/dn = -4 on the right."""

    mesh = triangle_mesh(6, 5, Lx=2.0, Ly=1.0)
    model = Model(name='patch', dim=2, mesh=mesh, mat=material('X', k=2.0, rho=1, Cp=1), psc=ht)
    model.physics.Convection = False
    exact = lambda x, y: 2 * x + 3 * y

    for selector in (lambda x, y: x < 1e-12, lambda x, y: y < 1e-12, lambda x, y: y > 1 - 1e-12):
        model.physics.addBC_TemperatureEdges(mesh.boundary_edges(selector), exact)
    model.physics.addBC_HeatFluxEdges(mesh.boundary_edges(lambda x, y: x > 2 - 1e-12), -4.0)
    model.physics.initField('T', 0.0)
    configure(model)
    model.solve()

    xy = nodes_xy(mesh)
    assert np.max(np.abs(np.asarray(model.sol['T']) - exact(xy[:, 0], xy[:, 1]))) < 1e-10


def poisson_error(n):

    """-k lap T = f with T = sin(pi x) sin(pi y) on triangles; max nodal error."""

    k = 1.5
    u = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    mesh = triangle_mesh(n, n)
    model = Model(name='mms', dim=2, mesh=mesh, mat=material('X', k=k, rho=1, Cp=1), psc=ht)
    model.physics.Convection = False
    model.physics.F_const = lambda x, y: 2 * k * np.pi ** 2 * u(x, y)
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(), 0.0)
    model.physics.initField('T', 0.0)
    configure(model)
    model.solve()

    xy = nodes_xy(mesh)
    return np.max(np.abs(np.asarray(model.sol['T']) - u(xy[:, 0], xy[:, 1])))


def test_poisson_on_triangles_converges_with_order_two():
    errors = [poisson_error(n) for n in (8, 16, 32)]

    assert errors[0] / errors[1] > 3.3 and errors[1] / errors[2] > 3.3, errors
    assert errors[2] < 1e-3, errors


def test_reaction_diffusion_on_triangles_matches_the_analytic_profile():

    """D A'' = k A, A(0) = 1, insulated end at x = 1 and sides: A = cosh(m (1 - x)) / cosh(m), 2D triangles."""

    D, kr = 0.5, 20.0
    m = np.sqrt(kr / D)
    mesh = triangle_mesh(40, 4, Lx=1.0, Ly=0.2)
    model = Model(name='rd', dim=2, mesh=mesh, mat=material('X', k=1, rho=1, Cp=1), psc=mt)
    model.physics.Convection = False
    model.physics.setChemSpecies('A', 'A')
    model.physics.setDiffusivity('A', D)
    model.physics.addReaction({'A': -1}, k0=kr, E_R=0.0, T=300.0, order={'A': 1})
    model.physics.addBC_ConcentrationEdges(mesh.boundary_edges(lambda x, y: x < 1e-12), 'A', 1.0)
    model.physics.initField('A', 0.5)
    configure(model, ProblemType.nonlinear)
    model.solve()

    x = nodes_xy(mesh)[:, 0]
    assert np.max(np.abs(np.asarray(model.sol['A']) - np.cosh(m * (1 - x)) / np.cosh(m))) < 2e-3


def test_supg_works_on_triangles():
    Pe, n = 50.0, 12
    mesh = triangle_mesh(n, 3, Lx=1.0, Ly=0.25)

    def solve(stab):
        model = Model(name='cd', dim=2, mesh=mesh, mat=material('X', k=1.0 / Pe, rho=1, Cp=1), psc=ht)
        model.physics.setVelocity([1.0, 0.0])
        model.physics.setStabilization(stab)
        model.physics.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x < 1e-12), 0.0)
        model.physics.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x > 1 - 1e-12), 1.0)
        model.physics.initField('T', 0.0)
        configure(model)
        model.solve()
        return np.asarray(model.sol['T'])

    galerkin, supg = solve(None), solve('SUPG')
    overshoot = lambda T: max(T.max() - 1.0, -T.min(), 0.0)

    assert overshoot(supg) < 0.25 * overshoot(galerkin), (overshoot(supg), overshoot(galerkin))

    x = nodes_xy(mesh)[:, 0]
    exact = np.expm1(Pe * x) / np.expm1(Pe)
    assert np.max(np.abs(supg - exact)) < 0.2 and np.max(np.abs(supg - exact)) < np.max(np.abs(galerkin - exact))


# ---------------------------------------------------------------------------------------------
# gmsh: curved geometry
# ---------------------------------------------------------------------------------------------

R1, R2, T1, T2 = 1.0, 2.0, 100.0, 20.0


def annulus_quarter(lc, path=None):

    """Quarter of an annulus meshed with gmsh. Groups: inner, outer, sides (1D) and plate (2D)."""

    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.model.add('annulus')
        geo = gmsh.model.geo
        c = geo.addPoint(0, 0, 0, lc)
        p = [geo.addPoint(R1, 0, 0, lc), geo.addPoint(R2, 0, 0, lc), geo.addPoint(0, R2, 0, lc), geo.addPoint(0, R1, 0, lc)]
        lines = [geo.addLine(p[0], p[1]), geo.addCircleArc(p[1], c, p[2]), geo.addLine(p[2], p[3]), geo.addCircleArc(p[3], c, p[0])]
        surface = geo.addPlaneSurface([geo.addCurveLoop(lines)])
        geo.synchronize()

        gmsh.model.addPhysicalGroup(1, [lines[3]], name='inner')
        gmsh.model.addPhysicalGroup(1, [lines[1]], name='outer')
        gmsh.model.addPhysicalGroup(1, [lines[0], lines[2]], name='sides')
        gmsh.model.addPhysicalGroup(2, [surface], name='plate')
        gmsh.model.mesh.generate(2)

        if path:
            gmsh.write(path)

        return Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()


def solve_annulus(mesh):
    model = Model(name='annulus', dim=2, mesh=mesh, mat=material('X', k=3.0, rho=1, Cp=1), psc=ht)
    model.physics.Convection = False
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges('inner'), T1)
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges('outer'), T2)
    model.physics.initField('T', 50.0)
    configure(model)
    model.solve()

    xy = nodes_xy(mesh)
    r = np.hypot(xy[:, 0], xy[:, 1])
    exact = T1 + (T2 - T1) * np.log(r / R1) / np.log(R2 / R1)
    return np.max(np.abs(np.asarray(model.sol['T']) - exact))


def test_gmsh_annulus_matches_the_logarithmic_profile_and_converges():
    coarse, fine = annulus_quarter(0.25), annulus_quarter(0.125)
    e_coarse, e_fine = solve_annulus(coarse), solve_annulus(fine)

    assert coarse.element_type == 'tri3' and len(fine.EL) > 3 * len(coarse.EL)
    assert e_fine < 0.5 and e_coarse / e_fine > 2.5, (e_coarse, e_fine)

    # The mesh is the polygonal approximation of the quarter annulus
    area = Discretization(fine).detJw.sum()
    assert abs(area - np.pi * (R2 ** 2 - R1 ** 2) / 4) / (np.pi * (R2 ** 2 - R1 ** 2) / 4) < 0.01


def test_gmsh_groups_and_file_round_trip():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, 'annulus.msh')
        direct = annulus_quarter(0.25, path)
        loaded = Mesh.read_msh(path)

    assert sorted(loaded.groups) == ['inner', 'outer', 'sides'] and 'plate' in loaded.regions
    assert (loaded.getNoN(), loaded.getNoE()) == (direct.getNoN(), direct.getNoE())

    xy = nodes_xy(loaded)
    r = np.hypot(xy[:, 0], xy[:, 1])
    for name, radius in (('inner', R1), ('outer', R2)):
        nodes = {n for edge in loaded.boundary_edges(name) for n in edge}
        assert nodes and np.allclose(r[list(nodes)], radius, atol=1e-9), name

    # Every group edge is a boundary facet of the mesh; together they cover the whole boundary
    whole = {tuple(sorted(e)) for e in loaded.boundary_edges()}
    named = {tuple(sorted(e)) for name in loaded.groups for e in loaded.boundary_edges(name)}
    assert named == whole

    try:
        loaded.boundary_edges('nope')
    except KeyError as error:
        assert 'nope' in str(error)
    else:
        raise AssertionError("expected KeyError for an unknown group")


def test_gmsh_quadrilateral_mesh_runs_with_the_q4_element():
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add('rect')
        geo = gmsh.model.geo
        pts = [geo.addPoint(0, 0, 0), geo.addPoint(2, 0, 0), geo.addPoint(2, 1, 0), geo.addPoint(0, 1, 0)]
        lines = [geo.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        surface = geo.addPlaneSurface([geo.addCurveLoop(lines)])
        for line in lines:
            geo.mesh.setTransfiniteCurve(line, 9)
        geo.mesh.setTransfiniteSurface(surface)
        geo.mesh.setRecombine(2, surface)
        geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [lines[3]], name='left')
        gmsh.model.addPhysicalGroup(1, [lines[1]], name='right')
        gmsh.model.mesh.generate(2)
        mesh = Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()

    assert mesh.element_type == 'quad4' and mesh.getNoE() == 64
    assert abs(Discretization(mesh).detJw.sum() - 2.0) < 1e-12

    model = Model(name='q', dim=2, mesh=mesh, mat=material('X', k=1.0, rho=1, Cp=1), psc=ht)
    model.physics.Convection = False
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges('left'), 10.0)
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges('right'), 30.0)
    model.physics.initField('T', 0.0)
    configure(model)
    model.solve()

    x = nodes_xy(mesh)[:, 0]
    assert np.allclose(np.asarray(model.sol['T']), 10.0 + 10.0 * x, atol=1e-9)


def test_mixed_element_meshes_are_rejected():
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add('mixed')
        geo = gmsh.model.geo
        pts = [geo.addPoint(0, 0, 0, 0.3), geo.addPoint(1, 0, 0, 0.3), geo.addPoint(1, 1, 0, 0.3), geo.addPoint(0, 1, 0, 0.3)]
        lines = [geo.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        surface = geo.addPlaneSurface([geo.addCurveLoop(lines)])
        geo.mesh.setRecombine(2, surface)
        geo.synchronize()
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)       # simple recombination leaves some triangles
        gmsh.model.mesh.generate(2)

        types = gmsh.model.mesh.getElements(2)[0]
        if len(types) < 2:
            return                                                     # gmsh produced a pure mesh: nothing to reject

        try:
            Mesh.from_gmsh_model()
        except ValueError as error:
            assert 'un solo tipo' in str(error)
        else:
            raise AssertionError("expected ValueError for a mixed mesh")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    tests = [test_tri3_reference_element, test_structured_triangle_mesh_geometry,
             test_patch_test_linear_field_with_dirichlet_and_flux, test_poisson_on_triangles_converges_with_order_two,
             test_reaction_diffusion_on_triangles_matches_the_analytic_profile, test_supg_works_on_triangles,
             test_gmsh_annulus_matches_the_logarithmic_profile_and_converges, test_gmsh_groups_and_file_round_trip,
             test_gmsh_quadrilateral_mesh_runs_with_the_q4_element, test_mixed_element_meshes_are_rejected]
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
