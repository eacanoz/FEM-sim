"""Checks for the RANS turbulence in 3D (phase 5 of docs/PLAN_3D.md): the wall distance to faces, Spalart-Allmaras and k-omega SST.

Run from the project root:  python Test/test_3d_turbulence.py
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.Turbulence import sa, sst, wall_distance
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers

DNS_CENTRELINE = {180.0: 18.3, 395.0: 20.1}   # Moser, Kim & Mansour (1999), as in Test/test_turbulence.py


def reichardt(y_plus):
    k = 0.41
    return np.log(1 + k * y_plus) / k + 7.8 * (1 - np.exp(-y_plus / 11) - y_plus / 11 * np.exp(-y_plus / 3))


# ----- wall distance -----

def test_wall_distance_to_faces():
    """Distance to the six faces of a cube: exact at every node on hex8, tet4, hex27, tet10 (flat faces); to the wall of a sphere of
    gmsh, d = R - |x|: flat tet4 faces within their chordal error, the curved tet10 ones (split into four triangles, chords of half
    the length) about 4 times closer (measured: 3.3e-2 and 8.3e-3 with size 0.3)."""

    kinds = {'hex8': (ElementType.hexahedron, 1), 'tet4': (ElementType.tetrahedron, 1), 'hex27': (ElementType.hexahedron, 2),
             'tet10': (ElementType.tetrahedron, 2)}
    for name, (kind, order) in kinds.items():
        mesh = Mesh()
        mesh.generate3DMesh((1.0, 2.0, 1.5), (3, 4, 3), kind, order=order)
        xyz = np.array([n.coor for n in mesh.NL])
        d = wall_distance(mesh, mesh.boundary_faces())
        exact = np.min(np.stack([xyz[:, 0], 1 - xyz[:, 0], xyz[:, 1], 2 - xyz[:, 1], xyz[:, 2], 1.5 - xyz[:, 2]]), axis=0)
        assert np.max(np.abs(d - exact)) < 1e-12, (name, np.max(np.abs(d - exact)))

    from test_3d_foundation import gmsh_mesh
    errors = {}
    for order in (1, 2):
        mesh = gmsh_mesh(order, sphere=True, size=0.3)
        xyz = np.array([n.coor for n in mesh.NL])
        d = wall_distance(mesh, mesh.boundary_faces('wall'))
        errors[order] = np.max(np.abs(d - (1.0 - np.linalg.norm(xyz, axis=1))))
        print(f"   sphere, order {order}: largest error of the wall distance {errors[order]:.1e}")
    assert errors[1] < 5e-2 and errors[2] < errors[1] / 3


# ----- channel: 3D slab against 2D -----

def channel(dim, model_kind, re_tau, ny, beta, W=0.1):

    """Half channel 0 < y < 1 (wall y = 0, symmetry y = 1), x-independent, unit body force (u is u+, y Re_tau is y+); in 3D a slab of
    one cell in z with symmetric z faces."""

    mesh = Mesh()
    if dim == 2:
        mesh.generate2DMesh((0.1, 1.0), (2, ny), ElementType.quadrilateral)
    else:
        mesh.generate3DMesh((0.1, 1.0, W), (2, ny, 1), ElementType.hexahedron)
    for node in mesh.NL:                                              # tanh clustering at the wall
        c = list(node.coor)
        c[1] = 1.0 + np.tanh(beta * (c[1] - 1.0)) / np.tanh(beta)
        node.coor = tuple(c)
    xyz = np.array([n.coor for n in mesh.NL])

    model = Model(name='channel', dim=dim, mesh=mesh, mat=material('X', miu=1.0 / re_tau, rho=1.0),
                  psc=[ff, sa if model_kind == 'sa' else sst])
    fluid, turbulence = model.physics.fluid, model.physics.turbulence
    fluid.setInertia(True)
    fluid.setBodyForce([1.0] + [0.0] * (dim - 1))
    wall = mesh.boundary_facets(lambda *c: c[1] < 1e-9)
    fluid.addBC_NoSlipEdges(wall)
    fluid.addBC_SymmetryEdges(mesh.boundary_facets(lambda *c: c[1] > 1 - 1e-9))
    fluid.addBC_OutletEdges(mesh.boundary_facets(lambda *c: c[0] < 1e-9))
    fluid.addBC_OutletEdges(mesh.boundary_facets(lambda *c: c[0] > 0.1 - 1e-9))
    if dim == 3:
        fluid.addBC_SymmetryEdges(mesh.boundary_faces(lambda x, y, z: z < 1e-9 or z > W - 1e-9))
    turbulence.setWalls(wall)
    turbulence.coupleFlow()

    zeros = np.zeros((len(xyz), dim - 1))
    if model_kind == 'sa':
        fluid.var['u'].updateField(np.c_[reichardt(xyz[:, 1] * re_tau), zeros])
        turbulence.var['nut'].updateField(0.41 * xyz[:, 1] * (1 - xyz[:, 1]) + 1e-3)
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, Method=SolverType.direct,
                                  Solver=DirectSolvers.PARDISO)
    else:
        from test_sst import guess
        u, k, g = guess(xyz[:, 1], re_tau)
        fluid.var['u'].updateField(np.c_[u, zeros])
        turbulence.var['k'].updateField(k)
        turbulence.var['ln_omega'].updateField(g)
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, Method=SolverType.direct,
                                  Solver=DirectSolvers.PARDISO, globalization='pseudo-transient')
    model.solve()
    return model, xyz


def profile(model, xyz):
    column = (np.abs(xyz[:, 0]) < 1e-12) & (np.abs(xyz[:, 2]) < 1e-12 if xyz.shape[1] == 3 else True)
    order = np.argsort(xyz[column, 1])
    return xyz[column, 1][order], np.asarray(model.sol['u'])[column, 0][order]


def test_channel_slab_matches_2d_and_the_dns():
    """Spalart-Allmaras and k-omega SST at Re_tau = 180: the 3D slab gives the u+ profile of the 2D channel and its centreline
    velocity against the DNS (SA within 2 %, SST within 5 %, as the 2D suites). Not to round-off: equal order with PSPG uses the
    element size (side of the square of the same area in 2D, of the cube of the same volume in 3D), which a graded mesh cannot make
    equal in every cell (measured with SA: 2.3e-6 in the profile, 7e-7 of cross flow)."""

    re_tau, ny, beta = 180.0, 48, 2.5
    for kind, tolerance in (('sa', 0.02), ('sst', 0.05)):
        start = time.perf_counter()
        results = {dim: channel(dim, kind, re_tau, ny, beta) for dim in (2, 3)}
        (m2, x2), (m3, x3) = results[2], results[3]
        y2, u2 = profile(m2, x2)
        y3, u3 = profile(m3, x3)
        assert np.allclose(y2, y3)
        diff = np.max(np.abs(u3 - u2)) / u2.max()
        cross = np.max(np.abs(np.asarray(m3.sol['u'])[:, 1:]))
        print(f"   {kind.upper()}: U_c+ 3D {u3.max():.3f}, 2D {u2.max():.3f} (DNS {DNS_CENTRELINE[re_tau]}), profile 3D - 2D "
              f"{diff:.1e}, cross flow {cross:.1e}, Newton {m3.newton_iterations} ({time.perf_counter() - start:.1f} s)")
        assert m3.nonlinear_converged in (None, True) and diff < 1e-4 and cross < 1e-4
        assert abs(u3.max() / DNS_CENTRELINE[re_tau] - 1) < tolerance


def test_tangent_in_3d():
    """The Newton tangent of the coupled flow and turbulence in 3D (equal order hex8 with PSPG/SUPG/LSIC and Taylor-Hood hex27) against
    central finite differences, for SA and SST, with tau frozen at the state."""

    for kind in ('sa', 'sst'):
        for order in (1, 2):
            mesh = Mesh()
            mesh.generate3DMesh((1.0, 1.0, 1.0), (1, 2, 1), ElementType.hexahedron, order=order)
            model = Model(name='tangent', dim=3, mesh=mesh, mat=material('X', miu=0.01, rho=1.0), psc=[ff, sa if kind == 'sa' else sst])
            fluid, turbulence = model.physics.fluid, model.physics.turbulence
            fluid.setInertia(True)
            wall = mesh.boundary_faces(lambda x, y, z: y < 1e-9)
            fluid.addBC_NoSlipEdges(wall)
            turbulence.setWalls(wall)
            turbulence.coupleFlow()
            model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
            system = model._system()
            layout = system.layout
            rng = np.random.default_rng(2)
            x = rng.normal(scale=0.5, size=layout.ndof)
            if kind == 'sa':
                x[layout.dofs('nut')[:, 0]] = 0.02 + 0.01 * rng.random(layout.n_nodes)
            else:
                x[layout.dofs('k')[:, 0]] = 0.05 + 0.02 * rng.random(layout.n_nodes)
                x[layout.dofs('ln_omega')[:, 0]] = 2.0 + 0.3 * rng.normal(size=layout.n_nodes)
            x[layout.dofs('wall_distance')[:, 0]] = np.asarray(turbulence.var['wall_distance'].values)
            for term in system._terms:
                if hasattr(term, 'tau_state'):
                    term.tau_state = x.copy()
            J = system._assemble(x)[0].toarray()
            FD = np.zeros_like(J)
            for k in range(layout.ndof):
                e = np.zeros(layout.ndof)
                e[k] = 1e-6 * max(1.0, abs(x[k]))
                FD[:, k] = (system._assemble(x + e)[1] - system._assemble(x - e)[1]) / (2 * e[k])
            free = ~system.is_dirichlet
            err = np.max(np.abs(J - FD)[:, free] / (np.max(np.abs(FD[:, free]), axis=0) + 1e-300))
            print(f"   {kind.upper()} {'hex27' if order == 2 else 'hex8'}: {layout.ndof} unknowns, largest tangent error {err:.1e}")
            assert err < 1e-5, (kind, order, err)


# ----- square duct -----

def test_square_duct_spalart_allmaras():
    """Fully developed turbulent flow in a square duct with Spalart-Allmaras on Taylor-Hood hex27 (quarter duct, walls y = 0 and z = 0,
    symmetry at y = z = 1, unit body force, nu = 1/180): a linear eddy-viscosity model has v = w = 0 as exact solution (no secondary
    flow of Prandtl's second kind, which needs an anisotropic model), the wall reactions balance the body force, and the velocity on
    the symmetry plane z = 1 stays below that of a channel (the corner slows the flow). Equal order (hex8 with PSPG, 24^2 cells, same
    unknowns) gives the same u_max (14.07 against 14.05) but a secondary flow of 4e-5 and a force 0.4 % off: an effect of its
    stabilization, measured apart."""

    re, n, beta = 180.0, 12, 2.0
    mesh = Mesh()
    mesh.generate3DMesh((0.1, 1.0, 1.0), (1, n, n), ElementType.hexahedron, order=2)
    stretch = lambda e: 1.0 + np.tanh(beta * (e - 1.0)) / np.tanh(beta)
    for node in mesh.NL:
        x, y, z = node.coor
        node.coor = (x, stretch(y), stretch(z))
    xyz = np.array([n.coor for n in mesh.NL])

    model = Model(name='duct', dim=3, mesh=mesh, mat=material('X', miu=1.0 / re, rho=1.0), psc=[ff, sa])
    fluid, turbulence = model.physics.fluid, model.physics.turbulence
    fluid.setInertia(True)
    fluid.setBodyForce((1.0, 0.0, 0.0))
    walls = mesh.boundary_faces(lambda x, y, z: y < 1e-9 or z < 1e-9)
    fluid.addBC_NoSlipEdges(walls)
    fluid.addBC_SymmetryEdges(mesh.boundary_faces(lambda x, y, z: y > 1 - 1e-9))
    fluid.addBC_SymmetryEdges(mesh.boundary_faces(lambda x, y, z: z > 1 - 1e-9))
    fluid.addBC_OutletEdges(mesh.boundary_faces(lambda x, y, z: x < 1e-9))
    fluid.addBC_OutletEdges(mesh.boundary_faces(lambda x, y, z: x > 0.1 - 1e-9))
    turbulence.setWalls(walls)
    turbulence.coupleFlow()
    d = np.minimum(xyz[:, 1], xyz[:, 2])
    fluid.var['u'].updateField(np.c_[reichardt(d * re), np.zeros((len(xyz), 2))])
    turbulence.var['nut'].updateField(0.41 * d * (1 - d) + 1e-3)
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
    start = time.perf_counter()
    model.solve()
    seconds = time.perf_counter() - start
    assert model.nonlinear_converged

    U = np.asarray(model.sol['u'])
    secondary = np.max(np.abs(U[:, 1:])) / U[:, 0].max()
    force = np.asarray(fluid.boundaryForce(walls))                  # force of the fluid on the walls
    body = 1.0 * 0.1 * 1.0 * 1.0                                     # body force times the volume
    plane = (np.abs(xyz[:, 0]) < 1e-12) & (np.abs(xyz[:, 2] - 1) < 1e-12)
    u_plane = U[plane, 0]
    print(f"   {model.layout.ndof} unknowns, Newton {model.newton_iterations}, {seconds:.1f} s: u_max {U[:, 0].max():.2f}, "
          f"secondary flow {secondary:.1e} of it, wall force {force[0]:.8f} against the body force {body:.8f}, centre of the "
          f"symmetry plane {u_plane.max():.2f} (channel at Re_tau 180: 18.3)")
    assert secondary < 1e-10
    assert abs(force[0] - body) < 1e-8 * body
    assert 12.0 < u_plane.max() < 18.3


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
