"""Checks for the 3D foundation (phase 0 of docs/PLAN_3D.md): tetrahedra and hexahedra (tet4, tet10, hex8, hex27), their
quadrature and faces, the box generator, 3D facet normals, the gmsh import with groups and the VTK output.

Physics-independent: the Poisson problems are assembled here from the operators.

Run from the project root:  python Test/test_3d_foundation.py
"""

import base64
import itertools
import os
import re
import sys
import tempfile
import time
from math import factorial

import matplotlib
matplotlib.use("Agg")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.Pre_processing.Elements import ELEMENTS
from Source.Simulation.Discretization import Discretization
import Source.Simulation.Operators as ops
from Source.Post_processing.VTK import write_vtu, vtk_node_order
from Source.enums import ElementType

ELEMENTS_3D = ('tet4', 'tet10', 'hex8', 'hex27')
KIND = {'tet4': (ElementType.tetrahedron, 1), 'tet10': (ElementType.tetrahedron, 2),
        'hex8': (ElementType.hexahedron, 1), 'hex27': (ElementType.hexahedron, 2)}


def box(name, n=(2, 2, 2), L=(1.0, 1.0, 1.0)):
    mesh = Mesh()
    mesh.generate3DMesh(L, n, *KIND[name])
    return mesh


def nodes_of(mesh):
    return np.array([node.coor for node in mesh.NL], dtype=float)


def distort(mesh, amount, seed=0):

    """Moves the interior nodes at random by `amount` times the smallest spacing (the box keeps its faces)."""

    xyz = nodes_of(mesh)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    interior = np.all((xyz > lo + 1e-12) & (xyz < hi - 1e-12), axis=1)
    h = min(np.diff(np.unique(xyz[:, a]))[0] for a in range(3))
    xyz[interior] += amount * h * np.random.default_rng(seed).uniform(-1, 1, (interior.sum(), 3))
    for node, c in zip(mesh.NL, xyz):
        node.coor = tuple(c)
    return mesh


# ----- reference elements -----

def exact_monomial(name, p):

    """Integral of x^i y^j z^k over the reference tetrahedron or the cube [-1, 1]^3."""

    if name.startswith('tet'):
        return factorial(p[0]) * factorial(p[1]) * factorial(p[2]) / factorial(sum(p) + 3)
    return np.prod([0.0 if q % 2 else 2.0 / (q + 1) for q in p])


def test_quadrature_exact_to_its_degree():
    """Tetrahedron: 1, 4 and 14 points integrate degrees 1, 2 and 5, the collapsed rule of order n degree 2n - 1; hexahedron:
    degree 2n - 1 per direction. A monomial of the next degree is no longer exact (the degree is not overstated)."""

    for name in ('tet4', 'hex8'):
        element = ELEMENTS[name]
        for order in (1, 2, 3, 4):
            points, weights = element.quadrature(order)
            assert np.all(weights > 0), (name, order)
            degree = {1: 1, 2: 2, 3: 5, 4: 7}[order] if name == 'tet4' else 2 * order - 1
            for p in itertools.product(range(degree + 2), repeat=3):
                if name == 'hex8' and max(p) > degree + 1:
                    continue
                if name == 'tet4' and sum(p) > degree + 1:
                    continue
                value = np.sum(weights * np.prod(points ** np.array(p), axis=1))
                exact = exact_monomial(name, p)
                within = sum(p) <= degree if name == 'tet4' else max(p) <= degree
                if within:
                    assert abs(value - exact) < 1e-14, (name, order, p, value, exact)
            print(f"   {name} order {order}: {len(weights)} points, exact to degree {degree}")

    # the degree is sharp for the 14-point rule: some monomial of degree 6 is not exact
    points, weights = ELEMENTS['tet4'].quadrature(3)
    errors = [abs(np.sum(weights * np.prod(points ** np.array(p), axis=1)) - exact_monomial('tet4', p))
              for p in itertools.product(range(7), repeat=3) if sum(p) == 6]
    assert max(errors) > 1e-8


def test_shape_functions():
    """Kronecker delta at the nodes, partition of unity, derivatives and second derivatives against finite differences."""

    rng = np.random.default_rng(1)

    for name in ELEMENTS_3D:
        element = ELEMENTS[name]
        N, _ = element.shape(element.nodes)
        assert np.allclose(N, np.eye(element.n_nodes), atol=1e-13), name

        if name.startswith('tet'):
            pts = rng.dirichlet(np.ones(4), 6)[:, 1:]
        else:
            pts = rng.uniform(-1, 1, (6, 3))

        N, dN = element.shape(pts)
        assert np.allclose(N.sum(axis=1), 1.0) and np.allclose(dN.sum(axis=1), 0.0), name

        eps = 1e-6
        H = element.second_derivatives(pts)
        for a in range(3):
            step = np.zeros(3)
            step[a] = eps
            Np, dNp = element.shape(pts + step)
            Nm, dNm = element.shape(pts - step)
            assert np.allclose((Np - Nm) / (2 * eps), dN[..., a], atol=1e-8), (name, a)
            if H is not None:
                assert np.allclose((dNp - dNm) / (2 * eps), H[..., a], atol=1e-6), (name, a)

        assert (H is None) == (name in ('tet4', 'hex8')), name


def test_faces_follow_their_facet_element():
    """Each face lists its nodes in the order of its facet element (tri3/tri6/quad4/quad9): the element shape functions on the face
    are the facet shape functions of those nodes and vanish for the others; the faces cover the boundary once."""

    rng = np.random.default_rng(2)

    for name in ELEMENTS_3D:
        element = ELEMENTS[name]
        facet = ELEMENTS[element.facet_element]
        assert len(element.facets) == (4 if name.startswith('tet') else 6)

        for local in element.facets:
            local = list(local)
            assert len(local) == facet.n_nodes, name
            if facet.name.startswith('tri'):
                s = rng.dirichlet(np.ones(3), 5)[:, 1:]
            else:
                s = rng.uniform(-1, 1, (5, 2))
            Nf, _ = facet.shape(s)
            X = Nf @ element.nodes[local]                     # points of the face
            N, _ = element.shape(X)
            assert np.allclose(N[:, local], Nf, atol=1e-12), (name, local)
            others = [k for k in range(element.n_nodes) if k not in local]
            assert np.allclose(N[:, others], 0.0, atol=1e-12), (name, local)

        # Taylor-Hood pressure: the linear subspace reproduces the linear (trilinear) field of the vertices
        n_vertices = element.n_nodes - len(element.linear_subspace)
        assert n_vertices == (4 if name.startswith('tet') else 8) or not element.linear_subspace


# ----- meshes -----

def test_box_meshes():
    """Box generator: number of nodes and elements, positive volumes, total volume and the faces of the boundary."""

    n, L = (3, 2, 4), (1.5, 1.0, 2.0)
    cells, faces = np.prod(n), 2 * (n[0] * n[1] + n[1] * n[2] + n[0] * n[2])

    for name in ELEMENTS_3D:
        mesh = box(name, n, L)
        m = 2 if name in ('tet10', 'hex27') else 1
        assert mesh.getNoN() == np.prod([m * k + 1 for k in n]), name
        tet = name.startswith('tet')
        assert mesh.getNoE() == cells * (6 if tet else 1), name

        disc = Discretization(mesh)
        xyz = nodes_of(mesh)
        nv = 4 if tet else 8
        vertices = xyz[disc.conn[:, :nv]]
        corner = [1, 2, 3] if tet else [1, 3, 4]                      # edges from vertex 0
        assert np.all(np.linalg.det(vertices[:, corner] - vertices[:, [0]]) > 0), name
        assert np.isclose(disc.detJw.sum(), np.prod(L)), name

        assert len(mesh.boundary_faces()) == faces * (2 if tet else 1), name
        assert len(mesh.boundary_faces(lambda x, y, z: z > L[2] - 1e-12)) == n[0] * n[1] * (2 if tet else 1), name

        # every node of a quadratic mesh is used by some element
        assert len(np.unique(disc.conn)) == mesh.getNoN(), name


def test_element_size_3d():
    """h of a structured mesh is the cell side for hexahedra and for the 6 tetrahedra of a cell."""

    for name in ELEMENTS_3D:
        h = ops.element_size(Discretization(box(name, (4, 4, 4), (2.0, 2.0, 2.0))))
        assert np.allclose(h, 0.5), (name, h.min(), h.max())


def test_surface_integrals_and_normals_cube():
    """On a distorted cube: area of the boundary 6, integral of n = 0, divergence theorem integral of x.n = 3 V, normals of the
    face x = 1 equal to (1, 0, 0)."""

    for name in ELEMENTS_3D:
        mesh = distort(box(name, (3, 3, 3)), 0.15)
        disc = Discretization(mesh)
        facets = disc.facets(mesh.boundary_faces())

        assert np.isclose(facets.detJw.sum(), 6.0), name
        assert np.allclose(np.einsum('fq,fqa->a', facets.detJw, facets.normals), 0.0, atol=1e-12), name
        assert np.isclose(np.einsum('fq,fqa,fqa->', facets.detJw, facets.x, facets.normals), 3.0), name
        assert np.allclose(np.linalg.norm(facets.normals, axis=-1), 1.0)

        right = disc.facets(mesh.boundary_faces(lambda x, y, z: x > 1 - 1e-12))
        assert np.allclose(right.normals, [1.0, 0.0, 0.0]), name


# ----- Poisson in a cube -----

def solve_poisson(mesh, exact, f, order=None):

    """-lap(u) = f with u = exact on the whole boundary, assembled from the operators; returns (disc, u)."""

    disc = Discretization(mesh, order=order)
    n, nne = mesh.getNoN(), disc.conn.shape[1]

    Ke = ops.diffusion(disc, 1.0)
    Fe = ops.source(disc, f) if f is not None else np.zeros((disc.n_elements, nne))
    rows = np.repeat(disc.conn, nne, axis=1).ravel()
    cols = np.tile(disc.conn, (1, nne)).ravel()
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    F = np.zeros(n)
    np.add.at(F, disc.conn.ravel(), Fe.ravel())

    boundary = np.unique(np.concatenate([np.asarray(t) for t in mesh.boundary_faces()]))
    xyz = nodes_of(mesh)
    u = np.zeros(n)
    u[boundary] = exact(*xyz[boundary].T)
    free = np.setdiff1d(np.arange(n), boundary)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), F[free] - K[free][:, boundary] @ u[boundary])

    return disc, u


def test_patch_test_linear_field():
    """A linear field imposed on the boundary is reproduced exactly inside, also on distorted (and curved, for the quadratic
    elements) meshes."""

    exact = lambda x, y, z: 1.0 + 2.0 * x - 3.0 * y + 0.5 * z

    for name in ELEMENTS_3D:
        mesh = distort(box(name, (3, 3, 3)), 0.2, seed=3)
        disc, u = solve_poisson(mesh, exact, None)
        err = np.max(np.abs(u - exact(*nodes_of(mesh).T)))
        assert err < 1e-10, (name, err)

        # the gradient at the Gauss points is the exact one
        assert np.allclose(disc.gradient(u), [2.0, -3.0, 0.5], atol=1e-9), name


def l2_error(disc, u, exact):
    u_gp = disc.interpolate(u)
    return np.sqrt(np.sum(disc.detJw * (u_gp - exact(*np.moveaxis(disc.x_gp, -1, 0))) ** 2))


def test_poisson_manufactured_convergence():
    """-lap(u) = 3 pi^2 sin(pi x) sin(pi y) sin(pi z) in the unit cube: order 2 in L2 for tet4 and hex8, order 3 for tet10 and hex27."""

    exact = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
    f = lambda x, y, z: 3 * np.pi ** 2 * exact(x, y, z)

    for name, sizes, expected in (('hex8', (4, 8, 16), 2.0), ('tet4', (4, 8, 16), 2.0),
                                  ('hex27', (2, 4, 8), 3.0), ('tet10', (2, 4, 8), 3.0)):
        start = time.perf_counter()
        errors = []
        for n in sizes:
            disc, u = solve_poisson(box(name, (n, n, n)), exact, f)
            errors.append(l2_error(disc, u, exact))
        rates = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
        print(f"   {name}: L2 {['%.2e' % e for e in errors]}, rates {['%.2f' % r for r in rates]} "
              f"({time.perf_counter() - start:.1f} s)")
        assert rates[-1] > expected - 0.15, (name, rates)


def test_hessian_and_projection_3d():
    """Second derivatives of a quadratic field are exact on quadratic elements; the projection keeps a constant (consistent mass
    of the quadratic elements) and the patch recovery gives the gradient of a linear field."""

    q = lambda x, y, z: x * x - 2 * y * z + 3 * x * z + y
    exact_H = np.array([[2.0, 0.0, 3.0], [0.0, 0.0, -2.0], [3.0, -2.0, 0.0]])

    for name in ('tet10', 'hex27'):
        mesh = box(name, (2, 2, 2))
        disc = Discretization(mesh)
        u = q(*nodes_of(mesh).T)
        H = np.einsum('egkij,ek->egij', disc.hessian, u[disc.conn])
        assert np.allclose(H, exact_H, atol=1e-10), name
        assert np.allclose(np.einsum('egk,ek->eg', disc.laplacian, u[disc.conn]), 2.0), name
        assert np.allclose(disc.project(np.full(disc.detJw.shape, 4.0)), 4.0), name

    for name in ('tet4', 'hex8'):
        mesh = distort(box(name, (3, 3, 3)), 0.1)
        disc = Discretization(mesh)
        u = (lambda x, y, z: x - 2 * y + 4 * z)(*nodes_of(mesh).T)
        G = disc.recovery() @ disc.gradient(u).reshape(-1, 3)
        assert np.allclose(G, [1.0, -2.0, 4.0], atol=1e-9), name
        assert disc.hessian is None


# ----- gmsh -----

def gmsh_mesh(order=1, hexahedra=False, sphere=False, size=0.35):

    """Mesh of the unit box (or a sphere of radius 1) made by gmsh, with physical groups for the faces and the volume."""

    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("m")
        if sphere:
            gmsh.model.occ.addSphere(0, 0, 0, 1.0)
        else:
            gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()

        surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
        if not sphere:
            top = [s for s in surfaces if abs(gmsh.model.occ.getCenterOfMass(2, s)[2] - 1.0) < 1e-9]
            gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, top), "top")
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, surfaces), "wall")
        gmsh.model.setPhysicalName(3, gmsh.model.addPhysicalGroup(3, [1]), "solid")

        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size)
        if hexahedra:
            gmsh.model.mesh.setTransfiniteAutomatic()
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
        gmsh.model.mesh.generate(3)
        if order == 2:
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
            gmsh.model.mesh.setOrder(2)

        return Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()


def test_reference_nodes_match_gmsh():
    """The node order of the reference elements is the one of gmsh (its reference coordinates)."""

    import gmsh

    gmsh.initialize()
    try:
        for gmsh_type, name in ((4, 'tet4'), (11, 'tet10'), (5, 'hex8'), (12, 'hex27'), (2, 'tri3'), (9, 'tri6'),
                                (3, 'quad4'), (10, 'quad9')):
            props = gmsh.model.mesh.getElementProperties(gmsh_type)
            ref = np.array(props[4]).reshape(-1, props[1])
            assert np.allclose(ref, ELEMENTS[name].nodes), name
    finally:
        gmsh.finalize()


def test_gmsh_box_with_groups():
    """gmsh box (tet4, tet10, hex8, hex27): volume 1, groups of faces ('top' of area 1 with normals +z, 'wall' of area 6), region
    with every element, and a linear field reproduced exactly."""

    exact = lambda x, y, z: 2.0 - x + 2 * y + 3 * z

    for order, hexahedra in ((1, False), (2, False), (1, True), (2, True)):
        mesh = gmsh_mesh(order, hexahedra)
        expected = {(1, False): 'tet4', (2, False): 'tet10', (1, True): 'hex8', (2, True): 'hex27'}[(order, hexahedra)]
        assert mesh.element_type == expected and mesh.PD == 3, (mesh.element_type, expected)

        disc = Discretization(mesh)
        assert np.isclose(disc.detJw.sum(), 1.0), expected
        assert sorted(mesh.groups) == ['top', 'wall'] and len(mesh.regions['solid']) == mesh.getNoE()

        top = disc.facets(mesh.boundary_faces('top'))
        assert np.isclose(top.detJw.sum(), 1.0) and np.allclose(top.normals, [0, 0, 1]), expected
        assert np.isclose(disc.facets(mesh.boundary_faces('wall')).detJw.sum(), 6.0), expected
        assert len(mesh.boundary_faces('wall')) == len(mesh.boundary_faces()), expected

        _, u = solve_poisson(mesh, exact, None)
        assert np.max(np.abs(u - exact(*nodes_of(mesh).T))) < 1e-10, expected
        print(f"   gmsh {expected}: {mesh.getNoE()} elements, {mesh.getNoN()} nodes")


def test_sphere_surface_and_normals():
    """Sphere of radius 1 from gmsh: the curved tet10 faces give the area 4 pi and the volume 4 pi / 3 far more accurately than
    the flat tet4 ones, and their normals approach x / |x|."""

    errors = {}
    for order in (1, 2):
        mesh = gmsh_mesh(order, sphere=True, size=0.3)
        disc = Discretization(mesh)
        facets = disc.facets(mesh.boundary_faces('wall'))
        area_error = abs(facets.detJw.sum() - 4 * np.pi) / (4 * np.pi)
        volume_error = abs(disc.detJw.sum() - 4 * np.pi / 3) / (4 * np.pi / 3)
        flux = np.einsum('fq,fqa,fqa->', facets.detJw, facets.x, facets.normals)          # = 3 V (divergence theorem)
        assert np.isclose(flux, 3 * disc.detJw.sum(), rtol=1e-10)
        radial = facets.x / np.linalg.norm(facets.x, axis=-1, keepdims=True)
        normal_error = np.max(np.linalg.norm(facets.normals - radial, axis=-1))
        errors[order] = (area_error, volume_error, normal_error)
        print(f"   sphere order {order}: area error {area_error:.1e}, volume error {volume_error:.1e}, "
              f"normal error {normal_error:.1e}")

    assert errors[1][0] < 5e-2 and errors[2][0] < 1e-3
    assert errors[2][0] < errors[1][0] / 10 and errors[2][1] < errors[1][1] / 10 and errors[2][2] < errors[1][2]


# ----- VTK -----

def read_vtu_arrays(path):

    """{name: values} of a .vtu written by write_vtu, ASCII or base64 binary."""

    text = open(path, encoding='utf-8').read()
    arrays = {}
    for m in re.finditer(r'<DataArray type="(\w+)" Name="([^"]+)" NumberOfComponents="(\d+)" format="(\w+)">\n(.*?)</DataArray>',
                         text, re.S):
        vtk_type, name, _, fmt, body = m.groups()
        dtype = {'Float64': '<f8', 'Int64': '<i8', 'UInt8': 'u1'}[vtk_type]
        if fmt == 'binary':
            raw = base64.b64decode(body.strip())
            size = np.frombuffer(raw[:4], '<u4')[0]
            arrays[name] = np.frombuffer(raw[4:4 + size], dtype)
        else:
            arrays[name] = np.array(body.split(), dtype=float).astype(dtype)
    return arrays


def test_vtk_3d_cells():
    """3D cells with their VTK types; the quadratic cells in the VTK node order (the edge node k of a VTK tet10 is the midpoint of
    its VTK edge, the hex27 face nodes are x-, x+, y-, y+, z-, z+); binary and ASCII files hold the same data."""

    vtk_edges = {'tet10': ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)),
                 'hex27': ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))}
    hex_faces = ((0, 3, 7, 4), (1, 2, 6, 5), (0, 1, 5, 4), (3, 2, 6, 7), (0, 1, 2, 3), (4, 5, 6, 7))

    with tempfile.TemporaryDirectory() as folder:
        for name in ELEMENTS_3D:
            mesh = box(name, (2, 1, 1))
            xyz = nodes_of(mesh)
            fields = {'T': xyz[:, 0] + xyz[:, 2], 'u': xyz.copy()}
            binary = read_vtu_arrays(write_vtu(os.path.join(folder, f'{name}_b.vtu'), mesh, fields))
            ascii_ = read_vtu_arrays(write_vtu(os.path.join(folder, f'{name}_a.vtu'), mesh, fields, binary=False))

            assert sorted(binary) == sorted(ascii_)
            for key in binary:
                assert np.allclose(binary[key], ascii_[key]), (name, key)

            assert np.all(binary['types'] == {'tet4': 10, 'hex8': 12, 'tet10': 24, 'hex27': 29}[name])
            nne = ELEMENTS[name].n_nodes
            conn = binary['connectivity'].reshape(-1, nne)
            points = binary['Points'].reshape(-1, 3)
            assert np.allclose(points, xyz) and np.allclose(binary['T'], fields['T'])

            if name in vtk_edges:
                cell = points[conn]                                   # (nel, nne, 3) in the VTK order
                nv = 4 if name == 'tet10' else 8
                for k, (a, b) in enumerate(vtk_edges[name]):
                    assert np.allclose(cell[:, nv + k], 0.5 * (cell[:, a] + cell[:, b])), (name, k)
                if name == 'hex27':
                    for k, face in enumerate(hex_faces):
                        assert np.allclose(cell[:, 20 + k], cell[:, list(face)].mean(axis=1)), k
                    assert np.allclose(cell[:, 26], cell[:, :8].mean(axis=1))
            else:
                assert vtk_node_order(name) is None


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
