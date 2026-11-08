"""Registry of reference elements (shape functions, quadrature and facets), independent of the physics.

Each ReferenceElement knows its nodes on the reference domain, how to evaluate N and dN/dxi at
points, a Gauss rule and the local node indices of its facets (the boundary entities: points for
lines, edges for triangles and quadrilaterals, faces for tetrahedra and hexahedra). Adding an element means adding one
entry to ELEMENTS; the discretization and the operators are generic in the
dimension.

Shapes: N (npts, nne); dN (npts, nne, dim) with dN[..., a] = dN / dxi_a.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss

from Source.Pre_processing.BasisFunctions import q4_shape_functions, Q4_NODES


def lagrange_1d(nodes, xi):

    """Values N (len(xi), n) and derivatives dN/dxi of the 1D Lagrange basis defined on `nodes`."""

    nodes = np.asarray(nodes, dtype=float)
    xi = np.asarray(xi, dtype=float)
    n = len(nodes)

    N = np.zeros((len(xi), n))
    dN = np.zeros((len(xi), n))

    for j in range(n):

        others = [k for k in range(n) if k != j]
        denom = np.prod([nodes[j] - nodes[k] for k in others])

        for g, x in enumerate(xi):

            N[g, j] = np.prod([x - nodes[k] for k in others]) / denom
            dN[g, j] = sum(np.prod([x - nodes[k] for k in others if k != m]) for m in others) / denom

    return N, dN


@dataclass(frozen=True)
class ReferenceElement:

    name: str
    dim: int
    nodes: np.ndarray          # (nne, dim) reference coordinates
    facets: tuple              # local node indices of each facet
    facet_element: str         # name of the reference element of the facets
    _shape: Callable
    _quadrature: Callable
    default_order: int = 2     # Gauss points per direction of the volume integrals
    # Quadratic elements: each non-vertex node as a combination of the vertices, (node, ((vertex, weight), ...)). A nodal
    # field that satisfies these relations is linear (bilinear) in the element: the pressure of Taylor-Hood.
    linear_subspace: tuple = ()
    # d2N (npts, nne, dim, dim) in the reference coordinates; None if they vanish in the reference element's sense of use
    _second: Callable = None

    @property
    def n_nodes(self):
        return len(self.nodes)

    def shape(self, points):

        """N (npts, nne) and dN (npts, nne, dim) at points (npts, dim)."""

        points = np.asarray(points, dtype=float)

        return self._shape(points.reshape(len(points), self.dim))

    def second_derivatives(self, points):

        """d2N / dxi_a dxi_b (npts, nne, dim, dim) at points (npts, dim); None for the linear elements."""

        if self._second is None:
            return None

        points = np.asarray(points, dtype=float)

        return self._second(points.reshape(len(points), self.dim))

    def quadrature(self, order=2):

        """Gauss points (nq, dim) and weights (nq,); `order` is the number of points per direction."""

        return self._quadrature(order)


# ----- point (facet of a line) -----

def _point_shape(points):
    npts = points.shape[0]
    return np.ones((npts, 1)), np.zeros((npts, 1, 0))


def _point_quadrature(order):
    return np.zeros((1, 0)), np.ones(1)


# ----- lines -----

def _line_shape(nodes_1d):
    def shape(points):
        N, dN = lagrange_1d(nodes_1d, points[:, 0])
        return N, dN[:, :, None]
    return shape


def _line_quadrature(order):
    x, w = leggauss(order)
    return x[:, None], w


# ----- bilinear quadrilateral -----

def _quad4_shape(points):
    return q4_shape_functions(points[:, 0], points[:, 1])


def _quad_quadrature(order):
    x, w = leggauss(order)
    xi, eta = np.meshgrid(x, x, indexing='ij')
    return np.stack([xi.ravel(), eta.ravel()], axis=1), np.outer(w, w).ravel()


# ----- biquadratic quadrilateral -----

QUAD9_NODES = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
                        [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
_QUAD9_I = np.array([0, 2, 2, 0, 1, 2, 1, 0, 1])     # position of each node on the 1D nodes -1, 0, 1
_QUAD9_J = np.array([0, 0, 2, 2, 0, 1, 2, 1, 1])


def _quad9_shape(points):
    Nx, dNx = lagrange_1d([-1.0, 0.0, 1.0], points[:, 0])
    Ny, dNy = lagrange_1d([-1.0, 0.0, 1.0], points[:, 1])
    N = Nx[:, _QUAD9_I] * Ny[:, _QUAD9_J]
    dN = np.stack([dNx[:, _QUAD9_I] * Ny[:, _QUAD9_J], Nx[:, _QUAD9_I] * dNy[:, _QUAD9_J]], axis=-1)
    return N, dN


def _quad9_second(points):
    nodes = [-1.0, 0.0, 1.0]
    Nx, dNx = lagrange_1d(nodes, points[:, 0])
    Ny, dNy = lagrange_1d(nodes, points[:, 1])
    d2 = np.array([1.0, -2.0, 1.0])                  # of a parabola on -1, 0, 1: constant
    xx = d2[_QUAD9_I][None] * Ny[:, _QUAD9_J]
    xy = dNx[:, _QUAD9_I] * dNy[:, _QUAD9_J]
    yy = Nx[:, _QUAD9_I] * d2[_QUAD9_J][None]
    return np.stack([np.stack([xx, xy], axis=-1), np.stack([xy, yy], axis=-1)], axis=-1)


# ----- linear triangle -----

TRI3_NODES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def _tri3_shape(points):
    xi, eta = points[:, 0], points[:, 1]
    N = np.stack([1.0 - xi - eta, xi, eta], axis=1)
    dN = np.broadcast_to(np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]), (len(points), 3, 2)).copy()
    return N, dN


def _tri_quadrature(order):

    """Interior Gauss rule on the reference triangle (area 1/2): 1 point for order 1, 3 points (degree 2) for order 2,
    7 points (degree 5, Radon) for a higher order."""

    if order <= 1:
        return np.array([[1.0 / 3.0, 1.0 / 3.0]]), np.array([0.5])

    if order >= 3:
        s = np.sqrt(15.0)
        a, b = (6.0 - s) / 21.0, (6.0 + s) / 21.0
        wa, wb = (155.0 - s) / 2400.0, (155.0 + s) / 2400.0
        points = [[1 / 3, 1 / 3], [a, a], [1 - 2 * a, a], [a, 1 - 2 * a], [b, b], [1 - 2 * b, b], [b, 1 - 2 * b]]
        return np.array(points), np.array([9.0 / 80.0, wa, wa, wa, wb, wb, wb])

    return np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]]), np.full(3, 1.0 / 6.0)


# ----- quadratic triangle -----

TRI6_NODES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])


def _tri6_shape(points):
    xi, eta = points[:, 0], points[:, 1]
    L = 1.0 - xi - eta
    N = np.stack([L * (2 * L - 1), xi * (2 * xi - 1), eta * (2 * eta - 1), 4 * L * xi, 4 * xi * eta, 4 * eta * L], axis=1)
    dxi = np.stack([1 - 4 * L, 4 * xi - 1, 0 * xi, 4 * (L - xi), 4 * eta, -4 * eta], axis=1)
    deta = np.stack([1 - 4 * L, 0 * xi, 4 * eta - 1, -4 * xi, 4 * xi, 4 * (L - eta)], axis=1)
    return N, np.stack([dxi, deta], axis=-1)


def _tri6_second(points):
    xx = np.array([4.0, 4.0, 0.0, -8.0, 0.0, 0.0])
    xy = np.array([4.0, 0.0, 0.0, -4.0, 4.0, -4.0])
    yy = np.array([4.0, 0.0, 4.0, 0.0, 0.0, -8.0])
    H = np.stack([np.stack([xx, xy], axis=-1), np.stack([xy, yy], axis=-1)], axis=-1)
    return np.broadcast_to(H, (len(points),) + H.shape).copy()


# ----- 3D: node orders of gmsh (also its reference coordinates: [0, 1] tetrahedra, [-1, 1] hexahedra) -----

TET4_NODES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
TET10_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (2, 3), (1, 3))          # vertices of the nodes 4..9
TET10_NODES = np.vstack([TET4_NODES] + [0.5 * (TET4_NODES[a] + TET4_NODES[b]) for a, b in TET10_EDGES])
TET_FACES = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))                 # vertices of each face

HEX8_NODES = np.array([[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
                       [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]])
HEX27_NODES = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
                        [0, -1, -1], [-1, 0, -1], [-1, -1, 0], [1, 0, -1], [1, -1, 0], [0, 1, -1], [1, 1, 0], [-1, 1, 0],
                        [0, -1, 1], [-1, 0, 1], [1, 0, 1], [0, 1, 1],
                        [0, 0, -1], [0, -1, 0], [-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
HEX_FACES = ((0, 3, 2, 1), (0, 1, 5, 4), (0, 4, 7, 3), (1, 2, 6, 5), (2, 3, 7, 6), (4, 5, 6, 7))


def _find_node(nodes, point):
    hit = np.flatnonzero(np.all(np.abs(nodes - point) < 1e-12, axis=1))
    if len(hit) != 1:
        raise ValueError(f"No hay un nodo en {point}.")
    return int(hit[0])


def _quadratic_facets(nodes, faces):

    """Local nodes of the faces of a quadratic element in the order of their facet element (tri6: the vertices, then the
    midpoints of the edges v0v1, v1v2, v2v0; quad9: the same and the centre), found by their reference coordinates."""

    facets = []
    for face in faces:
        v = list(face)
        n = len(v)
        mids = [_find_node(nodes, 0.5 * (nodes[v[k]] + nodes[v[(k + 1) % n]])) for k in range(n)]
        centre = [_find_node(nodes, nodes[v].mean(axis=0))] if n == 4 else []
        facets.append(tuple(v + mids + centre))
    return tuple(facets)


def _linear_subspace(nodes, n_vertices, linear_shape):

    """Each non-vertex node of a quadratic element as the combination of the vertices that a linear field takes there."""

    N, _ = linear_shape(nodes[n_vertices:])
    return tuple((n_vertices + i, tuple((k, float(w)) for k, w in enumerate(row) if abs(w) > 1e-14))
                 for i, row in enumerate(N))


# ----- tetrahedra -----

def _tet4_shape(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    N = np.stack([1.0 - x - y - z, x, y, z], axis=1)
    dN = np.broadcast_to(_TET_DL, (len(points), 4, 3)).copy()
    return N, dN


_TET_DL = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])   # d(barycentric)/dxi


def _tet10_shape(points):
    L, _ = _tet4_shape(points)                                           # barycentric coordinates (npts, 4)
    N = [L[:, k] * (2 * L[:, k] - 1) for k in range(4)] + [4 * L[:, a] * L[:, b] for a, b in TET10_EDGES]
    dN = [(4 * L[:, k] - 1)[:, None] * _TET_DL[k] for k in range(4)]
    dN += [4 * (L[:, b][:, None] * _TET_DL[a] + L[:, a][:, None] * _TET_DL[b]) for a, b in TET10_EDGES]
    return np.stack(N, axis=1), np.stack(dN, axis=1)


def _tet10_second(points):
    H = [4 * np.outer(_TET_DL[k], _TET_DL[k]) for k in range(4)]
    H += [4 * (np.outer(_TET_DL[a], _TET_DL[b]) + np.outer(_TET_DL[b], _TET_DL[a])) for a, b in TET10_EDGES]
    H = np.stack(H)
    return np.broadcast_to(H, (len(points),) + H.shape).copy()


# 14-point rule of degree 5 with positive weights (Walkington): orbits of 4 + 4 points (a, a, a, 1 - 3a) and of 6 points
# (c, c, 1/2 - c, 1/2 - c), solved to machine precision; Test/test_3d_foundation.py checks it against the exact moments.
_TET14 = (0.31088591926330078, 0.092735250310892053, 0.45449629587434925,
          0.0187813209530025, 0.01224884051939377, 0.0070910034628469199)


def _tet_orbit_4(a):
    L = np.array([[a, a, a, 1 - 3 * a], [a, a, 1 - 3 * a, a], [a, 1 - 3 * a, a, a], [1 - 3 * a, a, a, a]])
    return L[:, 1:]


def _tet_quadrature(order):

    """Gauss rules on the reference tetrahedron (volume 1/6): 1 point (degree 1) for order 1, 4 points (degree 2) for order 2,
    14 points (degree 5) for order 3, and a collapsed Gauss-Jacobi product of order**3 points (degree 2 order - 1) above."""

    if order <= 1:
        return np.array([[0.25, 0.25, 0.25]]), np.array([1.0 / 6.0])

    if order == 2:
        return _tet_orbit_4((5.0 - np.sqrt(5.0)) / 20.0), np.full(4, 1.0 / 24.0)

    if order == 3:
        a, b, c, wa, wb, wc = _TET14
        d = 0.5 - c
        edges = np.array([[c, c, d, d], [c, d, c, d], [c, d, d, c], [d, c, c, d], [d, c, d, c], [d, d, c, c]])[:, 1:]
        points = np.vstack([_tet_orbit_4(a), _tet_orbit_4(b), edges])
        return points, np.concatenate([np.full(4, wa), np.full(4, wb), np.full(6, wc)])

    from scipy.special import roots_jacobi

    # x = u, y = v (1 - u), z = w (1 - u)(1 - v): the Jacobian (1 - u)^2 (1 - v) goes into Gauss-Jacobi weights
    tu, wu = roots_jacobi(order, 2.0, 0.0)
    tv, wv = roots_jacobi(order, 1.0, 0.0)
    tw, ww = leggauss(order)
    u, v, w = (tu + 1) / 2, (tv + 1) / 2, (tw + 1) / 2
    U, V, W = np.meshgrid(u, v, w, indexing='ij')
    x, y, z = U, V * (1 - U), W * (1 - U) * (1 - V)
    weights = np.einsum('i,j,k->ijk', wu / 8.0, wv / 4.0, ww / 2.0)
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1), weights.ravel()


# ----- hexahedra -----

def _hex_shape(nodes):

    """Tensor-product Lagrange shape functions (and second derivatives) of a hexahedron whose nodes sit on the 1D nodes
    -1, (0,) 1."""

    nodes_1d = np.unique(nodes[:, 0])
    index = np.searchsorted(nodes_1d, nodes)                              # (nne, 3) position on the 1D nodes

    def factors(points):
        values = [lagrange_1d(nodes_1d, points[:, a]) for a in range(3)]
        return [v[0][:, index[:, a]] for a, v in enumerate(values)], [v[1][:, index[:, a]] for a, v in enumerate(values)]

    def shape(points):
        Ns, dNs = factors(points)
        N = Ns[0] * Ns[1] * Ns[2]
        dN = np.stack([dNs[0] * Ns[1] * Ns[2], Ns[0] * dNs[1] * Ns[2], Ns[0] * Ns[1] * dNs[2]], axis=-1)
        return N, dN

    def second(points):
        Ns, dNs = factors(points)
        d2 = np.array([1.0, -2.0, 1.0])                                   # of a parabola on -1, 0, 1: constant
        d2Ns = [np.broadcast_to(d2[index[:, a]], Ns[a].shape) for a in range(3)]
        H = np.empty(Ns[0].shape + (3, 3))
        for a in range(3):
            for b in range(3):
                f = [(Ns, dNs, d2Ns)[(c == a) + (c == b)][c] for c in range(3)]
                H[..., a, b] = f[0] * f[1] * f[2]
        return H

    return shape, second


def _hex_quadrature(order):
    x, w = leggauss(order)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1), np.einsum('i,j,k->ijk', w, w, w).ravel()


_hex8_shape, _ = _hex_shape(HEX8_NODES)
_hex27_shape, _hex27_second = _hex_shape(HEX27_NODES)


ELEMENTS = {
    'point': ReferenceElement('point', 0, np.zeros((1, 0)), (), 'point', _point_shape, _point_quadrature),
    'line2': ReferenceElement('line2', 1, np.array([[-1.0], [1.0]]), ((0,), (1,)), 'point',
                              _line_shape([-1.0, 1.0]), _line_quadrature),
    'line3': ReferenceElement('line3', 1, np.array([[-1.0], [0.0], [1.0]]), ((0,), (2,)), 'point',
                              _line_shape([-1.0, 0.0, 1.0]), _line_quadrature),
    'tri3': ReferenceElement('tri3', 2, TRI3_NODES.copy(), ((0, 1), (1, 2), (2, 0)), 'line2',
                             _tri3_shape, _tri_quadrature),
    'quad4': ReferenceElement('quad4', 2, Q4_NODES.copy(), ((0, 1), (1, 2), (2, 3), (3, 0)), 'line2',
                              _quad4_shape, _quad_quadrature),
    'tri6': ReferenceElement('tri6', 2, TRI6_NODES.copy(), ((0, 3, 1), (1, 4, 2), (2, 5, 0)), 'line3',
                             _tri6_shape, _tri_quadrature, default_order=3,
                             linear_subspace=((3, ((0, 0.5), (1, 0.5))), (4, ((1, 0.5), (2, 0.5))), (5, ((2, 0.5), (0, 0.5)))),
                             _second=_tri6_second),
    'quad9': ReferenceElement('quad9', 2, QUAD9_NODES.copy(), ((0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)), 'line3',
                              _quad9_shape, _quad_quadrature, default_order=3,
                              linear_subspace=((4, ((0, 0.5), (1, 0.5))), (5, ((1, 0.5), (2, 0.5))), (6, ((2, 0.5), (3, 0.5))),
                                               (7, ((3, 0.5), (0, 0.5))), (8, ((0, 0.25), (1, 0.25), (2, 0.25), (3, 0.25)))),
                              _second=_quad9_second),
    'tet4': ReferenceElement('tet4', 3, TET4_NODES.copy(), TET_FACES, 'tri3', _tet4_shape, _tet_quadrature),
    'tet10': ReferenceElement('tet10', 3, TET10_NODES.copy(), _quadratic_facets(TET10_NODES, TET_FACES), 'tri6',
                              _tet10_shape, _tet_quadrature, default_order=3,
                              linear_subspace=_linear_subspace(TET10_NODES, 4, _tet4_shape), _second=_tet10_second),
    'hex8': ReferenceElement('hex8', 3, HEX8_NODES.copy(), HEX_FACES, 'quad4', _hex8_shape, _hex_quadrature),
    'hex27': ReferenceElement('hex27', 3, HEX27_NODES.copy(), _quadratic_facets(HEX27_NODES, HEX_FACES), 'quad9',
                              _hex27_shape, _hex_quadrature, default_order=3,
                              linear_subspace=_linear_subspace(HEX27_NODES, 8, _hex8_shape), _second=_hex27_second),
}


def element_type_of(mesh):

    """Name of the reference element of a mesh (falls back to the default one for its dimension)."""

    name = getattr(mesh, 'element_type', None)

    if name is None:
        name = {1: 'line2', 2: 'quad4', 3: 'hex8'}[mesh.PD]

    return name
