"""Registry of reference elements (shape functions, quadrature and facets), independent of the physics.

Each ReferenceElement knows its nodes on the reference domain, how to evaluate N and dN/dxi at
points, a Gauss rule and the local node indices of its facets (the boundary entities: points for
lines, edges for triangles and quadrilaterals, faces for hexahedra). Adding an element (tet4, hex8, quad9,
...) means adding one entry to ELEMENTS; the discretization and the operators are generic in the
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

    @property
    def n_nodes(self):
        return len(self.nodes)

    def shape(self, points):

        """N (npts, nne) and dN (npts, nne, dim) at points (npts, dim)."""

        points = np.asarray(points, dtype=float)

        return self._shape(points.reshape(len(points), self.dim))

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


# ----- linear triangle -----

TRI3_NODES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def _tri3_shape(points):
    xi, eta = points[:, 0], points[:, 1]
    N = np.stack([1.0 - xi - eta, xi, eta], axis=1)
    dN = np.broadcast_to(np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]), (len(points), 3, 2)).copy()
    return N, dN


def _tri_quadrature(order):

    """Interior Gauss rule on the reference triangle (area 1/2): 1 point for order 1, 3 points (degree 2) otherwise."""

    if order <= 1:
        return np.array([[1.0 / 3.0, 1.0 / 3.0]]), np.array([0.5])

    return np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]]), np.full(3, 1.0 / 6.0)


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
}


def element_type_of(mesh):

    """Name of the reference element of a mesh (falls back to the default one for its dimension)."""

    name = getattr(mesh, 'element_type', None)

    if name is None:
        name = {1: 'line2', 2: 'quad4'}[mesh.PD]

    return name
