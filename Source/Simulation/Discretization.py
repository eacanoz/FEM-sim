"""Geometry of a mesh evaluated at the quadrature points, for any dimension and element type.

Computed once per mesh and reused by every assembly. Nothing here depends on the physics.

Volume data (nel elements, ngp Gauss points, nne nodes per element, dim dimensions):
    conn    (nel, nne)            global node ids
    N       (ngp, nne)            shape functions
    B       (nel, ngp, nne, dim)  physical gradients dN/dx
    detJw   (nel, ngp)            |detJ| * Gauss weight (* r if axisymmetric)
    area    (nel, ngp)            |detJ| * Gauss weight (plane measure)
    x_gp    (nel, ngp, dim)       physical coordinates of the Gauss points

Boundary data comes as a FacetSet (points in 1D, edges in 2D, faces in 3D).
"""

import numpy as np

from Source.Pre_processing.Elements import ELEMENTS, element_type_of


class FacetSet:

    """Boundary facets with their quadrature data.

    nodes   (nf, nnf)       global node ids, in the order of the owning element
    elems   (nf,)           index of the owning element
    N       (nq, nnf)       facet shape functions at the facet quadrature points
    detJw   (nf, nq)        facet measure * weight
    x       (nf, nq, dim)   physical coordinates of the quadrature points
    normals (nf, nq, dim)   outward unit normals
    """

    def __init__(self, nodes, elems, N, detJw, x, normals):
        self.nodes = nodes
        self.elems = elems
        self.N = N
        self.detJw = detJw
        self.x = x
        self.normals = normals

    @property
    def n_facets(self):
        return len(self.nodes)


class Discretization:

    """axisymmetric=True: the 2D mesh is a meridian plane (x = r, y = z, axis at x = 0). Every integral is then
    integral(... r dr dz): detJw and the facet measures carry the factor r (the 2 pi is left out everywhere).
    `area` keeps the plane measure (element sizes, recovery patches) and `hoop` = N / r gives the hoop strain
    u_r / r of vector fields."""

    def __init__(self, mesh, order=None, axisymmetric=False):

        self.element = ELEMENTS[element_type_of(mesh)]
        self.dim = self.element.dim
        self.order = self.element.default_order if order is None else order
        self.n_nodes = mesh.getNoN()
        self.axisymmetric = bool(axisymmetric)

        self.conn = np.array([e.getNodesId() for e in mesh.EL], dtype=int)
        self.node_xy = np.array([node.coor[:self.dim] for node in mesh.NL], dtype=float)

        points, weights = self.element.quadrature(self.order)
        N, dN = self.element.shape(points)  # (ngp, nne), (ngp, nne, dim)

        coords = self.node_xy[self.conn]    # (nel, nne, dim)

        # J[e, g, a, b] = dx_a / dxi_b
        J = np.einsum('eka,gkb->egab', coords, dN)
        detJ = np.linalg.det(J)
        if np.any(np.abs(detJ) < 1e-14):
            raise ValueError("Elemento degenerado: jacobiano casi nulo.")

        invJ = np.linalg.inv(J)  # invJ[e, g, b, a] = dxi_b / dx_a

        self._invJ, self._coords, self._points = invJ, coords, points

        self.N = N
        self.B = np.einsum('gkb,egba->egka', dN, invJ)
        self.area = np.abs(detJ) * weights          # plane measure (no r factor)
        self.x_gp = np.einsum('gk,eka->ega', N, coords)

        self.hoop = None
        if self.axisymmetric:
            if self.dim != 2:
                raise ValueError("La formulacion axisimetrica necesita una malla 2D del plano meridiano (x = r, y = z).")
            if np.any(self.node_xy[:, 0] < -1e-12 * max(1.0, np.abs(self.node_xy).max())):
                raise ValueError("Malla axisimetrica con radios negativos: el eje de simetria debe estar en x = 0.")
            r = self.x_gp[..., 0]
            self.detJw = self.area * r
            self.hoop = N[None, :, :] / r[..., None]    # (nel, ngp, nne): N_i / r
        else:
            self.detJw = self.area

        self._facet_owner = None

    @property
    def hessian(self):

        """Physical second derivatives d2N / dx_a dx_b (nel, ngp, nne, dim, dim) of a quadratic element, also curved; None for
        linear elements, where they vanish. Built once."""

        if getattr(self, '_hessian', None) is None:
            d2N = self.element.second_derivatives(self._points)
            if d2N is None:
                return None
            # d2x/dxi dxi of the isoparametric map, and d2N/dx dx = J^-T (d2N/dxi dxi - dN/dx_c d2x_c/dxi dxi) J^-1
            d2x = np.einsum('eka,gkbc->egabc', self._coords, d2N)
            reference = d2N[None] - np.einsum('egka,egabc->egkbc', self.B, d2x)
            self._hessian = np.einsum('egbi,egkbc,egcj->egkij', self._invJ, reference, self._invJ)

        return self._hessian

    @property
    def laplacian(self):

        """Laplacian of the shape functions (nel, ngp, nne), with the term dN/dr / r if axisymmetric; None for linear elements in
        the plane, where it vanishes."""

        hessian = self.hessian
        if hessian is None:
            return None

        return np.einsum('egkaa->egk', hessian) + (self.hoop_gradient if self.axisymmetric else 0.0)

    @property
    def hoop_gradient(self):

        """dN/dr / r (nel, ngp, nne) of an axisymmetric discretization."""

        return self.B[..., 0] / self.radius[..., None]

    @property
    def radius(self):

        """r at the Gauss points (nel, ngp) of an axisymmetric discretization."""

        return self.x_gp[..., 0]

    @property
    def n_elements(self):
        return len(self.conn)

    def interpolate(self, nodal):

        """Nodal values (n_nodes,) or (n_nodes, k) interpolated to the Gauss points: (nel, ngp[, k])."""

        nodal = np.asarray(nodal, dtype=float)

        if nodal.ndim == 1:
            return np.einsum('gi,ei->eg', self.N, nodal[self.conn])

        return np.einsum('gi,eik->egk', self.N, nodal[self.conn])

    def gradient(self, nodal):

        """Gradient of nodal values (n_nodes,) or (n_nodes, k) at the Gauss points: (nel, ngp, dim) or (nel, ngp, k, dim)."""

        nodal = np.asarray(nodal, dtype=float)

        if nodal.ndim == 1:
            return np.einsum('egia,ei->ega', self.B, nodal[self.conn])

        return np.einsum('egia,eik->egka', self.B, nodal[self.conn])

    def project(self, values):

        """Gauss-point values (nel, ngp) or (nel, ngp, ...) to nodal values by lumped L2 projection.

        v_i = sum_e integral(N_i v) / sum_e integral(N_i). Exact for fields that are constant on the
        mesh; a gradient of a linear field (Darcy velocity of a linear pressure) is recovered exactly.
        Quadratic elements use the consistent mass matrix: integral(N_i) vanishes at the vertices of a T6.
        """

        values = np.asarray(values, dtype=float)
        tail = values.shape[2:]

        if self.element.linear_subspace:
            contrib = np.einsum('eg,gi,eg...->ei...', self.detJw, self.N, values)
            rhs = np.zeros((self.n_nodes,) + tail)
            np.add.at(rhs, self.conn.ravel(), contrib.reshape((-1,) + tail))
            return self._mass_factorization().solve(rhs.reshape(self.n_nodes, -1)).reshape(rhs.shape)

        weights = np.zeros(self.n_nodes)
        np.add.at(weights, self.conn.ravel(), np.einsum('eg,gi->ei', self.detJw, self.N).ravel())

        nodal = np.zeros((self.n_nodes,) + tail)
        contrib = np.einsum('eg,gi,eg...->ei...', self.detJw, self.N, values)
        np.add.at(nodal, self.conn.ravel(), contrib.reshape((-1,) + tail))

        return nodal / weights.reshape((-1,) + (1,) * len(tail))

    def _mass_factorization(self):

        """Factorization of the consistent mass matrix, built once."""

        if getattr(self, '_mass_lu', None) is None:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla

            nne = self.conn.shape[1]
            Me = np.einsum('eg,gi,gj->eij', self.detJw, self.N, self.N)
            rows = np.repeat(self.conn, nne, axis=1).ravel()
            cols = np.tile(self.conn, (1, nne)).ravel()
            M = sp.coo_matrix((Me.ravel(), (rows, cols)), shape=(self.n_nodes, self.n_nodes)).tocsc()
            self._mass_lu = spla.splu(M)

        return self._mass_lu

    def recovery(self):

        """Sparse W (n_nodes, nel * ngp): nodal values recovered from Gauss-point values, v_k = sum_m W[k, m] v_m.

        Superconvergent patch recovery (Zienkiewicz-Zhu): a linear least-squares fit of the Gauss-point values of a
        patch of elements, evaluated at the node. The patch of an interior node is its elements; a boundary node
        uses the patches of its interior neighbours as well, so the fit sees two element layers across the
        boundary and also recovers the normal slope (a one-sided average would be first order there).
        Depends only on the geometry; built once.
        """

        if getattr(self, '_recovery', None) is not None:
            return self._recovery

        import scipy.sparse as sp

        nel, ngp = self.detJw.shape
        node_elems = [[] for _ in range(self.n_nodes)]
        for e, nodes in enumerate(self.conn):
            for k in nodes:
                node_elems[k].append(e)

        boundary = np.zeros(self.n_nodes, dtype=bool)
        for nodes in self.boundary_facet_nodes():
            boundary[list(nodes)] = True

        # Element means: the Gauss values of a gradient carry an O(h) error that is odd inside each element, so the
        # mean (the value at the centre) is accurate to O(h^2); fitting the means keeps that order on one-sided patches.
        volume = self.area.sum(axis=1)                                       # plane measure, also axisymmetric
        centres = np.einsum('eg,ega->ea', self.area, self.x_gp) / volume[:, None]
        mean_weights = self.area / volume[:, None]                           # (nel, ngp)

        rows, cols, vals = [], [], []

        for k in range(self.n_nodes):
            patch = set(node_elems[k])
            if boundary[k]:
                neighbours = {int(j) for e in node_elems[k] for j in self.conn[e]}
                interior = [j for j in neighbours if not boundary[j]] or list(neighbours)
                for j in interior:
                    patch.update(node_elems[j])

            elems = np.array(sorted(patch))
            A = np.hstack([np.ones((len(elems), 1)), centres[elems] - self.node_xy[k]])
            w = np.linalg.pinv(A)[0]           # value at the node of the linear function fitted to the means

            rows.append(np.full(len(elems) * ngp, k))
            cols.append((elems[:, None] * ngp + np.arange(ngp)).ravel())
            vals.append((w[:, None] * mean_weights[elems]).ravel())

        self._recovery = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                       shape=(self.n_nodes, nel * ngp)).tocsr()

        return self._recovery

    # ----- boundary facets -----

    def _owners(self):

        """Maps the sorted node ids of every element facet to its [(element, node ids in element order)]."""

        if self._facet_owner is None:
            owners = {}
            for e, conn in enumerate(self.conn):
                for local in self.element.facets:
                    ids = tuple(int(conn[k]) for k in local)
                    owners.setdefault(tuple(sorted(ids)), []).append((e, ids))
            self._facet_owner = owners

        return self._facet_owner

    def boundary_facet_nodes(self, selector=None):

        """Node tuples of the outer-boundary facets whose midpoint satisfies selector(*coords)."""

        found = []

        for key, owners in self._owners().items():
            if len(owners) != 1:
                continue

            e, ids = owners[0]
            mid = self.node_xy[list(ids)].mean(axis=0)

            if selector is None or selector(*mid):
                found.append((e, ids))

        return [ids for _, ids in sorted(found)]

    def facets(self, node_tuples):

        """FacetSet for the given boundary facets (tuples of node ids, any order)."""

        owners = self._owners()

        elems, nodes = [], []
        for t in node_tuples:
            key = tuple(sorted(int(n) for n in np.atleast_1d(t)))
            entries = owners.get(key, [])

            if len(entries) != 1:
                raise ValueError(f"{tuple(t)} no es una faceta del contorno de la malla.")

            e, ids = entries[0]
            elems.append(e)
            nodes.append(ids)

        elems = np.array(elems, dtype=int)
        nodes = np.array(nodes, dtype=int).reshape(len(elems), len(self.element.facets[0]))

        facet_element = ELEMENTS[self.element.facet_element]
        points, weights = facet_element.quadrature(self.order)
        Nf, dNf = facet_element.shape(points)  # (nq, nnf), (nq, nnf, dim_f)

        fcoords = self.node_xy[nodes]          # (nf, nnf, dim)
        x = np.einsum('qk,fka->fqa', Nf, fcoords)

        # Measure of the facet: sqrt(det(T^T T)) with T = dx/dxi_facet
        if facet_element.dim == 0:
            measure = np.ones(x.shape[:2])
            tangent = None
        else:
            tangent = np.einsum('fka,qkb->fqab', fcoords, dNf)
            measure = np.sqrt(np.linalg.det(np.einsum('fqab,fqac->fqbc', tangent, tangent)))

        # Outward normals, oriented with the centroid of the owning element
        centroid = self.node_xy[self.conn[elems]].mean(axis=1)[:, None, :]
        outward = x - centroid

        if self.dim == 1:
            normals = np.sign(outward)
        elif self.dim == 2:
            t = tangent[..., 0]
            normals = np.stack([t[..., 1], -t[..., 0]], axis=-1)
            normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
            flip = np.sign(np.sum(normals * outward, axis=-1, keepdims=True))
            normals = normals * flip
        else:
            raise NotImplementedError("Las normales de faceta solo estan implementadas en 1D y 2D.")

        detJw = measure * weights[None, :]
        if self.axisymmetric:
            detJw = detJw * x[..., 0]                # surface of revolution: ds -> r ds

        return FacetSet(nodes, elems, Nf, detJw, x, normals)
