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


def outer_facets(conn, element):

    """Facets of the outer boundary, [(element, node ids in element order)] sorted by element and ids: the facets that belong to
    a single element. Vectorized (sorted node ids of every facet and np.unique); depends only on the topology."""

    local = np.array(element.facets, dtype=int)                       # (nf, nnf)
    conn = np.asarray(conn, dtype=int)
    if local.size == 0 or len(conn) == 0:
        return []

    ids = conn[:, local].reshape(-1, local.shape[1])                  # (nel * nf, nnf)
    _, inverse, counts = np.unique(np.sort(ids, axis=1), axis=0, return_inverse=True, return_counts=True)
    single = np.flatnonzero(counts[np.ravel(inverse)] == 1)
    elems = single // len(local)

    return sorted((int(e), tuple(int(k) for k in row)) for e, row in zip(elems, ids[single]))


def select_facets(facets, node_xy, selector=None):

    """Node tuples of the facets [(element, ids)] whose midpoint satisfies selector(*coords) (None: all)."""

    if selector is None:
        return [ids for _, ids in facets]

    return [ids for _, ids in facets if selector(*node_xy[list(ids)].mean(axis=0))]


class _MassCG:

    """solve(rhs (n, k)) of a consistent mass matrix by CG with the Jacobi preconditioner, column by column (to rtol 1e-13)."""

    def __init__(self, M):
        import scipy.sparse.linalg as spla
        self.M = M.tocsr()
        inverse_diagonal = 1.0 / self.M.diagonal()
        self.P = spla.LinearOperator(self.M.shape, matvec=lambda r: inverse_diagonal * r)

    def solve(self, rhs):
        import scipy.sparse.linalg as spla
        rhs = np.asarray(rhs, dtype=float)
        columns = rhs.reshape(rhs.shape[0], -1)
        out = np.empty_like(columns)
        for k in range(columns.shape[1]):
            x, info = spla.cg(self.M, columns[:, k], M=self.P, rtol=1e-13, atol=0.0, maxiter=1000)
            if info != 0:
                raise RuntimeError("La proyeccion L2 (CG sobre la matriz de masa) no convergio.")
            out[:, k] = x
        return out.reshape(rhs.shape)


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
        width = max(len(node.coor) for node in mesh.NL)
        full = np.array([tuple(node.coor) + (0.0,) * (width - len(node.coor)) for node in mesh.NL], dtype=float)
        # A surface in space (the midplane of a folded part): elements of lower dimension than the coordinates, which are not all in
        # the plane of the first ones. Gradients are then tangential, of `dim` = the dimension of space.
        self.manifold = width > self.dim and bool(np.any(full[:, self.dim:] != 0.0))
        if self.manifold:
            self.dim = width
        self.node_xy = full[:, :self.dim]

        points, weights = self.element.quadrature(self.order)
        N, dN = self.element.shape(points)  # (ngp, nne), (ngp, nne, dim)

        coords = self.node_xy[self.conn]    # (nel, nne, dim)

        # J[e, g, a, b] = dx_a / dxi_b
        J = np.einsum('eka,gkb->egab', coords, dN)

        if self.manifold:
            # Metric M = J^T J; measure sqrt(det M); tangential gradient dN/dx = dN/dxi M^-1 J^T (exact for flat elements)
            M = np.einsum('egab,egac->egbc', J, J)
            detJ = np.sqrt(np.linalg.det(M))
            if np.any(detJ < 1e-14):
                raise ValueError("Elemento degenerado: jacobiano casi nulo.")
            invJ = np.einsum('egbc,egac->egba', np.linalg.inv(M), J)     # the left inverse, (e, g, b, a)
        else:
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
            if self.manifold:
                raise NotImplementedError("Segundas derivadas de elementos cuadraticos sobre superficies: no implementadas.")
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
            return nodal[self.conn] @ self.N.T

        return np.matmul(self.N, nodal[self.conn])          # faster than einsum('gi,eik->egk'), and in every residual

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
            # 3D: the LU of the mass matrix fills in (12 s of a 32 s Stokes solve with 31 k tet10); the mass matrix scaled by its
            # diagonal has a condition number independent of h, so CG + Jacobi converges in a few tens of iterations.
            self._mass_lu = _MassCG(M) if self.element.dim == 3 else spla.splu(M)

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
        n, nne = self.n_nodes, self.conn.shape[1]

        boundary = np.zeros(n, dtype=bool)
        for nodes in self.boundary_facet_nodes():
            boundary[list(nodes)] = True

        # Patches as sparse incidence products (a loop over the nodes with one pseudoinverse each took 3.3 s with 37 k nodes):
        # E[k, e] = 1 if node k belongs to element e; the neighbours of k are the nodes of its elements.
        E = sp.csr_matrix((np.ones(nel * nne), (self.conn.ravel(), np.repeat(np.arange(nel), nne))), shape=(n, nel))
        E.data[:] = 1.0                                                      # a node repeated in an element counts once
        neighbours = (E @ E.T).tocsr()
        interior = neighbours @ sp.diags((~boundary).astype(float))          # interior neighbours of each node
        interior.eliminate_zeros()
        has_interior = np.diff(interior.tocsr().indptr) > 0
        chosen = sp.diags((boundary & has_interior).astype(float)) @ interior \
            + sp.diags((boundary & ~has_interior).astype(float)) @ neighbours
        patches = (E + chosen @ E).tocsr()                                   # interior node: its elements
        patches.sort_indices()

        # Element means: the Gauss values of a gradient carry an O(h) error that is odd inside each element, so the
        # mean (the value at the centre) is accurate to O(h^2); fitting the means keeps that order on one-sided patches.
        volume = self.area.sum(axis=1)                                       # plane measure, also axisymmetric
        centres = np.einsum('eg,ega->ea', self.area, self.x_gp) / volume[:, None]
        mean_weights = self.area / volume[:, None]                           # (nel, ngp)

        rows, cols, vals = [], [], []
        sizes = np.diff(patches.indptr)

        for size in np.unique(sizes):                                        # the nodes with patches of the same size, at once
            nodes = np.flatnonzero(sizes == size)
            elems = patches.indices[patches.indptr[nodes][:, None] + np.arange(size)]          # (nodes, size), sorted
            A = np.concatenate([np.ones((len(nodes), size, 1)), centres[elems] - self.node_xy[nodes][:, None]], axis=-1)
            w = np.linalg.pinv(A)[:, 0]        # value at the node of the linear function fitted to the means

            rows.append(np.repeat(nodes, size * ngp))
            cols.append((elems[:, :, None] * ngp + np.arange(ngp)).ravel())
            vals.append((w[:, :, None] * mean_weights[elems]).ravel())

        self._recovery = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                       shape=(self.n_nodes, nel * ngp)).tocsr()

        return self._recovery

    # ----- boundary facets -----

    def _owners(self):

        """Maps the sorted node ids of every element facet to its [(element, node ids in element order)]."""

        if self._facet_owner is None:
            self._facet_owner = {tuple(sorted(ids)): [(e, ids)] for e, ids in outer_facets(self.conn, self.element)}

        return self._facet_owner

    def boundary_facet_nodes(self, selector=None):

        """Node tuples of the outer-boundary facets whose midpoint satisfies selector(*coords)."""

        return select_facets(outer_facets(self.conn, self.element), self.node_xy, selector)

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

        if self.manifold:
            # Edge of a surface: the outward normal in the plane of the owning element, the part of x - centroid across the edge
            t = tangent[..., 0] / np.linalg.norm(tangent[..., 0], axis=-1, keepdims=True)
            normals = outward - np.sum(outward * t, axis=-1, keepdims=True) * t
            normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        elif self.dim == 1:
            normals = np.sign(outward)
        elif self.dim == 2:
            t = tangent[..., 0]
            normals = np.stack([t[..., 1], -t[..., 0]], axis=-1)
            normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
            flip = np.sign(np.sum(normals * outward, axis=-1, keepdims=True))
            normals = normals * flip
        else:
            normals = np.cross(tangent[..., 0], tangent[..., 1])         # face: product of the two tangents
            normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
            flip = np.sign(np.sum(normals * outward, axis=-1, keepdims=True))
            normals = normals * flip

        detJw = measure * weights[None, :]
        if self.axisymmetric:
            detJw = detJw * x[..., 0]                # surface of revolution: ds -> r ds

        return FacetSet(nodes, elems, Nf, detJw, x, normals)
