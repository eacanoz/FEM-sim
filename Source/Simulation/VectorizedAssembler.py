"""Vectorized assembly for linear finite elements (NumPy + einsum).

Supported elements:
- 1D line elements (Lagrange, 2 Gauss points);
- 2D bilinear quadrilaterals Q4 (2x2 Gauss points), used for steady heat conduction.

The element matrices of all elements are computed at once from reference data that is
precomputed once per mesh, and then scattered into a sparse matrix.

Limits (kept on purpose, to be extended later):
- constant coefficients K and C (the source F may be a function of x, y);
- convection only in 1D;
- one scalar variable per model (heat transfer);
- point conditions (1D only): Dirichlet (identity rows), Newton h*(T - T_ext) and Neumann flux
  act on nodes. In 2D the Newton and Neumann conditions act on edges (integrals along the edge).
- nonlinear problems (residual_and_tangent) only in 1D.
"""

import numpy as np
import scipy.sparse as sp
import jax

from numpy.polynomial.legendre import leggauss

from Source.Pre_processing.BasisFunctions import shape_mappings, q4_shape_functions

GAUSS_XI, GAUSS_W = leggauss(2)  # same quadrature as the element-based code


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


def _constant(value, name):

    if callable(value):
        raise NotImplementedError(f"{name} como funcion no esta soportado en el ensamblaje vectorizado.")

    return float(value)


class LinearGeometry:

    """Reference data and element geometry of a 1D or 2D mesh. Computed once and reused by every assembly.

    Shapes: N (ngp, nne); B (nel, ngp, nne, dim) = dN/dx; detJw (nel, ngp) = |detJ| * weight;
    x_gp (nel, ngp, dim) physical coordinates of the Gauss points.
    """

    def __init__(self, mesh):

        self.dim = mesh.PD
        self.n_nodes = mesh.getNoN()

        self.conn = np.array([e.getNodesId() for e in mesh.EL], dtype=int)  # (nel, nne)
        self.node_xy = np.array([node.coor[:self.dim] for node in mesh.NL], dtype=float)

        coords = np.array([e.getCoor() for e in mesh.EL], dtype=float)

        if self.dim == 1:
            coords = coords.reshape(len(mesh.EL), -1, 1)
            N, dN = lagrange_1d(shape_mappings[mesh.shape_function], GAUSS_XI)  # (ngp, nne)
            dN = dN[:, :, None]
            gauss_w = GAUSS_W

        elif self.dim == 2:
            xi, eta = np.meshgrid(GAUSS_XI, GAUSS_XI, indexing='ij')
            N, dN = q4_shape_functions(xi.ravel(), eta.ravel())  # (ngp, 4), (ngp, 4, 2)
            gauss_w = np.outer(GAUSS_W, GAUSS_W).ravel()

        else:
            raise NotImplementedError("Solo se soportan mallas 1D y 2D.")

        # J[e, g, a, b] = dx_a / dxi_b
        J = np.einsum('eka,gkb->egab', coords, dN)
        detJ = np.linalg.det(J)
        if np.any(np.abs(detJ) < 1e-14):
            raise ValueError("Elemento degenerado: jacobiano casi nulo.")

        invJ = np.linalg.inv(J)  # invJ[e, g, b, a] = dxi_b / dx_a

        self.N = N
        self.B = np.einsum('gkb,egba->egka', dN, invJ)
        self.detJw = np.abs(detJ) * gauss_w
        self.x_gp = np.einsum('gk,eka->ega', N, coords)

        # 1D Lagrange basis on the reference edge, for 2D boundary integrals
        self.edge_N, _ = lagrange_1d((-1.0, 1.0), GAUSS_XI)

    # ----- element matrices -----

    def _scatter(self, Ke, conn=None):

        """Sums element matrices Ke (ne, nne, nne) into a global sparse matrix."""

        conn = self.conn if conn is None else conn
        nne = conn.shape[1]

        rows = np.repeat(conn[:, :, None], nne, axis=2).ravel()
        cols = np.repeat(conn[:, None, :], nne, axis=1).ravel()

        return sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(self.n_nodes, self.n_nodes)).tocsr()

    def _scatter_vector(self, fe, conn=None):

        """Sums element vectors fe (ne, nne) into a global vector."""

        conn = self.conn if conn is None else conn

        F = np.zeros(self.n_nodes)
        np.add.at(F, conn.ravel(), fe.ravel())

        return F

    def stiffness(self, k):

        """Integral of k * grad(N_i) . grad(N_j)."""

        return self._scatter(k * np.einsum('eg,egia,egja->eij', self.detJw, self.B, self.B))

    def convection(self, c, vel):

        """Integral of N_i * c * vel * dN_j/dx (test function i, trial function j). 1D only."""

        if self.dim != 1:
            raise NotImplementedError("La conveccion vectorial en 2D no esta soportada todavia.")

        return self._scatter(c * vel * np.einsum('eg,gi,egj->eij', self.detJw, self.N, self.B[..., 0]))

    def mass(self, c):

        """Integral of c * N_i * N_j."""

        return self._scatter(c * np.einsum('eg,gi,gj->eij', self.detJw, self.N, self.N))

    def _values_at_gauss(self, f):

        """Source values at the Gauss points; f is a number or a function of (x, y) / (x)."""

        if callable(f):
            values = f(*[self.x_gp[..., a] for a in range(self.dim)])
        else:
            values = f

        return np.broadcast_to(np.asarray(values, dtype=float), self.detJw.shape)

    def load(self, f):

        """Integral of f * N_i. f is a number or a function evaluated at the Gauss points."""

        fe = np.einsum('eg,gi,eg->ei', self.detJw, self.N, self._values_at_gauss(f))

        return self._scatter_vector(fe)

    # ----- boundary edges (2D) -----

    def edge_lengths(self, edges):

        """Length of each edge (edges: int array (E, 2) of node ids)."""

        return np.linalg.norm(self.node_xy[edges[:, 1]] - self.node_xy[edges[:, 0]], axis=1)

    def edge_mass(self, lengths):

        """Integral of N_i * N_j along each edge of length L: (E, 2, 2)."""

        return np.einsum('e,g,gi,gj->eij', lengths / 2, GAUSS_W, self.edge_N, self.edge_N)


class HeatSystem:

    """Discrete heat-conduction problem of a model, built from a LinearGeometry.

    linear(T)                 -> A, b          for the linear problem (A @ T = b)
    residual_and_tangent(T)   -> J, R          for Newton, 1D only (dT = -J^-1 R)

    The convention matches the element-based code: Dirichlet rows are identity rows with
    right-hand side equal to the prescribed value.
    """

    def __init__(self, model, geometry, variable=None):

        self.geo = geometry

        phys = model.physics
        self.variable = variable or next(iter(phys.var))

        n = geometry.n_nodes

        self.K = geometry.stiffness(_constant(phys.K_const, 'K'))

        if phys.Convection:
            self.C = geometry.convection(_constant(phys.C_const, 'C'), _constant(phys.vel, 'vel'))
        else:
            self.C = sp.csr_matrix((n, n))

        self.F = geometry.load(phys.F_const)

        # Conditions on nodes (Dirichlet, and Newton/Neumann in 1D)
        self.is_dirichlet = np.zeros(n, dtype=bool)
        self.dirichlet_value = np.zeros(n)
        self.newton = []   # (node id, h or h(T), T_ext)
        self.neumann = []  # (node id, flux or flux(T))

        for node in model.mesh.NL:

            bc = node.BC.get(self.variable) if node.BC else None

            if not bc:
                continue

            if bc['type'] == 'Dirichlet':
                self.is_dirichlet[node.id] = True
                self.dirichlet_value[node.id] = bc['value']

            elif geometry.dim == 2:
                raise ValueError("En 2D las BC Newton y Neumann se asignan a aristas (addBC_*Edges).")

            elif bc['type'] == 'Newton':
                self.newton.append((node.id, bc['h'], bc['var_ext']))

            elif bc['type'] == 'Neumann':
                self.neumann.append((node.id, bc['flux']))

        # Conditions on edges (2D): integrals along each edge
        self.edge_newton = []   # (edges, lengths, h, T_ext)
        self.edge_neumann = []  # (edges, lengths, flux)

        for bc in getattr(phys, 'edge_bcs', []):

            edges = np.array(bc['edges'], dtype=int).reshape(-1, 2)
            lengths = geometry.edge_lengths(edges)

            if bc['type'] == 'Newton':
                self.edge_newton.append((edges, lengths, _constant(bc['h'], 'h'), _constant(bc['var_ext'], 'T_ext')))
            else:
                self.edge_neumann.append((edges, lengths, _constant(bc['flux'], 'flux')))

    def _newton_terms(self, T):

        """h(T_i), dh/dT(T_i) and T_ext for every Newton node (1D point conditions)."""

        ids, h, dh, t_ext = [], [], [], []

        for node_id, h_bc, t_inf in self.newton:

            t = T[node_id]

            if callable(h_bc):
                h_val = float(h_bc(t))
                dh_val = float(jax.grad(h_bc)(t))
            else:
                h_val, dh_val = float(h_bc), 0.0

            ids.append(node_id)
            h.append(h_val)
            dh.append(dh_val)
            t_ext.append(float(t_inf))

        return np.array(ids, dtype=int), np.array(h), np.array(dh), np.array(t_ext)

    def _neumann_vector(self, T):

        """Neumann flux at every node (zero where there is none), 1D point conditions."""

        q = np.zeros(self.geo.n_nodes)

        for node_id, flux in self.neumann:
            q[node_id] = float(flux(T[node_id])) if callable(flux) else float(flux)

        return q

    def _apply_dirichlet_matrix(self, A):

        """Identity rows at Dirichlet nodes; the rest of the matrix is unchanged."""

        free = (~self.is_dirichlet).astype(float)

        return (sp.diags(free) @ A + sp.diags(1.0 - free)).tocsr()

    def linear(self, T):

        """(A, b) of the linear problem. T is used only by nonlinear point conditions (1D)."""

        n = self.geo.n_nodes
        ids, h, _, t_ext = self._newton_terms(T)

        A = self.K + self.C + sp.csr_matrix((h, (ids, ids)), shape=(n, n))

        b = self.F.copy()
        b[ids] += h * t_ext                   # Newton point: -G = h * T_ext
        b -= self._neumann_vector(T)          # Neumann point: -G = -flux

        for edges, lengths, h_e, t_ext_e in self.edge_newton:
            A = A + self.geo._scatter(h_e * self.geo.edge_mass(lengths), edges)
            b = b + self.geo._scatter_vector(np.repeat(lengths[:, None] / 2, 2, axis=1) * (h_e * t_ext_e), edges)

        for edges, lengths, q in self.edge_neumann:
            b = b - self.geo._scatter_vector(np.repeat(lengths[:, None] / 2, 2, axis=1) * q, edges)

        A = self._apply_dirichlet_matrix(A)
        b = np.where(self.is_dirichlet, self.dirichlet_value, b)

        return A, b

    def residual_and_tangent(self, T):

        """Residual R(T) = (K + C) T - F + h(T)(T - T_ext) + flux and its Jacobian dR/dT. 1D only."""

        if self.geo.dim != 1:
            raise NotImplementedError("El caso no lineal solo esta implementado en 1D.")

        n = self.geo.n_nodes
        ids, h, dh, t_ext = self._newton_terms(T)

        KC = self.K + self.C

        R = KC @ T - self.F + self._neumann_vector(T)
        R[ids] += h * (T[ids] - t_ext)

        J = KC + sp.csr_matrix((h + dh * (T[ids] - t_ext), (ids, ids)), shape=(n, n))
        J = self._apply_dirichlet_matrix(J)

        R = np.where(self.is_dirichlet, T - self.dirichlet_value, R)

        return J, R


# Names used by the 1D code and tests
LinearGeometry1D = LinearGeometry
HeatSystem1D = HeatSystem
