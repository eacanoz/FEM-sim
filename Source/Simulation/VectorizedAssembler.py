"""Vectorized assembly for 1D Lagrange line elements (NumPy + einsum).

The element matrices of all elements are computed at once from reference data that is
precomputed once per mesh, and then scattered into a sparse matrix. This replaces the
per-element JAX calls of the element-based path.

Limits (kept on purpose, to be extended later):
- constant coefficients only (K, C, M and F must be numbers);
- one scalar variable per model (heat transfer);
- point conditions act on nodes: Dirichlet (identity rows), Newton h(T)*(T - T_ext) and
  Neumann flux. Each boundary node is counted once, even if it belongs to several elements.
"""

import numpy as np
import scipy.sparse as sp
import jax

from numpy.polynomial.legendre import leggauss

from Source.Pre_processing.BasisFunctions import shape_mappings

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


class LinearGeometry1D:

    """Reference data and element geometry of a 1D mesh. Computed once and reused by every assembly."""

    def __init__(self, mesh):

        if mesh.PD != 1:
            raise NotImplementedError("LinearGeometry1D solo sirve para mallas 1D.")

        ref_nodes = shape_mappings[mesh.shape_function]

        N, dN = lagrange_1d(ref_nodes, GAUSS_XI)  # (ngp, nne): values and d/dxi at the Gauss points

        self.conn = np.array([e.getNodesId() for e in mesh.EL], dtype=int)  # (nel, nne)
        coords = np.array([e.getCoor() for e in mesh.EL], dtype=float)      # (nel, nne)

        J = coords @ dN.T  # (nel, ngp): dx/dxi at each Gauss point
        if np.any(np.abs(J) < 1e-14):
            raise ValueError("Elemento degenerado: jacobiano casi nulo.")

        self.N = N
        self.B = dN[None, :, :] / J[:, :, None]  # (nel, ngp, nne): dN/dx
        self.detJw = np.abs(J) * GAUSS_W        # (nel, ngp): |detJ| * weight
        self.n_nodes = mesh.getNoN()

    def _scatter(self, Ke):

        """Sums the element matrices Ke (nel, nne, nne) into a global sparse matrix."""

        nne = self.conn.shape[1]
        rows = np.repeat(self.conn[:, :, None], nne, axis=2).ravel()
        cols = np.repeat(self.conn[:, None, :], nne, axis=1).ravel()

        return sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(self.n_nodes, self.n_nodes)).tocsr()

    def stiffness(self, k):

        """Integral of k * dN_i/dx * dN_j/dx."""

        return self._scatter(k * np.einsum('eg,egi,egj->eij', self.detJw, self.B, self.B))

    def convection(self, c, vel):

        """Integral of N_i * c * vel * dN_j/dx (test function i, trial function j)."""

        return self._scatter(c * vel * np.einsum('eg,gi,egj->eij', self.detJw, self.N, self.B))

    def mass(self, c):

        """Integral of c * N_i * N_j."""

        return self._scatter(c * np.einsum('eg,gi,gj->eij', self.detJw, self.N, self.N))

    def load(self, f):

        """Integral of f * N_i (constant f)."""

        fe = f * np.einsum('eg,gi->ei', self.detJw, self.N)

        F = np.zeros(self.n_nodes)
        np.add.at(F, self.conn.ravel(), fe.ravel())

        return F


class HeatSystem1D:

    """Discrete heat-conduction problem of a 1D model, built from a LinearGeometry1D.

    linear(T)                 -> A, b          for the linear problem (A @ T = b)
    residual_and_tangent(T)   -> J, R          for Newton (dT = -J^-1 R)

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

        self.F = geometry.load(_constant(phys.F_const, 'F'))

        # Boundary conditions, read from the nodes
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

            elif bc['type'] == 'Newton':
                self.newton.append((node.id, bc['h'], bc['var_ext']))

            elif bc['type'] == 'Neumann':
                self.neumann.append((node.id, bc['flux']))

    def _newton_terms(self, T):

        """h(T_i), dh/dT(T_i) and T_ext for every Newton node."""

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

        """Neumann flux at every node (zero where there is none)."""

        q = np.zeros(self.geo.n_nodes)

        for node_id, flux in self.neumann:
            q[node_id] = float(flux(T[node_id])) if callable(flux) else float(flux)

        return q

    def _apply_dirichlet_matrix(self, A):

        """Identity rows at Dirichlet nodes; the rest of the matrix is unchanged."""

        free = (~self.is_dirichlet).astype(float)

        return (sp.diags(free) @ A + sp.diags(1.0 - free)).tocsr()

    def linear(self, T):

        """(A, b) of the linear problem for the nodal state T."""

        n = self.geo.n_nodes
        ids, h, _, t_ext = self._newton_terms(T)

        A = self.K + self.C + sp.csr_matrix((h, (ids, ids)), shape=(n, n))

        b = self.F.copy()
        b[ids] += h * t_ext                   # Newton: -G = h * T_ext
        b -= self._neumann_vector(T)          # Neumann: -G = -flux

        A = self._apply_dirichlet_matrix(A)
        b = np.where(self.is_dirichlet, self.dirichlet_value, b)

        return A, b

    def residual_and_tangent(self, T):

        """Residual R(T) = (K + C) T - F + h(T)(T - T_ext) + flux and its Jacobian dR/dT."""

        n = self.geo.n_nodes
        ids, h, dh, t_ext = self._newton_terms(T)

        KC = self.K + self.C

        R = KC @ T - self.F + self._neumann_vector(T)
        R[ids] += h * (T[ids] - t_ext)

        J = KC + sp.csr_matrix((h + dh * (T[ids] - t_ext), (ids, ids)), shape=(n, n))
        J = self._apply_dirichlet_matrix(J)

        R = np.where(self.is_dirichlet, T - self.dirichlet_value, R)

        return J, R
