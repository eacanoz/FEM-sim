"""Degrees of freedom of a model: which fields exist and where each of their components lives.

The numbering is interleaved by node: dof = node * n_total + offset(field) + component, which keeps
all the unknowns of a node together (good for ILU and block structure). A scalar field has one
component (temperature, concentration); a vector field has `dim` (displacement, velocity).
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class FieldSpec:

    name: str
    ncomp: int = 1


class DofLayout:

    def __init__(self, fields, n_nodes):

        self.fields = tuple(fields)
        self.n_nodes = n_nodes

        self._spec = {f.name: f for f in self.fields}
        self._offset = {}

        offset = 0
        for f in self.fields:
            self._offset[f.name] = offset
            offset += f.ncomp

        self.n_total = offset            # components per node, all fields together
        self.ndof = n_nodes * offset

    @property
    def names(self):
        return [f.name for f in self.fields]

    def spec(self, name):
        return self._spec[name]

    def node_dofs(self, name, nodes):

        """Dofs of `name` at the given nodes: shape nodes.shape + (ncomp,)."""

        nodes = np.asarray(nodes, dtype=int)
        comps = np.arange(self._spec[name].ncomp)

        return nodes[..., None] * self.n_total + self._offset[name] + comps

    def dofs(self, name):

        """Dofs of every node for `name`: (n_nodes, ncomp)."""

        return self.node_dofs(name, np.arange(self.n_nodes))

    def element_dofs(self, conn, name):

        """Dofs of each element for `name`, node-major: (nel, nne * ncomp)."""

        d = self.node_dofs(name, conn)

        return d.reshape(len(conn), -1)

    def split(self, x):

        """Solution vector -> {field: (n_nodes,) for scalars, (n_nodes, ncomp) for vectors}."""

        x = np.asarray(x)
        out = {}

        for f in self.fields:
            d = self.dofs(f.name)
            out[f.name] = x[d[:, 0]] if f.ncomp == 1 else x[d]

        return out

    def join(self, values):

        """{field: nodal values} -> solution vector."""

        x = np.zeros(self.ndof)

        for f in self.fields:
            d = self.dofs(f.name)
            v = np.asarray(values[f.name], dtype=float)
            x[d] = v.reshape(d.shape)

        return x

    def field_mask(self, name):

        """Boolean mask over all dofs that belong to `name`."""

        mask = np.zeros(self.ndof, dtype=bool)
        mask[self.dofs(name).ravel()] = True

        return mask

    def scatter_matrix(self, Ke, rows, cols):

        """Sums element blocks Ke (ne, nr, nc) into a global sparse matrix.

        rows (ne, nr) and cols (ne, nc) are the dofs of the test and of the solution field of each element,
        so blocks that couple two different fields use the same call.
        """

        nr, nc = rows.shape[1], cols.shape[1]

        r = np.repeat(rows[:, :, None], nc, axis=2).ravel()
        c = np.repeat(cols[:, None, :], nr, axis=1).ravel()

        return sp.coo_matrix((Ke.ravel(), (r, c)), shape=(self.ndof, self.ndof)).tocsr()

    def scatter_vector(self, fe, rows):

        """Sums element vectors fe (ne, nr) into a global vector."""

        F = np.zeros(self.ndof)
        np.add.at(F, rows.ravel(), fe.ravel())

        return F
