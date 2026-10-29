"""Discrete system of a model: matrices, nonlinear terms and boundary conditions, independent of the physics.

A physics builds a System from the operators (diffusion, convection, mass, sources), pointwise
laws (reactions, nonlinear boundary coefficients) and boundary conditions. The solvers only see:

    linear(x)                -> A, b      the system A x = b (Newton-linearized if there are nonlinear terms)
    residual_and_tangent(x)  -> J, R      Newton: dx = -J^-1 R
    mass()                   -> M         for transient studies
    is_dirichlet / dirichlet_value        constrained dofs

Conventions: the residual is R = A x - b + (nonlinear terms). Dirichlet dofs are identity rows with
R = x - value (b = value). Fluxes are outgoing: Neumann q = -k dT/dn, Robin q = h (T - T_ext).
Boundary conditions act on boundary facets (points in 1D, edges in 2D, faces in 3D).
"""

import numpy as np
import scipy.sparse as sp

import Source.Simulation.Operators as ops
from Source.Simulation.PointwiseLaw import PointwiseLaw, cached_law


class VolumeLawTerm:

    """R_out += sign * integral of N_i f_out(u), with u the input fields at the Gauss points."""

    def __init__(self, system, law, inputs, outputs, sign=-1.0):
        self.system = system
        self.law = law
        self.inputs = inputs
        self.outputs = outputs
        self.sign = sign

    def contribution(self, x):

        system, disc, layout = self.system, self.system.disc, self.system.layout

        fields = layout.split(x)
        U = disc.interpolate(np.stack([fields[n] for n in self.inputs], axis=-1))  # (nel, ngp, nin)

        values, derivatives = self.law(U.reshape(-1, len(self.inputs)))
        shape = disc.detJw.shape
        values = values.reshape(shape + (len(self.outputs),))
        derivatives = derivatives.reshape(shape + (len(self.outputs), len(self.inputs)))

        R = np.zeros(layout.ndof)
        J = sp.csr_matrix((layout.ndof, layout.ndof))

        for o, out in enumerate(self.outputs):
            rows = layout.element_dofs(disc.conn, out)

            fe = self.sign * np.einsum('eg,gi,eg->ei', disc.detJw, disc.N, values[..., o])
            R += layout.scatter_vector(fe, rows)

            for j, inp in enumerate(self.inputs):
                cols = layout.element_dofs(disc.conn, inp)
                Ke = self.sign * np.einsum('eg,gi,eg,gj->eij', disc.detJw, disc.N, derivatives[..., o, j], disc.N)
                J = J + layout.scatter_matrix(Ke, rows, cols)

        return R, J


class DiffusionLawTerm:

    """R_field += integral of k(u) grad(field) . grad(N_i), with k a pointwise law of the input fields.

    Nonlinear diffusion: conductivity k(T), diffusivity D(c, T), viscosity, ... The tangent has the
    usual stiffness part, integral of k grad(N_i) . grad(N_j), plus the dependence of k on the inputs,
    integral of (dk/du_m) (grad(field) . grad(N_i)) N_j, which couples the field with every input.
    """

    def __init__(self, system, law, field, inputs):
        self.system = system
        self.law = law
        self.field = field
        self.inputs = inputs

    def contribution(self, x):

        system, disc, layout = self.system, self.system.disc, self.system.layout
        n_in = len(self.inputs)

        fields = layout.split(x)
        U = disc.interpolate(np.stack([fields[n] for n in self.inputs], axis=-1))   # (nel, ngp, nin)

        k, dk = self.law(U.reshape(-1, n_in))
        shape = disc.detJw.shape
        k = k.reshape(shape)
        dk = dk.reshape(shape + (n_in,))

        phi = fields[self.field][disc.conn]                       # (nel, nne)
        grad = np.einsum('egia,ei->ega', disc.B, phi)             # (nel, ngp, dim)

        rows = layout.element_dofs(disc.conn, self.field)

        Re = np.einsum('eg,egia,ega,eg->ei', disc.detJw, disc.B, grad, k)
        R = layout.scatter_vector(Re, rows)

        Ke = np.einsum('eg,eg,egia,egja->eij', disc.detJw, k, disc.B, disc.B)
        J = layout.scatter_matrix(Ke, rows, rows)

        for m, name in enumerate(self.inputs):
            Km = np.einsum('eg,eg,egia,ega,gj->eij', disc.detJw, dk[..., m], disc.B, grad, disc.N)
            J = J + layout.scatter_matrix(Km, rows, layout.element_dofs(disc.conn, name))

        return R, J


def value_gradient_basis(disc, nc):

    """W (nel, ngp, nne, nc, nc * (1 + dim)) mapping the element values of a field to its values and gradients.

    At a Gauss point, w_k = sum_{j, c} W[j, c, k] x_{j c} with w = (u_0, ..., u_{nc-1}, du_0/dx_0, ..., du_0/dx_{dim-1},
    du_1/dx_0, ..., du_{nc-1}/dx_{dim-1}). The same array gives the test functions of a weak residual:
    R_{i a} = integral of sum_k W[i, a, k] r_k, with r = (s_0, ..., s_{nc-1}, F_00, ..., F_{nc-1, dim-1}) tested by
    N_i (s) and by dN_i/dx_b (F).
    """

    nel, ngp, nne, dim = disc.B.shape
    W = np.zeros((nel, ngp, nne, nc, nc * (1 + dim)))

    for c in range(nc):
        W[:, :, :, c, c] = disc.N[None]
        W[:, :, :, c, nc + c * dim: nc + (c + 1) * dim] = disc.B

    return W


class GradientLawTerm:

    """Weak residual given pointwise as a function of the values and gradients of the input fields.

    The law receives, for each input field in order, its values (nc) followed by its gradient (nc * dim, row
    by row: du_0/dx_0, du_0/dx_1, ...). It returns, for each output field in order, a source s (nc), tested by
    N_i, followed by a flux F (nc * dim, row by row), tested by grad(N_i):

        R_{i a} += sign * integral of (N_i s_a + dN_i/dx_b F_ab)

    Examples: nonlinear diffusion F = k(u, |grad u|) grad u (p-Laplacian, power-law fluid), convection
    s = c v . grad(u) with v another field, viscous stress F = 2 mu(gamma_dot) eps(u). The tangent comes from the
    law's automatic derivatives with respect to every value and gradient, so it couples outputs and inputs.
    """

    def __init__(self, system, law, inputs, outputs, sign=1.0):
        self.system = system
        self.law = law
        self.inputs = inputs
        self.outputs = outputs
        self.sign = sign
        self._basis = {}

    def _W(self, nc):

        if nc not in self._basis:
            self._basis[nc] = value_gradient_basis(self.system.disc, nc)

        return self._basis[nc]

    def _slices(self, names):

        dim, layout = self.system.disc.dim, self.system.layout
        out, start = [], 0

        for name in names:
            nc = layout.spec(name).ncomp
            out.append((name, nc, slice(start, start + nc * (1 + dim))))
            start += nc * (1 + dim)

        return out, start

    def contribution(self, x):

        disc, layout = self.system.disc, self.system.layout
        nel, ngp, nne, _ = disc.B.shape

        inputs, n_in = self._slices(self.inputs)
        outputs, n_out = self._slices(self.outputs)

        fields = layout.split(x)
        U = np.concatenate([np.einsum('egjck,ejc->egk', self._W(nc), fields[name][disc.conn].reshape(nel, nne, nc))
                            for name, nc, _ in inputs], axis=-1)

        values, derivatives = self.law(U.reshape(-1, n_in))

        if values.shape[1] != n_out:
            raise ValueError(f"La ley devuelve {values.shape[1]} valores por punto; se esperaban {n_out} "
                             f"(fuente y flujo de {self.outputs}).")

        values = values.reshape(nel, ngp, n_out)
        derivatives = derivatives.reshape(nel, ngp, n_out, n_in)

        R = np.zeros(layout.ndof)
        J = sp.csr_matrix((layout.ndof, layout.ndof))

        for out, nc_o, so in outputs:
            Wo = self._W(nc_o)
            rows = layout.element_dofs(disc.conn, out)

            Re = self.sign * np.einsum('eg,egiak,egk->eia', disc.detJw, Wo, values[..., so], optimize=True)
            R += layout.scatter_vector(Re.reshape(nel, -1), rows)

            for inp, nc_i, si in inputs:
                Ke = self.sign * np.einsum('eg,egiak,egkl,egjcl->eiajc', disc.detJw, Wo, derivatives[..., so, si],
                                           self._W(nc_i), optimize=True)
                J = J + layout.scatter_matrix(Ke.reshape(nel, nne * nc_o, nne * nc_i), rows,
                                              layout.element_dofs(disc.conn, inp))

        return R, J


class FacetLawTerm:

    """R_out += sign * boundary integral of N_i f_out(u), with u the input fields on the facets."""

    def __init__(self, system, law, facets, inputs, outputs, sign=1.0):
        self.system = system
        self.law = law
        self.facets = facets
        self.inputs = inputs
        self.outputs = outputs
        self.sign = sign

    def contribution(self, x):

        layout, fs = self.system.layout, self.facets

        fields = layout.split(x)
        U_nodes = np.stack([fields[n] for n in self.inputs], axis=-1)           # (n_nodes, nin)
        U = np.einsum('qk,fkn->fqn', fs.N, U_nodes[fs.nodes])                   # (nf, nq, nin)

        values, derivatives = self.law(U.reshape(-1, len(self.inputs)))
        shape = fs.detJw.shape
        values = values.reshape(shape + (len(self.outputs),))
        derivatives = derivatives.reshape(shape + (len(self.outputs), len(self.inputs)))

        R = np.zeros(layout.ndof)
        J = sp.csr_matrix((layout.ndof, layout.ndof))

        for o, out in enumerate(self.outputs):
            rows = layout.element_dofs(fs.nodes, out)

            fe = self.sign * np.einsum('fq,qi,fq->fi', fs.detJw, fs.N, values[..., o])
            R += layout.scatter_vector(fe, rows)

            for j, inp in enumerate(self.inputs):
                cols = layout.element_dofs(fs.nodes, inp)
                Ke = self.sign * np.einsum('fq,qi,fq,qj->fij', fs.detJw, fs.N, derivatives[..., o, j], fs.N)
                J = J + layout.scatter_matrix(Ke, rows, cols)

        return R, J


class System:

    def __init__(self, disc, layout):

        self.disc = disc
        self.layout = layout

        n = layout.ndof
        self._A = sp.csr_matrix((n, n))
        self._b = np.zeros(n)
        self._M = sp.csr_matrix((n, n))
        self._terms = []

        # Time derivative of the current step, set by the transient solver: x' = alpha x + beta, with the factor
        # `scale` for the terms that add it to their residual and the step `dt`. None in steady studies.
        self.time = None

        self.is_dirichlet = np.zeros(n, dtype=bool)
        self.dirichlet_value = np.zeros(n)

    @property
    def is_linear(self):

        """True if the system has no nonlinear terms: A, b and the tangent do not depend on the state."""

        return not self._terms

    # ----- building blocks -----

    def add_matrix(self, A):
        self._A = self._A + A

    def add_vector(self, b):

        """Adds b to the right-hand side (the residual is A x - b)."""

        self._b = self._b + b

    def add_mass(self, M):
        self._M = self._M + M

    def add_volume_law(self, law, inputs, outputs, sign=-1.0):
        self._terms.append(VolumeLawTerm(self, law, list(inputs), list(outputs), sign))

    def add_diffusion_law(self, law, field, inputs=None):

        """Nonlinear diffusion of `field` with coefficient k(inputs) given by a pointwise law (default input: the field)."""

        self._terms.append(DiffusionLawTerm(self, law, field, list(inputs) if inputs else [field]))

    def add_gradient_law(self, law, inputs, outputs, sign=1.0):

        """Weak residual sign * integral (N s + grad(N) . F) with s, F a law of the values and gradients of the inputs.

        See GradientLawTerm for how the law's inputs and outputs are packed.
        """

        self._terms.append(GradientLawTerm(self, law, list(inputs), list(outputs), sign))

    def add_term(self, term):

        """Any nonlinear term with contribution(x) -> (R, J), e.g. a physics-specific vectorized term."""

        self._terms.append(term)

    def add_facet_law(self, law, facets, inputs, outputs, sign=1.0):
        self._terms.append(FacetLawTerm(self, law, facets, list(inputs), list(outputs), sign))

    def set_dirichlet(self, dofs, values):
        self.is_dirichlet[dofs] = True
        self.dirichlet_value[dofs] = values

    def add_traction(self, field, facets, t):

        """Traction t (force per unit area acting on the domain) on a vector field: b += boundary integral of t . v."""

        rows = self.layout.element_dofs(facets.nodes, field)
        self.add_vector(self.layout.scatter_vector(ops.facet_vector_source(facets, t), rows))

    def add_normal_pressure(self, field, facets, p):

        """Normal traction t = -p n (p a number or a function of the coordinates): a pressure pushing on the boundary."""

        self.add_traction(field, facets, -ops.at_facets(facets, p)[..., None] * facets.normals)

    # ----- boundary conditions on facets (any scalar field) -----

    def add_flux(self, field, facets, q):

        """Outgoing flux q (number, or a function of the field value) through the facets."""

        rows = self.layout.element_dofs(facets.nodes, field)

        if callable(q):
            law = cached_law(('flux', q), lambda: PointwiseLaw(lambda u: q(u[0])))
            self.add_facet_law(law, facets, [field], [field], sign=1.0)
        else:
            self.add_vector(-self.layout.scatter_vector(ops.facet_source(facets, q), rows))

    def add_robin(self, field, facets, h, ext):

        """Outgoing flux h (T - ext) through the facets; h is a number or a function of the field value."""

        rows = self.layout.element_dofs(facets.nodes, field)

        if callable(h):
            law = cached_law(('robin', h, ext), lambda: PointwiseLaw(lambda u: h(u[0]) * (u[0] - ext)))
            self.add_facet_law(law, facets, [field], [field], sign=1.0)
        else:
            self.add_matrix(self.layout.scatter_matrix(ops.facet_mass(facets, h), rows, rows))
            self.add_vector(self.layout.scatter_vector(ops.facet_source(facets, h * ext), rows))

    def add_boundary_conditions(self, mesh, facet_bcs=()):

        """Reads the conditions of the mesh nodes and the facet conditions of the physics.

        Node conditions: Dirichlet on any dimension; Newton and Neumann on a node (1D only, where
        the boundary facet is a point). Facet conditions: dicts with 'type', 'field', 'facets' and
        the values ('h' and 'var_ext' for Newton, 'flux' for Neumann, 'traction' for Traction, 'pressure' for
        NormalTraction).

        A Dirichlet value of a vector field is a sequence with one entry per component; None leaves that
        component free (a symmetry or slip wall fixes only the normal component).
        """

        layout, disc = self.layout, self.disc
        entries = []

        for node in mesh.NL:
            for field in layout.names:

                bc = node.BC.get(field) if node.BC else None

                if not bc:
                    continue

                if bc['type'] == 'Dirichlet':
                    dofs = layout.node_dofs(field, node.id)
                    if len(dofs) == 1:
                        self.set_dirichlet(dofs[0], bc['value'])
                    else:
                        values = list(bc['value'])
                        if len(values) != len(dofs):
                            raise ValueError(f"El campo '{field}' tiene {len(dofs)} componentes y la condicion "
                                             f"del nodo {node.id} da {len(values)}.")
                        for dof, value in zip(dofs, values):
                            if value is not None:
                                self.set_dirichlet(dof, value)

                elif disc.dim != 1:
                    raise ValueError("En 2D y 3D las BC Newton y Neumann se asignan a aristas o caras (addBC_*Edges).")

                else:
                    entries.append(dict(bc, field=field, facets=[(node.id,)]))

        for bc in facet_bcs:
            entries.append(dict(bc, field=bc.get('field', layout.names[0])))

        for entry in entries:

            facets = disc.facets(entry['facets'])

            if entry['type'] == 'Newton':
                self.add_robin(entry['field'], facets, entry['h'], entry['var_ext'])
            elif entry['type'] == 'Neumann':
                self.add_flux(entry['field'], facets, entry['flux'])
            elif entry['type'] == 'Traction':
                self.add_traction(entry['field'], facets, entry['traction'])
            elif entry['type'] == 'NormalTraction':
                self.add_normal_pressure(entry['field'], facets, entry['pressure'])
            else:
                raise ValueError(f"Tipo de condicion de contorno desconocido: {entry['type']!r}.")

    # ----- what the solvers use -----

    def _apply_dirichlet_matrix(self, A):

        free = (~self.is_dirichlet).astype(float)

        return (sp.diags(free) @ A + sp.diags(1.0 - free)).tocsr()

    def _assemble(self, x):

        R = self._A @ x - self._b
        J = self._A

        for term in self._terms:
            Rt, Jt = term.contribution(x)
            R = R + Rt
            J = J + Jt

        return J, R

    def linear(self, x):

        """(A, b) of the system. With nonlinear terms it is the Newton linearization at x: A = J, b = J x - R."""

        if self._terms:
            J, R = self._assemble(x)
            A, b = J, J @ x - R
        else:
            A, b = self._A, self._b

        return self._apply_dirichlet_matrix(A), np.where(self.is_dirichlet, self.dirichlet_value, b)

    def residual_and_tangent(self, x):

        J, R = self._assemble(x)

        R = np.where(self.is_dirichlet, x - self.dirichlet_value, R)

        return self._apply_dirichlet_matrix(J), R

    def mass(self):
        return self._M
