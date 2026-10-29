# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++++

# Darcy flow in porous media: the pressure is the unknown and the velocity follows from its gradient.
#
#     u = -(kappa / mu) (grad p - rho g),      div u = Q,      S dp/dt + div u = Q (with storage)
#
# Weak form: integral of (kappa/mu) (grad p - rho g) . grad(q) - integral of Q q + boundary integral of (u . n) q = 0,
# so the outgoing normal velocity u . n plays the role of the outgoing flux of the scalar physics. Walls are
# impermeable by default (u . n = 0). The velocity can convect heat or species on the same mesh:
#
#     flow.solve()
#     transport.physics.setVelocity(flow.physics.velocity())     # Gauss-point array, same mesh


import numpy as np

from Source.Physics.Physics import physics
from Source.Primals.Scalar import scalarField
import Source.Simulation.Operators as ops
from Source.enums import ShapeFunctionType


class darcy(physics):

    part_name = 'flow'
    ConstRel = 'S dp/dt - ∇.((κ/μ)(∇p - ρ g)) == Q'

    def __init__(self, model):
        super().__init__(model)

        self.physics_description = 'Darcy flow'

        self.var = {'p': scalarField('p', 'Pressure', 'Pa', ShapeFunctionType.linear, model.mesh)}

        self.kappa = getattr(self.mat, 'kappa', 1.0)   # permeability [m2]: number, f(x, y) or tensor (dim, dim)
        self.mu = getattr(self.mat, 'miu', 1.0)        # dynamic viscosity [Pa s]
        self.rho = getattr(self.mat, 'rho', 0.0)       # density [kg/m3], only with gravity
        self.gravity = None                            # gravity vector [m/s2]
        self.M_const = 0.0                             # specific storage S [1/Pa] (transient)
        self.F_const = 0.0                             # volumetric source Q [1/s]

    # ----- properties -----

    def setPermeability(self, kappa):

        """Permeability [m2]: a number, a function of the coordinates or an anisotropic tensor (dim, dim)."""

        self.kappa = kappa

    def setViscosity(self, mu):
        self.mu = float(mu)

    def setGravity(self, g, rho=None):

        """Gravity vector [m/s2], e.g. (0, -9.81); rho [kg/m3] defaults to the material density."""

        self.gravity = np.asarray(g, dtype=float)
        if rho is not None:
            self.rho = float(rho)

    def setStorage(self, S):

        """Specific storage S [1/Pa] (compressibility of fluid and matrix) for transient studies."""

        self.M_const = S

    def setSource(self, Q):

        """Volumetric source Q [1/s] (injected volume per unit volume and time): number or function of the coordinates."""

        self.F_const = Q

    def _tensor(self):
        return np.ndim(self.kappa) == 2 and not callable(self.kappa)

    def _mobility(self, disc):

        """kappa / mu at the Gauss points: (nel, ngp), or (nel, ngp, dim, dim) for a tensor permeability."""

        if self._tensor():
            return np.broadcast_to(np.asarray(self.kappa, dtype=float) / self.mu, disc.detJw.shape + (disc.dim, disc.dim))

        return ops.at_gauss(disc, self.kappa) / self.mu

    # ----- weak form -----

    def add_terms(self, system, model, disc, layout):

        rows = layout.element_dofs(disc.conn, 'p')
        K = self._mobility(disc)

        if self._tensor():
            system.add_matrix(layout.scatter_matrix(ops.diffusion_tensor(disc, K), rows, rows))
        else:
            system.add_matrix(layout.scatter_matrix(ops.diffusion(disc, K), rows, rows))

        if self.gravity is not None:
            system.add_vector(layout.scatter_vector(ops.gradient_source(disc, self._body_flux(disc, K)), rows))

        system.add_vector(layout.scatter_vector(ops.source(disc, self.F_const), rows))
        system.add_mass(layout.scatter_matrix(ops.mass(disc, self.M_const), rows, rows))

    def _body_flux(self, disc, K):

        """(kappa/mu) rho g at the Gauss points: (nel, ngp, dim)."""

        g = self.rho * np.broadcast_to(self.gravity, disc.detJw.shape + (disc.dim,))

        return np.einsum('egab,egb->ega', K, g) if self._tensor() else K[..., None] * g

    # ----- velocity -----

    def velocity(self, at='gauss', pressure=None):

        """Darcy velocity u = -(kappa/mu)(grad p - rho g) of the current pressure (or of `pressure`, nodal values).

        at='gauss': (nel, ngp, dim) array, ready for setVelocity of a physics on the same mesh.
        at='nodes': (n_nodes, dim) by lumped L2 projection (exact for a linear pressure).
        """

        disc = self.modelRef.geometry
        p = self.var['p'].values if pressure is None else np.asarray(pressure, dtype=float)

        K = self._mobility(disc)
        grad = disc.gradient(p)
        u = -(np.einsum('egab,egb->ega', K, grad) if self._tensor() else K[..., None] * grad)

        if self.gravity is not None:
            u = u + self._body_flux(disc, K)

        if at == 'gauss':
            return u
        if at == 'nodes':
            return disc.project(u)

        raise ValueError(f"at debe ser 'gauss' o 'nodes', no {at!r}.")

    def derived_fields(self, model):
        return {'u': self.velocity('nodes')}

    # ----- boundary conditions -----

    def addBC_Pressure(self, id: int, p: float):

        self.modelRef._mesh.NL[id].BC['p'] = self.setDirichletBC(p)

    def addBC_PressureEdges(self, edges, p):

        """Pressure on every node of the edges [(a, b), ...]: a number or a function p(x, y)."""

        nodes = self.modelRef._mesh.NL

        for edge in edges:
            for node_id in edge:
                value = p(*nodes[node_id].coor[:2]) if callable(p) else p
                nodes[node_id].BC['p'] = self.setDirichletBC(value)

    def addBC_NormalVelocity(self, id: int, u_n: float):

        """Outgoing normal velocity at a node (1D): positive leaves the domain, negative is an injection."""

        self.modelRef._mesh.NL[id].BC['p'] = self.setNeumannBC(u_n)

    def addBC_NormalVelocityEdges(self, edges, u_n):

        """Outgoing normal Darcy velocity u . n [m/s] through the edges: positive leaves the domain (outflow),
        negative enters (injection). Edges without a condition are impermeable."""

        self.facet_bcs.append({'type': 'Neumann', 'field': 'p', 'facets': list(edges), 'flux': u_n})
