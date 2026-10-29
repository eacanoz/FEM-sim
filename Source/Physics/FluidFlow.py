# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++++

# Incompressible viscous flow: velocity u (vector) and pressure p on the same nodes (Q1/P1 equal order).
#
#     rho du/dt - div(2 mu eps(u)) + grad p = f,      div u = 0,      eps(u) = (grad u + grad u^T) / 2
#
# Equal-order elements do not satisfy the inf-sup condition, so the continuity equation carries the
# pressure-stabilizing Petrov-Galerkin term (PSPG, Hughes-Franca-Balestra / Tezduyar):
#
#     R_u = integral 2 mu eps(u):eps(v) - integral p div v + integral rho du/dt . v - integral f . v - boundary integral t . v
#     R_p = integral q div u + integral tau grad(q) . (rho du/dt + grad p - f)
#
# The viscous second derivatives of the momentum residual are left out of the PSPG term (they vanish on linear
# triangles and are small on bilinear quadrilaterals). tau = h^2 / (12 mu) for creeping flow (Operators.pspg_tau).
# A constant q gives integral div u = 0 exactly: the discrete flow rate is conserved globally.
#
# Boundary conditions: velocity (any component, None leaves it free), no slip, symmetry/slip on walls aligned
# with the axes (normal component fixed), outlet (normal traction -p n, optionally fully developed: tangential
# velocity zero) and general tractions. Walls without a condition are traction free. If no boundary sets the
# pressure level (all of it has velocity conditions) the pressure is fixed at one node.


import numpy as np
import scipy.sparse as sp

from Source.Physics.Physics import physics
from Source.Primals.Scalar import scalarField
from Source.Primals.Vector import vectorField
import Source.Simulation.Operators as ops
from Source.enums import ShapeFunctionType, StudyType
from Source.logger_config import sim_logger


class ff(physics):

    part_name = 'fluid'
    ConstRel = 'ρ ∂u/∂t - ∇.(2μ ε(u)) + ∇p == f,  ∇.u == 0'

    def __init__(self, model):
        super().__init__(model)

        self.physics_description = 'Creeping flow (Stokes)'

        dim = model._PD
        self.dim = dim

        self.var = {'u': vectorField('u', 'Velocity', 'm/s', ShapeFunctionType.linear, model.mesh, ncomp=dim),
                    'p': scalarField('p', 'Pressure', 'Pa', ShapeFunctionType.linear, model.mesh)}
        self.var['p'].initField(0.0)

        self.mu = getattr(self.mat, 'miu', 1.0)      # dynamic viscosity [Pa s]: number or f(x, y)
        self.rho = getattr(self.mat, 'rho', 0.0)     # density [kg/m3]: transient term and gravity
        self.body_force = None                       # force per unit volume [N/m3]: vector or f(x, y) -> components
        self.Stab = 'PSPG'
        self.viscous_projection = True               # PSPG with the projected viscous stress (consistent on linear elements)

        self.pressure_reference = None               # (node id, value)

    # ----- properties -----

    def setViscosity(self, mu):
        self.mu = mu

    def setDensity(self, rho):
        self.rho = rho

    def setBodyForce(self, f):

        """Force per unit volume [N/m3]: a constant vector, or a function f(x, y) returning the components."""

        self.body_force = f

    def setGravity(self, g):

        """Body force rho g with the material density (a constant density)."""

        self.body_force = self.rho * np.asarray(g, dtype=float)

    def setStabilization(self, method='PSPG'):

        """'PSPG' (default) or None (only stable with inf-sup elements, not with equal order)."""

        if method not in (None, 'PSPG'):
            raise ValueError(f"Estabilizacion desconocida: {method!r} (use 'PSPG' o None).")

        self.Stab = method

    def setVelocity(self, v):
        raise AttributeError("La velocidad de 'ff' es una incognita: use condiciones de contorno (addBC_Velocity...).")

    def setPressureReference(self, point=None, value=0.0, node=None):

        """Fixes the pressure at a node (the one closest to `point`, or `node`). Needed only when every boundary
        has velocity conditions; without it the first node is used automatically in that case."""

        if node is None:
            xy = np.array([n.coor[:self.dim] for n in self.modelRef.mesh.NL], dtype=float)
            node = int(np.argmin(np.linalg.norm(xy - np.asarray(point, dtype=float), axis=1)))

        self.pressure_reference = (int(node), float(value))

    # ----- weak form -----

    def _time_step(self, model):

        options = model.solverOptions or {}
        if options.get('Study') == StudyType.transient and np.any(np.asarray(self.rho) > 0):
            return float(options['Time Step'])
        return None

    def add_terms(self, system, model, disc, layout):

        ru = layout.element_dofs(disc.conn, 'u')
        rp = layout.element_dofs(disc.conn, 'p')

        D = ops.divergence(disc)
        system.add_matrix(layout.scatter_matrix(ops.viscous(disc, self.mu), ru, ru)
                          + layout.scatter_matrix(-np.swapaxes(D, 1, 2), ru, rp)
                          + layout.scatter_matrix(D, rp, ru))

        f = None if self.body_force is None else ops._velocity(disc, self.body_force)
        if f is not None:
            system.add_vector(layout.scatter_vector(ops.vector_source(disc, f), ru))

        transient = np.any(np.asarray(self.rho) > 0)
        if transient:
            system.add_mass(layout.scatter_matrix(ops.vector_mass(disc, self.rho), ru, ru))

        if self.Stab == 'PSPG':
            tau = ops.pspg_tau(disc, self.mu, self.rho, dt=self._time_step(model))
            system.add_matrix(layout.scatter_matrix(ops.diffusion(disc, tau), rp, rp))
            if f is not None:
                system.add_vector(layout.scatter_vector(ops.gradient_source(disc, tau[..., None] * f), rp))
            if transient:
                system.add_mass(layout.scatter_matrix(ops.pspg_mass(disc, tau * ops.at_gauss(disc, self.rho)), rp, ru))
            if self.viscous_projection:
                C, P = ops.projected_stress_divergence(disc, self.mu, tau)
                S = (C @ P).tocoo()
                p_dofs, u_dofs = layout.dofs('p')[:, 0], layout.dofs('u').ravel()
                system.add_matrix(sp.coo_matrix((-S.data, (p_dofs[S.row], u_dofs[S.col])),
                                                shape=(layout.ndof, layout.ndof)).tocsr())

        reference = self.pressure_reference
        if reference is None and not self._pressure_level_set(model):
            reference = (0, 0.0)
            sim_logger.info("Ningun contorno fija la presion: p = 0 en el nodo 0 (use setPressureReference).")
        if reference is not None:
            system.set_dirichlet(layout.node_dofs('p', reference[0])[0], reference[1])

    def _pressure_level_set(self, model):

        """True if a traction, an outlet or a Dirichlet pressure fixes the level of the pressure."""

        if any(bc['type'] in ('Traction', 'NormalTraction') for bc in self.facet_bcs):
            return True

        return any(node.BC and node.BC.get('p', {}).get('type') == 'Dirichlet' for node in model.mesh.NL)

    # ----- results -----

    def flowRate(self, edges, u=None):

        """Outgoing flow rate integral of u . n over the edges [m2/s in 2D]; u: nodal velocity (default: current)."""

        facets = self.modelRef.geometry.facets(edges)
        U = self.var['u'].values if u is None else np.asarray(u, dtype=float)
        uq = np.einsum('qk,fka->fqa', facets.N, U[facets.nodes])

        return float(np.sum(np.einsum('fqa,fqa->fq', uq, facets.normals) * facets.detJw))

    # ----- boundary conditions -----

    def _set_velocity(self, node_id, values):

        """Merges a velocity condition into the node: entries that are not None replace the previous ones."""

        node = self.modelRef._mesh.NL[node_id]
        values = list(values)

        if len(values) != self.dim:
            raise ValueError(f"La velocidad tiene {self.dim} componentes; se dieron {len(values)}.")

        old = node.BC.get('u')
        merged = list(old['value']) if old and old.get('type') == 'Dirichlet' else [None] * self.dim

        for c, v in enumerate(values):
            if v is not None:
                merged[c] = float(v)

        node.BC['u'] = self.setDirichletBC(merged)

    def _edge_nodes(self, edges):
        return sorted({int(n) for edge in edges for n in edge})

    def _normal_component(self, edges):

        """Index of the axis normal to edges that are all aligned with one axis."""

        xy = np.array([n.coor[:self.dim] for n in self.modelRef.mesh.NL], dtype=float)
        components = set()

        for a, b in edges:
            t = xy[b] - xy[a]
            t = t / np.linalg.norm(t)
            if abs(t[1]) < 1e-9:
                components.add(1)      # horizontal edge: normal along y
            elif abs(t[0]) < 1e-9:
                components.add(0)      # vertical edge: normal along x
            else:
                raise ValueError("La condicion de simetria/deslizamiento solo admite aristas alineadas con los ejes.")

        if len(components) != 1:
            raise ValueError("Las aristas de una condicion de simetria deben ser todas horizontales o todas verticales.")

        return components.pop()

    def addBC_Velocity(self, id: int, u):

        """Velocity at a node: a sequence with one value per component (None leaves the component free)."""

        self._set_velocity(id, u)

    def addBC_VelocityEdges(self, edges, u):

        """Velocity on every node of the edges: a sequence (None = free component) or a function u(x, y) returning one.

        Corners shared with another condition keep the last value assigned to each component.
        """

        nodes = self.modelRef._mesh.NL

        for node_id in self._edge_nodes(edges):
            self._set_velocity(node_id, u(*nodes[node_id].coor[:self.dim]) if callable(u) else u)

    def addBC_NoSlipEdges(self, edges):

        """Fixed wall: u = 0."""

        self.addBC_VelocityEdges(edges, [0.0] * self.dim)

    def addBC_SymmetryEdges(self, edges):

        """Symmetry plane or slip wall aligned with the axes: normal velocity zero, no shear traction."""

        normal = self._normal_component(edges)
        values = [None] * self.dim
        values[normal] = 0.0

        self.addBC_VelocityEdges(edges, values)

    addBC_SlipEdges = addBC_SymmetryEdges

    def addBC_OutletEdges(self, edges, p=0.0, fully_developed=True):

        """Outlet with the pressure p (normal traction -p n); p is a number or a function of the coordinates.

        fully_developed=True also fixes the tangential velocity to zero (edges aligned with the axes), which makes the
        stress form of the viscous term compatible with a fully developed (Poiseuille) profile. With False the outlet
        only prescribes the normal traction (and leaves no shear traction).
        """

        self.facet_bcs.append({'type': 'NormalTraction', 'field': 'u', 'facets': list(edges), 'pressure': p})

        if fully_developed:
            normal = self._normal_component(edges)
            values = [0.0] * self.dim
            values[normal] = None
            self.addBC_VelocityEdges(edges, values)

    def addBC_TractionEdges(self, edges, t):

        """Traction t = sigma . n [N/m2] acting on the fluid through the edges: a vector or a function t(x, y)."""

        self.facet_bcs.append({'type': 'Traction', 'field': 'u', 'facets': list(edges), 'traction': t})
