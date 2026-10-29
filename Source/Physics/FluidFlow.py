# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++++

# Incompressible viscous flow: velocity u (vector) and pressure p on the same nodes (Q1/P1 equal order).
#
#     rho (du/dt + (u . grad) u) - div(2 mu eps(u)) + grad p = f,      div u = 0,      eps(u) = (grad u + grad u^T) / 2
#
# Creeping flow (Stokes, default) leaves out the inertia (u . grad) u; setInertia(True) solves Navier-Stokes.
# Equal-order elements do not satisfy the inf-sup condition, so the continuity equation carries the
# pressure-stabilizing Petrov-Galerkin term (PSPG, Hughes-Franca-Balestra / Tezduyar), with the momentum residual
# R_M = rho (du/dt + (u . grad) u) + grad p - f - div(sigma~):
#
#     R_u = integral 2 mu eps(u):eps(v) - integral p div v + integral rho (du/dt + (u . grad) u) . v - integral f . v
#           - boundary integral t . v  [+ integral tau rho (u . grad v) . R_M + integral tau_c div u div v]
#     R_p = integral q div u + integral tau grad(q) . R_M
#
# The bracket (SUPG and grad-div/LSIC) is added only with inertia. div(2 mu eps(u_h)) vanishes inside a linear
# element, so R_M uses a recovered continuous stress sigma~ (viscous_projection, default) to stay consistent.
# tau = 1 / sqrt((2 rho / dt)^2 + (2 rho |u| / h)^2 + (12 mu / h^2)^2) (Operators.pspg_tau): h^2 / (12 mu) for creeping
# flow. A constant q gives integral div u = 0 exactly: the discrete flow rate is conserved globally.
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
        self.inertia = False                         # Navier-Stokes convection (setInertia)
        self.linearization = 'newton'                # 'newton' or 'picard'

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

    def setInertia(self, on=True, linearization='newton'):

        """Navier-Stokes: adds the convection rho (u . grad) u (with SUPG and grad-div stabilization). The problem
        becomes nonlinear (ProblemType.nonlinear). linearization: 'newton' (full tangent of the convection, fast near
        the solution) or 'picard' (u_old . grad u, slower but more robust). For high Reynolds numbers solve a sequence
        of increasing velocities, each starting from the previous solution."""

        if linearization not in ('newton', 'picard'):
            raise ValueError(f"Linealizacion desconocida: {linearization!r} (use 'newton' o 'picard').")
        if on and not np.all(np.asarray(self.rho) > 0):
            raise ValueError("Navier-Stokes necesita una densidad positiva (material rho o setDensity).")

        self.inertia = bool(on)
        self.linearization = linearization
        self.physics_description = 'Laminar flow (Navier-Stokes)' if on else 'Creeping flow (Stokes)'

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

        if self.inertia:
            self._add_navier_stokes(system, model, disc, layout, f)
        elif self.Stab == 'PSPG':
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

    def _add_navier_stokes(self, system, model, disc, layout, f):

        """Convection and the residual-based stabilization (PSPG, SUPG, LSIC) as one nonlinear term. With a time step
        the stabilized mass terms use tau and the SUPG weight of the state at the start of the solve."""

        V = ops.recovered_stress_divergence(disc, self.mu) if self.viscous_projection else None
        system.add_term(NavierStokesTerm(system, self.mu, self.rho, f, V, self.linearization, self.Stab == 'PSPG'))

        dt = self._time_step(model)
        if dt is not None:
            nel, _, nne, dim = disc.B.shape
            ru = layout.element_dofs(disc.conn, 'u')
            rp = layout.element_dofs(disc.conn, 'p')
            u = disc.interpolate(np.asarray(self.var['u'].values, dtype=float))
            tau = ops.pspg_tau(disc, self.mu, self.rho, speed=np.linalg.norm(u, axis=-1), dt=dt)
            rho = ops.at_gauss(disc, self.rho)
            uB = np.einsum('egb,egib->egi', u, disc.B)
            supg = np.einsum('eg,egi,gj,ac->eiajc', disc.detJw * tau * rho * rho, uB, disc.N, np.eye(dim))
            system.add_mass(layout.scatter_matrix(supg.reshape(nel, nne * dim, nne * dim), ru, ru))
            if self.Stab == 'PSPG':
                system.add_mass(layout.scatter_matrix(ops.pspg_mass(disc, tau * rho), rp, ru))

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


class NavierStokesTerm:

    """Nonlinear part of the stabilized Navier-Stokes equations, assembled on all elements at once.

    With the momentum residual at the Gauss points R_M = rho (u . grad) u + grad p - f - div(sigma~):

        R_u += integral rho ((u . grad) u) . v  +  integral tau rho (u . grad v) . R_M  +  integral tau_c div u div v
        R_p += integral tau grad(q) . R_M

    tau (PSPG/SUPG, Operators.pspg_tau) and tau_c (grad-div, Operators.lsic_tau) depend on |u|; the tangent keeps them
    frozen (their derivative is left out), the usual practice. 'picard' also leaves out the derivative of the
    convecting velocity (u_old . grad u) and of the SUPG weight. div(sigma~) is linear in u (V, a sparse matrix that
    couples the elements of each recovery patch); without the recovered stress it is zero.
    """

    def __init__(self, system, mu, rho, f, V, linearization='newton', pspg=True):

        self.system = system
        self.mu, self.rho = mu, rho
        self.f = f
        self.newton = linearization == 'newton'
        self.pspg = pspg
        self.tau_state = None        # evaluate tau at this state instead of x (to test the tangent)

        layout = system.layout
        self.V = None
        if V is not None:
            V = V.tocoo()
            u_dofs = layout.dofs('u').ravel()
            self.V = sp.coo_matrix((V.data, (V.row, u_dofs[V.col])), shape=(V.shape[0], layout.ndof)).tocsr()

    def contribution(self, x):

        disc, layout = self.system.disc, self.system.layout
        nel, ngp, nne, dim = disc.B.shape
        N, B, w = disc.N, disc.B, disc.detJw

        fields = layout.split(x)
        U = np.asarray(fields['u'], dtype=float)[disc.conn]                       # (nel, nne, dim)
        P = np.asarray(fields['p'], dtype=float)[disc.conn]                       # (nel, nne)

        u = np.einsum('gj,ejc->egc', N, U)
        G = np.einsum('egjb,eja->egab', B, U)                                     # G_ab = du_a/dx_b
        gp = np.einsum('egjb,ej->egb', B, P)

        u_tau = u if self.tau_state is None else disc.interpolate(layout.split(self.tau_state)['u'])
        speed = np.linalg.norm(u_tau, axis=-1)
        rho = ops.at_gauss(disc, self.rho)
        tau = ops.pspg_tau(disc, self.mu, self.rho, speed=speed)
        tau_c = ops.lsic_tau(disc, self.mu, self.rho, speed)

        conv = rho[..., None] * np.einsum('egb,egab->ega', u, G)
        RM = conv + gp
        if self.f is not None:
            RM = RM - self.f
        if self.V is not None:
            RM = RM - (self.V @ x).reshape(nel, ngp, dim)

        uB = np.einsum('egb,egib->egi', u, B)                                     # u . grad N_i
        divu = np.einsum('egaa->eg', G)
        ws = w * tau * rho                                                         # SUPG weight factor

        # ----- residual -----
        Ru = (np.einsum('eg,gi,ega->eia', w, N, conv)
              + np.einsum('eg,egi,ega->eia', ws, uB, RM)
              + np.einsum('eg,egia->eia', w * tau_c * divu, B))

        ru = layout.element_dofs(disc.conn, 'u')
        rp = layout.element_dofs(disc.conn, 'p')
        R = layout.scatter_vector(Ru.reshape(nel, -1), ru)
        if self.pspg:
            R = R + layout.scatter_vector(np.einsum('eg,egia,ega->ei', w * tau, B, RM), rp)

        # ----- tangent (tau frozen) -----
        eye = np.eye(dim)
        dconv = np.einsum('eg,ac,egj->egajc', rho, eye, uB)                        # rho u . grad N_j (Picard part)
        if self.newton:
            dconv = dconv + np.einsum('eg,gj,egac->egajc', rho, N, G)              # rho N_j du_a/dx_c

        Juu = (np.einsum('eg,gi,egajc->eiajc', w, N, dconv)
               + np.einsum('eg,egi,egajc->eiajc', ws, uB, dconv)
               + np.einsum('eg,egia,egjc->eiajc', w * tau_c, B, B))
        if self.newton:
            Juu = Juu + np.einsum('eg,gj,egic,ega->eiajc', ws, N, B, RM)            # derivative of the SUPG weight
        Jup = np.einsum('eg,egi,egja->eiaj', ws, uB, B)

        J = (layout.scatter_matrix(Juu.reshape(nel, nne * dim, nne * dim), ru, ru)
             + layout.scatter_matrix(Jup.reshape(nel, nne * dim, nne), ru, rp))

        if self.pspg:
            Jpu = np.einsum('eg,egia,egajc->eijc', w * tau, B, dconv)
            Jpp = np.einsum('eg,egia,egja->eij', w * tau, B, B)
            J = J + layout.scatter_matrix(Jpu.reshape(nel, nne, nne * dim), rp, ru) + layout.scatter_matrix(Jpp, rp, rp)

        if self.V is not None:
            # -(SUPG and PSPG weights) @ V: the recovered stress couples the elements of each patch
            cols = np.arange(nel * ngp).reshape(nel, ngp)[:, :, None] * dim + np.arange(dim)    # (nel, ngp, dim)
            Ws = np.einsum('eg,egi,ac->egiac', ws, uB, eye)                                      # (e, g, i, a, a')
            rows = np.broadcast_to(ru.reshape(nel, nne, dim)[:, None, :, :, None], Ws.shape)
            W = sp.coo_matrix((Ws.ravel(), (rows.ravel(), np.broadcast_to(cols[:, :, None, None, :], Ws.shape).ravel())),
                              shape=(layout.ndof, nel * ngp * dim)).tocsr()
            if self.pspg:
                Wp = np.einsum('eg,egia->egia', w * tau, B)
                W = W + sp.coo_matrix((Wp.ravel(), (np.broadcast_to(rp[:, None, :, None], Wp.shape).ravel(),
                                                    np.broadcast_to(cols[:, :, None, :], Wp.shape).ravel())),
                                      shape=(layout.ndof, nel * ngp * dim)).tocsr()
            J = J - W @ self.V

        return R, J
