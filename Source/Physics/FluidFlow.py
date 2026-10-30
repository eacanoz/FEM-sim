# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++++

# Incompressible viscous flow: velocity u (vector) and pressure p on the same nodes (Q1/P1 equal order), or Taylor-Hood
# on a quadratic mesh (Q9, T6): quadratic velocity and linear pressure, inf-sup stable (see below).
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
# tau = 1 / sqrt((2 rho |u| / h)^2 + (12 mu / h^2)^2) (Operators.pspg_tau, without the time step): h^2 / (12 mu) for
# creeping flow. A constant q gives integral div u = 0 exactly: the discrete flow rate is conserved globally.
# Transient studies: rho du/dt of the step enters R_M at the new time level (system.time, set by the solver).
#
# Generalized Newtonian fluids (setViscosityLaw): mu(gamma_dot, T) from Source.Physics.Rheology, gamma_dot = sqrt(2 eps:eps),
# with the Newton tangent through dmu/dgamma_dot and dmu/dT. Viscous heating (setViscousHeating): the dissipation
# Phi = mu gamma_dot^2 is a source of the heat equation. Combined with the heat physics, Model(psc=[ff, ht]) and
# heat.setVelocity('u'), velocity, pressure and temperature are solved together (monolithic coupling).
#
# Taylor-Hood (Q2/Q1, P2/P1), chosen by the mesh: Galerkin without PSPG, SUPG or a recovered stress, third order in u and
# second in p. The pressure keeps a value at every node, tied to the vertices (System.restrict_to_linear_subspace).
#
# Boundary conditions: velocity (any component, None leaves it free), no slip, symmetry/slip on walls aligned
# with the axes (normal component fixed), outlet (normal traction -p n, optionally fully developed: tangential
# velocity zero) and general tractions. Walls without a condition are traction free. If no boundary sets the
# pressure level (all of it has velocity conditions) the pressure is fixed at one node.


import numpy as np
import scipy.sparse as sp

from Source.Physics.Physics import physics
from Source.Pre_processing.Elements import ELEMENTS, element_type_of
from Source.Simulation.PointwiseLaw import PointwiseLaw, cached_law
from Source.Primals.Scalar import scalarField
from Source.Primals.Vector import vectorField
import Source.Simulation.Operators as ops
from Source.enums import ShapeFunctionType
from Source.logger_config import sim_logger


class ff(physics):

    part_name = 'fluid'
    ConstRel = 'ρ ∂u/∂t - ∇.(2μ ε(u)) + ∇p == f,  ∇.u == 0'
    computes_velocity = True     # its velocity is an unknown: CoupledPhysics.setVelocity does not set it

    def __init__(self, model):
        super().__init__(model)

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
        self.viscosity_law = None                    # generalized Newtonian mu(gamma_dot, T) (setViscosityLaw)
        self.viscosity_fields = []                   # fields read by the viscosity law (T, nut, k and omega, ...)
        self.heating_field = None                    # field that receives the viscous dissipation (setViscousHeating)
        self.buoyancy = []                           # Boussinesq terms (field, beta, reference, g) (setBoussinesq)

        self.pressure_reference = None               # (node id, value)

        # A quadratic mesh means Taylor-Hood: inf-sup stable, so without PSPG or the stress it needs
        self.taylor_hood = bool(ELEMENTS[element_type_of(model.mesh)].linear_subspace)
        if self.taylor_hood:
            self.Stab = None
            self.viscous_projection = False

    @property
    def physics_description(self):

        regime = 'Laminar flow (Navier-Stokes)' if self.inertia else 'Creeping flow (Stokes)'
        extras = []
        if self.viscosity_law is not None:
            extras.append(f"{type(self.viscosity_law).__name__} viscosity")
        if self.heating_field is not None:
            extras.append('viscous heating')
        if self.buoyancy:
            extras.append('Boussinesq buoyancy (' + ', '.join(b[0] for b in self.buoyancy) + ')')

        return regime + (f", {', '.join(extras)}" if extras else '')

    # ----- properties -----

    def setViscosity(self, mu):

        """Constant (Newtonian) viscosity [Pa s]: a number or a function of the coordinates. Removes a viscosity law."""

        self.mu = mu
        self.viscosity_law = None
        self.viscosity_fields = []

    def setViscosityLaw(self, law, temperature='T', fields=None):

        """Generalized Newtonian fluid: mu(gamma_dot, T) from Source.Physics.Rheology (power law, Carreau-Yasuda,
        Cross, Cross-WLF, Bingham, Herschel-Bulkley, Arrhenius shift, ...). The problem becomes nonlinear
        (ProblemType.nonlinear). A law that reads a field (uses_temperature) takes it from the scalar field named
        `temperature`: 'T' for the heat physics (Model(psc=[ff, ht])), or any other one, e.g. 'nut' for the eddy
        viscosity of a turbulence model. A law of several fields, law(gamma2, f1, f2, ...), names them in `fields`
        (e.g. ('k', 'omega', 'wall_distance') for k-omega SST). Shear-thinning laws converge best starting from a
        Newtonian solution (solve with setViscosity first, then switch the law on and solve again)."""

        self.viscosity_law = law
        if fields is not None:
            self.viscosity_fields = list(fields)
        else:
            self.viscosity_fields = [temperature] if getattr(law, 'uses_temperature', False) else []

    @property
    def temperature_field(self):

        """The field read by a one-field viscosity law (None if the law reads none or several)."""

        return self.viscosity_fields[0] if len(self.viscosity_fields) == 1 else None

    def setViscousHeating(self, on=True, field='T'):

        """Viscous dissipation Phi = mu gamma_dot^2 [W/m3] as a heat source of the field `field` (default 'T', the
        heat physics in the same model). Nonlinear (quadratic in the velocity gradient)."""

        self.heating_field = field if on else None

    def setBoussinesq(self, g, beta, reference, field='T'):

        """Boussinesq buoyancy: body force -rho0 beta (phi - reference) g, with phi the field `field` of the same model
        (default 'T': Model(psc=[ff, ht]); a concentration for solutal buoyancy). rho0 is the density of the material,
        beta the expansion coefficient [1/K or per unit of phi] and g the gravity vector, e.g. (0, -9.81). The
        hydrostatic part rho0 g is absorbed into the pressure, so p is the deviation from it. Call once per field
        (thermal and solutal buoyancy add up); setBoussinesq(..., beta=0) removes the field. Nonlinear coupling: use
        ProblemType.nonlinear."""

        self.buoyancy = [b for b in self.buoyancy if b[0] != field]
        if beta:
            self.buoyancy.append((field, float(beta), float(reference), np.asarray(g, dtype=float)))

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
        if method is not None and self.taylor_hood:
            raise ValueError("Taylor-Hood (malla cuadratica) es estable sin estabilizacion de la presion.")

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

    def setVelocity(self, v):
        raise AttributeError("La velocidad de 'ff' es una incognita: use condiciones de contorno (addBC_Velocity...).")

    def setPressureReference(self, point=None, value=0.0, node=None):

        """Fixes the pressure at a node (the one closest to `point`, or `node`). Needed only when every boundary
        has velocity conditions; without it the first node is used automatically in that case."""

        candidates = self._pressure_nodes()

        if node is None:
            xy = np.array([self.modelRef.mesh.NL[k].coor[:self.dim] for k in candidates], dtype=float)
            node = candidates[int(np.argmin(np.linalg.norm(xy - np.asarray(point, dtype=float), axis=1)))]
        elif int(node) not in candidates:
            raise ValueError(f"El nodo {node} no es un vertice: en Taylor-Hood la presion solo se fija en los vertices.")

        self.pressure_reference = (int(node), float(value))

    def _pressure_nodes(self):

        """Nodes where the pressure is an independent unknown: all of them, or the vertices with Taylor-Hood."""

        mesh = self.modelRef.mesh
        if not self.taylor_hood:
            return list(range(mesh.getNoN()))

        n_vertices = 3 if element_type_of(mesh).startswith('tri') else 4
        return sorted({n for element in mesh.EL for n in element.getNodesId()[:n_vertices]})

    # ----- weak form -----

    def _nonlinear(self):

        """True if the flow needs the nonlinear term: inertia, a viscosity law, viscous heating or buoyancy."""

        return (self.inertia or self.viscosity_law is not None or self.heating_field is not None
                or bool(self.buoyancy))

    def add_terms(self, system, model, disc, layout):

        ru = layout.element_dofs(disc.conn, 'u')
        rp = layout.element_dofs(disc.conn, 'p')

        for name in (*self.viscosity_fields, self.heating_field, *(b[0] for b in self.buoyancy)):
            if name is not None and name not in layout.names:
                raise ValueError(f"La fisica de flujo usa el campo '{name}', que no existe en el modelo "
                                 "(combine con la fisica de calor: Model(psc=[ff, ht])).")

        system.require_constraint(layout.dofs('p')[:, 0])
        if self.taylor_hood:
            system.restrict_to_linear_subspace('p')

        D = ops.divergence(disc)
        A = layout.scatter_matrix(-np.swapaxes(D, 1, 2), ru, rp) + layout.scatter_matrix(D, rp, ru)
        if self.viscosity_law is None:
            A = A + layout.scatter_matrix(ops.viscous(disc, self.mu), ru, ru)
        system.add_matrix(A)

        f = None if self.body_force is None else ops._velocity(disc, self.body_force)
        if f is not None:
            system.add_vector(layout.scatter_vector(ops.vector_source(disc, f), ru))

        has_density = np.any(np.asarray(self.rho) > 0)
        if has_density:
            system.add_mass(layout.scatter_matrix(ops.vector_mass(disc, self.rho), ru, ru))

        if self._nonlinear():
            self._add_flow_term(system, model, disc, layout, f)
        elif self.Stab == 'PSPG':
            tau = ops.pspg_tau(disc, self.mu, self.rho)                       # no dt: see FlowTerm
            system.add_matrix(layout.scatter_matrix(ops.diffusion(disc, tau), rp, rp))
            if f is not None:
                system.add_vector(layout.scatter_vector(ops.gradient_source(disc, tau[..., None] * f), rp))
            if has_density:
                system.add_mass(layout.scatter_matrix(ops.pspg_mass(disc, tau * ops.at_gauss(disc, self.rho)), rp, ru))
            if self.viscous_projection:
                S = (ops.pspg_weight(disc, tau) @ ops.recovered_stress_divergence(disc, self.mu)).tocoo()
                p_dofs, u_dofs = layout.dofs('p')[:, 0], layout.dofs('u').ravel()
                system.add_matrix(sp.coo_matrix((-S.data, (p_dofs[S.row], u_dofs[S.col])),
                                                shape=(layout.ndof, layout.ndof)).tocsr())

        reference = self.pressure_reference
        if reference is None and not self._pressure_level_set(model):
            reference = (self._pressure_nodes()[0], 0.0)
            sim_logger.info(f"Ningun contorno fija la presion: p = 0 en el nodo {reference[0]} (use setPressureReference).")
        if reference is not None:
            system.set_dirichlet(layout.node_dofs('p', reference[0])[0], reference[1])

    def _add_flow_term(self, system, model, disc, layout, f):

        """Viscosity law, convection, viscous heating and the residual-based stabilization (PSPG, SUPG, LSIC) as one
        nonlinear term. In transient studies the term reads the time derivative of the step from system.time, so
        rho du/dt enters the stabilized momentum residual at the new time level (no frozen stabilized masses)."""

        V = Q = None
        if self.viscous_projection:
            if self.viscosity_law is None:
                V = ops.recovered_stress_divergence(disc, self.mu)
            else:
                Q = ops.gauss_stress_divergence(disc)

        system.add_term(FlowTerm(system, self, f, V, Q))

    def _pressure_level_set(self, model):

        """True if a traction, an outlet or a Dirichlet pressure fixes the level of the pressure."""

        if any(bc['type'] in ('Traction', 'NormalTraction') for bc in self.facet_bcs):
            return True

        return any(node.BC and node.BC.get('p', {}).get('type') == 'Dirichlet' for node in model.mesh.NL)

    # ----- results -----

    def flowRate(self, edges, u=None):

        """Outgoing flow rate integral of u . n over the edges; u: nodal velocity (default: current).

        Plane: per unit depth [m2/s]. Axisymmetric: the volumetric flow rate through the surface of revolution,
        2 pi integral of u . n r ds [m3/s]."""

        disc = self.modelRef.geometry
        facets = disc.facets(edges)
        U = self.var['u'].values if u is None else np.asarray(u, dtype=float)
        uq = np.einsum('qk,fka->fqa', facets.N, U[facets.nodes])
        q = float(np.sum(np.einsum('fqa,fqa->fq', uq, facets.normals) * facets.detJw))

        return 2.0 * np.pi * q if disc.axisymmetric else q

    def boundaryForce(self, edges, rate=None):

        """Force of the fluid on the boundary formed by `edges` (a wall, an obstacle) [N per unit depth; N in
        axisymmetric], from the reactions: the momentum residual at the nodes of those edges, assembled without the
        boundary conditions, is the consistent integral of sigma . n (more accurate than integrating the stress of the
        discrete solution along the boundary). Returns (F_x, F_y): drag and lift for a flow along x.

        rate: time derivative of the solution vector (transient: e.g. (x_n - x_{n-1}) / dt) to include the inertia
        rho du/dt of the elements next to the boundary. Use the current state of the model's fields.
        """

        model = self.modelRef
        system, layout = model._system(), model.layout
        x = model._state()

        saved, system.time = system.time, None
        try:
            _, R = system._assemble(x)
        finally:
            system.time = saved
        if rate is not None:
            R = R + system.mass() @ np.asarray(rate, dtype=float)

        nodes = np.unique(np.ravel(edges))
        force = -R[layout.node_dofs('u', nodes)].sum(axis=0)

        return 2.0 * np.pi * force if model.geometry.axisymmetric else force

    def derived_fields(self, model):

        """Nodal shear rate gamma_dot = sqrt(2 eps:eps) and, with a viscosity law, the viscosity (both projected
        from the Gauss points)."""

        disc = model.geometry
        U = np.asarray(self.var['u'].values, dtype=float)
        G = disc.gradient(U)
        eps = 0.5 * (G + np.swapaxes(G, -1, -2))
        g2 = 2.0 * np.einsum('egab,egab->eg', eps, eps)
        if disc.axisymmetric:
            g2 = g2 + 2.0 * (disc.interpolate(U)[..., 0] / disc.radius) ** 2      # hoop strain u_r / r
        out = {'shear_rate': disc.project(np.sqrt(g2))}

        if self.viscosity_law is not None:
            F = [disc.interpolate(np.asarray(model.physics.var[name].values, dtype=float)) for name in self.viscosity_fields]
            law = _pointwise_viscosity(self.viscosity_law, len(F))
            values, _ = law(np.stack([g2] + (F or [np.zeros(g2.shape)]), axis=-1).reshape(-1, max(len(F), 1) + 1))
            out['viscosity'] = disc.project(values.reshape(g2.shape))

        return out

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

        for edge in edges:
            t = xy[edge[-1]] - xy[edge[0]]
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




def _pointwise_viscosity(law, n_fields):

    """PointwiseLaw of a viscosity law with input (gamma2, f_1, ..., f_n); with no field a dummy zero column is passed
    (one-field laws are written as law(gamma2, T))."""

    if n_fields == 0:
        return cached_law(('viscosity', law, 0), lambda: PointwiseLaw(lambda w: law(w[0], w[1])))
    return cached_law(('viscosity', law, n_fields), lambda: PointwiseLaw(lambda w: law(w[0], *[w[1 + i] for i in range(n_fields)])))


class FlowTerm:

    """Nonlinear part of the stabilized incompressible flow equations, assembled on all elements at once.

    Active pieces (by the physics settings):

        viscosity law      R_u += integral 2 mu(gamma_dot, T) eps(u) : grad v          (else the linear viscous matrix)
        inertia            R_u += integral rho ((u . grad) u) . v
                                  + integral tau rho (u . grad v) . R_M  (SUPG)  + integral tau_c div u div v  (LSIC)
        PSPG               R_p += integral tau grad(q) . R_M
        viscous heating    R_T -= integral N mu gamma_dot^2
        buoyancy           R_u -= integral f_b . v,  f_b = -rho0 beta (phi - reference) g  (Boussinesq; also in R_M)

    with the momentum residual at the Gauss points R_M = rho (u . grad) u + grad p - f - div(sigma~), sigma~ the recovered
    viscous stress (linear in u for a constant mu: V; through Q from the Gauss-point stress for a viscosity law).

    The tangent is exact except for tau and tau_c, which depend on |u| and on mu and are kept frozen (the usual
    practice); 'picard' also leaves out the derivative of the convecting velocity and of the SUPG weight.
    """

    def __init__(self, system, flow, f, V=None, Q=None):

        self.system = system
        self.mu_const = flow.mu
        self.rho = flow.rho
        self.f = f
        self.inertia = flow.inertia
        self.newton = flow.linearization == 'newton'
        self.pspg = flow.Stab == 'PSPG'
        self.supg = flow.inertia and not flow.taylor_hood
        self.fields = list(flow.viscosity_fields)       # fields read by the viscosity law
        self.heating = flow.heating_field
        # Boussinesq body forces f_b = c (phi - reference), c = -rho0 beta g
        self.buoyancy = [(name, -float(flow.rho) * beta * g, ref) for name, beta, ref, g in flow.buoyancy]
        self.tau_state = None        # evaluate tau (|u| and mu) at this state instead of x (to test the tangent)

        self.law = None
        if flow.viscosity_law is not None:
            self.law = _pointwise_viscosity(flow.viscosity_law, len(self.fields))

        layout = system.layout
        self.V = None
        if V is not None:
            V = V.tocoo()
            u_dofs = layout.dofs('u').ravel()
            self.V = sp.coo_matrix((V.data, (V.row, u_dofs[V.col])), shape=(V.shape[0], layout.ndof)).tocsr()
        self.Q = Q

    # ----- kinematics and viscosity at the Gauss points -----

    def _kinematics(self, x):

        disc, layout = self.system.disc, self.system.layout
        fields = layout.split(x)
        U = np.asarray(fields['u'], dtype=float)[disc.conn]                       # (nel, nne, dim)
        u = np.einsum('gj,ejc->egc', disc.N, U)
        G = np.einsum('egjb,eja->egab', disc.B, U)                                # G_ab = du_a/dx_b
        F = np.stack([disc.interpolate(np.asarray(fields[name], dtype=float)) for name in self.fields], axis=-1)             if self.fields else np.zeros(disc.detJw.shape + (0,))                 # (nel, ngp, n_fields)

        return fields, u, G, F

    def _strain(self, u, G, derivative=True):

        """eps (in-plane), the hoop strain u_r / r (zero on a plane mesh), gamma2 = 2 eps:eps and d(gamma2)/dU_jc
        (None without `derivative`)."""

        disc = self.system.disc
        eps = 0.5 * (G + np.swapaxes(G, -1, -2))
        dg2 = 4.0 * np.einsum('egcb,egjb->egjc', eps, disc.B) if derivative else None
        e_hh = np.zeros(disc.detJw.shape)

        if disc.axisymmetric:
            e_hh = u[..., 0] / disc.radius
            if derivative:
                dg2[..., 0] += 4.0 * e_hh[..., None] * disc.hoop

        g2 = 2.0 * (np.einsum('egab,egab->eg', eps, eps) + e_hh ** 2)

        return eps, e_hh, g2, dg2

    def viscosity(self, x, derivatives=True):

        """mu, dmu/dgamma2 (nel, ngp) and dmu/df (nel, ngp, n_fields) at the Gauss points of the state x, for the fields
        read by the viscosity law (the derivatives are zero for a constant mu, and None without `derivatives`)."""

        disc = self.system.disc
        _, u, G, F = self._kinematics(x)
        _, _, g2, _ = self._strain(u, G, derivative=False)
        shape = disc.detJw.shape
        nf = F.shape[-1]

        if self.law is None:
            return ops.at_gauss(disc, self.mu_const), np.zeros(shape), np.zeros(shape + (nf,))

        inputs = np.concatenate([g2[..., None], F if nf else np.zeros(shape + (1,))], axis=-1)

        if not derivatives:
            return self.law.values(inputs.reshape(-1, inputs.shape[-1])).reshape(shape), None, None

        values, jacobian = self.law(inputs.reshape(-1, inputs.shape[-1]))
        return (values.reshape(shape), jacobian[:, 0, 0].reshape(shape),
                jacobian[:, 0, 1:1 + nf].reshape(shape + (nf,)))

    # ----- residual and tangent -----

    def contribution(self, x, tangent=True):

        disc, layout = self.system.disc, self.system.layout
        nel, ngp, nne, dim = disc.B.shape
        N, B, w = disc.N, disc.B, disc.detJw
        eye = np.eye(dim)

        fields, u, G, _ = self._kinematics(x)
        P = np.asarray(fields['p'], dtype=float)[disc.conn]
        gp = np.einsum('egjb,ej->egb', B, P)

        eps, e_hh, g2, dg2 = self._strain(u, G, tangent)                          # dg2 = d(gamma2)/dU_jc
        mu, mu_g2, mu_F = self.viscosity(x, tangent)
        axi = disc.axisymmetric
        divB = B.copy()                                                           # div of the vector test function
        if axi:
            divB[..., 0] += disc.hoop

        state = x if self.tau_state is None else self.tau_state
        mu_tau = mu if self.tau_state is None else self.viscosity(state, False)[0]
        speed = np.linalg.norm(self._kinematics(state)[1], axis=-1) if self.inertia else np.zeros(w.shape)
        rho = ops.at_gauss(disc, self.rho)
        time = self.system.time
        # tau without the time step (Bochev, Gunzburger & Shadid 2004): a dt-dependent tau shifts the discrete
        # steady state, spoils the temporal convergence at a fixed mesh and weakens PSPG for small steps
        tau = ops.pspg_tau(disc, mu_tau, self.rho, speed=speed)

        ru = layout.element_dofs(disc.conn, 'u')
        rp = layout.element_dofs(disc.conn, 'p')
        rF = [layout.element_dofs(disc.conn, name) for name in self.fields]   # dofs of the viscosity fields

        R = np.zeros(layout.ndof)
        blocks = []                                                               # (Ke, rows, cols)
        Ru = np.zeros((nel, nne, dim))

        # ----- viscous stress of a viscosity law -----
        divs = np.zeros((nel, ngp, dim))
        d_divs = None                                                             # sparse (nel*ngp*dim, ndof)

        if self.law is not None:
            sigma = 2.0 * mu[..., None, None] * eps
            Ru += np.einsum('eg,egib,egab->eia', w, B, sigma)

            if tangent:
                dsig = (np.einsum('eg,egjb,ac->egabjc', mu, B, eye) + np.einsum('eg,egja,bc->egabjc', mu, B, eye)
                        + 2.0 * np.einsum('egab,eg,egjc->egabjc', eps, mu_g2, dg2))
                blocks.append((np.einsum('eg,egib,egabjc->eiajc', w, B, dsig).reshape(nel, nne * dim, nne * dim), ru, ru))

                dsigF = []                                                        # d sigma / d f_j per field
                for i, r_f in enumerate(rF):
                    dsigF.append(2.0 * np.einsum('egab,eg,gj->egabj', eps, mu_F[..., i], N))
                    blocks.append((np.einsum('eg,egib,egabj->eiaj', w, B, dsigF[-1]).reshape(nel, nne * dim, nne), ru, r_f))

            if axi:
                # hoop stress sigma_tt = 2 mu u_r / r, tested by v_r / r
                s_hh = 2.0 * mu * e_hh
                Ru[:, :, 0] += np.einsum('eg,egi,eg->ei', w, disc.hoop, s_hh)
                if tangent:
                    ds_hh = 2.0 * e_hh[..., None, None] * mu_g2[..., None, None] * dg2           # (e, g, j, c)
                    ds_hh[..., 0] += 2.0 * mu[..., None] * disc.hoop
                    Kh = np.zeros((nel, nne, dim, nne, dim))
                    Kh[:, :, 0] = np.einsum('eg,egi,egjc->eijc', w, disc.hoop, ds_hh)
                    blocks.append((Kh.reshape(nel, nne * dim, nne * dim), ru, ru))
                    ds_hhF = []
                    for i, r_f in enumerate(rF):
                        ds_hhF.append(2.0 * np.einsum('eg,eg,gj->egj', e_hh, mu_F[..., i], N))
                        KhF = np.zeros((nel, nne, dim, nne))
                        KhF[:, :, 0] = np.einsum('eg,egi,egj->eij', w, disc.hoop, ds_hhF[-1])
                        blocks.append((KhF.reshape(nel, nne * dim, nne), ru, r_f))

            if self.Q is not None:
                divs = (self.Q @ sigma.ravel()).reshape(nel, ngp, dim)
                if tangent:
                    d_divs = self.Q @ self._stress_tangent(dsig, dsigF, ru, rF)
                if axi and tangent:
                    extra, d_extra = self._hoop_divergence(sigma, s_hh, dsig, ds_hh, dsigF, ds_hhF, ru, rF)
                    divs = divs + extra
                    d_divs = d_divs + d_extra
                elif axi:
                    r = disc.radius
                    divs = divs + np.stack([(sigma[..., 0, 0] - s_hh) / r, sigma[..., 1, 0] / r], axis=-1)
        elif self.V is not None:
            divs = (self.V @ x).reshape(nel, ngp, dim)
            d_divs = self.V

        # ----- momentum residual R_M (without grad p: RM_space) and its local derivatives -----
        RM_space = -divs
        if self.f is not None:
            RM_space = RM_space - self.f

        # Boussinesq body forces f_b = c (phi - ref): Galerkin part here, stabilization through RM_space
        buoyancy_rows = []
        for name, c, ref in self.buoyancy:
            r_phi = layout.element_dofs(disc.conn, name)
            fb = (disc.interpolate(np.asarray(fields[name], dtype=float)) - ref)[..., None] * c
            RM_space = RM_space - fb
            Ru -= np.einsum('eg,gi,ega->eia', w, N, fb)
            if tangent:
                blocks.append((-np.einsum('eg,gi,a,gj->eiaj', w, N, c, N).reshape(nel, nne * dim, nne), ru, r_phi))
            buoyancy_rows.append((r_phi, c))

        dconv = None
        if self.inertia:
            conv = rho[..., None] * np.einsum('egb,egab->ega', u, G)
            RM_space = RM_space + conv
            uB = np.einsum('egb,egib->egi', u, B)                                 # u . grad N_i
            if tangent:
                dconv = np.einsum('eg,ac,egj->egajc', rho, eye, uB)                # rho u . grad N_j (Picard part)
                if self.newton:
                    dconv = dconv + np.einsum('eg,gj,egac->egajc', rho, N, G)      # rho N_j du_a/dx_c

        # The stabilization uses the time-discrete momentum residual of the step, evaluated once:
        #   R_M = rho (u - u_n)/dt-like x' + grad p + theta RM_space(x) + (1 - theta) RM_space(x_n)
        # c = (c_time, c_grad_p, c_new, c_old) weighs its parts: in the algebraic PSPG rows it enters whole; in the
        # momentum rows (weighted theta by the solver) it is divided by theta. Steady: (0, 1, 1, 0). The residual of
        # the old level (time['old_level']) leaves the SUPG out: it is already in the new level.
        old_level = time is not None and time.get('old_level', False)
        if time is None or old_level:
            c_p = c_s = (0.0, 1.0, 1.0, 0.0)
        else:
            th = time['theta']
            c_p = (1.0, 1.0, th, 1.0 - th)
            c_s = tuple(c / th for c in c_p)

        u_t = old = 0.0
        if c_p[0]:
            u_t = disc.interpolate(np.asarray(layout.split(time['alpha'] * x + time['beta'])['u'], dtype=float))
            if c_p[3]:
                old = self._old_space_residual(time)

        def residual(c):
            return c[0] * rho[..., None] * u_t + c[1] * gp + c[2] * RM_space + c[3] * old

        def local_derivative(c):
            d = None if dconv is None else c[2] * dconv
            if c[0]:
                d_t = c[0] * np.einsum('eg,ac,gj->egajc', time['alpha'] * rho, eye, N)
                d = d_t if d is None else d + d_t
            return d

        RM_s, RM_p = residual(c_s), residual(c_p)
        if tangent:
            dRM_s, dRM_p = local_derivative(c_s), local_derivative(c_p)

        if self.inertia:
            ws = w * tau * rho * (0.0 if old_level else 1.0)                      # SUPG weight factor
            tau_c = ops.lsic_tau(disc, mu_tau, self.rho, speed)
            divu = np.einsum('egaa->eg', G) + e_hh

            Ru += np.einsum('eg,gi,ega->eia', w, N, conv) + np.einsum('eg,egia->eia', w * tau_c * divu, divB)
            if self.supg:
                Ru += np.einsum('eg,egi,ega->eia', ws, uB, RM_s)

            if tangent:
                Juu = (np.einsum('eg,gi,egajc->eiajc', w, N, dconv)
                       + np.einsum('eg,egia,egjc->eiajc', w * tau_c, divB, divB))
                if self.supg:
                    Juu = Juu + np.einsum('eg,egi,egajc->eiajc', ws, uB, dRM_s)
                    if self.newton:
                        Juu = Juu + np.einsum('eg,gj,egic,ega->eiajc', ws, N, B, RM_s)   # derivative of the SUPG weight
                    blocks.append((c_s[1] * np.einsum('eg,egi,egja->eiaj', ws, uB, B).reshape(nel, nne * dim, nne), ru, rp))
                    for r_phi, c in buoyancy_rows:                                 # d(R_M)/dphi_j = -c N_j
                        blocks.append((-c_s[2] * np.einsum('eg,egi,a,gj->eiaj', ws, uB, c, N).reshape(nel, nne * dim, nne),
                                       ru, r_phi))
                blocks.append((Juu.reshape(nel, nne * dim, nne * dim), ru, ru))

        R += layout.scatter_vector(Ru.reshape(nel, -1), ru)

        # ----- PSPG -----
        if self.pspg:
            R += layout.scatter_vector(np.einsum('eg,egia,ega->ei', w * tau, B, RM_p), rp)
            if tangent:
                blocks.append((c_p[1] * np.einsum('eg,egia,egja->eij', w * tau, B, B), rp, rp))
                if dRM_p is not None:
                    blocks.append((np.einsum('eg,egia,egajc->eijc', w * tau, B, dRM_p).reshape(nel, nne, nne * dim), rp, ru))
                for r_phi, c in buoyancy_rows:
                    blocks.append((-c_p[2] * np.einsum('eg,egia,a,gj->eij', w * tau, B, c, N), rp, r_phi))

        # ----- viscous heating -----
        if self.heating is not None:
            rH = layout.element_dofs(disc.conn, self.heating)
            R -= layout.scatter_vector(np.einsum('eg,gi,eg->ei', w, N, mu * g2), rH)
            if tangent:
                dphi = np.einsum('eg,egjc->egjc', mu + g2 * mu_g2, dg2)
                blocks.append((-np.einsum('eg,gi,egjc->eijc', w, N, dphi).reshape(nel, nne, nne * dim), rH, ru))
                for i, r_f in enumerate(rF):
                    blocks.append((-np.einsum('eg,gi,eg,gj->eij', w, N, g2 * mu_F[..., i], N), rH, r_f))

        if not tangent:
            return R, None

        J = sp.csr_matrix((layout.ndof, layout.ndof))
        for Ke, rows, cols in blocks:
            J = J + layout.scatter_matrix(Ke, rows, cols)

        # ----- nonlocal part: -(SUPG and PSPG weights) @ d(div sigma~)/dx -----
        if d_divs is not None and (self.pspg or self.inertia):
            cols = np.arange(nel * ngp).reshape(nel, ngp)[:, :, None] * dim + np.arange(dim)    # (nel, ngp, dim)
            W = sp.csr_matrix((layout.ndof, nel * ngp * dim))
            if self.inertia:
                Ws = np.einsum('eg,egi,ac->egiac', c_s[2] * ws, uB, eye)                        # (e, g, i, a, a')
                rows = np.broadcast_to(ru.reshape(nel, nne, dim)[:, None, :, :, None], Ws.shape)
                W = W + sp.coo_matrix((Ws.ravel(), (rows.ravel(),
                                                    np.broadcast_to(cols[:, :, None, None, :], Ws.shape).ravel())),
                                      shape=W.shape).tocsr()
            if self.pspg:
                Wp = np.einsum('eg,egia->egia', c_p[2] * w * tau, B)
                W = W + sp.coo_matrix((Wp.ravel(), (np.broadcast_to(rp[:, None, :, None], Wp.shape).ravel(),
                                                    np.broadcast_to(cols[:, :, None, :], Wp.shape).ravel())),
                                      shape=W.shape).tocsr()
            J = J - W @ d_divs

        return R, J

    def _stress_tangent(self, dsig, dsigF, ru, rF):

        """Sparse d(sigma at the Gauss points)/dx (nel*ngp*dim*dim, ndof) from the local derivatives."""

        disc, layout = self.system.disc, self.system.layout
        nel, ngp, nne, dim = disc.B.shape
        dd = dim * dim

        rows = (np.arange(nel * ngp).reshape(nel, ngp)[:, :, None, None] * dd
                + np.arange(dim)[:, None] * dim + np.arange(dim)[None, :])                    # (nel, ngp, dim, dim)
        cols_u = ru.reshape(nel, nne, dim)
        D = sp.coo_matrix((dsig.ravel(), (np.broadcast_to(rows[..., None, None], dsig.shape).ravel(),
                                          np.broadcast_to(cols_u[:, None, None, None], dsig.shape).ravel())),
                          shape=(nel * ngp * dd, layout.ndof)).tocsr()
        for dsigT, rT in zip(dsigF, rF):
            D = D + sp.coo_matrix((dsigT.ravel(), (np.broadcast_to(rows[..., None], dsigT.shape).ravel(),
                                                   np.broadcast_to(rT[:, None, None, None, :], dsigT.shape).ravel())),
                                  shape=D.shape).tocsr()
        return D

    def _hoop_divergence(self, sigma, s_hh, dsig, ds_hh, dsigF, ds_hhF, ru, rF):

        """Non-derivative terms of div(sigma) in cylindrical coordinates at the Gauss points, (sigma_rr - sigma_tt) / r
        and sigma_rz / r, and their sparse derivative (nel*ngp*dim, ndof)."""

        disc, layout = self.system.disc, self.system.layout
        nel, ngp, nne, dim = disc.B.shape
        r = disc.radius

        extra = np.stack([(sigma[..., 0, 0] - s_hh) / r, sigma[..., 1, 0] / r], axis=-1)

        rows = np.arange(nel * ngp).reshape(nel, ngp)[:, :, None] * dim + np.arange(dim)       # (nel, ngp, dim)
        dU = np.stack([(dsig[:, :, 0, 0] - ds_hh) / r[..., None, None], dsig[:, :, 1, 0] / r[..., None, None]],
                      axis=2)                                                                   # (e, g, a, j, c)
        cols_u = ru.reshape(nel, nne, dim)
        D = sp.coo_matrix((dU.ravel(), (np.broadcast_to(rows[..., None, None], dU.shape).ravel(),
                                        np.broadcast_to(cols_u[:, None, None], dU.shape).ravel())),
                          shape=(nel * ngp * dim, layout.ndof)).tocsr()

        for dsigT, ds_hhT, rT in zip(dsigF, ds_hhF, rF):
            dT = np.stack([(dsigT[:, :, 0, 0] - ds_hhT) / r[..., None], dsigT[:, :, 1, 0] / r[..., None]], axis=2)
            D = D + sp.coo_matrix((dT.ravel(), (np.broadcast_to(rows[..., None], dT.shape).ravel(),
                                                np.broadcast_to(rT[:, None, None, :], dT.shape).ravel())),
                                  shape=D.shape).tocsr()

        return extra, D

    def _old_space_residual(self, time):

        """RM_space (momentum residual without grad p) at the old level time['x_old'], computed once per step."""

        x_old = time['x_old']
        cached = getattr(self, '_old_cache', None)
        if cached is not None and cached[0] is x_old:
            return cached[1]

        value = self._space_residual(x_old)
        self._old_cache = (x_old, value)

        return value

    def _space_residual(self, x):

        """rho (u . grad) u - f - f_buoyancy - div(sigma~) at the Gauss points of the state x (no derivatives)."""

        disc = self.system.disc
        nel, ngp, nne, dim = disc.B.shape
        fields, u, G, _ = self._kinematics(x)

        if self.law is not None:
            eps, e_hh, _, _ = self._strain(u, G, derivative=False)
            mu = self.viscosity(x, False)[0]
            sigma = 2.0 * mu[..., None, None] * eps
            divs = np.zeros((nel, ngp, dim)) if self.Q is None else (self.Q @ sigma.ravel()).reshape(nel, ngp, dim)
            if self.Q is not None and disc.axisymmetric:
                r = disc.radius
                divs = divs + np.stack([(sigma[..., 0, 0] - 2.0 * mu * e_hh) / r, sigma[..., 1, 0] / r], axis=-1)
        elif self.V is not None:
            divs = (self.V @ x).reshape(nel, ngp, dim)
        else:
            divs = np.zeros((nel, ngp, dim))

        value = -divs
        if self.f is not None:
            value = value - self.f
        for name, c, ref in self.buoyancy:
            value = value - (disc.interpolate(np.asarray(fields[name], dtype=float)) - ref)[..., None] * c
        if self.inertia:
            value = value + ops.at_gauss(disc, self.rho)[..., None] * np.einsum('egb,egab->ega', u, G)

        return value
