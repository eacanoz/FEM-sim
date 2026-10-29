# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++++

# Base class of the physics: variables, constants, boundary-condition dictionaries and the hook
# (build_system) that each physics implements to build its Source.Simulation.System.


import numpy as np

import Source.Simulation.Operators as ops
from Source.Simulation.DofLayout import FieldSpec
from Source.Simulation.System import System

KELVIN_OFFSET = 273.15  # Temperatures are stored in °C; Arrhenius kinetics and radiation need K


class physics:

    def __init__(self, model):

        self.modelRef = model
        self.mat = model.mat

        self.var = {}

        self.Convection = False
        self.Stab = None

        # Coefficients of the weak form (numbers, or functions of the coordinates)
        self.C_const = 0.0   # convection
        self.K_const = 0.0   # diffusion
        self.M_const = 0.0   # mass (transient)
        self.F_const = 0.0   # source

        self.vel = 1.0  # Velocity: a number in 1D, a vector in 2D

        self.Pe = 0  # Peclet number

        self.facet_bcs = []  # Newton/Neumann conditions on boundary facets: dicts with 'type', 'field', 'facets', ...

    def fields(self):

        """Field specifications (name and number of components) of the variables: scalars, or vectors with `ncomp`."""

        return [FieldSpec(name, getattr(field, 'ncomp', 1)) for name, field in self.var.items()]

    def derived_fields(self, model):

        """Nodal quantities computed from the solution (Darcy velocity, shear rate, ...), added to model.sol
        after a steady solve. Physics that have them override this."""

        return {}

    def add_terms(self, system, model, disc, layout):

        """Adds the operators and laws of this physics to `system` (no boundary conditions). Every physics overrides this."""

        raise NotImplementedError(f"{type(self).__name__} no implementa add_terms.")

    def build_system(self, model, disc, layout):

        """Builds the Source.Simulation.System of the model: the terms of this physics plus its boundary conditions."""

        system = System(disc, layout)

        self.add_terms(system, model, disc, layout)
        system.add_boundary_conditions(model.mesh, self.facet_bcs)

        return system

    def setVelocity(self, v):

        """Convection velocity: a number (1D), a vector (2D), a function of the coordinates, an array at the Gauss
        points, or the name of a vector field of the same model (e.g. 'u', the velocity solved by the flow physics in
        Model(psc=[ff, ht])): then the convection is nonlinear and coupled with that field. Switches convection on."""

        self.vel = v
        self.Convection = True

    def setStabilization(self, method='SUPG'):

        """Stabilization of the convection: 'SUPG' (streamline upwind Petrov-Galerkin) or None."""

        if method not in (None, 'SUPG'):
            raise ValueError(f"Estabilizacion desconocida: {method!r} (use 'SUPG' o None).")

        self.Stab = method

    def add_convection(self, system, layout, disc, field, k=None, f=None):

        """Convection of `field` with the velocity `vel` (coefficient C_const), stabilized if Stab = 'SUPG'.

        k: diffusion coefficient used only to set the SUPG parameter (None: pure advection).
        f: source of the equation (number or function of the coordinates), for the consistent SUPG term.
        The SUPG terms are linear: they stabilize convection, diffusion-free mass and source, not nonlinear reactions.
        """

        if not self.Convection:
            return

        rows = layout.element_dofs(disc.conn, field)

        if isinstance(self.vel, str):
            if self.vel not in layout.names:
                raise ValueError(f"La velocidad de conveccion es el campo '{self.vel}', que no existe en el modelo "
                                 "(combine con la fisica de flujo: Model(psc=[ff, ht])).")
            system.add_term(FieldConvectionTerm(system, field, self.vel, self.C_const, k, f, self.Stab == 'SUPG',
                                                m=self.M_const))
            return

        system.add_matrix(layout.scatter_matrix(ops.convection(disc, self.C_const, self.vel), rows, rows))

        if self.Stab == 'SUPG':
            tau = ops.supg_tau(disc, self.C_const, self.vel, k)
            K, M, F = ops.supg_terms(disc, self.C_const, self.vel, tau, m=self.M_const, f=f)
            system.add_matrix(layout.scatter_matrix(K, rows, rows))
            system.add_mass(layout.scatter_matrix(M, rows, rows))
            if F is not None:
                system.add_vector(layout.scatter_vector(F, rows))

    def boundaryFlux(self, edges, field=None, rate=None):

        """What enters the domain through the edges per unit time (heat [W per unit depth], species [mol/s per unit
        depth], ...; axisymmetric: through the whole surface of revolution), from the reactions: the residual of
        `field` at the nodes of those edges, assembled without the boundary conditions, is the consistent boundary
        integral of k dphi/dn (conduction plus what convection carries in). Positive: entering. Use it on edges with
        a Dirichlet condition (a heated wall): elsewhere the natural condition already sets the flux.

        rate: time derivative of the solution vector (transient), to include the storage of the boundary elements.
        """

        model = self.modelRef
        system, layout = model._system(), model.layout
        field = field or next(iter(self.var))

        saved, system.time = system.time, None
        try:
            _, R = system._assemble(model._state())
        finally:
            system.time = saved
        if rate is not None:
            R = R + system.mass() @ np.asarray(rate, dtype=float)

        nodes = np.unique(np.ravel(edges))
        flux = float(R[layout.node_dofs(field, nodes)[:, 0]].sum())

        return 2.0 * np.pi * flux if model.geometry.axisymmetric else flux

    def getVariables(self):
        return self.var.keys()

    def getNumOfVar(self):
        return len(self.var.keys())

    def initField(self, variable, value):

        """Initial value of a field: a number, or a function of the node coordinates f(x), f(x, y)."""

        if callable(value):
            nodes = self.modelRef.mesh.NL
            self.var[variable].updateField(np.array([value(*node.coor) for node in nodes], dtype=float))
        else:
            self.var[variable].initField(value)

    # Boundary conditions

    def setDirichletBC(self, val):

        """
        Define a Dirichlet Boundary Condition for the node

        :param T0: Temperature at the boundary
        :return: Dictionary with boundary information for the physics
        """

        return {'type': 'Dirichlet', 'value': val}

    def setNeumannBC(self, flux):

        """
        Define a Neumann Boundary Condition for the node

        :param q_flux: Heat flux at the boundary
        :return: Dictionary with boundary information for the physics
        """

        return {'type': 'Neumann', 'flux': flux}

    def setNewtonBC(self, h, var_ext):

        """
        Define a Newton Boundary Condition for the node

        :param h_c: Heat transfer coefficient
        :param T_ext: External temperature
        :return: Dictionary with boundary information for the physics
        """

        return {'type': 'Newton', 'h': h, 'var_ext': var_ext}


class FieldConvectionTerm:

    """Convection of a scalar field by a vector field of the same model, both unknowns (monolithic coupling):

        R_phi += integral N_i c u . grad(phi)  [+ integral tau (c u . grad N_i) (c u . grad(phi) - f)  with SUPG]

    The tangent includes the coupling blocks d/du (convecting velocity and SUPG weight); tau is kept frozen. The SUPG
    residual has the convection, the source f of the equation and, in transient studies, m dphi/dt of the step
    (system.time); other sources (reactions, viscous heating) and the diffusion (zero inside linear elements) are left
    out of it.
    """

    def __init__(self, system, field, velocity, c, k=None, f=None, supg=False, m=0.0):

        self.system = system
        self.field, self.velocity = field, velocity
        self.c, self.k, self.f = c, k, f
        self.m = m                   # capacity of the time derivative (rho Cp for heat): in the SUPG residual
        self.supg = supg
        self.tau_state = None        # evaluate tau at this state instead of x (to test the tangent)

    def contribution(self, x):

        disc, layout = self.system.disc, self.system.layout
        nel, ngp, nne, dim = disc.B.shape
        N, B, w = disc.N, disc.B, disc.detJw

        fields = layout.split(x)
        u = disc.interpolate(np.asarray(fields[self.velocity], dtype=float))      # (nel, ngp, dim)
        gphi = disc.gradient(np.asarray(fields[self.field], dtype=float))         # (nel, ngp, dim)
        c = ops.at_gauss(disc, self.c)

        a = c[..., None] * u
        conv = np.einsum('ega,ega->eg', a, gphi)
        aB = np.einsum('egb,egjb->egj', a, B)                                     # c u . grad N_j

        rows = layout.element_dofs(disc.conn, self.field)
        cols_u = layout.element_dofs(disc.conn, self.velocity)

        Re = np.einsum('eg,gi,eg->ei', w, N, conv)
        Kpp = np.einsum('eg,gi,egj->eij', w, N, aB)
        Kpu = np.einsum('eg,gi,gj,egc->eijc', w * c, N, N, gphi)

        time = self.system.time
        if self.supg and not (time is not None and time.get('old_level', False)):
            u_tau = u if self.tau_state is None else disc.interpolate(layout.split(self.tau_state)[self.velocity])
            tau = ops.supg_tau(disc, self.c, u_tau, self.k)
            res = conv - (ops.at_gauss(disc, self.f) if self.f is not None else 0.0)

            if time is not None:
                # time-discrete residual of the step, divided by theta (the solver weighs these rows by theta):
                # m phi'/theta + space(x) + (1 - theta)/theta space(x_n)
                th = time['theta']
                if th < 1.0:
                    res = res + (1.0 - th) / th * self._old_space_residual(time)
                if np.any(np.asarray(self.m) != 0):
                    phi_t = disc.interpolate(np.asarray(layout.split(time['alpha'] * x + time['beta'])[self.field],
                                                        dtype=float))
                    mt = ops.at_gauss(disc, self.m) / th
                    res = res + mt * phi_t
                    Kpp = Kpp + np.einsum('eg,egi,gj->eij', w * tau * mt * time['alpha'], aB, N)

            Re = Re + np.einsum('eg,egi,eg->ei', w * tau, aB, res)
            Kpp = Kpp + np.einsum('eg,egi,egj->eij', w * tau, aB, aB)
            Kpu = Kpu + (np.einsum('eg,gj,egic,eg->eijc', w * tau * c, N, B, res)
                         + np.einsum('eg,egi,gj,egc->eijc', w * tau * c, aB, N, gphi))

        R = layout.scatter_vector(Re, rows)
        J = (layout.scatter_matrix(Kpp, rows, rows)
             + layout.scatter_matrix(Kpu.reshape(nel, nne, nne * dim), rows, cols_u))

        return R, J

    def _old_space_residual(self, time):

        """c u . grad(phi) - f at the old level time['x_old'], computed once per step."""

        x_old = time['x_old']
        cached = getattr(self, '_old_cache', None)
        if cached is not None and cached[0] is x_old:
            return cached[1]

        disc, layout = self.system.disc, self.system.layout
        fields = layout.split(x_old)
        u = disc.interpolate(np.asarray(fields[self.velocity], dtype=float))
        gphi = disc.gradient(np.asarray(fields[self.field], dtype=float))
        value = ops.at_gauss(disc, self.c) * np.einsum('ega,ega->eg', u, gphi)
        if self.f is not None:
            value = value - ops.at_gauss(disc, self.f)

        self._old_cache = (x_old, value)
        return value
