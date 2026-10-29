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

        """Convection velocity: a number (1D), a vector (2D) or a function of the coordinates. Switches convection on."""

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
        system.add_matrix(layout.scatter_matrix(ops.convection(disc, self.C_const, self.vel), rows, rows))

        if self.Stab == 'SUPG':
            tau = ops.supg_tau(disc, self.C_const, self.vel, k)
            K, M, F = ops.supg_terms(disc, self.C_const, self.vel, tau, m=self.M_const, f=f)
            system.add_matrix(layout.scatter_matrix(K, rows, rows))
            system.add_mass(layout.scatter_matrix(M, rows, rows))
            if F is not None:
                system.add_vector(layout.scatter_vector(F, rows))

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
