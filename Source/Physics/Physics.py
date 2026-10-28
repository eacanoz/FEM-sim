# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++++

# Base class of the physics: variables, constants, boundary-condition dictionaries and the hook
# (build_system) that each physics implements to build its Source.Simulation.System.


from Source.Simulation.DofLayout import FieldSpec


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

        """Field specifications (name and number of components): one scalar field per variable."""

        return [FieldSpec(name) for name in self.var]

    def build_system(self, model, disc, layout):

        """Builds the Source.Simulation.System of the model. Every physics overrides this."""

        raise NotImplementedError(f"{type(self).__name__} no implementa build_system.")

    def getVariables(self):
        return self.var.keys()

    def getNumOfVar(self):
        return len(self.var.keys())

    def initField(self, variable, value):
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
