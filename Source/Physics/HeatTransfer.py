# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++++

# Class Definition for Heat transfer physics

from Source.Physics.Physics import physics, KELVIN_OFFSET
from Source.Primals.Scalar import scalarField
from Source.Simulation.PointwiseLaw import PointwiseLaw, cached_law
import Source.Simulation.Operators as ops
from Source.enums import ShapeFunctionType

import jax
jax.config.update("jax_enable_x64", True)
from jax import jit



@jit
def _calculate_radiative_heat_coef(epsilon, T_ext, T_surface):
    sigma = 5.6704e-8  # Stefan-Boltzmann constant
    h_rad = epsilon * sigma * ((T_surface)**3 + T_ext*(T_surface)**2 
                             + (T_surface)*T_ext**2 + T_ext**3)
    return h_rad


class ht(physics):

    part_name = 'heat'  # name used when this physics is combined with others (CoupledPhysics)
    ConstRel = '∇.(k * ∇T) + Q == 0'

    # Set material properties.

    def __init__(self, model):
        super().__init__(model)

        self.physics_description = 'Heat transfer'

        self.var = {'T': scalarField('T', 'Temperature', 'K', ShapeFunctionType.linear, model.mesh)}

        self.C_const = self.mat.rho * self.mat.Cp
        self.K_const = self.mat.k
        self.M_const = self.mat.rho * self.mat.Cp
        self.F_const = 0

        self.Pe = self.mat.rho * self.mat.Cp * self.vel / self.mat.k

        self.conductivity_law = None  # k(T), see setConductivityLaw
        self.conductivity_inputs = ('T',)

    def setConductivityLaw(self, k_of_T, depends_on=('T',)):

        """Conductivity law k(T) [W/(m K)], T in the units of the model (°C).

        k_of_T must be written with jax.numpy (it is traced and differentiated), e.g.
        `lambda T: 1.0 + 0.002 * T` or `lambda T: 2.0 * jnp.exp(0.01 * T)`. It replaces the constant
        conductivity of the material; the problem becomes nonlinear (use ProblemType.nonlinear).
        Pass None to go back to the constant conductivity.

        depends_on: names of the fields passed to the function, in order. With a coupled model the
        conductivity can depend on a species, e.g. `setConductivityLaw(lambda T, c: 1 + 0.5 * c,
        depends_on=('T', 'A'))`.
        """

        self.conductivity_law = k_of_T
        self.conductivity_inputs = tuple(depends_on)

    def add_terms(self, system, model, disc, layout):

        """Heat conduction with convection, source and mass. The conditions (Dirichlet, Neumann, Robin,
        radiation) are added by build_system.

        K and F can be numbers or functions of the coordinates; Newton h and Neumann flux can be
        numbers or functions of the temperature (radiation).
        """

        rows = layout.element_dofs(disc.conn, 'T')

        if self.conductivity_law is None:
            system.add_matrix(layout.scatter_matrix(ops.diffusion(disc, self.K_const), rows, rows))
        else:
            fn, inputs = self.conductivity_law, list(self.conductivity_inputs)
            for name in inputs:
                if name not in layout.names:
                    raise ValueError(f"La conductividad depende del campo '{name}', que no existe en el modelo.")
            law = cached_law(('conductivity', fn, tuple(inputs)),
                             lambda fn=fn, n_in=len(inputs): PointwiseLaw(lambda u: fn(*[u[i] for i in range(n_in)])))
            system.add_diffusion_law(law, 'T', inputs)

        if self.Convection:
            system.add_matrix(layout.scatter_matrix(ops.convection(disc, self.C_const, self.vel), rows, rows))

        system.add_vector(layout.scatter_vector(ops.source(disc, self.F_const), rows))
        system.add_mass(layout.scatter_matrix(ops.mass(disc, self.M_const), rows, rows))

## ---------- Boundary conditions ---------- ##

    def addBC_Temperature(self, id: int, T: float):

        self.modelRef._mesh.NL[id].BC['T'] = self.setDirichletBC(T)

    def addBC_Convection(self, id: int, h_c: float, T_ext: float):

        """
        Add convection boundary condition to the node with the given id.
        Parameters:
        -----------
        id (int): Node id to which the boundary condition will be applied
        h_c (float): Convection coefficient
        T_ext (float): External temperature
        """

        self.modelRef._mesh.NL[id].BC['T'] = self.setNewtonBC(h_c, T_ext)

    def addBC_HeatFlux(self, id: int, q_flux: float):

        self.modelRef._mesh.NL[id].BC['T'] = self.setNeumannBC(q_flux)

    ## ---------- Boundary conditions on edges (2D) ---------- ##

    def addBC_TemperatureEdges(self, edges, T):

        """Dirichlet temperature on every node of the given edges. edges: [(node_a, node_b), ...].

        T is a number or a function T(x, y) evaluated at each node.
        """

        nodes = self.modelRef._mesh.NL

        for a, b in edges:
            for node_id in (a, b):
                value = T(*nodes[node_id].coor[:2]) if callable(T) else T
                nodes[node_id].BC['T'] = self.setDirichletBC(value)

    def addBC_HeatFluxEdges(self, edges, q_flux: float):

        """Heat flux leaving the domain through the edges [W/m2], q = -k dT/dn with n pointing outwards.

        A positive value removes heat from the domain. Example: T = 2x + 1 on x = Lx with k = 1
        has q = -2 at that edge.
        """

        self.facet_bcs.append({'type': 'Neumann', 'field': 'T', 'facets': list(edges), 'flux': q_flux})

    def addBC_ConvectionEdges(self, edges, h_c: float, T_ext: float):

        """Convection through the edges: outgoing flux h_c (T - T_ext) [W/m2].

        h_c is a number, or a function of the temperature h_c(T) (nonlinear, solved with Newton).
        """

        self.facet_bcs.append({'type': 'Newton', 'field': 'T', 'facets': list(edges), 'h': h_c, 'var_ext': T_ext})

    @staticmethod
    def _radiation_coefficient(epsilon, T_ext):

        """h(T) of a radiating surface. Temperatures are given in °C; the Stefan-Boltzmann law needs
        Kelvin, so h(T) is evaluated with absolute temperatures. The flux is h(T)*(T - T_ext), which is
        the same in °C and K because only the difference appears."""

        T_ext_K = T_ext + KELVIN_OFFSET

        return lambda n: _calculate_radiative_heat_coef(epsilon, T_ext_K, n + KELVIN_OFFSET)

    def addBC_Radiation(self, id: int, epsilon: float, T_ext: float):

        """Radiation to the surroundings at T_ext from the node `id` (1D). See _radiation_coefficient."""

        self.modelRef._mesh.NL[id].BC['T'] = self.setNewtonBC(self._radiation_coefficient(epsilon, T_ext), T_ext)

    def addBC_RadiationEdges(self, edges, epsilon: float, T_ext: float):

        """Radiation to the surroundings at T_ext through the edges (2D), nonlinear in the temperature."""

        self.addBC_ConvectionEdges(edges, self._radiation_coefficient(epsilon, T_ext), T_ext)
