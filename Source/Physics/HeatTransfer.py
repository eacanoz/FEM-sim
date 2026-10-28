# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++++

# Class Definition for Heat transfer physics

from Source.Physics.Physics import physics
from Source.Primals.Scalar import scalarField
from Source.Simulation.System import System
import Source.Simulation.Operators as ops
from Source.enums import ShapeFunctionType

import jax
jax.config.update("jax_enable_x64", True)
from jax import jit


KELVIN_OFFSET = 273.15  # Temperatures are stored in °C; radiation needs K

@jit
def _calculate_radiative_heat_coef(epsilon, T_ext, T_surface):
    sigma = 5.6704e-8  # Stefan-Boltzmann constant
    h_rad = epsilon * sigma * ((T_surface)**3 + T_ext*(T_surface)**2 
                             + (T_surface)*T_ext**2 + T_ext**3)
    return h_rad


class ht(physics):
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

    def build_system(self, model, disc, layout):

        """Heat conduction with convection, source, Dirichlet, Neumann and Robin/radiation conditions.

        K and F can be numbers or functions of the coordinates; Newton h and Neumann flux can be
        numbers or functions of the temperature (radiation).
        """

        system = System(disc, layout)
        rows = layout.element_dofs(disc.conn, 'T')

        system.add_matrix(layout.scatter_matrix(ops.diffusion(disc, self.K_const), rows, rows))

        if self.Convection:
            system.add_matrix(layout.scatter_matrix(ops.convection(disc, self.C_const, self.vel), rows, rows))

        system.add_vector(layout.scatter_vector(ops.source(disc, self.F_const), rows))
        system.add_mass(layout.scatter_matrix(ops.mass(disc, self.M_const), rows, rows))

        system.add_boundary_conditions(model.mesh, self.facet_bcs)

        return system

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

        """Convection h_c (T_ext - T) through the edges [W/m2]. h_c must be a number."""

        self.facet_bcs.append({'type': 'Newton', 'field': 'T', 'facets': list(edges), 'h': h_c, 'var_ext': T_ext})

    def addBC_Radiation(self, id: int, epsilon: float, T_ext: float):

        """
        Radiation BC. Temperatures are given in °C; the Stefan-Boltzmann law needs Kelvin,
        so h(T) is evaluated with absolute temperatures. The flux is h(T)*(T - T_ext),
        which is the same in °C and K because only the difference appears.
        """

        T_ext_K = T_ext + KELVIN_OFFSET

        h_c = lambda n: _calculate_radiative_heat_coef(epsilon, T_ext_K, n + KELVIN_OFFSET)
        
        self.modelRef._mesh.NL[id].BC['T'] = self.setNewtonBC(h_c, T_ext)