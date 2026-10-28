# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++++

# Class Definition for Heat transfer physics

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit

from Source.Pre_processing.BasisFunctions import shape_functions_1d
from Source.enums import ShapeFunctionType

from Source.Physics.Physics import physics
from Source.Primals.Scalar import scalarField
from Source.Simulation.PointwiseLaw import PointwiseLaw
from Source.Simulation.System import System
import Source.Simulation.Operators as ops

u = 0.01  # Velocity for convection term, hardcoded for now


def build_reaction_function(param_dict: dict):

    required_keys = ['stoich', 'k0', 'E_R', 'T', 'order']

    try:
        for key in required_keys:
            if key not in param_dict or param_dict[key] is None:
                raise KeyError(f'{key} not defined')
    except KeyError as e:
        print(f'Execution error: {e}')

    else:
        stoich = param_dict['stoich']
        k0 = param_dict['k0']
        E_R = param_dict['E_R']
        T = param_dict['T']
        reaction_order = param_dict['order']
        components = tuple(stoich.keys())
        component_order = tuple(reaction_order[comp] for comp in components)
        k_rate = k0 * jnp.exp(-(E_R/T))


        @jit(static_argnums=(0,3))
        def compiled_reaction_func(element_shape:str, xi: float, state_at_p: dict, target_var: str):

            nu = stoich.get(target_var, 0.0)

            N_e = shape_functions_1d(element_shape, xi)

            rate_vector = k_rate

            for comp, order in zip(components, component_order):

                c = jnp.maximum(state_at_p[comp], 1e-12)

                rate_vector *= c**order

            rate_vector.reshape(-1, 1)

            rate = jnp.dot(N_e, rate_vector)

            return nu * rate

        return compiled_reaction_func


def build_reaction_law(species, reactions, min_concentration=1e-12):

    """Pointwise law c -> production rate of every species, for a list of reactions.

    Each reaction is a dict with 'stoich' {species: nu}, 'order' {species: n}, 'k0', 'E_R' and 'T':
    rate_k = k0 exp(-E_R / T) * prod_j c_j^n_j and r_s = sum_k nu_{s,k} rate_k.
    Concentrations are clamped to `min_concentration` before the power, so that non-integer orders
    never see negative values (the derivative is zero where the clamp is active).
    """

    nu = np.array([[r['stoich'].get(s, 0.0) for s in species] for r in reactions])   # (n_reactions, n_species)
    order = np.array([[r['order'].get(s, 0.0) for s in species] for r in reactions])
    k_rate = np.array([r['k0'] * np.exp(-r['E_R'] / r['T']) for r in reactions])

    def rates(c):
        c = jnp.maximum(c, min_concentration)
        rate = k_rate * jnp.prod(c[None, :] ** order, axis=1)

        return nu.T @ rate

    return PointwiseLaw(rates)


class mt(physics):

    supports_vectorized_assembly = True

    def __init__(self, model):
        super().__init__(model)

        self.physics_description = 'Transport of Chemical Species'

        self.var = {}

        self.C_const = 1
        self.K_const = 1
        self.M_const = 1

        self.ChemSpecies = []
        self.stoich = {}
   
        self.Pe = 1

        self.Diffusivities = {}

        self.source = 0.0

        self.reactions = []      # reaction dicts, see build_reaction_law
        self.sources = {}        # species -> external source (number or function of the coordinates)
        self._reaction_law = None

    def _check_species(self, chemSpecies):

        if chemSpecies not in self.var:
            raise ValueError(f"La especie '{chemSpecies}' no esta definida (use setChemSpecies).")

    def setDiffusivity(self, chemSpecies, value):

        """Diffusivity of a species: a number or a function of the coordinates f(x), f(x, y)."""

        self._check_species(chemSpecies)
        self.Diffusivities[chemSpecies] = value

    def setSource(self, chemSpecies, value):

        """External volumetric source of a species [mol/(m3 s)]: a number or a function of the coordinates."""

        self._check_species(chemSpecies)
        self.sources[chemSpecies] = value

    def setChemSpecies(self, chemSpecies, name):

        self.ChemSpecies.append(chemSpecies)
        self._reaction_law = None

        self.var[chemSpecies] = scalarField(chemSpecies, name, 'mol/m3', ShapeFunctionType.linear, self.modelRef.mesh)

    def reaction_law(self):

        """The reaction law of the current species and reactions, compiled once and reused."""

        if self._reaction_law is None:
            self._reaction_law = build_reaction_law(list(self.var), self.reactions)

        return self._reaction_law

    def build_system(self, model, disc, layout):

        """Convection-diffusion-reaction of every species, with their sources and boundary conditions.

        R_s = K_s c_s + C c_s - integral(N nu_s r(c)) - external source, with the reactions coupling
        all the species through one pointwise law (automatic Jacobian).
        """

        system = System(disc, layout)

        for name in self.var:

            if name not in self.Diffusivities:
                raise ValueError(f"Falta la difusividad de '{name}' (use setDiffusivity).")

            rows = layout.element_dofs(disc.conn, name)

            system.add_matrix(layout.scatter_matrix(ops.diffusion(disc, self.Diffusivities[name]), rows, rows))

            if self.Convection:
                system.add_matrix(layout.scatter_matrix(ops.convection(disc, self.C_const, self.vel), rows, rows))

            if name in self.sources:
                system.add_vector(layout.scatter_vector(ops.source(disc, self.sources[name]), rows))

            system.add_mass(layout.scatter_matrix(ops.mass(disc, self.M_const), rows, rows))

        if self.reactions:
            system.add_volume_law(self.reaction_law(), list(self.var), list(self.var), sign=-1.0)

        system.add_boundary_conditions(model.mesh, self.facet_bcs)

        return system

    def initializeMatrices(self, element, Variable):

        if self.Convection:
            self.C = self.div(self.var[Variable], element, self.C_const, u)  # 1 stands for velocity (u = 1)

        self.K = self.laplacian(self.Diffusivities[Variable], self.var[Variable], element)

        self.B = self.addBMatrix(element, Variable)

    def initializeVectors(self, element, Variable):

        self.F = self.forceVector(element, Variable)
        self.G = self.addGVector(element, Variable)        

## ------------- Source terms -------------- ##

    def addReaction(self, stoich: dict, k0: float, E_R: float, T: float, order: dict):

        """Adds a reaction rate = k0 exp(-E_R/T) prod(c^order). Can be called several times.

        stoich {species: nu} (negative for reactants), order {species: n}; species missing in
        `order` have order 0.
        """

        for species in list(stoich) + list(order):
            self._check_species(species)

        order = {**{species: 0 for species in stoich}, **order}

        param_dict = {'stoich': stoich, 'k0': k0, 'E_R': E_R, 'T': T, 'order': order}

        reaction_func = build_reaction_function(param_dict)

        self.reactions.append(dict(param_dict))
        self._reaction_law = None

        self.source = reaction_func
        self.F_const = reaction_func
        self.stoich = stoich
        self.has_reaction = True
        self.order = order

        return reaction_func



## ---------- Boundary conditions ---------- ##

    def addBC_Concentration(self, id:int, chemSpec: str, C0: float):
        self.modelRef._mesh.NL[id].BC[chemSpec] = self.setDirichletBC(C0)

            
    def addBC_Outflow(self, id: int, chemSpec: str):

        self.modelRef._mesh.NL[id].BC[chemSpec] = self.setNeumannBC(0.0)

    ## ---------- Boundary conditions on edges (2D) ---------- ##

    def addBC_ConcentrationEdges(self, edges, chemSpec: str, C0):

        """Dirichlet concentration on every node of the given edges (C0: number or function of x, y)."""

        self._check_species(chemSpec)
        nodes = self.modelRef._mesh.NL

        for a, b in edges:
            for node_id in (a, b):
                value = C0(*nodes[node_id].coor[:2]) if callable(C0) else C0
                nodes[node_id].BC[chemSpec] = self.setDirichletBC(value)

    def addBC_FluxEdges(self, edges, chemSpec: str, flux: float):

        """Species flux leaving the domain through the edges [mol/(m2 s)]."""

        self._check_species(chemSpec)
        self.facet_bcs.append({'type': 'Neumann', 'field': chemSpec, 'facets': list(edges), 'flux': flux})

    def addBC_MassTransferEdges(self, edges, chemSpec: str, k_m: float, C_ext: float):

        """Film transfer k_m (C - C_ext) leaving the domain through the edges."""

        self._check_species(chemSpec)
        self.facet_bcs.append({'type': 'Newton', 'field': chemSpec, 'facets': list(edges),
                               'h': k_m, 'var_ext': C_ext})
