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

from Source.enums import ShapeFunctionType

from Source.Physics.Physics import physics, KELVIN_OFFSET
from Source.Primals.Scalar import scalarField
from Source.Simulation.PointwiseLaw import PointwiseLaw
import Source.Simulation.Operators as ops

def build_reaction_law(species, reactions, min_concentration=1e-12):

    """Pointwise law of the reactions: (concentrations[, temperature]) -> (species production[, heat]).

    Each reaction is a dict with 'stoich' {species: nu}, 'order' {species: n}, 'k0', 'E_R', 'T' and 'dH':
    rate_k = k0 exp(-E_R / T_k) * prod_j c_j^n_j,  production of s = sum_k nu_{s,k} rate_k,
    heat released = sum_k (-dH_k) rate_k  [W/m3, dH in J/mol; negative dH = exothermic].
    T_k is a number in Kelvin, or the name of a temperature field (in °C, converted to Kelvin here).
    Concentrations are clamped to `min_concentration` before the power, so that non-integer orders
    never see negative values (the derivative is zero where the clamp is active).

    Returns (law, inputs, outputs): the field names that the law reads and writes. The inputs are the
    species plus the temperature field when some reaction uses one; the outputs are the species plus
    the temperature field when some reaction releases heat.
    """

    fields = {r['T'] for r in reactions if isinstance(r['T'], str)}

    if len(fields) > 1:
        raise ValueError(f"Las reacciones usan varios campos de temperatura: {sorted(fields)}.")

    T_field = fields.pop() if fields else None
    releases_heat = any(r['dH'] is not None for r in reactions)

    if releases_heat and T_field is None:
        raise ValueError("Una reaccion con dH necesita T como nombre del campo de temperatura (por ejemplo T='T').")

    n_species = len(species)
    nu = np.array([[r['stoich'].get(s, 0.0) for s in species] for r in reactions])   # (n_reactions, n_species)
    order = np.array([[r['order'].get(s, 0.0) for s in species] for r in reactions])
    k0 = np.array([r['k0'] for r in reactions])
    E_R = np.array([r['E_R'] for r in reactions])
    uses_field = np.array([isinstance(r['T'], str) for r in reactions])
    T_fixed = np.array([1.0 if isinstance(r['T'], str) else r['T'] for r in reactions])
    heat = np.array([-(r['dH'] or 0.0) for r in reactions])

    def reactions_law(u):
        c = jnp.maximum(u[:n_species], min_concentration)

        if T_field is None:
            T_k = T_fixed
        else:
            T_k = jnp.where(uses_field, u[n_species] + KELVIN_OFFSET, T_fixed)

        rate = k0 * jnp.exp(-E_R / T_k) * jnp.prod(c[None, :] ** order, axis=1)
        production = nu.T @ rate

        if releases_heat:
            return jnp.concatenate([production, jnp.sum(heat * rate)[None]])

        return production

    inputs = list(species) + ([T_field] if T_field else [])
    outputs = list(species) + ([T_field] if releases_heat else [])

    return PointwiseLaw(reactions_law), inputs, outputs


class mt(physics):

    part_name = 'mass'  # name used when this physics is combined with others (CoupledPhysics)

    def __init__(self, model):
        super().__init__(model)

        self.physics_description = 'Transport of Chemical Species'

        self.var = {}

        self.C_const = 1
        self.M_const = 1

        self.ChemSpecies = []

        self.Pe = 1

        self.Diffusivities = {}

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

        """(law, inputs, outputs) of the current species and reactions, compiled once and reused."""

        if self._reaction_law is None:
            self._reaction_law = build_reaction_law(list(self.var), self.reactions)

        return self._reaction_law

    def add_terms(self, system, model, disc, layout):

        """Convection-diffusion-reaction of every species, with their sources (conditions are added by build_system).

        R_s = K_s c_s + C c_s - integral(N nu_s r(c)) - external source, with the reactions coupling
        all the species through one pointwise law (automatic Jacobian).
        """

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
            law, inputs, outputs = self.reaction_law()

            for name in inputs + outputs:
                if name not in layout.names:
                    raise ValueError(f"Las reacciones usan el campo '{name}', que no existe en el modelo "
                                     "(combine esta fisica con la de calor: Model(psc=[ht, mt])).")

            system.add_volume_law(law, inputs, outputs, sign=-1.0)

## ------------- Source terms -------------- ##

    def addReaction(self, stoich: dict, k0: float, E_R: float, T, order: dict, dH: float = None):

        """Adds a reaction rate = k0 exp(-E_R/T) prod(c^order). Can be called several times.

        stoich {species: nu} (negative for reactants), order {species: n}; species missing in
        `order` have order 0. E_R is E/R in Kelvin.
        T: a temperature in Kelvin (isothermal kinetics), or the name of the temperature field
           (for example 'T'), whose values are in °C. The model must then include the heat physics.
        dH: heat of reaction [J/mol], negative if exothermic. It releases (-dH) * rate [W/m3] into the
            temperature field, so it needs T given as a field name.
        """

        for species in list(stoich) + list(order):
            self._check_species(species)

        order = {**{species: 0 for species in stoich}, **order}

        self.reactions.append({'stoich': stoich, 'k0': k0, 'E_R': E_R, 'T': T, 'order': order, 'dH': dH})
        self._reaction_law = None


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
