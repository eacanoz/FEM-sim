# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++++

# Class Definition for Heat transfer physics

import numpy as np
import scipy as sc
import sympy as sp
import math

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit

from scipy import integrate

from Source.Pre_processing.BasisFunctions import basisFunctions, shape_functions_1d, shape_functions_gradient_1d, mapping_function, jacobian_mapping_function, jacobian_determinant, jacobian_inverse
from Source.enums import ElementType, ShapeFunctionType, StudyType, ProblemType, SolverType


from Source.Physics.Physics import physics, get_gauss_points_weights
from Source.Primals.Scalar import scalarField

from Source.Pre_processing.Mesh import Mesh, Element, Node

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


class mt(physics):

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

    def setDiffusivity(self, chemSpecies, value):

        if chemSpecies in self.var:
            self.Diffusivities[chemSpecies] = value
        else:
            raise "Error: Chemical Specie not defined"


    def setChemSpecies(self, chemSpecies, name):

        self.ChemSpecies.append(chemSpecies)

        self.var[chemSpecies] = scalarField(chemSpecies, name, 'mol/m3', ShapeFunctionType.linear, self.modelRef.mesh)

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

        k0 = k0
        E_R = E_R
        T = T

        param_dict = {'stoich': stoich, 'k0': k0, 'E_R': E_R, 'T': T, 'order': order}

        reaction_func = build_reaction_function(param_dict)


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