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


from Source.Physics.Physics import physics, get_gauss_points_weights
from Source.Primals.Scalar import scalarField

from Source.Pre_processing.Mesh import Mesh, Element, Node

u = 1  # Velocity for convection term, hardcoded for now

@jit
def _calculate_reaction_rate(w_shape:str, element_shape:str, element_coors: list[float] | np.ndarray, k, a, b, A_values, B_values, nu_stoich):

    puntos_gauss, pesos_gauss = get_gauss_points_weights(2)
    
    y = jnp.zeros((len(A_values), 1))

    for p, w in zip(puntos_gauss, pesos_gauss):

        N = shape_functions_1d(w_shape, p)
        det_J = jacobian_determinant(element_shape, element_coors, p)

        A_p = jnp.dot(N, A_values)[0,0]
        B_p = jnp.dot(N, B_values)[0,0]

        A_gauss = jnp.maximum(A_gauss, 0.0)
        B_gauss = jnp.maximum(B_gauss, 0.0)

        r_p = k * (A_p**a) * (B_p**b)

        y += N.T * (nu_stoich * r_p) * det_J * w

    return y


class mt(physics):

    def __init__(self, model):
        super().__init__(model)

        self.var = {}

        self.C_const = 1
        self.K_const = 1
        self.M_const = 1

        self.Pe = 1

        self.Diffusivities = {}

        self.reaction = None

    def setDiffusivity(self, chemSpecies, value):

        if chemSpecies in self.var:
            self.Diffusivities[chemSpecies] = value
        else:
            raise "Error: Chemical Specie not defined"


    def setChemSpecies(self, chemSpecies, name):

        self.var[chemSpecies] = scalarField(chemSpecies, name, 'mol/m3', 'Linear', self.modelRef.mesh)

    def initializeMatrices(self, element, Variable):

        if self.Convection:
            self.C = self.div(self.var[Variable], element, self.C_const, u)  # 1 stands for velocity (u = 1)

        self.K = self.laplacian(self.Diffusivities[Variable], self.var[Variable], element)

        self.B = self.addBMatrix(element, Variable)

    def initializeVectors(self, element, Variable):

        self.F = self.forceVector(element, Variable)
        self.G = self.addGVector(element, Variable)        

## ------------- Source terms -------------- ##

    def addReaction(self, stoich):

        k0 = 10
        E_R = 500
        T = 303.15
        self.a_param = 1
        self.b_param = 1

        self.k_rate = k0 * math.exp(-(E_R/T))

        #self.rRate = lambda n: self.k_rate * (self.var['A'].values[n]**a) * (self.var['B'].values[n]**b)

        self.stoich = stoich

        self.has_reaction = True
        # self.reaction = lambda elem, chemSpec:  self.stoich[chemSpec] * self.rRate(elem)

        # def reaction(element):

        #     rates = [self.rRate(n.id) for n in element.nodes]

        #     rateVector = element.sF.N.transpose() * sp.Matrix(rates)

        #     rateVectorfunc = sp.lambdify(list(element.sF.N.free_symbols), rateVector, 'numpy')

        #     return  rateVectorfunc


        #self.source = reaction

    # Overwrite forceVector method

    def forceVector(self, element: Element, Variable):

        nu = self.stoich.get(Variable, 0.0)

        if not getattr(self, 'has_reaction', False) or nu == 0.0:
            return jnp.zeros((element.getNumberNodes(), 1))

        A_np = np.array(self.var['A'].getElementValues(element)).reshape(-1, 1)
        B_np = np.array(self.var['B'].getElementValues(element)).reshape(-1, 1)

        A_nodal = jnp.asarray(A_np)
        B_nodal = jnp.asarray(B_np)

        coors = jnp.asarray(element.getCoor())

        F_reaction = _calculate_reaction_rate(self.w_shape, element.shape, coors, self.k_rate, self.a_param, self.b_param, A_nodal, B_nodal, nu)

        return F_reaction

        # if callable(self.source):

        #     f = lambda *args: self.stoich[Variable] * self.source(element)(*args)
        # else:

        #     f = self.source


        # # diff_F = self.w.N * f * element.Jacobian()

        # # F = sp.integrate(diff_F, (e1, -1, 1)).tolist()

        # # return np.array(F).astype(np.float64).reshape((element.getNumberNodes(), 1))[0]  ## To fix!!!

        # diffF = lambda *args: self.w.N_func(*args) * f(*args) * element.J_func(*args)

        # y, err = integrate.quad_vec(diffF, -1, 1)

        # return y[0]


## ---------- Boundary conditions ---------- ##

    def addBC_Concentration(self, id:int, chemSpec: str, C0: float):
        self.modelRef._mesh.NL[id].BC[chemSpec] = self.setDirichletBC(C0)

            
    def addBC_Outflow(self, id: int, chemSpec: str):

        self.modelRef._mesh.NL[id].BC[chemSpec] = self.setNeumannBC(0.0)