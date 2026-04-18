# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++++

# Class physics: Contains all the functions necessary to create the conservative relations


import numpy as np
import sympy as sp
from scipy import integrate
import math


import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit

from numdifftools import Jacobian

from Source.Pre_processing.BasisFunctions import basisFunctions, shape_functions_1d, shape_functions_gradient_1d, mapping_function, jacobian_mapping_function, jacobian_determinant, jacobian_inverse

from Source.Pre_processing.Mesh import Mesh, Element, Node
from Source.Material import material
from Source.Primals.Scalar import scalarField
from Source.Primals.Vector import vectorField

from Source.enums import ShapeFunctionType, StudyType, ProblemType, SolverType

# from Source.core import Model

# Define basis functions variables
e1, e2, e3 = sp.symbols('e1 e2 e3')


def get_gauss_points_weights(num_points: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    xi, w = np.polynomial.legendre.leggauss(num_points)

    return jnp.array(xi), jnp.array(w)

@jit(static_argnums=(0,1,2))
def _calc_laplacian_term(w_shape:str, 
                        var_shape:str, 
                        element_shape:str, 
                        element_coors: list[float] | np.ndarray, 
                        const: float):
            
        puntos_gauss, pesos_gauss = get_gauss_points_weights(2)

        y = jnp.zeros((len(element_coors), len(element_coors)))  # Assuming square matrix for simplicity, adjust as needed

        for p, w in zip(puntos_gauss, pesos_gauss):

            B_w = shape_functions_gradient_1d(w_shape, p) * jacobian_inverse(element_shape, element_coors, p)     
            B_var = shape_functions_gradient_1d(var_shape, p) * jacobian_inverse(element_shape, element_coors, p)
            detJ = jacobian_determinant(element_shape, element_coors, p)

            y += const * jnp.dot(B_w.T, B_var) * detJ * w
        
        return y

@jit(static_argnums=(0,1,2))
def _calc_gradient_term(w_shape:str, 
                        var_shape:str, 
                        element_shape:str, 
                        element_coors: list[float] | np.ndarray):

        puntos_gauss, pesos_gauss = get_gauss_points_weights(2)

        y = jnp.zeros((len(element_coors), len(element_coors)))  # Assuming square matrix for simplicity, adjust as needed

        for p, w in zip(puntos_gauss, pesos_gauss):

            N_w = shape_functions_1d(w_shape, p)
            B_var = shape_functions_gradient_1d(var_shape, p) * jacobian_inverse(element_shape, element_coors, p)
            detJ = jacobian_determinant(element_shape, element_coors, p)

            y += jnp.dot(N_w.T, B_var) * detJ * w
        
        return y

@jit(static_argnums=(0,1,2))
def _calc_divergence_term(w_shape:str, 
                        var_shape:str, 
                        element_shape:str, 
                        element_coors: list[float] | np.ndarray, 
                        const: float,
                        Vel: float | np.ndarray):
    
        puntos_gauss, pesos_gauss = get_gauss_points_weights(2)

        y = jnp.zeros((len(element_coors), len(element_coors)))  # Assuming square matrix for simplicity, adjust as needed

        for p, w in zip(puntos_gauss, pesos_gauss):

            N_w = shape_functions_1d(w_shape, p)
            B_var = shape_functions_gradient_1d(var_shape, p) * jacobian_inverse(element_shape, element_coors, p)
            detJ = jacobian_determinant(element_shape, element_coors, p)

            y += const * jnp.dot(N_w.T, B_var) * Vel * detJ * w 
        
        return y

@jit(static_argnums=(0,1,2))
def _calc_mass_term(w_shape:str, 
                    var_shape:str, 
                    element_shape:str, 
                    element_coors: list[float] | np.ndarray, 
                    constM: float):
        puntos_gauss, pesos_gauss = get_gauss_points_weights(2)
        y = jnp.zeros((len(element_coors), len(element_coors)))  # Assuming square matrix for simplicity, adjust as needed
        for p, w in zip(puntos_gauss, pesos_gauss):
            N_w = shape_functions_1d(w_shape, p)
            B_var = shape_functions_gradient_1d(var_shape, p)   
            detJ = jacobian_determinant(element_shape, element_coors, p)
            y += constM * jnp.dot(N_w.T, B_var) * detJ * w
        return y

@jit(static_argnums=(0,1))
def _calc_force_vector(w_shape:str, 
                       element_shape:str,
                       element_coors: list[float] | np.ndarray,
                       f: float | np.ndarray):
    
    puntos_gauss, pesos_gauss = get_gauss_points_weights(2)
    y = jnp.zeros((len(element_coors),1))  # Assuming vector for simplicity, adjust as needed

    for p, w in zip(puntos_gauss, pesos_gauss):
        N_w = shape_functions_1d(w_shape, p)
        detJ = jacobian_determinant(element_shape, element_coors, p)

        y += N_w.T * f * detJ * w
    
    return y

def _calc_stabilization_term(w_shape:str, 
                            element_shape:str, 
                            element_coors: list[float] | np.ndarray, 
                            element_length: float,
                            Pe_h: float):

    alpha = (1 / math.tanh(Pe_h / 2)) - 2 / Pe_h

    puntos_gauss, pesos_gauss = get_gauss_points_weights(2)

    y = jnp.zeros((len(element_coors), len(element_coors)))  # Assuming square matrix for simplicity, adjust as needed

    for p, w in zip(puntos_gauss, pesos_gauss):

        B_w = shape_functions_gradient_1d(w_shape, p) * jacobian_inverse(element_shape, element_coors, p)     
        detJ = jacobian_determinant(element_shape, element_coors, p)

        y += alpha * element_length / 2 * jnp.dot(B_w.T, B_w) * detJ * w
    
    return y

#stab= lambda *args: alpha * element.getLength() / 2 * self.w.gradN_func(*args) * element.Jinv_func(*args)


class physics:

    def __init__(self, model):

        self.modelRef = model
        self.w: basisFunctions = model.w
        self.w_shape: str = model.w_shape # To fix: This is not the best way to do it, but for now it works. Next updates will show a better way to do it.
        self.mat = model.mat

        self.var = {}

        self.source = 0

        self.Convection = False
        self.Stab = None

        self.C = 0
        self.K = None
        self.B = 0
        self.F = None
        self.G = None

        self.Pe = 0  # Peclet number

    def getVariables(self):
        return self.var.keys()

    def getNumOfVar(self):
        return len(self.var.keys())

    def initField(self, variable, value):
        self.var[variable].initField(value)
    
    def getElementMatrix(self, element, Variable, solverOptions=None):

        self.initializeMatrices(element, Variable)
        
        return np.asarray(self.C + self.K + self.B)

    def getElementVector(self, element, Variable, solverOptions=None):

        self.initializeVectors(element, Variable)
        
        return np.asarray(self.F - self.G)   
    
    def getElementMassMatrix(self, element, solverOptions=None):
        
        self.initializeMassMatrix(element)

        return np.asarray(self.M)
    
    def getResidualVector(self, element, Variable, solverOptions=None):

        A_e = self.getElementMatrix(element, Variable, solverOptions)
        b_e = self.getElementVector(element, Variable, solverOptions)

        x_e = self.var[Variable].getElementValues(element)

        return A_e.dot(x_e) - b_e
    
    def aux_getTangentMatrix(self, element, Variable, x_values, solverOptions=None):

        x_e_orig = self.var[Variable].getElementValues(element)

        self.var[Variable].setElementValues(element, x_values)

        A_e = self.getElementMatrix(element, Variable, solverOptions)
        b_e = self.getElementVector(element, Variable, solverOptions)

        F_e = A_e.dot(x_values) - b_e

        self.var[Variable].setElementValues(element, x_e_orig)

        return F_e

    def getElementTangentMatrix(self, element, Variable, solverOptions=None):
        
        #func1 = lambda x: self.aux_getTangentMatrix(element, Variable, x, solverOptions)

        x_e = self.var[Variable].getElementValues(element)
        n = x_e.size
        K_e = np.zeros((n, n))
        f0 = self.aux_getTangentMatrix(element, Variable, x_e, solverOptions)

        eps = 1e-8

        for i in range(n):
            x_e_perturbed = np.copy(x_e)
            x_e_perturbed[i] += eps

            f1 = self.aux_getTangentMatrix(element, Variable, x_e_perturbed, solverOptions)

            K_e[:, i] = (f1 - f0) / eps

        #return Jacobian(func1)(x_e)

        return K_e

    def laplacian(self, const: float, var: scalarField, element: Element):
        """
        Define the element matrix from the weak form for the Laplacian term.


        e. g.
        ∇.(const * ∇T)  -> Integral(dNi/dx^Trans*const*dNj/dx*det(J^-1)de1, -1, 1)


        Note: Stills for 1D. Next updates will show the code for multiple dimensions

        :param var:
        :param const: material proporcionality constant
        :param element: Element
        :return: Matrix of laplacian term
        """

        # B_w = lambda x: shape_functions_gradient_1d(self.w_shape, x) * element.Jinv_func(x)

        # B_var = lambda x: shape_functions_gradient_1d(var.shape, x) * element.Jinv_func(x)



        # def diff_Laplacian(*args):
        #     B_w = self.w.gradN_func(*args) * element.Jinv_func(*args)
        #     B_var = var.gradN_func(*args) * element.Jinv_func(*args)
        #     detJ = element.Jdet_func(*args)

        #     return const * jnp.dot(B_w.T, B_var) * detJ

        #diffA1 = lambda *args: self.w.gradN_func(*args)*element.Jinv_func(*args)*\
                               #const*(var.gradN_func(*args)*element.Jinv_func(*args)).T * element.Jdet_func(*args)

        #diffA2 = lambda *args: const*np.dot((self.w.gradN_func(*args)*element.Jinv_func(*args)).T, (var.gradN_func(*args)*element.Jinv_func(*args)))*element.Jdet_func(*args)
        
        #y, err = integrate.quad_vec(diffA1, -1, 1)

        # puntos_gauss = jnp.array([-0.5773502691896257, 0.5773502691896257])
        # pesos_gauss = jnp.array([1.0, 1.0])

        # y = 0.0

        # for p, w in zip(puntos_gauss, pesos_gauss):
        #     y += w * diff_Laplacian(p)

        jax_coors = jnp.array(element.getCoor())

        return _calc_laplacian_term(self.w_shape, var.shape, element.shape, jax_coors, const)

    def Grad(self, var: scalarField, element: Element):

        """
        Define de element matrix from the weak form for the Gradient term.

        e. g.
        ∇T -> Integral(Ni*dNj/dx*det(J^-1)de1, -1, 1)

        :param element: Element
        :return: Matrix form of the Gradient term
        """        
        # if self.Stab == 'PG':
        #     self.w.addStab(self.Stab, self.stabilization(element))


        # diff_Grad = (self.w.N) * \
        #             (var.gradN * element.Jinv).transpose() * \
        #             element.Jacobian().det()

        # Grad = sp.integrate(diff_Grad, (e1, -1, 1)).tolist()

        jax_coors = jnp.array(element.getCoor())

        return _calc_gradient_term(self.w_shape, var.shape, element.shape, jax_coors)

    def div(self, var: scalarField, element: Element, const, Vel):

        """
        Define de element matrix from the weak form for the Divergence term.

        e. g.
        const*_u∇T -> Integral(const*Ni*dNj/dx*vel*det(J^-1)de1, -1, 1)

        :param element: Element
        :return: Matrix form of the Gradient term
        """

        # if self.Stab == 'PG':
        #     self.w.addStab(self.Stab, self.stabilization(element))


        # diff_Div = lambda *args: const * self.w.N_func(*args)* \
        #           (var.gradN_func(*args)*element.Jinv_func(*args)).T * Vel * element.Jdet_func(*args)


        # y, err = integrate.quad_vec(diff_Div, -1, 1)

        jax_coors = jnp.array(element.getCoor())

        return _calc_divergence_term(self.w_shape, var.shape, element.shape, jax_coors, const, Vel)

    def mass(self, var: scalarField, element: Element, constM):

        # diff_M = constM * (self.w.N) * (var.bf.N).transpose() * element.Jacobian().det()

        # Mass_Matrix = sp.integrate(diff_M, (e1, -1, 1)).tolist()

        jax_coors = jnp.array(element.getCoor())

        return _calc_mass_term(self.w_shape, var.shape, element.shape, jax_coors, constM)

    def forceVector(self, element: Element, Variable):

        if callable(self.source):

            f = self.source(element, Variable)
        else:

            f = self.source


        # diff_F = self.w.N * f * element.Jacobian()

        # F = sp.integrate(diff_F, (e1, -1, 1)).tolist()

        # return np.array(F).astype(np.float64).reshape((element.getNumberNodes(), 1))[0]  ## To fix!!!

        # diffF = lambda *args: self.w.N_func(*args) * np.array(f).astype(np.float64) * element.J_func(*args)

        # y, err = integrate.quad_vec(diffF, -1, 1)

        jax_coors = jnp.array(element.getCoor())

        return _calc_force_vector(self.w_shape, element.shape, jax_coors, f)

    def addBMatrix(self, element, Variable):

        B = jnp.zeros((element.getNumberNodes(), element.getNumberNodes()))

        for i, node in enumerate(element.nodes):

            if node.BC and node.BC[Variable]['type'] == 'Newton':

                if callable(node.BC[Variable]['h']):

                    B.at[i, i].set(node.BC[Variable]['h'](i))

                    #B[i, i] = node.BC[Variable]['h'](i)
                    # print('Radiaction BC used')
                else:
                    B.at[i, i].set(node.BC[Variable]['h'])

                    #B[i, i] = node.BC[Variable]['h']

        return B

    def addGVector(self, element, Variable):

        G = jnp.zeros((element.getNumberNodes(),1))

        for i, node in enumerate(element.nodes):

            if node.BC:

                if node.BC[Variable]['type'] == 'Newton':

                    if callable(node.BC[Variable]['h']):

                        G.at[i].set(- node.BC[Variable]['h'](i) * node.BC[Variable]['var_ext'])

                        # G[i] = - node.BC[Variable]['h'](i) * node.BC[Variable]['var_ext']
                        # print('Radiaction BC used')
                    else:
                        G.at[i].set(- node.BC[Variable]['h'] * node.BC[Variable]['var_ext'])

                        # G[i] = - node.BC[Variable]['h'] * node.BC[Variable]['var_ext']

                elif node.BC[Variable]['type'] == 'Neumann':

                    G.at[i].set(node.BC[Variable]['flux'])
                    # G[i] = node.BC[Variable]['flux']

        return G

    def stabilization(self, element: Element):

        Pe_h = self.Pe * element.getLength()
        alpha = (1 / math.tanh(Pe_h / 2)) - 2 / Pe_h
        
        #stab = alpha * element.getLength() / 2 * self.w.gradN \
        #    * element.Jinv

        stab= lambda *args: alpha * element.getLength() / 2 * self.w.gradN_func(*args) * element.Jinv_func(*args)

        return stab
    
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
