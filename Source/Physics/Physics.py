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

from typing import Callable

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

@jit(static_argnums=(0,1,2,3,6))
def _calc_laplacian_term(w_shape:str, 
                        var_shape:str, 
                        element_shape:str,
                        const: Callable[[jnp.ndarray], float], 
                        element_coors: list[float] | np.ndarray,
                        state_at_x: dict,
                        target_variable: str):
        
        """Calculate the Laplacian term for the given element and variable.
        This function is JIT-compiled using JAX for improved performance.
        
        e. g.
        ∇.(const * ∇T)  -> Integral(dNi/dx^Trans*const*dNj/dx*det(J^-1)de1, -1, 1)

        Parameters:
        w_shape (str): Shape of the basis functions for the test function
        var_shape (str): Shape of the basis functions for the variable
        element_shape (str): Shape of the basis functions for the element   
        const (Callable[[jnp.ndarray], float]): A function that takes the variable values and returns the constant value for the Laplacian term
        element_coors (list[float] | np.ndarray): Coordinates of the element nodes
        var_values (float | np.ndarray): Values of the variable at the element nodes
        
        Returns:
        jnp.ndarray: The Laplacian term matrix for the element
        """

        #_calc_force_vector(self.w_shape, element.shape, f_func, jax_coors, state_at_x, Variable)
            
        puntos_gauss, pesos_gauss = get_gauss_points_weights(2)

        y = jnp.zeros((len(element_coors), len(element_coors)))  # Assuming square matrix for simplicity, adjust as needed

        #const_func = const(var_values)

        for p, w in zip(puntos_gauss, pesos_gauss):

            const_func = const(element_shape, p, state_at_x, target_variable)

            B_w = shape_functions_gradient_1d(w_shape, p) * jacobian_inverse(element_shape, element_coors, p)     
            B_var = shape_functions_gradient_1d(var_shape, p) * jacobian_inverse(element_shape, element_coors, p)
            detJ = jacobian_determinant(element_shape, element_coors, p)

            y += const_func * jnp.dot(B_w.T, B_var) * detJ * w
        
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

@jit(static_argnums=(0,1,2,3,6))
def _calc_divergence_term(w_shape:str, 
                        var_shape:str, 
                        element_shape:str, 
                        const: Callable[[jnp.ndarray], float], 
                        element_coors: list[float] | np.ndarray,
                        state_at_x: dict,
                        target_variable: str,
                        Vel: float | np.ndarray):
        
        """
        Calculate the Divergence term for the given element and variable.
        This function is JIT-compiled using JAX for improved performance.

        e. g.
        const*_u∇T -> Integral(const*Ni*dNj/dx*vel*det(J^-1)de1, -1, 1)
        Parameters:
        w_shape (str): Shape of the basis functions for the test function
        var_shape (str): Shape of the basis functions for the variable
        element_shape (str): Shape of the basis functions for the element
        const (Callable[[jnp.ndarray], float]): A function that takes the variable values and returns the constant value for the Divergence term
        element_coors (list[float] | np.ndarray): Coordinates of the element nodes
        var_values (float | np.ndarray): Values of the variable at the element nodes
        Vel (float | np.ndarray): Velocity value for the convection term

        Returns:
        jnp.ndarray: The Divergence term matrix for the element
        """        
    
        puntos_gauss, pesos_gauss = get_gauss_points_weights(2)

        y = jnp.zeros((len(element_coors), len(element_coors)))  # Assuming square matrix for simplicity, adjust as needed

        #const_func = const(var_values)

        for p, w in zip(puntos_gauss, pesos_gauss):

            const_func = const(element_shape, p, state_at_x, target_variable)

            N_w = shape_functions_1d(w_shape, p)
            B_var = shape_functions_gradient_1d(var_shape, p) * jacobian_inverse(element_shape, element_coors, p)
            detJ = jacobian_determinant(element_shape, element_coors, p)

            y += const_func * jnp.dot(N_w.T, B_var) * Vel * detJ * w 
        
        return y

@jit(static_argnums=(0,1,2,3,6))
def _calc_mass_term(w_shape:str, 
                    var_shape:str, 
                    element_shape:str,
                    constM: Callable[[jnp.ndarray], float], 
                    element_coors: list[float] | np.ndarray,
                    state_at_x: dict,
                    target_variable: str,):
        
        """ Calculate the Mass term for the given element and variable.
        This function is JIT-compiled using JAX for improved performance.
        
        e. g.
        
        constM*Ni*N_j -> Integral(constM*Ni*N_j*det(J^-1)de1, -1, 1)
        
        Parameters:
        w_shape (str): Shape of the basis functions for the test function
        var_shape (str): Shape of the basis functions for the variable
        element_shape (str): Shape of the basis functions for the element
        constM (Callable[[jnp.ndarray], float]): A function that takes the variable values and returns the constant value for the Mass term
        element_coors (list[float] | np.ndarray): Coordinates of the element nodes
        var_values (float | np.ndarray): Values of the variable at the element nodes

        Returns:
        jnp.ndarray: The Mass term matrix for the element
        
        """

        puntos_gauss, pesos_gauss = get_gauss_points_weights(2)
        
        y = jnp.zeros((len(element_coors), len(element_coors)))  # Assuming square matrix for simplicity, adjust as needed

        #constM_func = constM(var_values)

        for p, w in zip(puntos_gauss, pesos_gauss):

            constM_func = constM(element_shape, p, state_at_x, target_variable)

            N_w = shape_functions_1d(w_shape, p)
            B_var = shape_functions_gradient_1d(var_shape, p)   
            detJ = jacobian_determinant(element_shape, element_coors, p)
            y += constM_func * jnp.dot(N_w.T, B_var) * detJ * w
        return y

@jit(static_argnums=(0,1,2,5))
def _calc_force_vector(w_shape:str, 
                       element_shape:str,
                       f: Callable[[jnp.ndarray], float],
                       element_coors: list[float] | np.ndarray,
                       #var_values: float | np.ndarray,
                       state_at_x: dict[str, np.ndarray],
                       target_variable: str):
    
    """Calculate the Force vector for the given element and variable.
    This function is JIT-compiled using JAX for improved performance.
    
    e. g.
    f*Ni -> Integral(f*Ni*det(J^-1)de1, -1, 1)
    
    Parameters:
    w_shape (str): Shape of the basis functions for the test function
    element_shape (str): Shape of the basis functions for the element
    f (Callable[[jnp.ndarray], float]): A function that takes the variable values and returns the value of the source term
    element_coors (list[float] | np.ndarray): Coordinates of the element nodes
    var_values (float | np.ndarray): Values of the variable at the element nodes

    Returns:
    jnp.ndarray: The Force vector for the element
    
    """

    puntos_gauss, pesos_gauss = get_gauss_points_weights(2)

    y = jnp.zeros((len(element_coors),1))  # Assuming vector for simplicity, adjust as needed

    #const_func = f(var_values)

    for p, w in zip(puntos_gauss, pesos_gauss):
        N_w = shape_functions_1d(w_shape, p)
        detJ = jacobian_determinant(element_shape, element_coors, p)

        const_func = f(element_shape, p, state_at_x, target_variable)

        y += N_w.T * const_func * detJ * w
    
    return y.flatten()

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

@jit(static_argnums=(0,1,3))
def _calc_b_matrix(w_shape:str, 
                   element_shape:str,
                   id: list[int],
                   h_c: Callable[[jnp.ndarray], float],
                   element_coors: list[float] | np.ndarray, # Ojala recibir un único lambda que ya tenga los otros valores y solo depende del valor de la variable en el nodo, para evitar tener que pasar muchos parámetros a esta función
                   var_values: float | np.ndarray):
    
    # To fix: This function is currently only for 1D and for a single node with Newton BC. 
    # Next updates will show the code for multiple dimensions and for multiple nodes with Newton BC.

    #puntos_gauss, pesos_gauss = get_gauss_points_weights(2)

    # For simplicity, we will assume that the B matrix is only affected by the boundary condition at the node with the given id. 
    # In a real implementation, this would need to be integrated over the boundary of the element.

    y = jnp.zeros((len(element_coors), len(element_coors)))  # Assuming square matrix for simplicity, adjust as needed

    id = jnp.atleast_1d(jnp.array(id, dtype=int))

    if id.size > 0:
        for i in id:

            h_c_calc = h_c(var_values[i])

            y = y.at[i, i].set(h_c_calc)
    
    return y

@jit(static_argnums=(0,1,3))
def _calc_g_vector(w_shape:str,
                   element_shape:str,
                   id: list[int],
                   g_c: Callable[[jnp.ndarray], float],
                   element_coors: list[float] | np.ndarray, # Ojala recibir un único lambda que ya tenga los otros valores y solo depende del valor de la variable en el nodo, para evitar tener que pasar muchos parámetros a esta función
                   var_values: float | np.ndarray):
    

    y = jnp.zeros((len(element_coors),1))  # Assuming vector for simplicity, adjust as needed
    id = jnp.atleast_1d(jnp.array(id, dtype=int))


    if id.size > 0:
        for i in id:

            g_c_calc = g_c(var_values[i])

            y = y.at[i, 0].set(g_c_calc)  # Assuming var_values is an array with the variable values at the element nodes
    
    return y.flatten()
                   

@jit(static_argnums=(0,1,2,3,4,5,6,7,10))
def _calc_element_residual_vector(w_shape:str, 
                        var_shape:str, 
                        element_shape:str,
                        const_C: Callable[[jnp.ndarray], float],
                        const_K: Callable[[jnp.ndarray], float],
                        const_B: Callable[[jnp.ndarray], float],
                        const_F: Callable[[jnp.ndarray], float],
                        const_G: Callable[[jnp.ndarray], float],
                        ids_for_B: list[int],
                        ids_for_G: list[int],
                        target_var: str,
                        element_coors: list[float] | np.ndarray,
                        state_dict: dict,
                        Vel: float | np.ndarray):
    

    var_values = state_dict[target_var]

    #M = _calc_mass_term(w_shape, var_shape, element_shape, const_C, element_coors, var_values) # To fix: This is currently not used, but it could be used in a transient problem. Next updates will show how to use it in a transient problem.
    C = _calc_divergence_term(w_shape, var_shape, element_shape, const_C, element_coors, state_dict, target_var, Vel)
    K = _calc_laplacian_term(w_shape, var_shape, element_shape, const_K, element_coors, state_dict, target_var)
    B = _calc_b_matrix(w_shape, element_shape, ids_for_B, const_B, element_coors, var_values)  # To fix: This is currently only for a single node with Newton BC. Next updates will show the code for multiple nodes with Newton BC.
    
    #_calc_force_vector(self.w_shape, element.shape, f_func, jax_coors, state_at_x, Variable)

    F = _calc_force_vector(w_shape, element_shape, const_F, element_coors, state_dict, target_var)
    G = _calc_g_vector(w_shape, element_shape, ids_for_G, const_G, element_coors, var_values) # To fix: This is currently only for a single node with Newton BC. Next updates will show the code for multiple nodes with Newton BC.

    A_e = C + K + B
    b_e = (F - G).flatten()
    
    R = jnp.dot(A_e, var_values) - b_e  

    return R


_calc_element_tangent_matrix = jax.jit(jax.jacfwd(_calc_element_residual_vector, argnums=12), static_argnums= (0,1,2,3,4,5,6,7,10))
                       
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

        self.C_const = 0.0
        self.K_const = 0.0
        self.B_const = 0.0
        self.F_const = 0.0
        self.G_const = 0.0

        self.vel = 1.0  # Velocity

        self.Pe = 0  # Peclet number

    def func_normalization(self, term):

        if isinstance(term, (float, int)):
            h_c = lambda *args: float(term)

        elif callable(term):
            h_c = term

        else:
            raise TypeError("La propiedad debe ser un float o una función.")

        return h_c

    def getVariables(self):
        return self.var.keys()

    def getNumOfVar(self):
        return len(self.var.keys())

    def normalize_constants(self):
        self.C_const = self.func_normalization(self.C_const)
        self.K_const = self.func_normalization(self.K_const)
        self.B_const = self.func_normalization(self.B_const)
        self.F_const = self.func_normalization(self.F_const)
        self.G_const = self.func_normalization(self.G_const)

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


        #x_e = self.var[Variable].getElementValues(element)

        self.F_const = self.source
        self.normalize_constants()

        jax_coors = jnp.array(element.getCoor())

        ids_for_B, h_c_for_B = self.check_boundary_conditions_B(element, Variable)
        ids_for_G, g_c_for_G = self.check_boundary_conditions_G(element, Variable)

        state_at_x = {variable: self.var[variable].getElementValues(element) for variable in self.var.keys()}

        R_e = _calc_element_residual_vector(self.w_shape, self.var[Variable].shape, element.shape,
                        self.C_const, self.K_const, h_c_for_B, self.F_const, g_c_for_G, ids_for_B, ids_for_G,
                        Variable, jax_coors, state_at_x, self.vel)

        return np.asarray(R_e)
    
    def get_variable_element_values(self, element, Variable):

        return self.var[Variable].getElementValues(element)
    
    def aux_getTangentMatrix(self, element, Variable, x_values, solverOptions=None):

        x_e_orig = self.var[Variable].getElementValues(element)

        self.var[Variable].setElementValues(element, x_values)

        A_e = self.getElementMatrix(element, Variable, solverOptions)
        b_e = self.getElementVector(element, Variable, solverOptions)

        F_e = A_e.dot(x_values) - b_e

        self.var[Variable].setElementValues(element, x_e_orig)

        return F_e

    def getElementTangentMatrix(self, element, Variable, solverOptions=None):
        
        self.F_const = self.source
        self.normalize_constants()

        jax_coors = jnp.array(element.getCoor())

        ids_for_B, h_c_for_B = self.check_boundary_conditions_B(element, Variable)
        ids_for_G, g_c_for_G = self.check_boundary_conditions_G(element, Variable)

        state_at_x = {variable: self.var[variable].getElementValues(element) for variable in self.var.keys()}

        K_e = _calc_element_tangent_matrix(self.w_shape, self.var[Variable].shape, element.shape,
                        self.C_const, self.K_const, h_c_for_B, self.F_const, g_c_for_G, ids_for_B, ids_for_G,
                        Variable, jax_coors, state_at_x, self.vel)  # To fix: This is currently only for a single node with Newton BC and no convection. Next updates will show the code for multiple nodes with Newton BC and for convection.

        return K_e

    def laplacian(self, const: float, var: scalarField, element: Element):
        """
        Define the element matrix from the weak form for the Laplacian term.

        Note: Stills for 1D. Next updates will show the code for multiple dimensions

        :param var:
        :param const: material proporcionality constant
        :param element: Element
        :return: Matrix of laplacian term
        """

        jax_coors = jnp.array(element.getCoor())

        const_func = self.func_normalization(const)

        #var_values = self.var[var.name].getElementValues(element)

        state_at_x = {variable: self.var[variable].getElementValues(element) for variable in self.var.keys()}

        return _calc_laplacian_term(self.w_shape, var.shape, element.shape, const_func, jax_coors, state_at_x, var.name)

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

        const_func = self.func_normalization(const)

        state_at_x = {variable: self.var[variable].getElementValues(element) for variable in self.var.keys()}

        #var_values = self.var[var.name].getElementValues(element)

        return _calc_divergence_term(self.w_shape, var.shape, element.shape, const_func, jax_coors, state_at_x, var.name, Vel)

    def mass(self, var: scalarField, element: Element, constM):

        # diff_M = constM * (self.w.N) * (var.bf.N).transpose() * element.Jacobian().det()

        # Mass_Matrix = sp.integrate(diff_M, (e1, -1, 1)).tolist()

        jax_coors = jnp.array(element.getCoor())

        constM_func = self.func_normalization(constM)

        var_values = self.var[var.name].getElementValues(element)

        return _calc_mass_term(self.w_shape, var.shape, element.shape, constM_func, jax_coors, var_values)

    def forceVector(self, element: Element, Variable):


        f = self.source

        jax_coors = jnp.array(element.getCoor())

        f_func = self.func_normalization(self.source)

        var_values = self.var[Variable].getElementValues(element)

        state_at_x = {variable: self.var[variable].getElementValues(element) for variable in self.var.keys()}
       

        return _calc_force_vector(self.w_shape, element.shape, f_func, jax_coors, state_at_x, Variable)

    def addBMatrix(self, element, Variable):

        B = jnp.zeros((element.getNumberNodes(), element.getNumberNodes()))

        jax_coors = jnp.array(element.getCoor())

        ids = []
        h_c_funcs = []

        h_c = lambda n: 0.0

        var_values = self.var[Variable].getElementValues(element)

        for i, node in enumerate(element.nodes):

            #var_value = self.var[Variable].getElementValues(element)[i]

            if node.BC and node.BC[Variable]['type'] == 'Newton':

                h_c = self.func_normalization(node.BC[Variable]['h'])

                ids.append(i)
                h_c_funcs.append(h_c)

        B += _calc_b_matrix(self.w_shape, element.shape, ids, h_c, jax_coors, var_values)


        return B

    def addGVector(self, element, Variable):

        G = jnp.zeros((element.getNumberNodes()))

        jax_coors = jnp.array(element.getCoor())

        ids = []

        g_c_funcs = []

        var_values = self.var[Variable].getElementValues(element)

        g_c = lambda *args: 0.0  # Default value for g_c, can be overwritten if there are Newton BCs

        for i, node in enumerate(element.nodes):

            #var_value = self.var[Variable].getElementValues(element)[i]

            if node.BC and node.BC[Variable]['type'] == 'Newton':

                h_c = self.func_normalization(node.BC[Variable]['h'])

                if callable(node.BC[Variable]['h']):

                    g_c = lambda *args: -h_c(*args) * node.BC[Variable]['var_ext']

                    g_c = self.func_normalization(g_c)

                    ids.append(i)
                    g_c_funcs.append(g_c)


                    #G.at[i].set(- node.BC[Variable]['h'](i) * node.BC[Variable]['var_ext'])

                    # G[i] = - node.BC[Variable]['h'](i) * node.BC[Variable]['var_ext']
                    # print('Radiaction BC used')
                else:
                    g_c = -node.BC[Variable]['h'] * node.BC[Variable]['var_ext']

                    g_c = self.func_normalization(g_c)

                    ids.append(i)
                    g_c_funcs.append(g_c)
                    #G.at[i].set(g_c(i))

                    # G[i] = - node.BC[Variable]['h'] * node.BC[Variable]['var_ext']

                #g_c = self.func_normalization(g_c)

                #G += _calc_g_vector(self.w_shape, element.shape, i, g_c, jax_coors, var_value)

            elif node.BC and node.BC[Variable]['type'] == 'Neumann':

                g_c = lambda *args: node.BC[Variable]['flux']

                g_c = self.func_normalization(g_c)

                ids.append(i)
                g_c_funcs.append(g_c)
                # G[i] = node.BC[Variable]['flux']

                #g_c = self.func_normalization(g_c)

                #G += _calc_g_vector(self.w_shape, element.shape, i, g_c, jax_coors, var_value)

        #g_tup = tuple(g_c_funcs)

        G += _calc_g_vector(self.w_shape, element.shape, ids, g_c, jax_coors, var_values)

        return G

    def check_boundary_conditions_B(self, element, Variable):

        ids = []
        h_c_funcs = []

        h_c = lambda n: 0.0  # Default value for h_c, can be overwritten if there are Newton BCs


        for i, node in enumerate(element.nodes):

            if node.BC and node.BC[Variable]['type'] == 'Newton':

                h_c = self.func_normalization(node.BC[Variable]['h'])

                ids.append(i)
                h_c_funcs.append(h_c)

        return ids, h_c
    
    def check_boundary_conditions_G(self, element, Variable):
        ids = []
        g_c_funcs = []

        g_c = lambda n: 0.0  # Default value for g_c, can be overwritten if there are Newton BCs
        for i, node in enumerate(element.nodes):

            if node.BC and node.BC[Variable]['type'] == 'Newton':

                if callable(node.BC[Variable]['h']):

                    g_c = lambda n: -node.BC[Variable]['h'](n) * node.BC[Variable]['var_ext']

                    g_c = self.func_normalization(g_c)

                    ids.append(i)
                    g_c_funcs.append(g_c)

                else:
                    g_c = -node.BC[Variable]['h'] * node.BC[Variable]['var_ext']

                    g_c = self.func_normalization(g_c)

                    ids.append(i)
                    g_c_funcs.append(g_c)

            elif node.BC and node.BC[Variable]['type'] == 'Neumann':

                g_c = lambda n: node.BC[Variable]['flux']

                g_c = self.func_normalization(g_c)

                ids.append(i)
                g_c_funcs.append(g_c)

        return ids, g_c # To fix: This is currently only for a single node with Newton BC. Next updates will show the code for multiple nodes with Newton BC.

    
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
