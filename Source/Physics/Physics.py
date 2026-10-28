# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++++

# Class physics: Contains all the functions necessary to create the conservative relations


import numpy as np

from typing import Callable

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit

from Source.Pre_processing.BasisFunctions import basisFunctions, shape_functions_1d, shape_functions_gradient_1d, jacobian_determinant, jacobian_inverse

from Source.Pre_processing.Mesh import Element
from Source.Primals.Scalar import scalarField

u = 1

_zero_const = lambda *args: 0.0

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

            
        puntos_gauss, pesos_gauss = get_gauss_points_weights(2)

        y = jnp.zeros((len(element_coors), len(element_coors)))  # Assuming square matrix for simplicity, adjust as needed


        for p, w in zip(puntos_gauss, pesos_gauss):

            const_func = const(element_shape, p, state_at_x, target_variable)

            B_w = shape_functions_gradient_1d(w_shape, p) * jacobian_inverse(element_shape, element_coors, p)     
            B_var = shape_functions_gradient_1d(var_shape, p) * jacobian_inverse(element_shape, element_coors, p)
            detJ = jacobian_determinant(element_shape, element_coors, p)

            y += const_func * jnp.dot(B_w.T, B_var) * detJ * w
        
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

        puntos_gauss, pesos_gauss = get_gauss_points_weights(2) #fix
        
        y = jnp.zeros((len(element_coors), len(element_coors)))  # Assuming square matrix for simplicity, adjust as needed


        for p, w in zip(puntos_gauss, pesos_gauss):

            constM_func = constM(element_shape, p, state_at_x, target_variable)

            N_w = shape_functions_1d(w_shape, p)
            N_var = shape_functions_1d(var_shape, p)   
            detJ = jacobian_determinant(element_shape, element_coors, p)
            y += constM_func * jnp.dot(N_w.T, N_var) * detJ * w
        return y

@jit(static_argnums=(0,1,2,5))
def _calc_force_vector(w_shape:str, 
                       element_shape:str,
                       f: Callable[[jnp.ndarray], float],
                       element_coors: list[float] | np.ndarray,
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


    for p, w in zip(puntos_gauss, pesos_gauss):
        N_w = shape_functions_1d(w_shape, p)
        detJ = jacobian_determinant(element_shape, element_coors, p)

        const_func = f(element_shape, p, state_at_x, target_variable)

        y += N_w.T * const_func * detJ * w
    
    return y.flatten()


@jit(static_argnums=(0,1,3))
def _calc_b_matrix(w_shape:str, 
                   element_shape:str,
                   id: list[int],
                   h_c: Callable[[jnp.ndarray], float],
                   element_coors: list[float] | np.ndarray, # Ojala recibir un único lambda que ya tenga los otros valores y solo depende del valor de la variable en el nodo, para evitar tener que pasar muchos parámetros a esta función
                   var_values: float | np.ndarray):
    
    # To fix: This function is currently only for 1D and for a single node with Newton BC. 
    # Next updates will show the code for multiple dimensions and for multiple nodes with Newton BC.


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

    C = _calc_divergence_term(w_shape, var_shape, element_shape, const_C, element_coors, state_dict, target_var, Vel)
    K = _calc_laplacian_term(w_shape, var_shape, element_shape, const_K, element_coors, state_dict, target_var)
    B = _calc_b_matrix(w_shape, element_shape, ids_for_B, const_B, element_coors, var_values)  # To fix: This is currently only for a single node with Newton BC. Next updates will show the code for multiple nodes with Newton BC.
    

    F = _calc_force_vector(w_shape, element_shape, const_F, element_coors, state_dict, target_var)
    G = _calc_g_vector(w_shape, element_shape, ids_for_G, const_G, element_coors, var_values) # To fix: This is currently only for a single node with Newton BC. Next updates will show the code for multiple nodes with Newton BC.

    A_e = C + K + B
    b_e = (F - G).flatten()
    
    R = jnp.dot(A_e, var_values) - b_e  

    return R

# Tangent matrix calculation
_calc_element_tangent_matrix = jax.jit(jax.jacfwd(_calc_element_residual_vector, argnums=12), static_argnums= (0,1,2,3,4,5,6,7,10))
                       


class physics:

    # Set to True by physics whose system can be assembled with Source.Simulation.VectorizedAssembler
    supports_vectorized_assembly = False

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

        self.edge_bcs = []  # Newton/Neumann conditions on edges (2D): dicts with 'type', 'edges', ...


    
    


    def func_normalization(self, term):

        if isinstance(term, (float, int)):
            h_c = lambda *args: float(term)

        elif callable(term):
            h_c = term

        else:
            raise TypeError("La propiedad debe ser un float o una función.")

        return h_c

    def _C_term(self):

        """Convection coefficient for the element-based residual: zero when Convection is off.

        A module-level zero function is used so the jitted kernels are not recompiled.
        """

        return self.C_const if self.Convection else _zero_const

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
    
    def initializeMatrices(self, element, Variable):

        if self.Convection:
            self.C = self.div(self.var[Variable], element, self.C_const, u)  # 1 stands for velocity (u = 1)

        self.K = self.laplacian(self.K_const, self.var[Variable], element)

        self.B = self.addBMatrix(element, Variable)


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


        self.F_const = self.source
        self.normalize_constants()

        jax_coors = jnp.array(element.getCoor())

        ids_for_B, h_c_for_B = self.check_boundary_conditions_B(element, Variable)
        ids_for_G, g_c_for_G = self.check_boundary_conditions_G(element, Variable)

        state_at_x = {variable: self.var[variable].getElementValues(element) for variable in self.var.keys()}

        R_e = _calc_element_residual_vector(self.w_shape, self.var[Variable].shape, element.shape,
                        self._C_term(), self.K_const, h_c_for_B, self.F_const, g_c_for_G, ids_for_B, ids_for_G,
                        Variable, jax_coors, state_at_x, self.vel)

        return np.asarray(R_e)
    
    def get_variable_element_values(self, element, Variable):

        return self.var[Variable].getElementValues(element)
    

    def getElementTangentMatrix(self, element, Variable, solverOptions=None):
        
        self.F_const = self.source
        self.normalize_constants()

        jax_coors = jnp.array(element.getCoor())

        ids_for_B, h_c_for_B = self.check_boundary_conditions_B(element, Variable)
        ids_for_G, g_c_for_G = self.check_boundary_conditions_G(element, Variable)

        state_at_x = {variable: self.var[variable].getElementValues(element) for variable in self.var.keys()}

        K_e = _calc_element_tangent_matrix(self.w_shape, self.var[Variable].shape, element.shape,
                        self._C_term(), self.K_const, h_c_for_B, self.F_const, g_c_for_G, ids_for_B, ids_for_G,
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


        state_at_x = {variable: self.var[variable].getElementValues(element) for variable in self.var.keys()}

        return _calc_laplacian_term(self.w_shape, var.shape, element.shape, const_func, jax_coors, state_at_x, var.name)


    def div(self, var: scalarField, element: Element, const, Vel):

        """
        Define de element matrix from the weak form for the Divergence term.

        e. g.
        const*_u∇T -> Integral(const*Ni*dNj/dx*vel*det(J^-1)de1, -1, 1)

        :param element: Element
        :return: Matrix form of the Gradient term
        """


        jax_coors = jnp.array(element.getCoor())

        const_func = self.func_normalization(const)

        state_at_x = {variable: self.var[variable].getElementValues(element) for variable in self.var.keys()}


        return _calc_divergence_term(self.w_shape, var.shape, element.shape, const_func, jax_coors, state_at_x, var.name, Vel)

    def mass(self, var: scalarField, element: Element, constM):


        jax_coors = jnp.array(element.getCoor())

        constM_func = self.func_normalization(constM)

        var_values = self.var[var.name].getElementValues(element)

        state_at_x = {variable: self.var[variable].getElementValues(element) for variable in self.var.keys()}

        return _calc_mass_term(self.w_shape, var.shape, element.shape, constM_func, jax_coors, state_at_x, var.name)


    def forceVector(self, element: Element, Variable):


        f = self.source

        jax_coors = jnp.array(element.getCoor())

        f_func = self.func_normalization(self.source)

        var_values = self.var[Variable].getElementValues(element)

        state_at_x = {variable: self.var[variable].getElementValues(element) for variable in self.var.keys()}
       

        return _calc_force_vector(self.w_shape, element.shape, f_func, jax_coors, state_at_x, Variable)


    def _node_bc(self, node, Variable):

        """Returns the boundary-condition dict of `node` for `Variable`, or None if it has none."""

        if not node.BC:
            return None

        return node.BC.get(Variable)

    def _flagged_nodes(self, element, Variable):

        """Returns [(local_index, bc)] for the Newton and Neumann nodes of an element.

        The batched/jitted kernels apply one coefficient function to every flagged node of
        an element, so an element can carry at most one such node for now.
        """

        found = []

        for i, node in enumerate(element.nodes):

            bc = self._node_bc(node, Variable)

            if bc and bc['type'] in ('Newton', 'Neumann'):
                found.append((i, bc))

        if len(found) > 1:
            raise NotImplementedError("Un elemento con mas de un nodo Newton/Neumann aun no esta soportado.")

        return found

    @staticmethod
    def _newton_rhs(bc):

        """Right-hand-side term -h(T)*T_ext of a Newton BC, with its own closure (no late binding)."""

        h, var_ext = bc['h'], bc['var_ext']

        if callable(h):
            return lambda n: -h(n) * var_ext

        return lambda n: -h * var_ext

    def addBMatrix(self, element, Variable):

        jax_coors = jnp.array(element.getCoor())

        var_values = self.var[Variable].getElementValues(element)

        ids, h_c = self.check_boundary_conditions_B(element, Variable)

        return _calc_b_matrix(self.w_shape, element.shape, ids, h_c, jax_coors, var_values)

    def addGVector(self, element, Variable):

        jax_coors = jnp.array(element.getCoor())

        var_values = self.var[Variable].getElementValues(element)

        ids, g_c = self.check_boundary_conditions_G(element, Variable)

        return _calc_g_vector(self.w_shape, element.shape, ids, g_c, jax_coors, var_values)

    def check_boundary_conditions_B(self, element, Variable):

        """Newton nodes of the element and their coefficient h(T)."""

        ids = []
        h_c = lambda n: 0.0  # No Newton BC on this element

        for i, bc in self._flagged_nodes(element, Variable):

            if bc['type'] == 'Newton':
                ids.append(i)
                h_c = self.func_normalization(bc['h'])

        return ids, h_c

    def check_boundary_conditions_G(self, element, Variable):

        """Newton and Neumann nodes of the element and their right-hand-side term g(T)."""

        ids = []
        g_c = lambda n: 0.0  # No flux BC on this element

        for i, bc in self._flagged_nodes(element, Variable):

            ids.append(i)

            if bc['type'] == 'Newton':
                g_c = self.func_normalization(self._newton_rhs(bc))
            else:
                flux = bc['flux']
                g_c = self.func_normalization(lambda n, flux=flux: flux)

        return ids, g_c


    
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
