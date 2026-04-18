## Basis functions used for Finite element analysis

import sympy as sp
import numpy as np
#import jax.jnp as jnp

import jax
jax.config.update("jax_enable_x64", True)


import jax.numpy as jnp
from jax import jit


from Source.enums import ShapeFunctionType


e1, e2, e3 = sp.symbols('e1 e2 e3')

map_dim = [e1, e2, e3]

shape_mappings = {ShapeFunctionType.linear: [-1, 1],
                  ShapeFunctionType.quadratic: [-1, 0, 1]}

class basisFunctions:

    def __init__(self, mesh, shape:str | None):

        if mesh != None and shape != None:
            self.dim = mesh.PD
            self.set_basisFunctions(mesh, shape)

        else:
            self.N = sp.Matrix([])

        self.stabilized = False


    def set_basisFunctions(self, mesh, shape:str):

        """
        Set basis functions for the given mesh and shape.

        Parameters:
        -----------
        mesh (Mesh): Mesh object
        shape (str): Shape of the basis functions
        """
        
        """ 
        Previous implementation
        ---------------------------------------------   
        if shape == 'Linear':
            mapping = [n for n in range(-1, 2, 2)]

            self.constructBFVector(mesh, mapping) 
        ---------------------------------------------    
        """
        
        if shape in shape_mappings:
            self.constructBFVector(shape_mappings[shape])
            self.bfGrad(shape_mappings[shape])


    def constructBFVector(self, mapping):

        Nj = []

        """ for i in range(len(mapping)):
                j = 1
                for dim in range(mesh.PD):

                    j *= self.lagrangePoly(map_dim[dim], i, mapping)

                Nj.append(j) """

        for i in range(len(mapping)):
            j = 1
            for dim in range(self.dim):

                j *= self.lagrangePoly(map_dim[dim], i, mapping)

            Nj.append(j)

        self.N = sp.Matrix(Nj)

        #self.N_func = sp.lambdify(map_dim[:self.dim], self.N, 'numpy')
        self.N_func = lambda x: shape_functions_1d(mapping, x)

    def bfGrad(self, mapping):

        self.gradN = self.N.jacobian(sp.Matrix(list(self.N.free_symbols)))
        #self.gradN_func = sp.lambdify(map_dim[:self.dim], self.gradN, 'numpy')
        self.gradN_func = lambda x: shape_functions_gradient_1d(mapping, x)

        return self.gradN


    # Add stabilization function

    def addStab(self, type:str, stab):

        if not self.stabilized:
            if type == 'PG':
                #self.N_func += stab
                self.N_func_original = self.N_func
                self.N_func = None
                self.N_func = lambda x: self.N_func_original(x) + stab(x)
                #self.gradN_func = sp.lambdify(map_dim[:self.dim], self.gradN, 'numpy')
                self.stabilized = True

    def lagrangePoly(self, var: sp.core.symbol.Symbol, node: int, nodes: list):
        L = 1
        for i in range(len(nodes)):
            if i != node:
                L *= (var - nodes[i])/(nodes[node] - nodes[i])

        return L



@jit(static_argnums=(1,))
def lagrange_basis(nodes: list[float] | np.ndarray, i: int, x: float | np.ndarray) -> float | np.ndarray:
    """
    Calcula el i-ésimo polinomio base de Lagrange.
    nodes: lista o array con las coordenadas de los nodos del elemento.
    i: índice del nodo (0 a n).
    x: punto(s) donde se evalúa la función.
    """
    xi = nodes[i]
    basis = 1.0
    for j, xj in enumerate(nodes):
        if i != j:
            basis *= (x - xj) / (xi - xj)
    return basis

@jit(static_argnums=(0,))
def shape_functions_1d(shape:str, x: float | np.ndarray) -> jnp.ndarray:
    """
    Retorna un vector con todas las funciones de forma evaluadas en x.
    """
    nodes_array = jnp.array(shape_mappings[shape])

    n = len(nodes_array)
    N = jnp.array([lagrange_basis(nodes_array, i, x) for i in range(n)])
    return N.reshape(1, -1)  # Retornamos como matriz [1 x num_nodos] para mantener consistencia


@jit(static_argnums=(1,))
def lagrange_derivative(nodes: list[float] | np.ndarray, i: int, x: float | np.ndarray) -> float | np.ndarray:
    """Derivada del i-ésimo polinomio de Lagrange en el punto x."""
    xi = nodes[i]
    deriv = 0.0
    for j in range(len(nodes)):
        if i == j: continue
        
        # Calculamos el producto para el término j
        term = 1.0 / (xi - nodes[j])
        for k in range(len(nodes)):
            if k != i and k != j:
                term *= (x - nodes[k]) / (xi - nodes[k])
        deriv += term
    return deriv

@jit(static_argnums=(0,))
def shape_functions_gradient_1d(shape:str, xi: float | np.ndarray) -> jnp.ndarray:
    """Retorna dN/dxi para un elemento de línea."""

    nodes_array = jnp.array(shape_mappings[shape])

    # Simplemente llamamos a la derivada de Lagrange que ya definimos
    dN_dxi = jnp.array([lagrange_derivative(nodes_array, i, xi) for i in range(len(nodes_array))])
    
    # Retornamos como matriz [1 x num_nodos] para mantener consistencia
    return dN_dxi.reshape(1, -1)

@jit(static_argnums=(0,))
def mapping_function(shape:str, element_coor: list[float] | np.ndarray, xi: float | np.ndarray) -> jnp.ndarray:
    """Función de mapeo para un elemento de línea."""
  
    N = shape_functions_1d(shape, xi)  # Funciones de forma en el punto xi
    x = jnp.array(element_coor)  # Coordenadas de los nodos del elemento
    
    return jnp.dot(N, x)  # Mapeo a coordenadas físicas

@jit(static_argnums=(0,))
def jacobian_mapping_function(shape:str, element_coor: list[float] | np.ndarray, xi: float | np.ndarray) -> jnp.ndarray:
    """Calcula el Jacobiano de la función de mapeo."""
    
    dN_dxi = shape_functions_gradient_1d(shape, xi)  # Derivadas de las funciones de forma
    x = jnp.array(element_coor)  # Coordenadas de los nodos del elemento
    
    return jnp.dot(dN_dxi, x)  # Jacobiano del mapeo

@jit(static_argnums=(0,))
def jacobian_determinant(shape:str, element_coor: list[float] | np.ndarray, xi: float | np.ndarray) -> jnp.ndarray:
    """Calcula el determinante del Jacobiano."""

    J = jacobian_mapping_function(shape, element_coor, xi)

    if J.shape == (1,):
        return jnp.abs(J)  # En 1D, el Jacobiano es un escalar, así que tomamos su valor absoluto

    return jnp.linalg.det(J)  # En dimensiones superiores, calculamos el determinante normalmente

@jit(static_argnums=(0,))
def jacobian_inverse(shape:str, element_coor: list[float] | np.ndarray, xi: float | np.ndarray) -> jnp.ndarray:
    """Calcula la inversa del Jacobiano."""

    J = jacobian_mapping_function(shape, element_coor, xi)

    if J.shape == (1,):
        return 1.0 / J  # En 1D, la inversa del Jacobiano es simplemente el recíproco

    return jnp.linalg.inv(J)  # En dimensiones superiores, calculamos la inversa normalmente