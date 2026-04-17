## Basis functions used for Finite element analysis

import sympy as sp
import numpy as np
#import jax.jnp as jnp

from numba import jit, prange, int64, float64
# from Source.Pre_processing.Mesh import Mesh

e1, e2, e3 = sp.symbols('e1 e2 e3')

map_dim = [e1, e2, e3]

shape_mappings = {'Linear': [-1, 1],
                  'Quadratic': [-1, 0, 1]}

class BasisFunctions:

    def __init__(self, mesh, shape:str | None):

        if mesh != None and shape != None:
            self.dim = mesh.PD
            self.set_basis_functions(shape)

        else:
            self.N = sp.Matrix([])

        self.stabilized = False


    def set_basis_functions(self, shape:str):

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
            self.construct_BF_vector(shape_mappings[shape])
            self.calculate_BF_gradient(shape_mappings[shape])


    def construct_BF_vector(self, mapping):

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

    def calculate_BF_gradient(self, mapping):

        self.gradN = self.N.jacobian(sp.Matrix(list(self.N.free_symbols)))
        #self.gradN_func = sp.lambdify(map_dim[:self.dim], self.gradN, 'numpy')
        self.gradN_func = lambda x: shape_functions_gradient_1d(mapping, x)

        return self.gradN


    # Add stabilization function

    def add_stabilization(self, type:str, stab):

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


"""     
@staticmethod
    def get_basisFunctions(mesh, shape:str):
        if mesh.meshType == "1DROD2P" and shape == 'Linear':
            N = sp.Matrix([(1 - e1) / 2, (1 + e1) / 2])
        return N

 """

@jit(cache=True)
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


def shape_functions_1d(nodes: list[float] | np.ndarray, x: float | np.ndarray) -> np.ndarray:
    """
    Retorna un vector con todas las funciones de forma evaluadas en x.
    """
    n = len(nodes)
    N = np.array([lagrange_basis(nodes, i, x) for i in range(n)])
    return N


@jit(cache=True)
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

def shape_functions_gradient_1d(nodes_xi: list[float] | np.ndarray, xi: float | np.ndarray) -> np.ndarray:
    """Retorna dN/dxi para un elemento de línea."""
    # Simplemente llamamos a la derivada de Lagrange que ya definimos
    dN_dxi = np.array([lagrange_derivative(nodes_xi, i, xi) for i in range(len(nodes_xi))])
    
    # Retornamos como matriz [1 x num_nodos] para mantener consistencia
    return dN_dxi.reshape(1, -1)