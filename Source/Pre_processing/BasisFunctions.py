## Basis functions used for Finite element analysis

import sympy as sp
# from Source.Pre_processing.Mesh import Mesh

e1, e2, e3 = sp.symbols('e1 e2 e3')


class basisFunctions:

    def __init__(self):
        pass

    @staticmethod
    def get_basisFunctions(mesh, shape:str):
        if mesh.meshType == "1DROD2P" and shape == 'Linear':
            N = sp.Matrix([(1 - e1) / 2, (1 + e1) / 2])
        return N

