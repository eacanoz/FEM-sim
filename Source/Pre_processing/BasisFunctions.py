## Basis functions used for Finite element analysis

import sympy as sp
# from Source.Pre_processing.Mesh import Mesh

e1, e2, e3 = sp.symbols('e1 e2 e3')

map_dim = [e1, e2, e3]

class basisFunctions:

    def __init__(self, mesh, shape:str | None):

        if mesh != None and shape != None:
            self.set_basisFunctions(mesh, shape)

        else:
            self.N = sp.Matrix([])

        self.stabilized = False


    def set_basisFunctions(self, mesh, shape:str):

        if shape == 'Linear':
            mapping = [n for n in range(-1, 2, 2)]

            self.constructBFVector(mesh, mapping)


    def constructBFVector(self, mesh, mapping):

        Nj = []

        for i in range(len(mapping)):
                j = 1
                for dim in range(mesh.PD):

                    j *= self.lagrangePoly(map_dim[dim], i, mapping)

                Nj.append(j)

        self.N = sp.Matrix(Nj)

    def bfGrad(self):
        return self.N.jacobian(sp.Matrix(list(self.N.free_symbols)))


    # Add stabilization function

    def addStab(self, type:str, stab):

        if not self.stabilized:
            if type == 'PG':
                self.N += stab
                self.stabilized = True

    def lagrangePoly(self, var: sp.core.symbol.Symbol, node: int, nodes: list):
        L = 1
        for i in range(len(nodes)):
            if i != node:
                L *= (var - nodes[i])/(nodes[node] - nodes[i])

        return L


    @staticmethod
    def get_basisFunctions(mesh, shape:str):
        if mesh.meshType == "1DROD2P" and shape == 'Linear':
            N = sp.Matrix([(1 - e1) / 2, (1 + e1) / 2])
        return N

