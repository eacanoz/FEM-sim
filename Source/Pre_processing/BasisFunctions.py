## Basis functions used for Finite element analysis

import sympy as sp
# from Source.Pre_processing.Mesh import Mesh

e1, e2, e3 = sp.symbols('e1 e2 e3')


class basisFunctions:

    def __init__(self, mesh, shape:str | None):

        if mesh != None and shape != None:
            self.set_basisFunctions(mesh, shape)

        else:
            self.N = sp.Matrix([])

        self.stabilized = False


    def set_basisFunctions(self, mesh, shape:str):
        if mesh.meshType == "1DROD2P" and shape == 'Linear':
            self.N = sp.Matrix([(1 - e1) / 2, (1 + e1) / 2])


    def bfGrad(self):
        return self.N.jacobian(sp.Matrix(list(self.N.free_symbols)))


    # Add stabilization function

    def addStab(self, type:str, stab):

        if not self.stabilized:
            if type == 'PG':
                self.N += stab
                self.stabilized = True


    @staticmethod
    def get_basisFunctions(mesh, shape:str):
        if mesh.meshType == "1DROD2P" and shape == 'Linear':
            N = sp.Matrix([(1 - e1) / 2, (1 + e1) / 2])
        return N

