import numpy as np
import scipy as sc
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as scla

class IterativeSolver:

    def __init__(self, A, b, x0, options=None):

        self.A = A
        self.b = b

        self.x0 = x0

        self.options = options

    def solve(self):

        if self.options['Solver'] == 'BicgStab':
            return self.BicgStab()
        elif self.options['Solver'] == 'GMRES':
            return self.GMres()

    def BicgStab(self):
        print('Solver: BicgStab (Biconjugate gradient stabilized method)')

        x, info = spla.bicgstab(self.A, self.b, x0=self.x0)
        
        return x
    
    def GMres(self):
        print('Solver: GMRES (Generalized Minimal Residual method)')

        x, info = spla.gmres(self.A, self.b, x0=self.x0)

        return x
    
    def Preconditioner(self):

        if self.options == 'ILU Factorization':
            return spla.spilu(self.A)
        