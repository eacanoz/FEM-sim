import numpy as np
import scipy as sc
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as scla

import pypardiso

class DirectSolver:

    def __init__(self, A, b, x0, options=None):

        self.A = A
        self.b = b

        self.x0 = x0

        self.options = options

    def solve(self):

        if self.options['Solver'] == 'PARDISO':
            return self.PARDISO()

    def PARDISO(self):
        print('Solver: PARDISO (Parallel Direct Sparse Solver)')

        x = pypardiso.spsolve(self.A, self.b)
        
        return x
    
    def SuperLU(self):
        print('Solver: Super LU')

        x = spla.spsolve(self.A, self.b)