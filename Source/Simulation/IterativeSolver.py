import numpy as np
import scipy as sc
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as scla

import matplotlib.pyplot as plt

class IterativeSolver:

    def __init__(self, A, b, x0, options=None):

        self.A = A
        self.b = b

        self.x0 = x0

        self.options = options

        self.nIter=1
        self.error = []
        self.iter_list = []


    def solve(self):

        self.convRes()

        if self.options['Solver'] == 'BicgStab':
            return self.BicgStab()
        elif self.options['Solver'] == 'GMRES':
            return self.GMres()

    def BicgStab(self):
        print('Solver: BicgStab (Biconjugate gradient stabilized method)')

        M = self.Preconditioner()

        x, info = spla.bicgstab(self.A, self.b, x0=self.x0, M=M, callback=self.callBackFunc)
        
        self.updConvergencePlot()

        return x
    
    def GMres(self):
        print('Solver: GMRES (Generalized Minimal Residual method)')

        M = self.Preconditioner()

        x, info = spla.gmres(self.A, self.b, x0=self.x0, M=M, callback=self.callBackFunc)

        self.updConvergencePlot()

        return x
    
    def Preconditioner(self):

        if self.options['Preconditioner'] == 'iLU Factorization':
            return self.iLUFactorization()
        elif self.options['Preconditioner'] == None:
            return None
            
  
    def iLUFactorization(self):

        print('Preconditioner: iLU Factorization')

        sA_iLU = spla.spilu(self.A)

        return spla.LinearOperator(sA_iLU.shape, sA_iLU.solve)

    def convRes(self):
        self.nIter=1
        self.error = []
        self.iter_list = []

    def callBackFunc(self, xk):
        error = np.linalg.norm(self.A.dot(xk)-self.b)
        self.error.append(error)
        self.iter_list.append(self.nIter)

        # print('{0:4d}   {1:3.14f}'.format(self.nIter, error))

        self.nIter += 1

    def updConvergencePlot(self):
        plt.clf() # Limpiar la figura actual
        plt.plot(self.iter_list, self.error) # Graficar los datos
        plt.yscale('log')
        plt.grid(True)
        # plt.title('Convergencia') # Añadir título
        plt.xlabel('Iteration') # Etiqueta del eje x
        plt.ylabel('Error') # Etiqueta del eje y
        plt.show() # Mostrar la figura