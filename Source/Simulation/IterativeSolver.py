import numpy as np
import scipy as sc
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as scla

from Source.enums import  IterativeSolvers, Preconditioners
from Source.logger_config import sim_logger


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

        if self.options['Solver'] == IterativeSolvers.BiCGSTAB:
            return self.BicgStab()
        elif self.options['Solver'] == IterativeSolvers.GMRES:
            return self.GMres()
        elif self.options['Solver'] == IterativeSolvers.CG:
            return self.Cg()

        raise ValueError(f"Iterative solver not implemented: {self.options['Solver']}")

    def _check_info(self, info, name):

        """scipy returns info=0 on success, >0 if not converged after maxiter, <0 on illegal input."""

        if info != 0:
            sim_logger.warning(f"{name} did not converge (info={info}).")

    def Cg(self):

        M = self.Preconditioner()

        x, info = spla.cg(self.A, self.b, x0=self.x0, M=M, callback=self.callBackFunc)

        self._check_info(info, "CG")

        return x

    def BicgStab(self):
        # print('Solver: BicgStab (Biconjugate gradient stabilized method)')

        M = self.Preconditioner()

        x, info = spla.bicgstab(self.A, self.b, x0=self.x0, M=M, callback=self.callBackFunc)

        self._check_info(info, "BiCGSTAB")

        #self.updConvergencePlot()

        return x

    def GMres(self):
        # print('Solver: GMRES (Generalized Minimal Residual method)')

        M = self.Preconditioner()

        x, info = spla.gmres(self.A, self.b, x0=self.x0, M=M, callback=self.callBackFunc)

        self._check_info(info, "GMRES")

        #self.updConvergencePlot()

        return x
    
    def Preconditioner(self):

        if self.options['Preconditioner'] == Preconditioners.iLU:
            return self.iLUFactorization()
        elif self.options['Preconditioner'] == None:
            return None
            
  
    def iLUFactorization(self):

        # print('Preconditioner: iLU Factorization')

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