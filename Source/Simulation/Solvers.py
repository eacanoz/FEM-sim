import numpy as np
import scipy as sc
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as scla

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Source.Simulation.DirectSolver import DirectSolver
from Source.Simulation.IterativeSolver import IterativeSolver

import time

# from Source.core import Model

Nfeval = 1
xi_1 = []

def tictoc(method):
        def wrapper(ref):
            t1 = time.time()
            method(ref)
            t2 = time.time() - t1
            print(f'Simulation finished on {t2} seconds')
        
        return wrapper

class modelSolver():

    def __init__(self, model):

        self.nIter=1
        self.error = []
        self.iter_list = []

        self.model = model

        self.A = None
        self.b = None

        #self.x0 = {}
        self.x0 = np.array([])
        self.sol = None


    def construcProblem(self):
 
        self.A, self.b = self.model.assembleGlobalSystem()

        self.applyBC()

        self.A = self.A.tocsc()


    def applyBC(self):
        for node in self.model._mesh.NL:
            if node.BC is not None and node.BC['type'] == 'Dirichlet':
                self.A[node.id, :] = 0
                self.A[node.id, node.id] = 1

                self.b[node.id] = node.BC['value']

    def convRes(self):
        self.nIter=1
        self.error = []
        self.iter_list = []

    @tictoc
    def solve(self):

        self.convRes()

        self.getFieldVariables()

        if self.model.solverOptions['Type'] == 'Linear':
            self.linearSolver()
        elif self.model.solverOptions['Type'] == 'Nonlinear':
            self.nonlinearSolver()

    def linearSolver(self):


        self.construcProblem()
        self.getInitialField()

        if self.model.solverOptions['Method'] == 'Direct':
            solutionMethod = DirectSolver(self.A, self.b, self.x0, self.model.solverOptions)

            self.sol = solutionMethod.solve()


        elif self.model.solverOptions['Method'] == 'Iterative':
            solutionMethod = IterativeSolver(self.A, self.b, self.x0, self.model.solverOptions)

            self.sol = solutionMethod.solve()

        # self.updConvergencePlot()

        self.model.sol = self.sol


    def nonlinearSolver(self):

        print('---------- Nonlinear Solver ----------')

        self.tolerance = 1
        self.numbIterations = 0

        while (self.tolerance > 1e-4 or self.numbIterations < 200):

            self.numbIterations += 1

            print(f'----- Iteration number: {self.numbIterations} -----')

            self.linearSolver()

            self.tolerance = np.linalg.norm(self.sol - self.x0) 

            print(f'Tolerance for iteration number {self.numbIterations}: {self.tolerance}')

            if self.tolerance < 1e-4 or self.numbIterations > 200:
                break
            else:
                self.model.physics.var['T'].updateField(self.sol) # Fix -- Hardcoded
                # x0 = self.sol

    def callBackFunc(self, xk):
        error = np.linalg.norm(self.A.dot(xk)-self.b)
        self.error.append(error)
        self.iter_list.append(self.nIter)

        print('{0:4d}   {1:3.14f}'.format(self.nIter, error))

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
        

    def getFieldVariables(self):
        self.fieldVariables = self.model.physics.var.keys()

    def getInitialField(self):

        # print('\nCollecting Initial field\n')

        self.x0 = np.array([])

        for fieldVar in self.fieldVariables:

            self.x0 = np.append(self.x0, self.model.physics.var[fieldVar].values)
#------------------------------------------------------------------------------
# Converting self.x0 to np.array that append different values from self.model.physics.var[fieldVar].values

