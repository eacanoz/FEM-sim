import numpy as np
import scipy as sc
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as scla

from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from numdifftools import Jacobian


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Source.Simulation.DirectSolver import DirectSolver
from Source.Simulation.IterativeSolver import IterativeSolver

import time

# from Source.core import Model

Nfeval = 1
xi_1 = []

def tictoc(method):
        def wrapper(*args, **kwargs):
            t1 = time.time()
            method(*args, **kwargs)
            t2 = time.time() - t1
            print(f'method finished on {t2} seconds')
        
        return wrapper

class modelSolver():

    def __init__(self, model):

        self.nIter=1
        self.error = []
        self.iter_list = []

        self.model = model

        self.A = None
        self.b = None
        self.M = None

        #self.x0 = {}
        self.x0 = np.array([])
        self.sol = None

        self.solverOptions = model.solverOptions
        self.nDoF = model.mesh.getNoN()
        self.nVar = None

        self.timeVector = None


    def construcProblem(self):
 
        self.A, self.b = self.model.assembleGlobalSystem(self.solverOptions)

        if self.model.solverOptions['Study'] == 'Transient':
            self.M = self.model.assembleMassMatrix(self.solverOptions)
            
            self.M = self.M.tocsc()

        self.A = self.A.tocsc()

    def convRes(self):
        self.nIter=1
        self.error = []
        self.iter_list = []

    @tictoc
    def solve(self):

        solDict = {}

        self.convRes()

        self.getFieldVariables()

        if self.model.solverOptions['Study'] == 'Steady state':
            self.steadyStateSolver()
        elif self.model.solverOptions['Study'] == 'Transient':
            self.transientSolver()

        for fieldVar in self.fieldVariables:
            solDict[fieldVar] = self.model.physics.var[fieldVar].getFieldValues(self.solverOptions['Study'])

        self.model.sol = solDict


    def steadyStateSolver(self):

        if self.model.solverOptions['Type'] == 'Linear':
            self.linearSolver()
        elif self.model.solverOptions['Type'] == 'Nonlinear':
            self.nonlinearSolver()

        nls = self.nonlinearSolver

        

    def transientSolver(self):

        self.getInitialField()

        sol = solve_ivp(self.transientRHS, [0, self.solverOptions['Time']], self.x0, method='BDF')

        self.timeVector = sol.t 

        self.model.physics.var['T'].updateTimeValues(sol.y) # Fix




    def transientRHS(self, t, y):

        self.updateSolution(y)

        self.construcProblem()

        dy_dt = spla.inv(self.M) @ (self.b - self.A.dot(y))

        self.updateSolution(y)

        return dy_dt

    def linearSolver(self):

        self.construcProblem()
        self.getInitialField()

        if self.model.solverOptions['Method'] == 'Direct':
            solutionMethod = DirectSolver(self.A, self.b, self.x0, self.model.solverOptions)

            self.sol = solutionMethod.solve()


        elif self.model.solverOptions['Method'] == 'Iterative':
            solutionMethod = IterativeSolver(self.A, self.b, self.x0, self.model.solverOptions)

            self.sol = solutionMethod.solve()

        self.updateSolution(self.sol)
        # self.updConvergencePlot()


    def nonlinearSolver(self):

        # print('---------- Nonlinear Solver ----------')

        self.getInitialField()

        sol = fsolve(self.createNonlinearSystem, self.x0)
        # sol = self.fixPointSolver()

        # sol = self.newtonSolver()

        self.updateSolution(sol)


    def createNonlinearSystem(self, x):

        self.updateSolution(x)
        # F = self.model.assembleResidualVector(self.solverOptions)

        self.construcProblem()

        F_a = self.A.dot(x) - self.b

        return F_a

    def callBackFunc(self, xk):
        error = np.linalg.norm(self.A.dot(xk)-self.b)
        self.error.append(error)
        self.iter_list.append(self.nIter)

        print('{0:4d}   {1:3.14f}'.format(self.nIter, error))

        self.nIter += 1

    def fixPointSolver(self):
        print('---------- Nonlinear Solver ----------')

        self.tolerance = 1
        self.numbIterations = 0

        alpha = 0.7

        while (self.tolerance > 1e-4 or self.numbIterations < 200):

            self.numbIterations += 1

            print(f'----- Iteration number: {self.numbIterations} -----')

            self.linearSolver()

            self.tolerance = np.linalg.norm(self.sol - self.x0) 

            print(f'Tolerance for iteration number {self.numbIterations}: {self.tolerance}')

            if self.tolerance < 1e-4 or self.numbIterations > 200:
                break
            else:

                xi = self.x0 + alpha *(self.sol - self.x0)

                self.updateSolution(xi) # Fix -- Hardcoded

        return self.sol


    def newtonSolver(self):
        self.tolerance = 1
        self.numbIterations = 0

        nLS = self.createNonlinearSystem

        self.jac = Jacobian(nLS)

        while (self.tolerance > 1e-4 or self.numbIterations < 200):

            lamb = 0.8

            self.numbIterations += 1
            print(self.numbIterations)

            #print(f'----- Iteration number: {self.numbIterations} -----')
            # self.construcProblem()
            # self.getInitialField()

            #J = self.Jacobian()

            # Fi_1 = self.A.dot(self.x0) - self.b

            if self.model.solverOptions['Method'] == 'Direct':
                solutionMethod = DirectSolver(self.spJacobian(self.x0), (-nLS(self.x0)), self.x0, self.model.solverOptions)

                dX = solutionMethod.solve()


            elif self.model.solverOptions['Method'] == 'Iterative':
                solutionMethod = IterativeSolver(self.spJacobian(self.x0), (-nLS(self.x0)), self.x0, self.model.solverOptions)

                dX = solutionMethod.solve()            

            # self.linearSolver()

            Ui = self.x0 + lamb * dX

            # estimating error for new iteration.

            #self.model.physics.var['T'].updateField(Ui)
            # self.updateSolution(Ui)

            # self.construcProblem()

            # Fi = self.A.dot(Ui) - self.b

            if self.model.solverOptions['Method'] == 'Direct':
                solutionMethod = DirectSolver(self.spJacobian(self.x0), (-nLS(Ui)), dX, self.model.solverOptions)

                error = solutionMethod.solve()


            elif self.model.solverOptions['Method'] == 'Iterative':
                solutionMethod = IterativeSolver(self.spJacobian(self.x0), (-nLS(Ui)), dX, self.model.solverOptions)

                error = solutionMethod.solve()         

            #self.tolerance = np.linalg.norm(self.sol - self.x0)
            self.tolerance = np.linalg.norm(error) 

            print(f'\nTolerance for iteration number {self.numbIterations}: {self.tolerance}\n')

            if self.tolerance < 1e-4 or self.numbIterations > 100:
                break
            else:
                #alpha = 0.7
                #xi = self.x0 + alpha *(self.sol - self.x0)

                self.x0 = Ui

                #self.model.physics.var['T'].updateField(xi) # Fix -- Hardcoded
                #self.updateSolution(xi)
                # x0 = self.sol

        return Ui

    def updateSolution(self, x):

        # nVar = self.model.physics.getNumOfVar()

        for i, fieldVar in enumerate(self.fieldVariables):

            sol = [x[i + self.nVar*j] for j in range(self.model._mesh.getNoN())]

            self.model.physics.var[fieldVar].updateField(sol)


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

        self.nVar = len(self.fieldVariables)

    def getInitialField(self):

        # print('\nCollecting Initial field\n')

        self.x0 = np.zeros(self.model._mesh.getNoN()*self.nVar)

        for idxVar, fieldVar in enumerate(self.fieldVariables):

            for idxValue, value in enumerate(self.model.physics.var[fieldVar].values):

                self.x0[idxVar + self.nVar * idxValue] = value

            #self.x0 = np.append(self.x0, self.model.physics.var[fieldVar].values)
#------------------------------------------------------------------------------
# Converting self.x0 to np.array that append different values from self.model.physics.var[fieldVar].values
# Fix Jacobian matrix

    # def Jacobian(self, eps=1e-6):

    #     """Jacobian matrix is taking to long to assemble"""

    #     Ai_1 = self.A
    #     bi_1 = self.b

    #     x0 = self.x0.copy()

    #     Fi_1 = Ai_1.dot(x0) - bi_1

    #     J = sc.sparse.lil_matrix((self.model._mesh.getNoN(), self.model._mesh.getNoN()))

    #     for i in range(self.x0.size):
            
    #         x0[i] += eps
    #         self.model.physics.var['T'].updateField(x0)

    #         self.construcProblem()

    #         Ai = self.A
    #         bi = self.b

    #         Fi = Ai.dot(x0)-bi

    #         J[:, i] = (Fi - Fi_1)

    #         J *= (1/eps)

    #         x0 = self.x0.copy()

            
    #     self.A = Ai_1
    #     self.b = bi_1

    #     return J.tocsr()

    @tictoc
    def spJacobian(self, x):

        return sp.csr_matrix(self.jac(x))

        

