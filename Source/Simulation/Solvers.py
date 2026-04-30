import numpy as np
import scipy as sc
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as scla

from scipy.integrate import solve_ivp
from scipy.optimize import root

from numba import jit, njit, prange, guvectorize, vectorize, float64

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Source.Simulation.DirectSolver import DirectSolver
from Source.Simulation.IterativeSolver import IterativeSolver
from Source.enums import SolverType, StudyType, ProblemType
from Source.logger_config import sim_logger


import time
from progress.bar import Bar

# from Source.core import Model

Nfeval = 1
xi_1 = []

def tictoc(method):
        def wrapper(*args, **kwargs):
            t1 = time.time()
            method(*args, **kwargs)
            t2 = time.time() - t1
            #print(f'method finished on {round(t2, 3)} seconds')
            sim_logger.info(f'method finished on {round(t2, 3)} seconds')
            return t2
        
        return wrapper

class modelSolver():

    """Class responsible for solving the model based on the physics and mesh. 
    It includes methods for assembling the global system, solving linear and nonlinear problems, and handling transient simulations.
    """

    def __init__(self, model):

        """Initializes the model solver with a reference to the model and sets up necessary attributes."""

        self.nIter=1
        self.error = []
        self.iter_list = []

        self.model = model

        self.A = None
        self.b = None
        self.M = None


        self.K = None
        self.f = None

        #self.x0 = {}
        self.x0 = np.array([])
        self.sol = None

        self.solverOptions = model.solverOptions
        self.nDoF = model.mesh.getNoN()
        self.nVar = None

        self.timeVector = None


    def construcProblem(self):

        """Assembles the global system of equations for the model based on the physics and mesh."""
 
        self.A, self.b = self.model.assembleGlobalSystem(self.solverOptions)

        if self.model.solverOptions['Study'] == StudyType.transient:
            self.M = self.model.assembleMassMatrix(self.solverOptions)
            
            self.M = self.M.tocsc()

        self.A = self.A.tocsc()

    def convRes(self):
        self.nIter=1
        self.error = []
        self.iter_list = []

    @tictoc
    def solve(self):

        """Solves the model based on the configured solver options and updates the solution in the model."""

        solDict = {}

        self.convRes()

        self.getFieldVariables()

        if self.model.solverOptions['Study'] == StudyType.steady_state:
            self.steadyStateSolver()
        elif self.model.solverOptions['Study'] == StudyType.transient:
            self.transientSolver()

        for fieldVar in self.fieldVariables:
            solDict[fieldVar] = self.model.physics.var[fieldVar].getFieldValues(self.solverOptions['Study'])

        self.model.sol = solDict


    def steadyStateSolver(self):

        """Solves the steady-state problem for the model based on the configured solver options."""

        if self.model.solverOptions['Type'] == ProblemType.linear:
            self.linearSolver()
        elif self.model.solverOptions['Type'] == ProblemType.nonlinear:
            self.nonlinearSolver()

        #nls = self.nonlinearSolver

        

    def transientSolver(self):

        """Solves the transient problem for the model based on the configured solver options using an ODE solver."""

        self.getInitialField()

        sol = solve_ivp(self.transientRHS, [0, self.solverOptions['Time']], self.x0, method='BDF')

        self.timeVector = sol.t 

        self.model.physics.var['T'].updateTimeValues(sol.y) # Fix


    def transientRHS(self, t, y):

        """Defines the right-hand side of the transient problem for the ODE solver."""

        self.updateSolution(y)

        self.construcProblem()

        dy_dt = spla.inv(self.M) @ (self.b - self.A.dot(y))

        self.updateSolution(y)

        return dy_dt

    def linearSolver(self):

        """Solves the linear problem for the model based on the configured solver options.
        It constructs the global system and uses either a direct or iterative solver to find the solution.
        """

        self.construcProblem()
        self.getInitialField()

        if self.model.solverOptions['Method'] == SolverType.direct:
            solutionMethod = DirectSolver(self.A, self.b, self.x0, self.model.solverOptions)

            self.sol = solutionMethod.solve()


        elif self.model.solverOptions['Method'] == SolverType.iterative:
            solutionMethod = IterativeSolver(self.A, self.b, self.x0, self.model.solverOptions)

            self.sol = solutionMethod.solve()

        self.updateSolution(self.sol)
        # self.updConvergencePlot()


    def nonlinearSolver(self):

        """Solves the nonlinear problem for the model based on the configured solver options.
        It constructs the global system, computes the Jacobian, and uses a Newton solver to find the solution iteratively until convergence."""

        # print('---------- Nonlinear Solver ----------')

        self.getInitialField()

        # sol = fsolve(self.createNonlinearSystem, self.x0)
        # sol = self.fixPointSolver()

        sol = self.newtonSolver()

        self.updateSolution(sol)

    def createNonlinearSystem(self, x):

        """Creates the nonlinear system of equations for the model based on the current solution vector x.
        It updates the solution in the model, constructs the global system, and returns the residual vector for the nonlinear problem."""

        x_init = self.x0.copy()

        self.updateSolution(x)
        F = self.model.assembleResidualVector(self.solverOptions)

        self.updateSolution(x_init)
        return F

    def callBackFunc(self, xk):

        """Callback function for monitoring the convergence of the nonlinear solver.
        It computes the error at each iteration and updates the convergence plot."""

        error = np.linalg.norm(self.A.dot(xk)-self.b)
        self.error.append(error)
        self.iter_list.append(self.nIter)

        print('{0:4d}   {1:3.14f}'.format(self.nIter, error))

        self.nIter += 1

    def fixPointSolver(self):

        """Implements a fixed-point iteration solver for the nonlinear problem.
        It iteratively updates the solution vector until convergence based on a relaxation parameter alpha."""
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


    def assembleNonlinearSystem(self):

        """Assembles the nonlinear system of equations for the model based on the current solution vector.
        It constructs the global system and returns the residual vector for the nonlinear problem."""

        self.K, self.f = self.model.assembleGlobalSystemNonLinear(self.solverOptions)


    def newtonSolver(self):

        """Implements a Newton-Raphson solver for the nonlinear problem.
        It iteratively computes the Jacobian matrix, solves the linearized system, and updates the solution vector until convergence based on a relaxation parameter lambda."""

        sim_logger.info("------- Starting Newton-Raphson nonlinear solver -------")


        self.tolerance = 1
        self.numbIterations = 0

        lamb_min = 0.05
        lamb_max = 0.9

        error_1 = 100

        lamb = 0.8

        min_tolerance = 1e-4
        max_numb_iterations = 200


        sim_logger.info(f"Minimum Tolerance: {min_tolerance:.4e}, Max Number of Iterations: {max_numb_iterations}")

        # 1. IMPRIMIR EL ENCABEZADO DE LA TABLA ANTES DEL BUCLE
        # Explicación del formato:
        # :^6  -> Centrado ocupando 6 espacios
        # :^15 -> Centrado ocupando 15 espacios
        header = f"| {'Iter':^6} | {'Error (Norm R)':^15} | {'Alpha (Step)':^15}"
        separator = "-" * len(header)
        
        sim_logger.info(separator)
        sim_logger.info(header)
        sim_logger.info(separator)
        
        # Fila para la iteración 0 (Condición Inicial)
        #row_0 = f"| {0:^6} | {error:^15.4e} | {'-':^15} | {'Condición Inicial':^20} |"
        #sim_logger.info(row_0)

        while (self.tolerance > min_tolerance and self.numbIterations < max_numb_iterations):

            self.numbIterations += 1

            if lamb == lamb_min:
                lamb = 0.8

            #print(f'----- Iteration number: {self.numbIterations} -----')
            #self.construcProblem()
            #self.getInitialField()


            #if self.numbIterations == 1 or self.numbIterations % 3 == 0:
            #    self.assembleNonlinearSystem()
            #    J = self.K
            #    Fi_1 = self.f

            self.assembleNonlinearSystem()
            J = self.K
            Fi_1 = self.f

            #J = self.jac(self.x0)

            #J = self.Jacobian()

            #Fi_1 = self.A.dot(self.x0) - self.b

            if self.model.solverOptions['Method'] == SolverType.direct:

                dX = DirectSolver(J, -Fi_1, self.x0, self.model.solverOptions).solve()

            elif self.model.solverOptions['Method'] == SolverType.iterative:

                dX = IterativeSolver(J, -Fi_1, self.x0, self.model.solverOptions).solve()         

            # self.linearSolver()

            xi = self.x0 + lamb * dX
            
            # while (True):

            #     xi = self.x0 + lamb * dX

            #     Fi = self.createNonlinearSystem(xi)

            # # estimating error for new iteration.


            #     if self.model.solverOptions['Method'] == SolverType.direct:

            #         error = DirectSolver(J, -Fi, dX, self.model.solverOptions).solve()      


            #     elif self.model.solverOptions['Method'] == SolverType.iterative:

            #         error = IterativeSolver(J, -Fi, dX, self.model.solverOptions).solve()

                
            #     if np.linalg.norm(error) < error_1:

            #         lamb = min(lamb * 1.2, lamb_max)

            #         error_1 = np.linalg.norm(error)

            #         break

            #     else:
            #         lamb = max(lamb * 0.5, lamb_min)

            #         error_1 = np.linalg.norm(error)

            #         if lamb == lamb_min:
            #             break

            #self.tolerance = np.linalg.norm(self.sol - self.x0)
            #self.tolerance = np.linalg.norm(Fi) 

            #self.tolerance = np.linalg.norm(Fi) / np.linalg.norm(Fi_1)
            self.tolerance = np.linalg.norm(dX)

            self.updateSolution(xi)

            self.x0 = xi

            row = f"| {self.numbIterations:^6} | {self.tolerance:^15.4e} | {lamb:^15.4f}"
            sim_logger.info(row)

            #print(f'\nTolerance for iteration number {self.numbIterations}: {self.tolerance}\n')

        sim_logger.info(separator)

        if self.tolerance <= min_tolerance:
            sim_logger.info(f"Succefully converged in {self.numbIterations} iterations.")
        else:
            sim_logger.warning("Solver did not converge at the specified tolerance and number of iterations.")

        return xi

    def updateSolution(self, x):

        """Updates the solution in the model based on the current solution vector x.
        It extracts the values for each field variable from the solution vector and updates the corresponding fields in"""

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

        """Retrieves the field variables from the model's physics and stores them in an attribute for later use."""

        self.fieldVariables = self.model.physics.var.keys()

        self.nVar = len(self.fieldVariables)

    def getInitialField(self):

        """Collects the initial field values for each variable from the model's physics and stores them in a single solution vector x0 for use in the solvers."""

        # print('\nCollecting Initial field\n')

        self.x0 = np.zeros(self.model._mesh.getNoN()*self.nVar)

        for idxVar, fieldVar in enumerate(self.fieldVariables):

            for idxValue, value in enumerate(self.model.physics.var[fieldVar].values):

                self.x0[idxVar + self.nVar * idxValue] = value

            #self.x0 = np.append(self.x0, self.model.physics.var[fieldVar].values)
#------------------------------------------------------------------------------


        

