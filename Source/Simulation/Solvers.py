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

        self.x0 = {}
        self.sol = None


    def construcProblem(self):
        self.A = self.model.assembleGlobalMatrix()
        self.b = self.model.assembleGlobalVector()

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

        self.getFieldVariables()

        self.convRes()

        self.construcProblem()
        self.getInitialField()

        x0 = self.x0['T'] # Fix when adding a physics with multiple variables


        options = {'Method': 'Direct', 'Solver':'PARDISO'}
        # options = {'Method': 'Iterative', 'Solver':'BicgStab'}

        if options['Method'] == 'Direct':
            solutionMethod = DirectSolver(self.A, self.b, x0, options)

            self.sol = solutionMethod.solve()


        elif options['Method'] == 'Iterative':
            solutionMethod = IterativeSolver(self.A, self.b, x0, options)

            self.sol = solutionMethod.solve()


        #print('{0:4s}   {1:9s}'.format('Iter', 'error'))
        #self.sol, exitCode = spla.bicgstab(self.A, self.b, x0=x0, callback=self.callBackFunc)
        #print('\n')

        # self.updConvergencePlot()

        self.model.sol = self.sol


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

        print('\nCollecting Initial field\n')

        for fieldVar in self.fieldVariables:

            self.x0[fieldVar] = self.model.physics.var[fieldVar].values
#------------------------------------------------------------------------------

    


# def pyPARDISO(model:Model):
    
#     # A = model.A
#     b = model.b.todense()

#     A = sp.csr_matrix([[4, 0, 0, 0], [0, 2, 0, 1], [0, 0, 3, 0], [0, 1, 0, 2]])

#     lu = spla.splu(A)

#     y = lu.solve(b)
#     x = lu.solve(y)
    
#     spla.spsolve(A, b, use_umfpack = False)
    
#     pass


