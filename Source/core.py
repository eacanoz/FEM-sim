# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++++

# Main Class for FEA model

import numpy as np
import sympy as sp
import scipy as sc
import matplotlib.pyplot as plt

from Source.Pre_processing.Mesh import Mesh, Element, Node
from Source.Physics.Physics import physics
from Source.Material import material
from Source.Pre_processing.BasisFunctions import basisFunctions
import Source.Simulation.Solvers as Solution


# Model


class Model(object):
    """
    Superclass for all FEA models
    """

    area = 1

    def __init__(self, name: str, mtype=None, dim: int = 1, mesh: Mesh = None, mat: material = None, psc=None):
        """
        Parameters
        -----------
        :param name:  Name of model
        :param mtype: Type of Model (Not defined)
        :param dim: Problem dimension: 1->1D; 2->2D(!); 3->3D(!)
        :param mesh Mesh (Class: Mesh)
        """

        self._mesh = mesh  # Model mesh
        self._PD = dim  # Model dimension
        self.name = name  # Name of the model
        self.mtype = mtype  # Type of model
        self.w = basisFunctions(self._mesh, 'Linear') # Test Function
        self.mat = mat  # Material domain
        self.physics = psc(self)  # Model physics

        self.A = None
        self.b = None

        self.solverOptions = None

        #self.solverOptions = {'Study': 'Steady state', 'Type': 'Nonlinear', 'Method': 'Direct', 'Solver':'PARDISO'}
        # self.solverOptions = {'Type': 'Linear', 'Method': 'Direct', 'Solver':'PARDISO'}
        # self.solverOptions = {'Type': 'Linear', 'Method': 'Iterative', 'Solver':'BicgStab', 'Preconditioner': 'iLU Factorization'}
        # self.solverOptions = {'Type': 'Linear', 'Method': 'Iterative', 'Solver':'BicgStab', 'Preconditioner': None}

        self.sol = None

        self.timeVector = None

    @property
    def mesh(self):
        return self._mesh

    def add_material(self):
        pass

    def set_physics(self, physics):
        pass

    def __str__(self):
        if self.physics is None:
            physics_status = 'Not defined'

        custom_str = ("Model: " + self.name + "\nDimension: " + str(self._PD) + "D" + "\nNodes: "
                      + str(self.mesh.getNoN()) + "\nElements: " + str(self.mesh.getNoE()) + "\nPhysics: "
                      + physics_status)

        return custom_str

    def assembleGlobalSystem(self, solverOptions = None):

        Var = self.physics.getVariables()
        nVar = self.physics.getNumOfVar()

        A = sc.sparse.lil_matrix((self._mesh.getNoN()*nVar, self._mesh.getNoN()*nVar))
        b = np.zeros(self._mesh.getNoN()*nVar)

        for idxVar, Variable in enumerate(Var):

            for element in self._mesh.EL:
                A_e = self.physics.getElementMatrix(element, Variable, solverOptions)
                b_e = self.physics.getElementVector(element, Variable, solverOptions)

                for idx, node_i in enumerate(element.nodes):

                    if node_i.BC and node_i.BC[Variable]['type'] == 'Dirichlet':

                        b[idxVar + nVar*node_i.id] += node_i.BC[Variable]['value']
                        A[idxVar + nVar*node_i.id, :] = 0
                        A[idxVar + nVar*node_i.id, idxVar + nVar*node_i.id] = 1
                    else:

                        b[idxVar + nVar*node_i.id] += b_e[idx]

                        for jdx, node_j in enumerate(element.nodes):
                        
                            A[idxVar + nVar*node_i.id, idxVar + nVar*node_j.id] += A_e[idx, jdx]
         
        return A, b

    def assembleResidualVector(self, solverOptions = None):
        
        Var = self.physics.getVariables()
        nVar = self.physics.getNumOfVar()

        F = np.zeros(self._mesh.getNoN()*nVar)

        for idxVar, Variable in enumerate(Var):

            for element in self._mesh.EL:
                
                F_e = self.physics.getResidualVector(element, Variable, solverOptions)

                for idx, node_i in enumerate(element.nodes):

                    if node_i.BC and node_i.BC[Variable]['type'] == 'Dirichlet':
                        F[idxVar + nVar*node_i.id] = 0
                    else:
                        F[idxVar + nVar*node_i.id] += F_e[idx]

        return F

    def assembleMassMatrix(self, solverOptions = None):

        M = sc.sparse.lil_matrix((self._mesh.getNoN(), self._mesh.getNoN()))

        for element in self._mesh.EL:

            M_e = self.physics.getElementMassMatrix(element, solverOptions)

            for idx, node_i in enumerate(element.nodes):
                for jdx, node_j in enumerate(element.nodes):

                    M[node_i.id, node_j.id] += M_e[idx, jdx]

        return M

    def solverConfiguration(self, Study='Steady state', Type='Linear', Method = 'Direct', Solver = 'PARDISO', timeDisc = 1, timeStep=0.05, totalTime = 3, prec = 'iLU Factorization'):

        self.solverOptions = {
            'Study': Study,
            'Type': Type,
            'Method': Method,
            'Solver': Solver,
            'Preconditioner': prec,
            'Time Discretization': timeDisc,
            'Time Step': timeStep,
            'Time': totalTime
        }


    def solve(self):

        print('Simulation started')

        if self.solverOptions is None:
            self.solverConfiguration()

        solver = Solution.modelSolver(self)

        solver.solve()

        self.timeVector = solver.timeVector

        print('Simulation finished')

    def postProcess(self):

        plt.plot(self.mesh.getXCoor(), self.sol['T'], marker='o')
        plt.xlabel("x-axis [m]")
        plt.ylabel("Temperature[°C]")
        plt.show()
