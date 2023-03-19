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
        # self.bF = basisFunctions().get_basisFunctions(self._mesh, 'Linear')
        self.mat = mat  # Material domain
        self.physics = psc(self)  # Model physics

        self.A = None
        self.b = None

        self.sol = None

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

    def setBC(self, id: int, type=None, **kwargs):

        if type == 'Dirichlet':
            self._mesh.NL[id].BC = self.physics.setDirichletBC(**kwargs)

        elif type == 'Neumann':
            self._mesh.NL[id].BC = self.physics.setNeumannBC(**kwargs)

        elif type == 'Newton':
            self._mesh.NL[id].BC = self.physics.setNewtonBC(**kwargs)

    def constructProblem(self):

        self.A = self.assembleGlobalMatrix()
        self.b = self.assembleGlobalVector()
        self.applyDirichletBC()
        self.A = self.A.tocsc()
        self.b = self.b.tocsc()

    def assembleGlobalMatrix(self):

        print('Assembling Global matrix')

        A = sc.sparse.lil_matrix((self._mesh.getNoN(), self._mesh.getNoN()))

        for element in self._mesh.EL:

            A_e = self.physics.getElementMatrix(element)

            for idx, node_i in enumerate(element.nodes):
                for jdx, node_j in enumerate(element.nodes):
                    
                    A[node_i.id, node_j.id] += A_e[idx, jdx]

        return A

    def assembleGlobalVector(self):

        print('Assembling Global vector')

        b = np.zeros(self._mesh.getNoN())

        for element in self._mesh.EL:

            b_e = self.physics.getElementVector(element)

            for idx, node_i in enumerate(element.nodes):
                                    
                b[node_i.id] += b_e[idx]

        return b

    def applyDirichletBC(self):

        print('Applying Boundary Conditions')

        for j in range(self._mesh.getNoN()):

            if self._mesh.NL[j].BC is not None and self._mesh.NL[j].BC['type'] == 'Dirichlet':
                self.A[j, :] = 0
                self.A[j, j] = 1

                self.b[j] = self._mesh.NL[j].BC['value']

    def solve(self):

        print('Simulation started')

        # self.constructProblem()

        # Initial values: Remove it
        # x0 = np.array([2, 8/3, 10/3, 4])
        # x0 = np.ones(shape=self._mesh.getNoN())*200

        # x, exitCode = sc.sparse.linalg.bicgstab(self.A, self.b.todense(), x0=x0)

        # The problem is in the initial values of the field.

        # x = sc.sparse.linalg.inv(self.A) @ self.b.todense()

        # x, exitCode = Solution.solve(self, x0)

        solver = Solution.modelSolver(self)

        solver.solve()

        # print(x)
        # self.sol = x

        # print(self.sol)
        #print('Simulation finished')

    def postProcess(self):

        plt.plot(self.mesh.getXCoor(), self.sol, marker='o')
        plt.xlabel("x-axis [m]")
        plt.ylabel("Temperature[°C]")
        plt.show()
