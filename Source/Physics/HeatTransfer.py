# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++++

# Class Definition for Heat transfer physics

import numpy as np
from Source.Physics.Physics import physics
from Source.Primals.Scalar import scalarField


class ht(physics):
    ConstRel = 'lap(k,T) + Q == 0'

    # Set material properties.

    def __init__(self, model):
        super().__init__(model)

        self.var = {'T': scalarField('T', 'Temperature', 'K', 'Linear', model.mesh)}

        self.Pe = self.mat.rho * self.mat.Cp * 1 / self.mat.k

    def getElementMatrix(self, element):

        if self.Convection:
            self.C = self.div(self.var['T'], element, self.mat.rho * self.mat.Cp, 1)  # 1 stands for velocity (u = 1)

        self.K = self.laplacian(self.mat.k, self.var['T'], element)

        self.B = self.addBMatrix(element)

        elementMatrix = self.C + self.K + self.B

        return elementMatrix

    def getElementVector(self, element):

        self.F = self.forceVector(element)
        self.G = self.addGVector(element)

        return self.F - self.G

    def setVariables(self):

        return {'T': []}

    def setDirichletBC(self, T0):

        """
        Define a Dirichlet Boundary Condition for the node

        :param T0: Temperature at the boundary
        :return: Dictionary with boundary information for the physics
        """

        return {'type': 'Dirichlet', 'value': T0}

    def setNeumannBC(self, q_flux):

        """
        Define a Neumann Boundary Condition for the node

        :param q_flux: Heat flux at the boundary
        :return: Dictionary with boundary information for the physics
        """

        return {'type': 'Neumann', 'q_flux': q_flux}

    def setNewtonBC(self, h_c, T_ext):

        """
        Define a Newton Boundary Condition for the node

        :param h_c: Heat transfer coefficient
        :param T_ext: External temperature
        :return: Dictionary with boundary information for the physics
        """

        return {'type': 'Newton', 'h_c': h_c, 'T_ext': T_ext}

    def addBMatrix(self, element):

        B = np.zeros((len(element.nodes), len(element.nodes)))

        for i in range(len(element.nodes)):

            if element.nodes[i].BC is not None and element.nodes[i].BC['type'] == 'Newton':
                B[i, i] = element.nodes[i].BC['h_c']

        return B

    def addGVector(self, element):

        G = np.zeros((len(element.nodes))).transpose()

        for i in range(len(element.nodes)):

            if element.nodes[i].BC is not None:

                if element.nodes[i].BC['type'] == 'Newton':

                    G[i] = - element.nodes[i].BC['h_c'] * element.nodes[i].BC['T_ext']

                elif element.nodes[i].BC['type'] == 'Neumann':

                    G[i] = element.nodes[i].BC['q_flux']

        return G
