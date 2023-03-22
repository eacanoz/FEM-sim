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
    ConstRel = '∇.(k * ∇T) + Q == 0'

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

        B = np.zeros((element.getNumberNodes(), element.getNumberNodes()))

        for i, node in enumerate(element.nodes):

            if node.BC is not None and node.BC['type'] == 'Newton':

                if callable(node.BC['h_c']):
                    B[i, i] = node.BC['h_c'](i)
                    # print('Radiaction BC used')
                else:
                    B[i, i] = node.BC['h_c']

        return B

    def addGVector(self, element):

        G = np.zeros((element.getNumberNodes())).transpose()

        for i, node in enumerate(element.nodes):

            if node.BC is not None:

                if node.BC['type'] == 'Newton':

                    if callable(node.BC['h_c']):
                        G[i] = - node.BC['h_c'](i) * node.BC['T_ext']
                        # print('Radiaction BC used')
                    else:
                        G[i] = - node.BC['h_c'] * node.BC['T_ext']

                elif node.BC['type'] == 'Neumann':

                    G[i] = node.BC['q_flux']

        return G

    def addBC_Temperature(self, id: int, T: float):

        self.modelRef._mesh.NL[id].BC = self.setDirichletBC(T0=T)

    def addBC_Convection(self, id: int, h_c: float, T_ext: float):

        self.modelRef._mesh.NL[id].BC = self.setNewtonBC(h_c, T_ext)

    def addBC_HeatFlux(self, id: int, q_flux: float):

        self.modelRef._mesh.NL[id].BC = self.setNeumannBC(q_flux)

    def addBC_Radiation(self, id: int, epsilon: float, T_ext: float):

        sigma = 5.6704e-8

        h_c = lambda n: sigma*epsilon*((self.var['T'].values[n])**3 + T_ext*(self.var['T'].values[n])**2 
                             + (self.var['T'].values[n])*T_ext**2 + T_ext**3)
        
        self.modelRef._mesh.NL[id].BC = self.setNewtonBC(h_c, T_ext)

        # print('Radiaction BC added')