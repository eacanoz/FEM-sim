# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++++

# Class Definition for Heat transfer physics

import numpy as np
from Source.Physics.Physics import physics
from Source.Primals.Scalar import scalarField

u = 1  # Velocity for convection term, hardcoded for now


class ht(physics):
    ConstRel = '∇.(k * ∇T) + Q == 0'

    # Set material properties.

    def __init__(self, model):
        super().__init__(model)

        self.var = {'T': scalarField('T', 'Temperature', 'K', 'Linear', model.mesh)}

        self.Pe = self.mat.rho * self.mat.Cp * u / self.mat.k

    def initializeMatrices(self, element, Variable):

        if self.Convection:
            self.C = self.div(self.var[Variable], element, self.mat.rho * self.mat.Cp, u)  # 1 stands for velocity (u = 1)

        self.K = self.laplacian(self.mat.k, self.var[Variable], element)

        self.B = self.addBMatrix(element, Variable)

    def initializeMassMatrix(self, element):

        self.M = self.mass(self.var['T'], element, self.mat.rho * self.mat.Cp)

    def initializeVectors(self, element, Variable):

        self.F = self.forceVector(element, Variable)
        self.G = self.addGVector(element, Variable)

## ---------- Boundary conditions ---------- ##

    def addBC_Temperature(self, id: int, T: float):

        self.modelRef._mesh.NL[id].BC['T'] = self.setDirichletBC(T)

    def addBC_Convection(self, id: int, h_c: float, T_ext: float):

        """
        Add convection boundary condition to the node with the given id.
        Parameters:
        -----------
        id (int): Node id to which the boundary condition will be applied
        h_c (float): Convection coefficient
        T_ext (float): External temperature
        """

        self.modelRef._mesh.NL[id].BC['T'] = self.setNewtonBC(h_c, T_ext)

    def addBC_HeatFlux(self, id: int, q_flux: float):

        self.modelRef._mesh.NL[id].BC['T'] = self.setNeumannBC(q_flux)

    def addBC_Radiation(self, id: int, epsilon: float, T_ext: float):

        sigma = 5.6704e-8

        h_c = lambda n: sigma*epsilon*((self.var['T'].values[n])**3 + T_ext*(self.var['T'].values[n])**2 
                             + (self.var['T'].values[n])*T_ext**2 + T_ext**3)
        
        self.modelRef._mesh.NL[id].BC['T'] = self.setNewtonBC(h_c, T_ext)