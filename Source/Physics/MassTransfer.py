# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++++

# Class Definition for Heat transfer physics

import numpy as np
import scipy as sc
import sympy as sp
import math

from Source.Physics.Physics import physics
from Source.Primals.Scalar import scalarField

class mt(physics):

    def __init__(self, model):
        super().__init__(model)

        self.var = {}

        self.Pe = 1

        self.Diffusivities = {}

        self.reaction = None

    def setDiffusivity(self, chemSpecies, value):

        if chemSpecies in self.var:
            self.Diffusivities[chemSpecies] = value
        else:
            raise "Error: Chemical Specie not defined"


    def setChemSpecies(self, chemSpecies, name):

        self.var[chemSpecies] = scalarField(chemSpecies, name, 'mol/m3', 'Linear', self.modelRef.mesh)

    def initializeMatrices(self, element, Variable):

        if self.Convection:
            self.C = self.div(self.var[Variable], element, 1, 0.01)  # 1 stands for velocity (u = 1)

        self.K = self.laplacian(self.Diffusivities[Variable], self.var[Variable], element)

        self.B = self.addBMatrix(element, Variable)

    def initializeVectors(self, element, Variable):

        self.F = self.forceVector(element, Variable)
        self.G = self.addGVector(element, Variable)        

## ------------- Source terms -------------- ##

    def addReaction(self, stoich):

        k0 = 10
        E_R = 500
        T = 303.15
        a = 1
        b = 1

        self.rRate = lambda n: k0 * math.exp(-(E_R/T)) * (self.var['A'].values[n]**a) * (self.var['B'].values[n]**b)

        self.stoich = stoich
        # self.reaction = lambda elem, chemSpec:  self.stoich[chemSpec] * self.rRate(elem)

        def reaction(element, chemSpec):

            rates = [self.rRate(n.id) for n in element.nodes]

            rateVector = element.sF.N.transpose() * sp.Matrix(rates)

            return self.stoich[chemSpec] * rateVector


        self.source = reaction

## ---------- Boundary conditions ---------- ##

    def addBC_Concentration(self, id:int, chemSpec: str, C0: float):
        self.modelRef._mesh.NL[id].BC[chemSpec] = self.setDirichletBC(C0)

            
    def addBC_Outflow(self, id: int, chemSpec: str):

        self.modelRef._mesh.NL[id].BC[chemSpec] = self.setNeumannBC(0.0)