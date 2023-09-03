import numpy as np
import sympy as sp

from Source.Pre_processing.Mesh import Mesh
from Source.Pre_processing.BasisFunctions import basisFunctions

class scalarField:

    def __init__(self, name:str=None, desc:str=None, unit:str=None, basisFunction:str=None, mesh:Mesh = None):
        self.name = name
        self.desc = desc
        self.unit = unit
        self.bf = basisFunctions(mesh, basisFunction)
        self.values = np.ones(mesh.getNoN())

        self.timeValues = None

    def bfGrad(self):
        return self.bf.bfGrad()
    
    def initField(self, value):
        self.values *= value

    def updateField(self, values):
        self.values = values

    def getElementValues(self, element):

        return np.array([self.values[i] for i in element.getNodesId()])
    
    def updateTimeValues(self, values=None):
        
        self.timeValues = values

    def getFieldValues(self, study):
        if study == 'Steady state':
            return self.values
        elif study == 'Transient':
            return self.timeValues