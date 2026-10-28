import numpy as np

from Source.Pre_processing.Mesh import Mesh
from Source.enums import StudyType

class scalarField:

    def __init__(self, name:str=None, desc:str=None, unit:str=None, basisFunction:str=None, mesh:Mesh = None):
        self.name = name
        self.desc = desc
        self.unit = unit
        #self.bf = basisFunctions(mesh, basisFunction)
        self.shape = basisFunction
        #self.gradN = self.bf.gradN
        #self.gradN_func = self.bf.gradN_func
        self.values = np.ones(mesh.getNoN())

        self.timeValues = None

    
    def initField(self, value):
        self.values *= value

    def updateField(self, values):
        self.values = values


    
    def updateTimeValues(self, values=None):
        
        self.timeValues = values

    def getFieldValues(self, study):
        if study == StudyType.steady_state:
            return self.values
        elif study == StudyType.transient:
            return self.timeValues