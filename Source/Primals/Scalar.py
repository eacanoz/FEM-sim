import numpy as np
from Source.Pre_processing.Mesh import Mesh
from Source.Pre_processing.BasisFunctions import basisFunctions

class scalarField:

    def __init__(self, name:str=None, desc:str=None, unit:str=None, basisFunction:str=None, mesh:Mesh = None):
        self.name = name
        self.desc = desc
        self.unit = unit
        self.bf = basisFunctions().get_basisFunctions(mesh, basisFunction)
        self.values = np.zeros(mesh.getNoN())
