# Class Mesh

import numpy as np
import sympy as sp

from Source.Pre_processing.BasisFunctions import basisFunctions


class Mesh:

    def __init__(self):

        self.NL = None
        self.EL = None

    def Generate_Mesh(self, PD, L, NoE, MeshType):

        self.NoE = NoE
        self.meshType = MeshType

        if PD == 1:
            if self.meshType == '1DROD2P':
                self.PpE = 2

        self.NL = []

        self.EL = []

        dx = L / NoE

        for i in range(0, self.NoE + 1):
            self.NL.append(Node(id=i, coor=(i * dx,)))

        for i in range(0, self.NoE):
            self.EL.append(Element(id=i, nodes=[self.NL[i], self.NL[i + 1]]))

    def getNoN(self):

        return len(self.NL)

    def getNoE(self):

        return len(self.EL)

    def getElementsSizes(self):
        sizes = []
        for element in self.EL:
            sizes.append(element.getLength())

        return sizes

    def getXCoor(self):

        x_coor = []

        for node in self.NL:
            x_coor.append(node.getXCoor())

        return x_coor

    def getElementvalue(self, variable):
        pass

    def setElementShapeFunction(self, shapeFunction):
        for element in self.EL:
            element.sF = basisFunctions(self, shapeFunction)

class Node:

    def __init__(self, id=None, coor: tuple = None):
        self.id = id
        self.coor = coor
        self.type = 'DoF'
        self.variable = {}
        self.BC = None


    def getVariable(self, psc):

        self.variable = psc.setVariables()


    def getXCoor(self):

        return self.coor[0]

class Element:

    def __init__(self, id=None, nodes: list = None):
        self.id = id
        self.nodes = nodes
        self.sF = None
    def getLength(self):
        return self.nodes[1].coor[0] - self.nodes[0].coor[0]

    def getCoor(self):
        coordinates = []

        for node in self.nodes:
            coordinates.append(node.coor[0])

        return coordinates

    def getNumberNodes(self):

        return len(self.nodes)
    
    def Jacobian(self):
        
        x_map = self.sF.N.transpose() * sp.Matrix(self.getCoor())
        return x_map.jacobian(sp.Matrix(list(x_map.free_symbols)))