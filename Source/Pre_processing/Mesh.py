# Class Mesh

import numpy as np
import sympy as sp

from Source.Pre_processing.BasisFunctions import basisFunctions


class Mesh:

    def __init__(self):

        self.NL = None
        self.EL = None

        self.boundaries = {}

    def Generate_Mesh(self, PD, L, NoE, MeshType):

        self.NoE = NoE
        self.meshType = MeshType
        self.PD = PD

        if PD == 1:
            if self.meshType == '1DROD2P':
                self.EpP = 2

        self.NL = []

        self.EL = []

        dx = L / NoE

        for i in range(0, self.NoE + 1):
            self.NL.append(Node(id=i, coor=(i * dx,)))

        for i in range(0, self.NoE):
            self.EL.append(Element(id=i, nodes=[self.NL[i], self.NL[i + 1]]))

    def generate1DMesh():
        pass

    def generate2DMesh(self, dim: tuple, div: tuple, element_type):

        # Only for rectangular geometries

        PD = 2
        q = np.array([[0, 0], [dim[0], 0], [0, dim[1]], [dim[0], dim[1]]]) # corners

        self.NoN = (div[0] + 1)*(div[1] + 1)

        self.NoE = (div[0])*(div[1])

        NPE = 4

        self.NL = np.zeros([self.NoN, PD])

        a = (q[1,0] - q[0,0])/ div[0]  # Increment in the horizontal direction
        b = (q[2,1] - q[0,1])/ div[1]  # Increment in the vertical direction

        n = 0 # Through rows in NL

        for i in range(1, div[1]+2):

            for j in range(1, div[0] + 2):

                self.NL[n, 0] = q[0, 0] + (j-1)*a
                self.NL[n, 1] = q[0, 1] + (i-1)*b

                n +=1

        ### Elements ###

        self.EL = np.zeros([self.NoE, NPE])

        for i in range(1, div[1]+1):

            for j in range(1, div[0] +1):

                if j ==1:

                    self.EL[(i-1)*div[0]+j-1, 0] = (i-1)*(div[0]+1) + j
                    self.EL[(i-1)*div[0]+j-1, 1] = self.EL[(i-1)*div[0]+j-1, 0] + 1
                    self.EL[(i-1)*div[0]+j-1, 2] = self.EL[(i-1)*div[0]+j-1, 0] + (div[0]+2)
                    self.EL[(i-1)*div[0]+j-1, 3] = self.EL[(i-1)*div[0]+j-1, 0] + (div[0]+1)
                else:

                    self.EL[(i-1)*div[0]+j-1, 0] = self.EL[(i-1)*div[0]+j-2, 1]
                    self.EL[(i-1)*div[0]+j-1, 3] = self.EL[(i-1)*div[0]+j-2, 2]
                    self.EL[(i-1)*div[0]+j-1, 1] = self.EL[(i-1)*div[0]+j-2, 1] + 1
                    self.EL[(i-1)*div[0]+j-1, 2] = self.EL[(i-1)*div[0]+j-2, 2] + 1

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

    def defineBoundary(self, name, nodes_id):

        self.boundaries[name] = nodes_id

    def checkBoundaries(self):

        boundaries = []

        for node in self.NL:

            nodeOccurrences = 0

            for element in self.EL:

                nodeOccurrences += element.nodes.count(node)

            if nodeOccurrences < self.EpP:
                    
                boundaries.append(node.id)


        print(boundaries)

        return boundaries

    

class Node:

    def __init__(self, id=None, coor: tuple = None):
        self.id = id
        self.coor = coor
        self.type = 'DoF'
        self.variable = {}
        self.BC = {}


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
    
    def getNodesId(self):

        N = [node.id for node in self.nodes]
        return N

    def getNumberNodes(self):

        return len(self.nodes)
    
    def Jacobian(self):
        
        x_map = self.sF.N.transpose() * sp.Matrix(self.getCoor())
        return x_map.jacobian(sp.Matrix(list(x_map.free_symbols)))