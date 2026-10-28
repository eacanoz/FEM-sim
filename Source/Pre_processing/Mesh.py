# Class Mesh

import numpy as np


from Source.enums import ElementType
from Source.Simulation.Discretization import Discretization


class Mesh:

    def __init__(self):

        self.NL = None
        self.EL = None

        self.boundaries = {}

    def Generate_Mesh(self, PD, L, NoE, MeshType):

        self.NoE = NoE
        self.meshType = MeshType
        self.PD = PD
        self.element_type = 'line2'  # reference element used by the discretization

        if PD == 1:
            if self.meshType == ElementType.line:
                self.EpP = 2

            elif self.meshType == ElementType.triangle:
                self.EpP = 3

        self.NL = []

        self.EL = []

        dx = L / NoE

        for i in range(0, self.NoE + 1):
            self.NL.append(Node(id=i, coor=(i * dx,)))

        for i in range(0, self.NoE):
            self.EL.append(Element(id=i, nodes=[self.NL[i], self.NL[i + 1]]))


    def generate2DMesh(self, dim: tuple, div: tuple, element_type):

        """Rectangular mesh of bilinear quadrilaterals (Q4).

        dim = (Lx, Ly) size of the domain, div = (nx, ny) number of elements in each direction.
        Node numbering runs along x first (as in 2DFEM.py); each element lists its nodes
        counter-clockwise: bottom-left, bottom-right, top-right, top-left.
        """

        Lx, Ly = dim
        nx, ny = div

        self.PD = 2
        self.meshType = element_type
        self.element_type = 'quad4'

        self.NL = []
        for j in range(ny + 1):
            for i in range(nx + 1):
                self.NL.append(Node(id=len(self.NL), coor=(i * Lx / nx, j * Ly / ny)))

        self.EL = []
        for j in range(ny):
            for i in range(nx):
                bl = j * (nx + 1) + i
                self.EL.append(Element(id=len(self.EL),
                                       nodes=[self.NL[bl], self.NL[bl + 1],
                                              self.NL[bl + nx + 2], self.NL[bl + nx + 1]]))

        self.NoN = len(self.NL)
        self.NoE = len(self.EL)


    def boundary_facets(self, selector=None):

        """Outer-boundary facets (points in 1D, edges in 2D) whose midpoint satisfies selector(*coords).

        Returns a list of node-id tuples, one per facet. selector(x) in 1D, selector(x, y) in 2D;
        None selects the whole boundary.
        """

        return Discretization(self).boundary_facet_nodes(selector)

    def boundary_edges(self, selector=None):

        """Same as boundary_facets (name used for 2D meshes)."""

        return self.boundary_facets(selector)

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

        """Records the shape-function type (the reference element is chosen by `element_type`)."""

        self.shape_function = shapeFunction

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
    def __init__(self, id: int =None, nodes: list = None):
        self.id = id
        self.nodes = nodes
    def getLength(self):
        if len(self.nodes) != 2:
            raise ValueError("Element must have exactly two nodes to compute length.")
        coor0 = self.nodes[0].coor
        coor1 = self.nodes[1].coor
        if len(coor0) != len(coor1):
            raise ValueError("Node coordinates must have the same dimension.")
        # Handle 1D and higher dimensions
        return float(np.linalg.norm(np.array(coor1) - np.array(coor0)))
        
        #return self.nodes[1].coor[0] - self.nodes[0].coor[0]

    def getCoor(self):

        """Coordinates of the nodes: scalars in 1D, (x, y) tuples in 2D."""

        if len(self.nodes[0].coor) == 1:
            return [node.coor[0] for node in self.nodes]

        return [tuple(node.coor) for node in self.nodes]
    
    def getNodesId(self):

        N = [node.id for node in self.nodes]
        return N

    def getNumberNodes(self):

        return len(self.nodes)
    


