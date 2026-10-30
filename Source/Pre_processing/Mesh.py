# Class Mesh

import numpy as np


from Source.enums import ElementType
from Source.Simulation.Discretization import Discretization


class Mesh:

    def __init__(self):

        self.NL = None
        self.EL = None

        self.boundaries = {}
        self.groups = {}     # name -> boundary facets (node tuples), from imported meshes
        self.regions = {}    # name -> gmsh element tags, from imported meshes

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


    def generate2DMesh(self, dim: tuple, div: tuple, element_type, order=1):

        """Rectangular mesh of bilinear quadrilaterals (Q4), or of linear triangles (T3) with ElementType.triangle.
        order=2 gives biquadratic quadrilaterals (Q9) or quadratic triangles (T6): vertices first, then the midside
        nodes and, in Q9, the centre (the order of gmsh and VTK).

        dim = (Lx, Ly) size of the domain, div = (nx, ny) number of elements in each direction.
        Node numbering runs along x first (as in 2DFEM.py); each element lists its nodes
        counter-clockwise: bottom-left, bottom-right, top-right, top-left. Triangles split every
        cell along the bottom-left / top-right diagonal into (bl, br, tr) and (bl, tr, tl).
        """

        Lx, Ly = dim
        nx, ny = div

        self.PD = 2
        self.meshType = element_type

        if order == 2:
            self._generate_quadratic(Lx, Ly, nx, ny, element_type)
            return

        if order != 1:
            raise ValueError(f"Orden de malla no disponible: {order} (use 1 o 2).")

        self.element_type = 'tri3' if element_type == ElementType.triangle else 'quad4'

        self.NL = []
        for j in range(ny + 1):
            for i in range(nx + 1):
                self.NL.append(Node(id=len(self.NL), coor=(i * Lx / nx, j * Ly / ny)))

        self.EL = []
        for j in range(ny):
            for i in range(nx):
                bl = j * (nx + 1) + i
                corners = [self.NL[bl], self.NL[bl + 1], self.NL[bl + nx + 2], self.NL[bl + nx + 1]]

                if self.element_type == 'tri3':
                    self.EL.append(Element(id=len(self.EL), nodes=[corners[0], corners[1], corners[2]]))
                    self.EL.append(Element(id=len(self.EL), nodes=[corners[0], corners[2], corners[3]]))
                else:
                    self.EL.append(Element(id=len(self.EL), nodes=corners))

        self.NoN = len(self.NL)
        self.NoE = len(self.EL)


    def _generate_quadratic(self, Lx, Ly, nx, ny, element_type):

        self.element_type = 'tri6' if element_type == ElementType.triangle else 'quad9'

        self.NL = []
        for j in range(2 * ny + 1):
            for i in range(2 * nx + 1):
                self.NL.append(Node(id=len(self.NL), coor=(i * Lx / (2 * nx), j * Ly / (2 * ny))))

        self.EL = []
        for j in range(ny):
            for i in range(nx):

                def at(a, b):
                    return self.NL[(2 * j + b) * (2 * nx + 1) + 2 * i + a]

                if self.element_type == 'tri6':
                    self.EL.append(Element(id=len(self.EL), nodes=[at(0, 0), at(2, 0), at(2, 2), at(1, 0), at(2, 1), at(1, 1)]))
                    self.EL.append(Element(id=len(self.EL), nodes=[at(0, 0), at(2, 2), at(0, 2), at(1, 1), at(1, 2), at(0, 1)]))
                else:
                    self.EL.append(Element(id=len(self.EL), nodes=[at(0, 0), at(2, 0), at(2, 2), at(0, 2), at(1, 0), at(2, 1),
                                                                   at(1, 2), at(0, 1), at(1, 1)]))

        self.NoN = len(self.NL)
        self.NoE = len(self.EL)

    def boundary_facets(self, selector=None):

        """Outer-boundary facets (points in 1D, edges in 2D) whose midpoint satisfies selector(*coords).

        Returns a list of node-id tuples, one per facet. selector(x) in 1D, selector(x, y) in 2D;
        None selects the whole boundary. A string selects a named group of an imported mesh
        (the physical groups of a gmsh file, see `read_msh`).
        """

        if isinstance(selector, str):
            if selector not in self.groups:
                raise KeyError(f"El grupo '{selector}' no existe en la malla (grupos: {sorted(self.groups)}).")
            return list(self.groups[selector])

        return Discretization(self).boundary_facet_nodes(selector)

    # ----- meshes from gmsh -----

    @classmethod
    def read_msh(cls, path):

        """Reads a 2D gmsh mesh (.msh) made of triangles (T3, T6) or quadrilaterals (Q4, Q9).

        The physical groups of dimension 1 become named boundary groups, available as
        `mesh.boundary_edges('name')` and in `mesh.groups`; the 2D ones are kept in `mesh.regions`
        (name -> gmsh element tags). Needs the gmsh package.
        """

        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        try:
            gmsh.open(str(path))
            return cls.from_gmsh_model()
        finally:
            gmsh.finalize()

    @classmethod
    def from_gmsh_model(cls):

        """Mesh from the gmsh model that is currently open (mesh it first with gmsh.model.mesh.generate(2); for quadratic
        elements, also gmsh.model.mesh.setOrder(2))."""

        import gmsh

        types, _, node_lists = gmsh.model.mesh.getElements(2)
        # gmsh type -> (element, nodes, vertices, the node order after reversing the orientation)
        supported = {2: ('tri3', 3, 3, [2, 1, 0]), 3: ('quad4', 4, 4, [3, 2, 1, 0]),
                     9: ('tri6', 6, 3, [0, 2, 1, 5, 4, 3]), 10: ('quad9', 9, 4, [0, 3, 2, 1, 7, 6, 5, 4, 8])}

        if len(types) != 1 or types[0] not in supported:
            raise ValueError("La malla debe tener un solo tipo de elemento 2D (triangulos o cuadrilateros).")

        element_type, nne, n_vertices, reversed_order = supported[types[0]]

        tags = np.array(node_lists[0], dtype=int).reshape(-1, nne)
        all_tags, all_xyz, _ = gmsh.model.mesh.getNodes()
        coords = dict(zip((int(t) for t in all_tags), np.asarray(all_xyz).reshape(-1, 3)[:, :2]))

        used = sorted(set(tags.ravel().tolist()))
        index = {tag: i for i, tag in enumerate(used)}

        mesh = cls()
        mesh.PD = 2
        mesh.element_type = element_type
        mesh.meshType = ElementType.triangle if element_type.startswith('tri') else ElementType.quadrilateral
        mesh.NL = [Node(id=i, coor=tuple(float(c) for c in coords[tag])) for i, tag in enumerate(used)]
        mesh.EL = []

        for row in tags:
            nodes = [mesh.NL[index[int(t)]] for t in row]
            xy = np.array([n.coor for n in nodes[:n_vertices]])
            twice_area = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
            if twice_area < 0:
                nodes = [nodes[k] for k in reversed_order]   # counter-clockwise
            mesh.EL.append(Element(id=len(mesh.EL), nodes=nodes))

        mesh.NoN, mesh.NoE = len(mesh.NL), len(mesh.EL)

        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag) or f"group_{dim}_{tag}"

            if dim == 1:
                edges = []
                for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
                    etypes, _, enodes = gmsh.model.mesh.getElements(1, entity)
                    for et, nl in zip(etypes, enodes):
                        if et == 1:
                            edges += [tuple(index[int(t)] for t in pair) for pair in np.asarray(nl).reshape(-1, 2)]
                        elif et == 8:                # 3-node line: gmsh lists the ends first, the facets the middle
                            edges += [(index[int(a)], index[int(m)], index[int(b)])
                                      for a, b, m in np.asarray(nl).reshape(-1, 3)]
                mesh.groups[name] = edges

            elif dim == 2:
                ids = []
                for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
                    _, etags, _ = gmsh.model.mesh.getElements(2, entity)
                    ids += [int(t) for chunk in etags for t in chunk]
                mesh.regions[name] = ids

        return mesh

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
    


