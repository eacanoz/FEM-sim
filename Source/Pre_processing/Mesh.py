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
        self.regions = {}    # name -> element indices, from imported meshes

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

    def generate3DMesh(self, dim: tuple, div: tuple, element_type=ElementType.hexahedron, order=1, origin=(0.0, 0.0, 0.0)):

        """Box mesh of hexahedra (hex8; hex27 with order=2) or tetrahedra (tet4; tet10 with order=2).

        dim = (Lx, Ly, Lz) size of the box, div = (nx, ny, nz) number of cells in each direction, origin its lowest corner. Nodes run along x, then y,
        then z, on a grid of order * n + 1 points per direction. Each element lists its nodes in the order of gmsh (see
        Elements.py). Tetrahedra split every cell into the 6 tetrahedra around its main diagonal (Kuhn), which match across
        the cells; all of them have positive volume.
        """

        Lx, Ly, Lz = dim
        nx, ny, nz = div

        if order not in (1, 2):
            raise ValueError(f"Orden de malla no disponible: {order} (use 1 o 2).")
        if element_type not in (ElementType.hexahedron, ElementType.tetrahedron):
            raise ValueError("generate3DMesh necesita ElementType.hexahedron o ElementType.tetrahedron.")

        from Source.Pre_processing.Elements import ELEMENTS, TET10_EDGES

        tet = element_type == ElementType.tetrahedron
        self.PD = 3
        self.meshType = element_type
        self.element_type = {(False, 1): 'hex8', (False, 2): 'hex27', (True, 1): 'tet4', (True, 2): 'tet10'}[(tet, order)]

        m = order
        gx, gy, gz = m * nx + 1, m * ny + 1, m * nz + 1

        self.NL = []
        for k in range(gz):
            for j in range(gy):
                for i in range(gx):
                    self.NL.append(Node(id=len(self.NL), coor=(origin[0] + i * Lx / (gx - 1), origin[1] + j * Ly / (gy - 1),
                                                               origin[2] + k * Lz / (gz - 1))))

        def at(p):
            return self.NL[int(p[0]) + gx * (int(p[1]) + gy * int(p[2]))]

        # Kuhn split of the unit cell: vertices 0, e_a, e_a + e_b, (1, 1, 1) for each permutation (a, b, c) of the axes
        eye = np.eye(3, dtype=int)
        kuhn = []
        for a, b in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)):
            v = [np.zeros(3, dtype=int), eye[a], eye[a] + eye[b], np.ones(3, dtype=int)]
            if np.linalg.det(np.array(v[1:]) - v[0]) < 0:
                v[1], v[2] = v[2], v[1]
            kuhn.append(v)

        hex_offsets = ((ELEMENTS[self.element_type].nodes + 1.0) / 2.0 * m).astype(int) if not tet else None

        self.EL = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    origin = m * np.array([i, j, k])
                    if not tet:
                        self.EL.append(Element(id=len(self.EL), nodes=[at(origin + o) for o in hex_offsets]))
                        continue
                    for v in kuhn:
                        corners = [origin + m * c for c in v]
                        nodes = [at(c) for c in corners]
                        if order == 2:
                            nodes += [at((corners[a] + corners[b]) // 2) for a, b in TET10_EDGES]
                        self.EL.append(Element(id=len(self.EL), nodes=nodes))

        self.NoN = len(self.NL)
        self.NoE = len(self.EL)

    def boundary_facets(self, selector=None):

        """Outer-boundary facets (points in 1D, edges in 2D, faces in 3D) whose midpoint satisfies selector(*coords).

        Returns a list of node-id tuples, one per facet. selector(x) in 1D, selector(x, y) in 2D, selector(x, y, z) in 3D;
        None selects the whole boundary. A string selects a named group of an imported mesh
        (the physical groups of a gmsh file, see `read_msh`).
        """

        if isinstance(selector, str):
            if selector not in self.groups:
                raise KeyError(f"El grupo '{selector}' no existe en la malla (grupos: {sorted(self.groups)}).")
            return list(self.groups[selector])

        # Topology of the boundary, kept while the elements are the same (the coordinates may move: midpoints are recomputed)
        from Source.Pre_processing.Elements import ELEMENTS, element_type_of
        from Source.Simulation.Discretization import outer_facets, select_facets

        key = (id(self.EL), len(self.EL), element_type_of(self))
        cached = getattr(self, '_outer_facets', None)
        if cached is None or cached[0] != key:
            conn = np.array([e.getNodesId() for e in self.EL], dtype=int)
            cached = (key, outer_facets(conn, ELEMENTS[key[2]]))
            self._outer_facets = cached

        width = max(len(node.coor) for node in self.NL)        # all the coordinates: a surface in space passes (x, y, z)
        node_xy = np.array([tuple(node.coor) + (0.0,) * (width - len(node.coor)) for node in self.NL], dtype=float)
        if not np.any(node_xy[:, ELEMENTS[key[2]].dim:] != 0.0):
            node_xy = node_xy[:, :ELEMENTS[key[2]].dim]
        return select_facets(cached[1], node_xy, selector)

    # ----- meshes from gmsh -----

    @classmethod
    def read_msh(cls, path, surface=False):

        """Reads a gmsh mesh (.msh): 2D of triangles (T3, T6) or quadrilaterals (Q4, Q9), or 3D of tetrahedra (tet4, tet10) or
        hexahedra (hex8, hex27); a mesh with 3D elements is read as 3D.

        The physical groups of the dimension of the boundary (1 in 2D, 2 in 3D) become named boundary groups, available as
        `mesh.boundary_edges('name')` / `mesh.boundary_faces('name')` and in `mesh.groups`; the groups of the dimension of
        the mesh are kept in `mesh.regions` (name -> indices in mesh.EL, for Model.region_field). surface=True keeps the z of a 2D mesh that is a surface in space (the
        midplane of a folded part, for Hele-Shaw): see from_gmsh_model. Needs the gmsh package.
        """

        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        try:
            gmsh.open(str(path))
            return cls.from_gmsh_model(surface=surface)
        finally:
            gmsh.finalize()

    @classmethod
    def from_arrays(cls, coords, conn, element_type='tri3', regions=None):

        """Mesh from node coordinates (n, 2 or 3) and the element connectivity (nel, nne) in the node order of Elements.py:
        'tri3' or 'quad4' (2D elements; with three coordinates, a surface in space). regions: name -> element indices."""

        coords = np.asarray(coords, dtype=float)
        conn = np.asarray(conn, dtype=int)
        mesh = cls()
        mesh.PD = 2
        mesh.element_type = element_type
        mesh.meshType = {'tri': ElementType.triangle, 'qua': ElementType.quadrilateral}[element_type[:3]]
        mesh.NL = [Node(id=i, coor=tuple(float(c) for c in xyz)) for i, xyz in enumerate(coords)]
        mesh.EL = [Element(id=e, nodes=[mesh.NL[i] for i in row]) for e, row in enumerate(conn)]
        mesh.NoN, mesh.NoE = len(mesh.NL), len(mesh.EL)
        mesh.regions = {name: list(ids) for name, ids in (regions or {}).items()}
        return mesh

    @classmethod
    def from_gmsh_model(cls, surface=False):

        """Mesh from the gmsh model that is currently open (mesh it first with gmsh.model.mesh.generate(2), or generate(3) for a
        3D mesh; for quadratic elements, also gmsh.model.mesh.setOrder(2)). surface=True: a mesh of 2D elements keeps its three
        coordinates, a surface in space (the plates of a part joined at edges, T joints included); the Discretization then works with
        tangential gradients. The orientation of its elements is kept (on a folded surface there is no common sense of rotation)."""

        import gmsh

        mesh_dim = 3 if len(gmsh.model.mesh.getElements(3)[0]) else 2
        types, element_tags, node_lists = gmsh.model.mesh.getElements(mesh_dim)
        # gmsh type -> (element, nodes, vertices, the node order after reversing the orientation). The node order of gmsh is
        # the one of the reference elements (Elements.py); tetrahedra are reversed by swapping the vertices 1 and 2.
        if mesh_dim == 2:
            supported = {2: ('tri3', 3, 3, [2, 1, 0]), 3: ('quad4', 4, 4, [3, 2, 1, 0]),
                         9: ('tri6', 6, 3, [0, 2, 1, 5, 4, 3]), 10: ('quad9', 9, 4, [0, 3, 2, 1, 7, 6, 5, 4, 8])}
        else:
            supported = {4: ('tet4', 4, 4, [0, 2, 1, 3]), 11: ('tet10', 10, 4, [0, 2, 1, 3, 6, 5, 4, 7, 9, 8]),
                         5: ('hex8', 8, 8, None), 12: ('hex27', 27, 8, None)}

        if len(types) != 1 or types[0] not in supported:
            raise ValueError("La malla debe tener un solo tipo de elemento de su dimension (triangulos o cuadrilateros en 2D, "
                             "tetraedros o hexaedros en 3D).")

        element_type, nne, n_vertices, reversed_order = supported[types[0]]

        tags = np.array(node_lists[0], dtype=int).reshape(-1, nne)
        all_tags, all_xyz, _ = gmsh.model.mesh.getNodes()
        width = 3 if (surface and mesh_dim == 2) else mesh_dim
        coords = dict(zip((int(t) for t in all_tags), np.asarray(all_xyz).reshape(-1, 3)[:, :width]))

        used = sorted(set(tags.ravel().tolist()))
        index = {tag: i for i, tag in enumerate(used)}

        mesh = cls()
        mesh.PD = mesh_dim
        mesh.element_type = element_type
        mesh.meshType = {'tri': ElementType.triangle, 'qua': ElementType.quadrilateral, 'tet': ElementType.tetrahedron,
                         'hex': ElementType.hexahedron}[element_type[:3]]
        mesh.NL = [Node(id=i, coor=tuple(float(c) for c in coords[tag])) for i, tag in enumerate(used)]
        mesh.EL = []

        for row in tags:
            nodes = [mesh.NL[index[int(t)]] for t in row]
            xy = np.array([n.coor for n in nodes[:n_vertices]])
            if mesh_dim == 2 and width == 3:
                negative = False
            elif mesh_dim == 2:
                negative = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]) < 0
            else:
                negative = reversed_order is not None and np.linalg.det(xy[1:4] - xy[0]) < 0
            if negative:
                nodes = [nodes[k] for k in reversed_order]   # counter-clockwise (2D), positive volume (tetrahedra)
            mesh.EL.append(Element(id=len(mesh.EL), nodes=nodes))

        mesh.NoN, mesh.NoE = len(mesh.NL), len(mesh.EL)

        position = {int(tag): e for e, tag in enumerate(element_tags[0])}       # gmsh tag -> index in mesh.EL

        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag) or f"group_{dim}_{tag}"

            if mesh_dim == 3 and dim == 2:           # faces of the boundary, in any node order (the facets sort them)
                faces = []
                for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
                    etypes, _, enodes = gmsh.model.mesh.getElements(2, entity)
                    for et, nl in zip(etypes, enodes):
                        n = gmsh.model.mesh.getElementProperties(et)[3]
                        faces += [tuple(index[int(t)] for t in face) for face in np.asarray(nl).reshape(-1, n)]
                mesh.groups[name] = faces

            elif mesh_dim == 3 and dim == 3:
                ids = []
                for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
                    _, etags, _ = gmsh.model.mesh.getElements(3, entity)
                    ids += [position[int(t)] for chunk in etags for t in chunk]
                mesh.regions[name] = ids

            elif mesh_dim == 3:
                continue

            elif dim == 1:
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
                    ids += [position[int(t)] for chunk in etags for t in chunk]
                mesh.regions[name] = ids

        return mesh

    def boundary_edges(self, selector=None):

        """Same as boundary_facets (name used for 2D meshes)."""

        return self.boundary_facets(selector)

    def boundary_faces(self, selector=None):

        """Same as boundary_facets (name used for 3D meshes): selector(x, y, z) or the name of a group of faces."""

        return self.boundary_facets(selector)

    def plot_triangles(self):

        """Triangles of nodes that cover the mesh, to draw a nodal field: every node is used, so a T6 gives 4 and a Q9 gives 8. Q4 gives 2, T3 itself."""

        if getattr(self, 'PD', 2) == 3:
            raise ValueError("Los graficos de matplotlib son solo 2D: una malla 3D se ve en ParaView (Post_processing/VTK.py).")

        pattern = {3: [(0, 1, 2)], 4: [(0, 1, 2), (0, 2, 3)],
                   6: [(0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)],
                   9: [(0, 4, 8), (0, 8, 7), (4, 1, 5), (4, 5, 8), (7, 8, 6), (7, 6, 3), (8, 5, 2), (8, 2, 6)]}
        triangles = []
        for element in self.EL:
            ids = element.getNodesId()
            triangles += [[ids[i] for i in tri] for tri in pattern[len(ids)]]
        return np.array(triangles, dtype=int)

    def triangulation(self, scale=1.0):

        """matplotlib Triangulation of the nodes, for tricontourf and the like, on any 2D mesh. scale: multiplies the coordinates (1e3: mm)."""

        import matplotlib.tri as mtri

        xy = scale * np.array([node.coor[:2] for node in self.NL], dtype=float)
        return mtri.Triangulation(xy[:, 0], xy[:, 1], self.plot_triangles())

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
    


