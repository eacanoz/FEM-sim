"""Arrays cached by the mesh (Mesh.coords, Mesh.conn): equal to what the node and element objects hold, built once, read-only, and rebuilt
after any change of a coordinate, a node or an element, however it is made.

Run from the project root:  python Test/test_mesh_arrays.py
"""

import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh, Node, Element
from Source.enums import ElementType


def quads(n=6):
    mesh = Mesh()
    mesh.generate2DMesh((1.0, 2.0), (n, n), ElementType.quadrilateral)
    return mesh


def test_arrays_equal_the_objects_and_are_built_once():
    for mesh in (quads(), quads(5)):
        coords, conn = mesh.coords, mesh.conn
        assert np.array_equal(coords, np.array([n.coor for n in mesh.NL])) and coords.dtype == float
        assert np.array_equal(conn, np.array([e.getNodesId() for e in mesh.EL])) and conn.dtype == np.int64
        assert mesh.coords is coords and mesh.conn is conn                   # no rebuild without a change
        for array in (coords, conn):
            assert not array.flags.writeable
            try:
                array[0] = 0
            except ValueError:
                pass
            else:
                raise AssertionError("a cached array was modified")

    solid = Mesh()
    solid.generate3DMesh((1.0, 1.0, 1.0), (3, 3, 3), ElementType.hexahedron)
    assert solid.coords.shape == (64, 3) and solid.conn.shape == (27, 8)

    line = Mesh()
    line.Generate_Mesh(1, 1.0, 4, ElementType.line)
    assert line.coords.shape == (5, 1) and line.conn.shape == (4, 2)


def test_every_kind_of_change_rebuilds_them():
    mesh = quads()
    before = mesh.coords

    mesh.NL[3].coor = (9.0, 9.0)                                            # a script moving a node
    assert mesh.coords is not before and tuple(mesh.coords[3]) == (9.0, 9.0)

    again = mesh.coords
    xy = np.array(mesh.coords) + 1.0
    from Source.core import Model
    from Source.Material import material
    from Source.Physics.HeatTransfer import ht
    model = Model(name='m', dim=2, mesh=mesh, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
    model.moveNodes(xy)
    assert mesh.coords is not again and np.allclose(mesh.coords, xy)

    conn = mesh.conn
    mesh.EL[0].nodes = [mesh.NL[i] for i in (1, 0, 5, 6)]                   # a new connectivity
    assert mesh.conn is not conn and list(mesh.conn[0]) == [1, 0, 5, 6]

    count = len(mesh.NL)
    mesh.NL.append(Node(id=count, coor=(5.0, 5.0)))                         # a node added
    assert mesh.coords.shape[0] == count + 1

    mesh.EL.append(Element(id=len(mesh.EL), nodes=[mesh.NL[i] for i in (0, 1, 2, count)]))
    assert mesh.conn.shape[0] == len(mesh.EL)

    other = quads(3)                                                        # another mesh was made: this one is rebuilt, not wrong
    assert np.array_equal(mesh.coords, np.array([n.coor for n in mesh.NL]))
    assert other.coords.shape == (16, 2)


def test_ragged_coordinates_are_padded():
    mesh = Mesh()
    mesh.NL = [Node(id=0, coor=(0.0, 0.0)), Node(id=1, coor=(1.0, 0.0, 0.5)), Node(id=2, coor=(0.0, 1.0))]
    mesh.EL = [Element(id=0, nodes=list(mesh.NL))]
    assert np.array_equal(mesh.coords, [[0, 0, 0], [1, 0, 0.5], [0, 1, 0]])


def test_discretization_and_boundaries_use_them_unchanged():
    from Source.Simulation.Discretization import Discretization

    mesh = quads()
    disc = Discretization(mesh)
    assert np.array_equal(disc.conn, mesh.conn) and np.array_equal(disc.node_xy, mesh.coords)
    disc.conn[0, 0] = 99                                                    # the discretization has its own copy
    assert mesh.conn[0, 0] == 0

    left = mesh.boundary_edges(lambda x, y: x < 1e-12)
    assert len(left) == 6 and all(mesh.coords[a, 0] == 0.0 and mesh.coords[b, 0] == 0.0 for a, b in left)
    mesh.NL[0].coor = (0.5, 0.0)                                            # moved: the selection follows the new coordinates
    assert len(mesh.boundary_edges(lambda x, y: x < 1e-12)) == 5          # the edge of node 0 no longer lies on x = 0


if __name__ == "__main__":
    tests = [test_arrays_equal_the_objects_and_are_built_once, test_every_kind_of_change_rebuilds_them,
             test_ragged_coordinates_are_padded, test_discretization_and_boundaries_use_them_unchanged]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS ", t.__name__)
        except Exception as e:
            traceback.print_exc()
            failed += 1
            print("FAIL ", t.__name__, repr(e))
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
