"""Output of results for ParaView: VTK XML unstructured grids (.vtu) and time series (.pvd).

Written directly (XML, no extra dependencies): ASCII, or base64 binary (the default for 3D meshes, whose ASCII files are large). Any nodal field can be exported, scalar (n,) or vector (n, k) with k <= 3; a transient
solution gives one .vtu per exported step plus a .pvd collection that ParaView opens as an animation.

    files = write_vtu('out/plate.vtu', mesh, {'T': values})
    files = export_model(model, 'out/plate', every=5)          # steady: plate.vtu; transient: plate.pvd + plate_0000.vtu ...
"""

import base64
import os
from io import StringIO

import numpy as np

# VTK cell types
VTK_CELL = {'line2': 3, 'tri3': 5, 'quad4': 9, 'tri6': 22, 'quad9': 28, 'tet4': 10, 'hex8': 12, 'tet10': 24, 'hex27': 29}

# Quadratic 3D cells: VTK orders the edge (and face) nodes differently from gmsh (the order of Elements.py). Edges and faces of
# VTK by their vertices; the face centres of the triquadratic hexahedron are x-, x+, y-, y+, z-, z+.
_VTK_TET10 = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
_VTK_HEX27 = (((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)),
              ((0, 3, 7, 4), (1, 2, 6, 5), (0, 1, 5, 4), (3, 2, 6, 7), (0, 1, 2, 3), (4, 5, 6, 7)))


def vtk_node_order(element_type):

    """Permutation p with VTK node k = node p[k] of the element in the order of Elements.py; None if both orders agree."""

    from Source.Pre_processing.Elements import ELEMENTS, _find_node

    if element_type not in ('tet10', 'hex27'):
        return None

    nodes = ELEMENTS[element_type].nodes
    groups = [_VTK_TET10] if element_type == 'tet10' else list(_VTK_HEX27) + [((0, 1, 2, 3, 4, 5, 6, 7),)]
    n_vertices = 4 if element_type == 'tet10' else 8
    order = list(range(n_vertices))
    for group in groups:
        order += [_find_node(nodes, nodes[list(v)].mean(axis=0)) for v in group]
    return np.array(order)


def _array_xml(name, values, dtype, components=1, binary=False):

    values = np.asarray(values)
    vtk_type = {'int64': 'Int64', 'uint8': 'UInt8', 'float64': 'Float64'}[np.dtype(dtype).name]

    if binary:
        raw = np.ascontiguousarray(values, dtype=np.dtype(dtype).newbyteorder('<')).tobytes()
        data = base64.b64encode(np.uint32(len(raw)).tobytes() + raw).decode('ascii')     # UInt32 header with the size
        return (f'<DataArray type="{vtk_type}" Name="{name}" NumberOfComponents="{components}" format="binary">\n'
                f'{data}\n</DataArray>\n')

    buffer = StringIO()

    fmt = '%d' if np.issubdtype(dtype, np.integer) else '%.9g'
    np.savetxt(buffer, values.reshape(-1, components) if values.ndim > 1 else values.reshape(-1, 1), fmt=fmt)

    return (f'<DataArray type="{vtk_type}" Name="{name}" NumberOfComponents="{components}" format="ascii">\n'
            f'{buffer.getvalue()}</DataArray>\n')


def mesh_dimension(mesh):

    """Number of coordinates of the nodes of a mesh (3 for 3D meshes and surfaces in space)."""

    return mesh.coords.shape[1]


def grid_arrays(mesh):

    """(points (n, 3), connectivity (nel, nne) in VTK node order, element type name) of a mesh: what a VTK grid is made of (write_vtu,
    and Source/Post_processing/Viewer.py for pyvista)."""

    from Source.Pre_processing.Elements import element_type_of

    element_type = element_type_of(mesh)
    if element_type not in VTK_CELL:
        raise ValueError(f"Tipo de elemento sin equivalente VTK: {element_type}")

    coords = mesh.coords
    points = np.zeros((len(coords), 3))
    points[:, :coords.shape[1]] = coords

    conn = np.array(mesh.conn)
    order = vtk_node_order(element_type)
    if order is not None:
        conn = conn[:, order]

    return points, conn, element_type


def write_vtu(path, mesh, point_data=None, binary=None):

    """Writes the mesh and the nodal fields {name: (n_nodes,)} to a .vtu file. Returns the path.

    binary: base64-encoded arrays instead of ASCII text; None chooses binary for 3D meshes only."""

    points, conn, element_type = grid_arrays(mesh)
    nel, nne = conn.shape
    if binary is None:
        binary = mesh_dimension(mesh) == 3

    def array(name, values, dtype, components=1):
        return _array_xml(name, values, dtype, components, binary)

    path = str(path)
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n<UnstructuredGrid>\n')
        f.write(f'<Piece NumberOfPoints="{len(points)}" NumberOfCells="{nel}">\n')

        f.write('<PointData>\n')
        for name, values in (point_data or {}).items():
            values = np.asarray(values, dtype=float)
            if values.ndim == 2 and values.shape[0] == len(points) and values.shape[1] <= 3:
                vector = np.zeros((len(points), 3))     # ParaView expects 3 components for a vector
                vector[:, :values.shape[1]] = values
                f.write(array(name, vector, np.float64, 3))
                continue
            if values.ndim == 2 and values.shape[0] == len(points):
                for c in range(values.shape[1]):            # more components than a vector: one array each
                    f.write(array(f'{name}_{c}', values[:, c], np.float64))
                continue
            if values.shape != (len(points),):
                raise ValueError(f"El campo '{name}' tiene forma {values.shape}; se esperaba ({len(points)},) "
                                 f"o ({len(points)}, k) con k <= 3.")
            f.write(array(name, values, np.float64))
        f.write('</PointData>\n')

        f.write('<Points>\n' + array('Points', points, np.float64, 3) + '</Points>\n')

        f.write('<Cells>\n')
        f.write(array('connectivity', conn, np.int64))
        f.write(array('offsets', np.arange(1, nel + 1) * nne, np.int64))
        f.write(array('types', np.full(nel, VTK_CELL[element_type]), np.uint8))
        f.write('</Cells>\n')

        f.write('</Piece>\n</UnstructuredGrid>\n</VTKFile>\n')

    return path


def write_pvd(path, entries):

    """Collection of (time, file) pairs; file paths are stored relative to the .pvd."""

    base = os.path.dirname(os.path.abspath(path))

    with open(path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n<Collection>\n')
        for time, file in entries:
            f.write(f'<DataSet timestep="{time:.12g}" part="0" file="{os.path.relpath(file, base).replace(os.sep, "/")}"/>\n')
        f.write('</Collection>\n</VTKFile>\n')

    return path


def export_model(model, base, every=1, fields=None):

    """Exports the solution of a solved model. Returns the list of files written.

    base: path without extension. Steady solutions give base.vtu; transient ones give base_0000.vtu, ...
    (the first, every `every`-th and the last step) and base.pvd. fields: names to export (default: all).
    """

    if model.sol is None:
        raise RuntimeError("El modelo no tiene solucion: llame a solve() antes de exportar.")

    names = list(fields) if fields else list(model.sol)
    unknown = [n for n in names if n not in model.sol]
    if unknown:
        raise KeyError(f"Campos desconocidos: {unknown} (disponibles: {list(model.sol)}).")

    if model.timeVector is None:
        return [write_vtu(f"{base}.vtu", model.mesh, {n: np.asarray(model.sol[n]) for n in names})]

    n_steps = len(model.timeVector)
    steps = sorted(set(range(0, n_steps, max(1, int(every)))) | {n_steps - 1})
    times = np.asarray(model.timeVector)

    files, entries = [], []
    for k, step in enumerate(steps):
        file = write_vtu(f"{base}_{k:04d}.vtu", model.mesh, {n: np.asarray(model.sol[n])[..., step] for n in names})
        files.append(file)
        entries.append((times[step], file))

    files.append(write_pvd(f"{base}.pvd", entries))

    return files
