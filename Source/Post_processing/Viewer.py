"""Viewer of results with pyvista (optional: `pip install pyvista`, or femsim[view]), for 3D meshes and surfaces; 2D meshes also have
the matplotlib views of the model (Model.postProcess2D).

    from femsim.Post_processing.Viewer import to_pyvista, show

    grid = to_pyvista(model.mesh, {'T': model.sol['T']})      # pyvista.UnstructuredGrid with the nodal fields
    show(model, 'T')                                           # interactive window
    show(model, 'u', screenshot='u.png')                       # no window: a PNG (off screen)
    show(model, 'T', step=-1, warp=None)                       # a step of a transient solution; warp: field that displaces the points
    show(filling, 'fill_time')                                 # a HeleShawFilling: any nodal array of it ('fill_time', 'p', 'fill_factor')

The field may be a scalar (n,) or a vector (n, k); a vector is drawn by its magnitude. Without pyvista the functions raise an error that
says how to install it; ParaView reads what Model.exportVTK writes in any case.
"""

import numpy as np

from Source.Post_processing.VTK import VTK_CELL, grid_arrays

INSTALL_HINT = "El visor usa pyvista: pip install pyvista  (o pip install femsim[view]). Mientras, Model.exportVTK escribe archivos para ParaView."


def _pyvista():

    try:
        import pyvista
    except ImportError:
        raise ImportError(INSTALL_HINT) from None

    return pyvista


def to_pyvista(mesh, point_data=None):

    """pyvista.UnstructuredGrid of a mesh (any element type of the VTK export) with nodal fields {name: (n,) or (n, k)}; vectors get
    three components."""

    pv = _pyvista()

    points, conn, element_type = grid_arrays(mesh)
    nel, nne = conn.shape
    cells = np.hstack([np.full((nel, 1), nne, dtype=np.int64), conn]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(nel, VTK_CELL[element_type], dtype=np.uint8), points)

    for name, values in (point_data or {}).items():
        values = np.asarray(values, dtype=float)
        if values.ndim == 2 and values.shape[1] < 3:
            padded = np.zeros((values.shape[0], 3))
            padded[:, :values.shape[1]] = values
            values = padded
        grid.point_data[name] = values

    return grid


def _field_values(source, name, step):

    """(mesh, values of the field) of a Model (solution fields), a HeleShawFilling (its nodal arrays) or a Mesh with a dict."""

    if hasattr(source, 'sol') and hasattr(source, 'mesh'):
        if source.sol is None:
            raise RuntimeError("El modelo no tiene solucion: llame a solve() antes de mostrarlo.")
        if name not in source.sol:
            raise KeyError(f"Campo {name!r} desconocido (disponibles: {list(source.sol)}).")
        values = np.asarray(source.sol[name], dtype=float)
        if source.timeVector is not None and values.shape[-1] == len(source.timeVector):
            values = values[..., -1 if step is None else step]
        return source.mesh, values

    if hasattr(source, 'fill_time') and hasattr(source, 'mesh'):                    # HeleShawFilling
        values = getattr(source, name, None)
        if values is None or np.ndim(values) != 1:
            raise KeyError(f"El llenado no tiene un campo nodal {name!r} (pruebe 'fill_time', 'p' o 'fill_factor').")
        return source.mesh, np.asarray(values, dtype=float)

    raise TypeError("show: se esperaba un Model resuelto o un HeleShawFilling.")


def show(source, field, step=None, warp=None, screenshot=None, off_screen=None, show_edges=False, cmap='viridis', title=None,
         window_size=(900, 650), cut='auto'):

    """Draws a nodal field of a solved model (or of a filling) with pyvista. step: index of a transient solution (default the last).
    warp: name of a vector field of the model that displaces the points (scaled to a tenth of the size of the part). screenshot: path of a
    PNG (implies off_screen). cut: a solid (3D cells) shows only its skin, so by default ('auto') half of it is removed with a plane
    through its centre normal to y and the field on the cut face is seen; 'x', 'y' or 'z' choose the normal, None or False show the
    whole part. Returns the pyvista plotter."""

    pv = _pyvista()

    mesh, values = _field_values(source, field, step)
    if values.ndim == 2:
        values = np.linalg.norm(values, axis=1)

    grid = to_pyvista(mesh, {field: values})

    if warp is not None:
        displacement = np.zeros((len(grid.points), 3))
        _, vector = _field_values(source, warp, step)
        displacement[:, :vector.shape[1]] = vector
        size = np.ptp(grid.points, axis=0).max()
        peak = max(np.abs(displacement).max(), 1e-300)
        grid.points = grid.points + displacement * (0.1 * size / peak)

    if off_screen is None:
        off_screen = screenshot is not None

    if cut == 'auto':
        cut = 'y' if _is_solid(mesh) else None
    if cut:
        if cut not in ('x', 'y', 'z'):
            raise ValueError(f"cut: 'auto', 'x', 'y', 'z' o None, no {cut!r}.")
        grid = grid.clip(normal=cut, origin=grid.center, invert=True)

    plotter = pv.Plotter(off_screen=off_screen, window_size=list(window_size))
    plotter.add_mesh(grid, scalars=field, cmap=cmap, show_edges=show_edges,
                     scalar_bar_args={'title': title or field, 'vertical': True, 'position_x': 0.88, 'position_y': 0.15,
                                      'height': 0.7, 'width': 0.06})
    plotter.add_axes()

    if mesh_is_planar(grid):
        plotter.view_xy()

    if screenshot is not None:
        plotter.show(screenshot=screenshot)
        plotter.close()
    elif not off_screen:
        plotter.show()

    return plotter


def _is_solid(mesh):

    from Source.Post_processing.VTK import grid_arrays as arrays

    return arrays(mesh)[2] in ('tet4', 'hex8', 'tet10', 'hex27')


def mesh_is_planar(grid):
    return bool(np.ptp(grid.points[:, 2]) < 1e-12 * max(np.ptp(grid.points, axis=0).max(), 1e-300))


def available():

    """True if pyvista can be imported."""

    try:
        _pyvista()
    except ImportError:
        return False
    return True
