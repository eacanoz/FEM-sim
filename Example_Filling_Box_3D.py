"""Non-isothermal filling of a 3D part on its midplane surface (Hele-Shaw on a surface in space): an open box 100 x 60 x 20 mm with an inner
rib, POLYFORT FPP 30 GFC K1079 (Cross-WLF fitted to its viscosity data), mold at 50 °C.

Gaps: base 2.0 mm, walls 1.5 mm, rib 1.0 mm. The rib joins the base along a line (a T joint: three elements share each edge there),
the walls join the base and each other at the edges of the box. The melt enters through the rim of one short wall (fill in 1 s).

Result (3582 triangles of 3 mm, 1850 nodes, about 29 min: one node per step, with temperature): gate pressure 24.3 MPa at the end, mean
frozen fraction 32.5 %; the base fills from 0.05 to 0.91 s, the rib from 0.26 to 0.89 s, and the last point is the rim of the wall opposite
the gate, at 1.00 s. No measurements to compare with: a demonstration of the surface model, not a validation.

Writes Filling_Box_3D.vtu (fill time, pressure, frozen fraction, gap) for ParaView and llenado_caja_3d.png.

Run from the project root:  python -u Example_Filling_Box_3D.py
"""

import time

import gmsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Source.Pre_processing.Mesh import Mesh
from Source.Physics.HeleShaw import HeleShawFilling
from Source.Physics.Rheology import CrossWLF
from Source.Post_processing.VTK import write_vtu

Lx, Ly, Lz = 0.10, 0.06, 0.02
RIB_Y, RIB_X0, RIB_X1, RIB_H = 0.03, 0.02, 0.08, 0.012
H_BASE, H_WALL, H_RIB = 2.0e-3, 1.5e-3, 1.0e-3
SIZE = 3e-3
T_MELT, T_MOLD = 230.0, 50.0
RHO, CP, K, HTC = 970.0, 2000.0, 0.22, 5000.0
T_NOFLOW = 453.9 - 273.15                              # b5 of the Tait fit of the same material
LAW = CrossWLF(n=0.2619, tau_star=1.778e4, D1=4.378e17, D2=266.7, A1=39.22, A2=51.6, gamma_min=1e-2)
FILL_TIME = 1.0

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
try:
    occ = gmsh.model.occ
    box = occ.addBox(0, 0, 0, Lx, Ly, Lz)
    occ.synchronize()
    faces = [tag for _, tag in gmsh.model.getEntities(2)]
    lid = [s for s in faces if abs(occ.getCenterOfMass(2, s)[2] - Lz) < 1e-12]
    occ.remove([(3, box)])
    occ.remove([(2, s) for s in lid])
    rib = occ.addRectangle(RIB_X0, 0.0, 0.0, RIB_X1 - RIB_X0, RIB_H)    # in the plane z = 0, then stood up at y = RIB_Y
    occ.rotate([(2, rib)], 0, 0, 0, 1, 0, 0, np.pi / 2)
    occ.translate([(2, rib)], 0, RIB_Y, 0)
    occ.synchronize()
    surfaces = [(2, tag) for _, tag in gmsh.model.getEntities(2)]
    occ.fragment(surfaces, [])                                          # conformal: the rib shares its nodes with the base
    occ.synchronize()
    gmsh.option.setNumber('Mesh.MeshSizeMax', SIZE)
    gmsh.model.mesh.generate(2)
    mesh = Mesh.from_gmsh_model(surface=True)
finally:
    gmsh.finalize()


def gap(x, y, z):
    on_rib = (np.abs(y - RIB_Y) < 1e-9) & (z > 1e-9) & (x > RIB_X0 - 1e-9) & (x < RIB_X1 + 1e-9)
    on_base = z < 1e-9
    return np.where(on_rib, H_RIB, np.where(on_base, H_BASE, H_WALL))


fill = HeleShawFilling(mesh, gap, LAW, T=T_MELT)
fill.setThermal(T_MELT, T_MOLD, RHO, CP, K, htc=HTC, T_noflow=T_NOFLOW, cells=20)
gate = mesh.boundary_facets(lambda x, y, z: x < 1e-9 and z > Lz - 1e-9 and abs(y - Ly / 2) < 0.01)
volume = float(fill.volume.sum())
fill.addGate(gate, volume / FILL_TIME)
print(f"{mesh.getNoE()} triangles, {mesh.getNoN()} nodes, cavity {volume * 1e6:.2f} cm3, gate on {len(gate)} edges", flush=True)

start = time.perf_counter()
fill.fill(progress=True, every=500)
print(f"gate pressure at the end {fill.history[-1][2] / 1e6:.2f} MPa, mean frozen fraction {fill.frozen_fraction().mean():.1%}"
      + (f", SHORT SHOT at {fill.short_shot:.2f} s" if fill.short_shot else "") + f" ({time.perf_counter() - start:.0f} s)")

xyz = fill.xy
on_rib = (np.abs(xyz[:, 1] - RIB_Y) < 1e-9) & (xyz[:, 2] > 1e-9) & (xyz[:, 0] > RIB_X0 - 1e-9) & (xyz[:, 0] < RIB_X1 + 1e-9)
print(f"rib: filled from {np.nanmin(fill.fill_time[on_rib]):.3f} to {np.nanmax(fill.fill_time[on_rib]):.3f} s; "
      f"last node of the part at {np.nanmax(fill.fill_time):.3f} s")

nodal_gap = gap(*xyz.T)
write_vtu('Filling_Box_3D.vtu', mesh, {'fill_time': np.nan_to_num(fill.fill_time, nan=-1.0), 'pressure': fill.p,
                                       'frozen_fraction': fill.frozen_fraction(), 'gap': nodal_gap})

def plot_fill_time(xyz, triangles, fill_time, path):

    """Fill time on the surface: each triangle with the colour of its mean fill time; the walls facing the viewer drawn faint so that the
    base and the rib show through the open top."""

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    t_elem = fill_time[triangles].mean(axis=1)
    norm = plt.Normalize(0.0, np.nanmax(fill_time))
    colours = plt.cm.viridis(norm(t_elem))
    centre = xyz[triangles].mean(axis=1)
    front = (centre[:, 1] < 1e-9) | (centre[:, 0] > Lx - 1e-9)               # the walls y = 0 and x = Lx face the viewer
    colours[front, 3] = 0.25
    fig = plt.figure(figsize=(9, 5.5), constrained_layout=True)
    ax = fig.add_subplot(projection='3d')
    ax.add_collection3d(Poly3DCollection(xyz[triangles] * 1e3, facecolors=colours, edgecolors=(0, 0, 0, 0.08), linewidths=0.2))
    ax.set_xlim(0, Lx * 1e3); ax.set_ylim(0, Ly * 1e3); ax.set_zlim(0, Lz * 1e3)
    ax.set_box_aspect((Lx, Ly, Lz * 1.5))
    ax.view_init(elev=30, azim=-55)
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_zlabel('z [mm]')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax, shrink=0.6, label='tiempo de llenado [s]')
    ax.set_title('Caja con nervio: tiempo de llenado (entrada en el borde de la pared x = 0)')
    fig.savefig(path, dpi=120)


triangles = np.array([e.getNodesId() for e in mesh.EL])
plot_fill_time(xyz, triangles, fill.fill_time, 'llenado_caja_3d.png')
print("wrote Filling_Box_3D.vtu and llenado_caja_3d.png")
