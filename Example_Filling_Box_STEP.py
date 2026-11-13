"""Non-isothermal filling of a thin-walled solid read from a STEP file, on its two skins (dual domain): the open box with a rib of
Example_Filling_Box_3D.py, but as the solid itself instead of its midplane. The solid is built here with gmsh (OpenCASCADE) so that its
midplane is exactly the surface of that example (base 2 mm, walls 1.5 mm, rib 1 mm thick), written to caja_3d.step and read back with
DualDomain.from_step: skins, edges, thickness by rays and the links across the walls come from the solid alone.

POLYFORT FPP 30 GFC K1079, melt 230 °C, mold 50 °C, filled in 1 s through the rim of one short wall, in batches of 10 % of the front.
Result (3 mm, 7296 skin triangles, 3786 nodes; 12 skin faces, 4 edge faces: the rim of the walls and the top and ends of the rib;
314 nodes without a link, along the corners of the walls and the root of the rib; about 4 min): cavity 22.23 cm3 (midplane 22.32),
gate pressure at the end 23.62 MPa (midplane 24.34, -3 %), mean frozen fraction 31.9 % (32.5 %), the base fills from 0.049 to 0.925 s
(0.05 to 0.91), the rib from 0.257 to 0.875 s (0.258 to 0.888), the last node at 0.998 s (0.998). The two models differ at the corners
(the midplane overlaps the walls there, the solid does not) and by the half wall of each skin; no measurements of the part.

Prints the comparison with the midplane of Example_Filling_Box_3D.py (its numbers in its docstring) and writes Filling_Box_STEP.vtu
(fill time, pressure, frozen fraction and the nodal mean of the thickness on the skins) for ParaView.

Run from the project root:  python -u Example_Filling_Box_STEP.py
"""

import time

import gmsh
import numpy as np

from Source.Pre_processing.DualDomain import DualDomain
from Source.Physics.HeleShaw import HeleShawFilling
from Source.Physics.Rheology import CrossWLF
from Source.Post_processing.VTK import write_vtu

# the midplane of Example_Filling_Box_3D.py, in mm, and the thicknesses
Lx, Ly, Lz = 100.0, 60.0, 20.0
RIB_Y, RIB_X0, RIB_X1, RIB_H = 30.0, 20.0, 80.0, 12.0
H_BASE, H_WALL, H_RIB = 2.0, 1.5, 1.0
SIZE = 3.0
T_MELT, T_MOLD = 230.0, 50.0
RHO, CP, K, HTC = 970.0, 2000.0, 0.22, 5000.0
T_NOFLOW = 453.9 - 273.15
LAW = CrossWLF(n=0.2619, tau_star=1.778e4, D1=4.378e17, D2=266.7, A1=39.22, A2=51.6, gamma_min=1e-2)
FILL_TIME = 1.0
STEP = "caja_3d.step"

# the solid: an outer box minus the inside (the walls centred on the midplane, the base below z = 0 by half its thickness, the walls up
# to z = Lz), and the rib on the base
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
try:
    occ = gmsh.model.occ
    w, b = H_WALL / 2, H_BASE / 2
    outer = occ.addBox(-w, -w, -b, Lx + 2 * w, Ly + 2 * w, Lz + b)
    inner = occ.addBox(w, w, b, Lx - 2 * w, Ly - 2 * w, Lz)
    box, _ = occ.cut([(3, outer)], [(3, inner)])
    rib = occ.addBox(RIB_X0, RIB_Y - H_RIB / 2, b, RIB_X1 - RIB_X0, H_RIB, RIB_H - b)
    occ.fuse(box, [(3, rib)])
    occ.synchronize()
    gmsh.write(STEP)
finally:
    gmsh.finalize()

start = time.perf_counter()
dual = DualDomain.from_step(STEP, size=SIZE, curvature=0, scale=1e-3)
print(f"{STEP}: {len(dual.conn)} skin triangles, {len(dual.xyz)} nodes; {len(dual.skins)} skin faces, {len(dual.edges)} edge faces; "
      f"thickness {dual.thickness.min() * 1e3:.2f} to {dual.thickness.max() * 1e3:.2f} mm ({np.sum(~dual.measured)} triangles from their "
      f"neighbours); {len(dual.link_source)} links, {len(dual.unjoined)} nodes without one ({time.perf_counter() - start:.1f} s)", flush=True)

fill = HeleShawFilling.from_dual_domain(dual, LAW, T=T_MELT)
fill.setThermal(T_MELT, T_MOLD, RHO, CP, K, htc=HTC, T_noflow=T_NOFLOW, cells=20)
gate = dual.mesh.boundary_facets(lambda x, y, z: z > Lz * 1e-3 - 1e-6 and x < 1e-3 and abs(y - Ly / 2 * 1e-3) < 0.01)
volume = float(fill.volume.sum())
fill.addGate(gate, volume / FILL_TIME)
print(f"cavity {volume * 1e6:.2f} cm3 (midplane 22.32 cm3), gate on {len(gate)} edges", flush=True)

start = time.perf_counter()
fill.fill(progress=True, every=100, batch=0.1)
print(f"gate pressure at the end {fill.history[-1][2] / 1e6:.2f} MPa (midplane 24.34), mean frozen fraction {fill.frozen_fraction().mean():.1%} "
      f"(midplane 32.5 %)" + (f", SHORT SHOT at {fill.short_shot:.2f} s" if fill.short_shot else "") + f" ({time.perf_counter() - start:.0f} s)")
xyz = fill.xy
rib = ((np.abs(xyz[:, 1] - RIB_Y * 1e-3) <= H_RIB / 2 * 1e-3 + 1e-9) & (xyz[:, 2] > H_BASE / 2 * 1e-3 + 1e-9)
       & (xyz[:, 0] > RIB_X0 * 1e-3 - 1e-9) & (xyz[:, 0] < RIB_X1 * 1e-3 + 1e-9))
base = xyz[:, 2] < 1e-9 + H_BASE / 2 * 1e-3
print(f"base: filled from {np.nanmin(fill.fill_time[base & (fill.fill_time > 0)]):.3f} to {np.nanmax(fill.fill_time[base]):.3f} s (midplane 0.05 to 0.91); "
      f"rib: {np.nanmin(fill.fill_time[rib]):.3f} to {np.nanmax(fill.fill_time[rib]):.3f} s (midplane 0.258 to 0.888); last node at "
      f"{np.nanmax(fill.fill_time):.3f} s (midplane 0.998)")

node_h = np.zeros(len(xyz))
np.add.at(node_h, dual.conn.ravel(), np.repeat(dual.thickness, 3))
node_h /= np.bincount(dual.conn.ravel(), minlength=len(xyz))
write_vtu('Filling_Box_STEP.vtu', dual.mesh, {'fill_time': np.nan_to_num(fill.fill_time, nan=-1.0), 'pressure': fill.p,
                                             'frozen_fraction': fill.frozen_fraction(), 'thickness': node_h}, binary=True)
print("wrote Filling_Box_STEP.vtu")
