"""Time of the non-isothermal Hele-Shaw filling of a 3D part on its midplane as the mesh grows, node by node and in batches
(fill(batch=...); docs/PLAN_INYECCION.md, phase 0). The part of Ejemplo_Llenado_Pieza_3D.ipynb: an open box 120 x 80 x 25 mm with two
ribs and a hole, pin gate at the centre of the base, POLYFORT Cross-WLF, melt 230 °C, mold 50 °C, filled in 1 s, 12 cells across the gap.

Run from the project root:  python -u Test/manual/hele_shaw_scaling.py size_mm[,size_mm...] batch[,batch...]
    e.g.  python -u Test/manual/hele_shaw_scaling.py 8,5,3.5,2.5,1.75,1.25 0.1
Prints per run: nodes, steps, seconds, gate pressure at the end, mean frozen fraction; and the peak memory of the process.
"""

import os
import sys
import time

import gmsh
import numpy as np
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.Physics.HeleShaw import HeleShawFilling
from Source.Physics.Rheology import CrossWLF

Lx, Ly, Lz = 0.12, 0.08, 0.025
GATE, HOLE = (0.06, 0.04, 0.002), (0.03, 0.04, 0.006)
RIBS, RIB_X0, RIB_X1, RIB_H = (0.027, 0.053), 0.02, 0.10, 0.012
H_BASE, H_WALL, H_RIB = 2.0e-3, 1.6e-3, 1.2e-3
LAW = CrossWLF(n=0.2619, tau_star=1.778e4, D1=4.378e17, D2=266.7, A1=39.22, A2=51.6, gamma_min=1e-2)


def part_mesh(size):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        occ = gmsh.model.occ
        box = occ.addBox(0, 0, 0, Lx, Ly, Lz)
        occ.synchronize()
        faces = [t for _, t in gmsh.model.getEntities(2)]
        lid = [s for s in faces if abs(occ.getCenterOfMass(2, s)[2] - Lz) < 1e-12]
        base = [s for s in faces if abs(occ.getCenterOfMass(2, s)[2]) < 1e-12][0]
        occ.remove([(3, box)])
        occ.remove([(2, s) for s in lid])
        holes = [occ.addDisk(x, y, 0, r, r) for x, y, r in (GATE, HOLE)]
        occ.cut([(2, base)], [(2, h) for h in holes])
        for y in RIBS:
            rib = occ.addRectangle(RIB_X0, 0.0, 0.0, RIB_X1 - RIB_X0, RIB_H)
            occ.rotate([(2, rib)], 0, 0, 0, 1, 0, 0, np.pi / 2)
            occ.translate([(2, rib)], 0, y, 0)
        occ.synchronize()
        occ.fragment([(2, t) for _, t in gmsh.model.getEntities(2)], [])
        occ.synchronize()
        gmsh.option.setNumber('Mesh.MeshSizeMax', size)
        gmsh.option.setNumber('Mesh.MeshSizeMin', size / 3)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 16)
        gmsh.model.mesh.generate(2)
        return Mesh.from_gmsh_model(surface=True)
    finally:
        gmsh.finalize()


def gap(x, y, z):
    on_rib = np.zeros(np.shape(x), dtype=bool)
    for yr in RIBS:
        on_rib |= (np.abs(y - yr) < 1e-9) & (z > 1e-9)
    return np.where(on_rib, H_RIB, np.where(z < 1e-9, H_BASE, H_WALL))


if __name__ == "__main__":
    sizes = [float(s) * 1e-3 for s in sys.argv[1].split(',')]
    batches = [float(b) for b in sys.argv[2].split(',')]
    process = psutil.Process()
    for size in sizes:
        mesh = part_mesh(size)
        gate = mesh.boundary_facets(lambda x, y, z: z < 1e-9 and np.hypot(x - GATE[0], y - GATE[1]) < GATE[2] + 1e-6)
        for batch in batches:
            fill = HeleShawFilling(mesh, gap, LAW, T=230.0)
            fill.setThermal(230.0, 50.0, 970.0, 2000.0, 0.22, htc=5000.0, T_noflow=453.9 - 273.15, cells=12)
            fill.addGate(gate, float(fill.volume.sum()) / 1.0)
            start = time.perf_counter()
            fill.fill(progress=True, every=1000, batch=batch)
            print(f"size {size * 1e3:.2f} mm, batch {batch}: {mesh.getNoN()} nodes, {len(fill.history)} steps, "
                  f"{time.perf_counter() - start:.0f} s, gate pressure {fill.history[-1][2] / 1e6:.3f} MPa, frozen "
                  f"{fill.frozen_fraction().mean():.4f}, peak memory {process.memory_info().peak_wset / 2**30:.2f} GB", flush=True)
