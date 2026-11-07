"""Non-isothermal filling of a plate in the Hele-Shaw approximation: POLYFORT FPP 30 GFC K1079 (PP + 30 % glass fibre, Cross-WLF fitted
to its viscosity data), a 100 x 60 x 2 mm plate with an 8 mm insert, gated on the left edge, mold at 50 °C.

The gate pressure against the fill time: a fast fill thins the melt and heats it by shear; a slow one lets the frozen layer grow and
narrow the channel. Between them the pressure goes through a minimum (the U-curve of injection molding), which an isothermal filling
cannot show: its pressure only falls as the fill slows down. Too slow a fill freezes before the end (a short shot).

About 1200 steps per run (one node per step), some 5 min per run with temperature.

Run from the project root:  python -u Example_Filling_NonIsothermal.py
"""

import time

import gmsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from Source.Pre_processing.Mesh import Mesh
from Source.Physics.HeleShaw import HeleShawFilling
from Source.Physics.Rheology import CrossWLF

Lx, Ly, h = 0.10, 0.06, 2e-3
xc, yc, rc = 0.05, 0.03, 0.008
SIZE = 2.5e-3
T_MELT, T_MOLD = 230.0, 50.0
RHO, CP, K, HTC = 970.0, 2000.0, 0.22, 5000.0        # melt density (Tait at 230 °C, 0 Pa), heat capacity, conductivity, mold contact
T_NOFLOW = 453.9 - 273.15                             # the transition temperature b5 of the Tait fit of the same material
LAW = CrossWLF(n=0.2619, tau_star=1.778e4, D1=4.378e17, D2=266.7, A1=39.22, A2=51.6, gamma_min=1e-2)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
try:
    occ = gmsh.model.occ
    occ.cut([(2, occ.addRectangle(0, 0, 0, Lx, Ly))], [(2, occ.addDisk(xc, yc, 0, rc, rc))])
    occ.synchronize()
    gmsh.option.setNumber('Mesh.MeshSizeMax', SIZE)
    gmsh.model.mesh.generate(2)
    mesh = Mesh.from_gmsh_model()
finally:
    gmsh.finalize()
gate = mesh.boundary_edges(lambda x, y: x < 1e-9 and abs(y - Ly / 2) < 0.005)
volume = Lx * Ly * h - np.pi * rc ** 2 * h


def run(fill_time, thermal):
    fill = HeleShawFilling(mesh, h, LAW, T=T_MELT)
    if thermal:
        fill.setThermal(T_MELT, T_MOLD, RHO, CP, K, htc=HTC, T_noflow=T_NOFLOW, cells=20)
    fill.addGate(gate, volume / fill_time)
    start = time.perf_counter()
    fill.fill(progress=True, every=400)
    print(f"   fill time {fill_time:.2f} s, {'with temperature' if thermal else 'isothermal'}: gate pressure {fill.history[-1][2] / 1e6:.2f} MPa"
          + (f", frozen fraction mean {fill.frozen_fraction().mean():.1%}" if thermal else "")
          + (f", SHORT SHOT at {fill.short_shot:.2f} s with {fill.melt_volume() / volume:.1%} of the cavity" if fill.short_shot else "")
          + f" ({time.perf_counter() - start:.0f} s)", flush=True)
    return fill


print(f"{len(mesh.NL)} nodes", flush=True)
times = [0.25, 0.5, 1.0, 2.0, 4.0]
results = {t: (run(t, False), run(t, True)) for t in times}

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
ax.plot(times, [results[t][0].history[-1][2] / 1e6 for t in times], 'o--', label='isotermo')
full = [t for t in times if results[t][1].short_shot is None]
ax.plot(full, [results[t][1].history[-1][2] / 1e6 for t in full], 's-', label='no isotermo (molde a 50 °C)')
for t in times:
    if results[t][1].short_shot is not None:
        ax.plot(t, results[t][1].history[-1][2] / 1e6, 'rx', ms=10, label=f'tiro corto ({results[t][1].melt_volume() / volume:.0%} lleno)')
ax.set_xscale('log'); ax.set_xlabel('tiempo de llenado [s]'); ax.set_ylabel('presión de entrada al final [MPa]'); ax.legend()
fig.savefig('llenado_curva_U.png', dpi=120)

fill = results[1.0][1]
xy = fill.xy * 1e3
tri = mtri.Triangulation(xy[:, 0], xy[:, 1], mesh.plot_triangles())
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
c = axs[0].tricontourf(tri, fill.mean_temperature(), 20, cmap='inferno')
fig.colorbar(c, ax=axs[0], label='T media en el espesor [°C]')
c = axs[1].tricontourf(tri, 100 * fill.frozen_fraction(), 20, cmap='Blues')
fig.colorbar(c, ax=axs[1], label='capa congelada [% del espesor]')
for a in axs:
    a.set_aspect('equal'); a.set_xlabel('x [mm]')
fig.savefig('llenado_temperatura.png', dpi=120)
print("figures: llenado_curva_U.png, llenado_temperatura.png")
