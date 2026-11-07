"""Filling, packing and cooling of a plate in the Hele-Shaw approximation: POLYFORT FPP 30 GFC K1079 (Cross-WLF and two-domain Tait,
both fitted to the data of the material), the 100 x 60 x 2 mm plate with its 8 mm insert, mold at 50 °C.

The plate is filled in 1 s, then packed through the gate at 80 % of the pressure at the end of the filling for 4 s and left to cool
with the gate closed. The same filled state is also left to cool with no packing. The packing pushes in the melt that the cooling
shrinks away: the part is heavier and its volumetric shrinkage, 1 - v(1 bar, 25 °C) / v at freezing, is smaller and more even.

Run from the project root:  python -u Example_Packing_HeleShaw.py   (some 6 min: the filling takes most of it)
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
from Source.Physics.Rheology import CrossWLF, TaitPVT

Lx, Ly, h = 0.10, 0.06, 2e-3
xc, yc, rc = 0.05, 0.03, 0.008
SIZE = 2.5e-3
T_MELT, T_MOLD = 230.0, 50.0
RHO, CP, K, HTC = 970.0, 2000.0, 0.22, 5000.0
LAW = CrossWLF(n=0.2619, tau_star=1.778e4, D1=4.378e17, D2=266.7, A1=39.22, A2=51.6, gamma_min=1e-2)
TAIT = TaitPVT(1.006e-3, 4.936e-7, 7.989e7, 3.53e-3, 9.073e-4, 1.49e-7, 2.601e8, 2.353e-3, 453.9, 1.709e-7, 1.009e-4, 4.788e-2, 1.553e-8)
T_NOFLOW = TAIT.b['b5'] - 273.15                     # the transition of the Tait law at 0 Pa, 180.7 °C
FILL_TIME, T_PACK, T_COOL = 1.0, 4.0, 11.0

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


def model():
    fill = HeleShawFilling(mesh, h, LAW, T=T_MELT)
    fill.setThermal(T_MELT, T_MOLD, RHO, CP, K, htc=HTC, T_noflow=T_NOFLOW, cells=20)
    fill.setDensity(TAIT.density)
    fill.addGate(gate, volume / FILL_TIME)
    return fill


start = time.perf_counter()
filled = model()
filled.fill(progress=True, every=400)
p_fill = filled.history[-1][2]
print(f"filled in {filled.time:.3f} s, gate pressure {p_fill / 1e6:.2f} MPa ({time.perf_counter() - start:.0f} s)", flush=True)

far = np.argmin(np.hypot(filled.xy[:, 0] - 0.09, filled.xy[:, 1] - 0.03))     # a pressure sensor near the end of the flow
cases = {}
for name, t_pack in (('empaque a 80 %', T_PACK), ('sin empaque', 0.0)):
    fill = model()
    for attr in ('fill_factor', 'fill_time', 'p', 'T', 'time', 'history'):
        value = getattr(filled, attr)
        setattr(fill, attr, value.copy() if hasattr(value, 'copy') else value)
    trace = []
    fill.pack(0.8 * p_fill, t_pack, t_cool=T_PACK + T_COOL - t_pack, dt_max=0.05, progress=True, every=50,
              on_step=lambda f: trace.append((f.time, f.p[far])))
    cases[name] = (fill, np.array(fill.pack_history), np.array(trace))
    sh = fill.shrinkage()
    print(f"{name}: part mass {fill.mass.sum() * 1e3:.3f} g, frozen nodes {np.mean(~np.isnan(fill.v_frozen)):.0%}, "
          f"volumetric shrinkage {sh.min():.2%} to {sh.max():.2%} (mean {sh.mean():.2%}) ({time.perf_counter() - start:.0f} s)", flush=True)

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
for name, (fill, hist, trace) in cases.items():
    line, = axs[0].plot(hist[:, 0], hist[:, 2] / 1e6, label=f'{name}: entrada')
    axs[0].plot(trace[:, 0], trace[:, 1] / 1e6, '--', color=line.get_color(), label=f'{name}: x = 90 mm')
    axs[1].plot(hist[:, 0], hist[:, 1] * 1e3, color=line.get_color(), label=name)
axs[0].set_xlabel('t [s]'); axs[0].set_ylabel('presión [MPa]'); axs[0].legend(fontsize=8)
axs[1].set_xlabel('t [s]'); axs[1].set_ylabel('masa de la pieza [g]'); axs[1].legend(fontsize=8)
fig.savefig('empaque_historia.png', dpi=120)

tri = mtri.Triangulation(filled.xy[:, 0] * 1e3, filled.xy[:, 1] * 1e3, mesh.plot_triangles())
shrink = {name: 100 * c[0].shrinkage() for name, c in cases.items()}
levels = np.linspace(min(s.min() for s in shrink.values()), max(s.max() for s in shrink.values()), 21)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
for a, (name, s) in zip(axs, shrink.items()):
    c = a.tricontourf(tri, s, levels, cmap='viridis')
    a.set_title(name); a.set_aspect('equal'); a.set_xlabel('x [mm]')
fig.colorbar(c, ax=axs, label='contracción volumétrica [%]')
fig.savefig('empaque_contraccion.png', dpi=120)
print("figures: empaque_historia.png, empaque_contraccion.png")
