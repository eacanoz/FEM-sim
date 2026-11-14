"""Filling, packing and cooling of a plate with crystallization, in the Hele-Shaw approximation: POLYFORT FPP 30 GFC K1079 (Cross-WLF and
two-domain Tait fitted to the data of the material), the 100 x 60 x 2 mm plate with its 8 mm insert, mold at 50 °C, filled in 1 s to
the switchover at 98 %, packed at 80 % of that pressure for 4 s and cooled to 16 s.

Two models of the solidification:
  - transition: the Tait law with its transition temperature, frozen below it (180.7 °C), no latent heat (the model of the stage 3);
  - crystallization: Nakamura / Hoffman-Lauritzen kinetics, latent heat, no flow beyond X = 0.2, and the density following X,
    (1 - X) v_melt + X v_solid.

The kinetics is NOT that of this grade (no DSC data of it here): it is the Flash-DSC fit at 10 K/s of another iPP, Sabic PP505P
(Ferrer Fabon et al., PAMM 22, e202200193, 2022), and the glass fibre of this grade speeds crystallization up (nucleation). The latent
heat is that of the polymer, 0.49 x 1.9e5 J/kg, times its 70 % mass fraction. The example shows what the model does, not this part.

Run from the project root:  python -u Example_Crystallization_HeleShaw.py   (some 15 min: two fillings)
"""

import time

import gmsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from Source.Pre_processing.Mesh import Mesh
from Source.Physics.Crystallization import NakamuraHoffman
from Source.Physics.HeleShaw import HeleShawFilling
from Source.Physics.Rheology import CrossWLF, TaitPVT

Lx, Ly, h = 0.10, 0.06, 2e-3
xc, yc, rc = 0.05, 0.03, 0.008
SIZE = 2.5e-3
T_MELT, T_MOLD = 230.0, 50.0
RHO, CP, K, HTC = 970.0, 2000.0, 0.22, 5000.0
LAW = CrossWLF(n=0.2619, tau_star=1.778e4, D1=4.378e17, D2=266.7, A1=39.22, A2=51.6, gamma_min=1e-2)
TAIT = TaitPVT(1.006e-3, 4.936e-7, 7.989e7, 3.53e-3, 9.073e-4, 1.49e-7, 2.601e8, 2.353e-3, 453.9, 1.709e-7, 1.009e-4, 4.788e-2, 1.553e-8)
KINETICS = NakamuraHoffman(33.6, 0.58e6, 17200.0, 231.15, 467.15, 3.0)       # iPP PP505P, Flash-DSC 10 K/s (see above)
LATENT = 0.7 * 0.49 * 1.9e5                                                    # J/kg of the filled grade
FILL_TIME, T_PACK, T_END = 1.0, 4.0, 16.0

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


def model(crystallizing):
    fill = HeleShawFilling(mesh, h, LAW, T=T_MELT)
    if crystallizing:
        fill.setThermal(T_MELT, T_MOLD, RHO, CP, K, htc=HTC, cells=20)
        fill.setCrystallization(KINETICS, LATENT, X_noflow=0.2, pvt=TAIT)
    else:
        fill.setThermal(T_MELT, T_MOLD, RHO, CP, K, htc=HTC, T_noflow=TAIT.b['b5'] - 273.15, cells=20)
        fill.setDensity(TAIT.density)
    fill.addGate(gate, volume / FILL_TIME)
    return fill


start = time.perf_counter()
far = None
cases = {}
for name, crystallizing in (('transición del Tait', False), ('cristalización', True)):
    fill = model(crystallizing)
    fill.fill(progress=True, every=600, switchover=0.98)
    p_switch = fill.history[-1][2]
    if far is None:
        far = np.argmin(np.hypot(fill.xy[:, 0] - 0.09, fill.xy[:, 1] - 0.03))
    trace = []
    fill.pack(0.8 * p_switch, T_PACK, t_cool=T_END - fill.time - T_PACK, dt_max=0.05, progress=True, every=100,
              on_step=lambda f: trace.append((f.time, f.p[far], f.mean_temperature()[far], (f.crystallinity() @ f.thermal['dz'])[far])))
    sh = fill.shrinkage()
    cases[name] = (fill, np.array(fill.pack_history), np.array(trace))
    X = fill.crystallinity() @ fill.thermal['dz']
    print(f"{name}: switchover {p_switch / 1e6:.2f} MPa, part mass {fill.mass.sum() * 1e3:.3f} g, mean crystallinity {X.mean():.2f} "
          f"(skin cell {fill.crystallinity()[:, -1].mean():.2f}, midplane {fill.crystallinity()[:, 0].mean():.2f}), volumetric shrinkage "
          f"{np.nanmin(sh):.2%} to {np.nanmax(sh):.2%} (mean {np.nanmean(sh):.2%}) ({time.perf_counter() - start:.0f} s)", flush=True)

crys = cases['cristalización'][0]
z = 0.5 * (crys.thermal['edges'][1:] + crys.thermal['edges'][:-1])
fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
for name, (fill, hist, trace) in cases.items():
    line, = axs[0].plot(hist[:, 0], hist[:, 2] / 1e6, label=f'{name}: entrada')
    axs[0].plot(trace[:, 0], trace[:, 1] / 1e6, '--', color=line.get_color(), label=f'{name}: x = 90 mm')
    axs[1].plot(trace[:, 0], trace[:, 2], color=line.get_color(), label=f'{name}: T media a 90 mm')
trace = cases['cristalización'][2]
axs[1].plot(trace[:, 0], 100 + 100 * trace[:, 3], 'k:', label='X media a 90 mm (100 + 100 X)')
for xi in (0.01, 0.05, 0.09):
    i = np.argmin(np.hypot(crys.xy[:, 0] - xi, crys.xy[:, 1] - 0.03))
    axs[2].plot(z, crys.crystallinity()[i], label=f'x = {xi * 1e3:.0f} mm')
axs[0].set_xlabel('t [s]'); axs[0].set_ylabel('presión [MPa]'); axs[0].legend(fontsize=7)
axs[1].set_xlabel('t [s]'); axs[1].set_ylabel('T [°C]'); axs[1].legend(fontsize=7)
axs[2].set_xlabel('z / b (0: plano medio, 1: pared)'); axs[2].set_ylabel('cristalinidad relativa al final'); axs[2].legend(fontsize=8)
fig.savefig('cristalizacion_historia.png', dpi=120)

tri = mtri.Triangulation(crys.xy[:, 0] * 1e3, crys.xy[:, 1] * 1e3, mesh.plot_triangles())
shrink = {name: 100 * c[0].shrinkage() for name, c in cases.items()}
levels = np.linspace(min(np.nanmin(s) for s in shrink.values()), max(np.nanmax(s) for s in shrink.values()), 21)
fig, axs = plt.subplots(1, 3, figsize=(16, 3.6), constrained_layout=True)
for a, (name, s) in zip(axs[:2], shrink.items()):
    c = a.tricontourf(tri, np.nan_to_num(s, nan=levels[-1]), levels, cmap='viridis')
    a.set_title(f'contracción, {name}')
fig.colorbar(c, ax=axs[:2], label='contracción volumétrica [%]')
c = axs[2].tricontourf(tri, crys.crystallinity() @ crys.thermal['dz'], 21, cmap='magma')
axs[2].set_title('cristalinidad media en el espesor')
fig.colorbar(c, ax=axs[2])
for a in axs:
    a.set_aspect('equal'); a.set_xlabel('x [mm]')
fig.savefig('cristalizacion_contraccion.png', dpi=120)
print("figures: cristalizacion_historia.png, cristalizacion_contraccion.png")
