## Ejemplo: cavidad con tapa movil, Navier-Stokes estacionario a Re = 100, 400 y 1000
#
# Cuadrado unitario, tapa superior con u = (1, 0), resto de paredes fijas, rho = 1 y mu = 1/Re.
# - Continuacion en el numero de Reynolds: cada solucion parte de la anterior, de modo que Newton converge en
#   pocas iteraciones aun a Re = 1000.
# - Validacion a Re = 100 con la tabla de Ghia, Ghia y Shin (1982) de u en la linea vertical central.
# - A mayor Re el vortice principal baja hacia el centro y aparecen vortices secundarios en las esquinas inferiores.
#
# Elementos Q4 de igual orden con SUPG/PSPG/LSIC y el esfuerzo viscoso recuperado en el residuo.

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers

N = 64
REYNOLDS = (100.0, 400.0, 1000.0)

GHIA_RE100 = np.array([[1.0000, 1.00000], [0.9766, 0.84123], [0.9688, 0.78871], [0.9609, 0.73722], [0.9531, 0.68717],
                       [0.8516, 0.23151], [0.7344, 0.00332], [0.6172, -0.13641], [0.5000, -0.20581],
                       [0.4531, -0.21090], [0.2813, -0.15662], [0.1719, -0.10150], [0.1016, -0.06434],
                       [0.0703, -0.04775], [0.0625, -0.04192], [0.0547, -0.03717], [0.0000, 0.00000]])

mesh = Mesh()
mesh.generate2DMesh((1.0, 1.0), (N, N), ElementType.quadrilateral)
xy = np.array([n.coor for n in mesh.NL])

model = Model(name='cavidad', dim=2, mesh=mesh, mat=material('fluido', miu=1.0 / REYNOLDS[0], rho=1.0), psc=ff)
model.physics.setInertia(True)
model.physics.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), (1.0, 0.0))
model.physics.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or x < 1e-9 or x > 1 - 1e-9))
model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, Method=SolverType.direct,
                          Solver=DirectSolvers.PARDISO)

line = np.abs(xy[:, 0] - 0.5) < 1e-9
order = np.argsort(xy[line, 1])
y_line = xy[line, 1][order]

solutions = {}
for Re in REYNOLDS:
    model.physics.setViscosity(1.0 / Re)       # the previous solution stays in the fields: continuation in Re
    start = time.perf_counter()
    model.solve()
    u_line = model.sol['u'][line, 0][order]
    solutions[Re] = (model.sol['u'].copy(), model.sol['p'].copy(), u_line)

    k = np.argmin(u_line)
    print(f"Re = {Re:6.0f}: {model.newton_iterations} iteraciones de Newton, {time.perf_counter() - start:.1f} s; "
          f"u_min en x = 0.5: {u_line[k]:.4f} en y = {y_line[k]:.3f}")

diff = np.interp(GHIA_RE100[:, 0], y_line, solutions[100.0][2]) - GHIA_RE100[:, 1]
print(f"Re = 100 frente a Ghia et al. (1982): diferencia maxima {np.max(np.abs(diff)):.4f}")

files = model.exportVTK(os.path.join('resultados_vtk', 'cavidad_re1000'))
print("Archivos para ParaView:", files)

# ----- Graficos -----
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

for Re in REYNOLDS:
    axes[0].plot(solutions[Re][2], y_line, label=f'Re = {Re:.0f}')
axes[0].plot(GHIA_RE100[:, 1], GHIA_RE100[:, 0], 'ko', mfc='none', label='Ghia et al. (1982), Re = 100')
axes[0].set_xlabel('u en x = 0.5')
axes[0].set_ylabel('y')
axes[0].grid(True)
axes[0].legend()

u = solutions[REYNOLDS[-1]][0]
triang = mtri.Triangulation(xy[:, 0], xy[:, 1])
cs = axes[1].tricontourf(triang, np.linalg.norm(u, axis=1), levels=40, cmap='viridis')
fig.colorbar(cs, ax=axes[1], label='|u|')
g = np.linspace(0, 1, 200)
gx, gy = np.meshgrid(g, g)
ux = mtri.LinearTriInterpolator(triang, u[:, 0])(gx, gy)
uy = mtri.LinearTriInterpolator(triang, u[:, 1])(gx, gy)
axes[1].streamplot(gx, gy, ux, uy, color='w', density=2.0, linewidth=0.5)
axes[1].set_aspect('equal')
axes[1].set_title(f'Re = {REYNOLDS[-1]:.0f}: velocidad y lineas de corriente')

plt.show()
