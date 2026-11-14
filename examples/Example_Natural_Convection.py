## Ejemplo: convección natural en una cavidad calentada lateralmente (de Vahl Davis, Ra = 1e3 a 1e6)
#
# Cavidad cuadrada llena de aire (Pr = 0.71): pared izquierda caliente, derecha fría, superior e inferior adiabáticas.
# El aire caliente sube por la pared izquierda, cruza por arriba, baja por la derecha y vuelve por abajo. Al crecer el
# número de Rayleigh, Ra = g beta dT L^3 / (nu alpha), el transporte pasa de conducción a capas límite delgadas.
#
# Formulación adimensional: L = rho = Cp = k = 1 (difusividad térmica alpha = 1), mu = Pr, beta = 1, dT = 1 y
# gravedad Ra Pr. Las velocidades quedan escaladas con alpha / L, como en la referencia.
#
# Modelo: Model(psc=[ff, ht]) con Navier-Stokes (setInertia), flotación de Boussinesq (setBoussinesq) y el calor
# convectado por la velocidad calculada (setVelocity('u'), SUPG). Todo se resuelve junto con Newton y continuación
# en Ra. El número de Nusselt sale del flujo de calor por reacciones en la pared caliente (heat.boundaryFlux).
# Referencia: G. de Vahl Davis, "Natural convection of air in a square cavity: a bench mark numerical solution", 1983.

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.HeatTransfer import ht
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers

PR, N = 0.71, 48
REFERENCE = {1e3: (1.118, 3.649, 3.697), 1e4: (2.243, 16.178, 19.617), 1e5: (4.519, 34.73, 68.59),
             1e6: (8.800, 64.63, 219.36)}           # Nusselt, u_max (x = 0.5), v_max (y = 0.5)

mesh = Mesh()
mesh.generate2DMesh((1.0, 1.0), (N, N), ElementType.quadrilateral)
for node in mesh.NL:                                # nodes clustered at the walls (Chebyshev) for the boundary layers
    node.coor = tuple(0.5 * (1 - np.cos(np.pi * c)) for c in node.coor)
xy = np.array([node.coor for node in mesh.NL])

model = Model(name='cavidad', dim=2, mesh=mesh, mat=material('aire', miu=PR, rho=1.0, Cp=1.0, k=1.0), psc=[ff, ht])
fluid, heat = model.physics.fluid, model.physics.heat
fluid.setInertia(True)
model.physics.setVelocity('u')
model.physics.setStabilization('SUPG')
hot = mesh.boundary_edges(lambda x, y: x < 1e-9)
cold = mesh.boundary_edges(lambda x, y: x > 1 - 1e-9)
fluid.addBC_NoSlipEdges(mesh.boundary_edges(None))
heat.addBC_TemperatureEdges(hot, 1.0)
heat.addBC_TemperatureEdges(cold, 0.0)
model.physics.initField('T', 0.5)
model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, Method=SolverType.direct,
                          Solver=DirectSolvers.PARDISO)

vertical, horizontal = np.abs(xy[:, 0] - 0.5) < 1e-9, np.abs(xy[:, 1] - 0.5) < 1e-9
results = {}
print(f"Malla {N}x{N} (estirada hacia las paredes), {model.layout.ndof} incógnitas")
print(f"{'Ra':>6} | {'Nu':>7} {'ref':>6} {'dif':>7} | {'u_max':>7} {'ref':>6} {'dif':>7} | {'v_max':>7} {'ref':>7} {'dif':>7} | Newton")
for Ra, (nu_ref, u_ref, v_ref) in REFERENCE.items():
    fluid.setBoussinesq((0.0, -Ra * PR), beta=1.0, reference=0.5)      # continuación: parte de la solución anterior
    start = time.perf_counter()
    model.solve()
    nu = heat.boundaryFlux(hot)
    u_max, v_max = model.sol['u'][vertical, 0].max(), model.sol['u'][horizontal, 1].max()
    results[Ra] = (model.sol['u'].copy(), model.sol['T'].copy())
    print(f"{Ra:6.0e} | {nu:7.4f} {nu_ref:6.3f} {nu / nu_ref - 1:+7.2%} | {u_max:7.3f} {u_ref:6.2f} {u_max / u_ref - 1:+7.2%} | "
          f"{v_max:7.2f} {v_ref:7.2f} {v_max / v_ref - 1:+7.2%} | {model.newton_iterations} ({time.perf_counter() - start:.1f} s)")
print(f"Balance de energía (Ra = 1e6): calor entrante {heat.boundaryFlux(hot):.6f}, saliente {-heat.boundaryFlux(cold):.6f}")

files = model.exportVTK(os.path.join('resultados_vtk', 'conveccion_natural'))
print("Archivos para ParaView:", files)

# ----- Gráficos: isotermas y líneas de corriente para cada Ra -----
triang = mtri.Triangulation(xy[:, 0], xy[:, 1])
g = np.linspace(0, 1, 160)
GX, GY = np.meshgrid(g, g)
fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
for col, (Ra, (u, T)) in enumerate(results.items()):
    ax = axes[0, col]
    cs = ax.tricontourf(triang, T, levels=np.linspace(0, 1, 21), cmap='coolwarm')
    ax.tricontour(triang, T, levels=np.linspace(0.05, 0.95, 10), colors='k', linewidths=0.4)
    ax.set_title(f'Ra = {Ra:.0e}: temperatura'); ax.set_aspect('equal')

    ax = axes[1, col]
    speed = np.linalg.norm(u, axis=1)
    ax.tricontourf(triang, speed, levels=30, cmap='viridis')
    UX = mtri.LinearTriInterpolator(triang, u[:, 0])(GX, GY)
    UY = mtri.LinearTriInterpolator(triang, u[:, 1])(GX, GY)
    ax.streamplot(GX, GY, UX, UY, color='w', density=1.0, linewidth=0.5)
    ax.set_title(f'|u| máx = {speed.max():.1f}'); ax.set_aspect('equal'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.colorbar(cs, ax=axes[0, :], label='T', shrink=0.8)
plt.show()
