## Ejemplo: fundido de polímero en un dado de rendija (Cross-WLF, calentamiento viscoso y acoplamiento térmico)
#
# Rendija plana de 2 mm de apertura y 20 mm de largo; por simetría se resuelve la mitad (semiapertura H = 1 mm).
# El fundido entra a 230 °C con un perfil parabólico y las paredes del dado se mantienen a 230 °C.
#
# 1. Reología: curvas de viscosidad Cross-WLF a varias temperaturas (zona newtoniana y adelgazamiento por corte).
# 2. Isotérmico: perfil de velocidad newtoniano frente a Cross-WLF (más plano) y caída de presión.
# 3. No isotérmico: velocidad, presión y temperatura juntas (Model(psc=[ff, ht])), con disipación viscosa como
#    fuente de calor y la viscosidad dependiente de la temperatura. El calor generado cerca de la pared, donde la
#    tasa de corte es mayor, eleva la temperatura y baja la viscosidad.
# 4. Curva caudal-caída de presión: el adelgazamiento hace que la presión crezca mucho menos que el caudal.
#
# Los parámetros Cross-WLF son ilustrativos (del orden de un polipropileno), no de un grado comercial concreto.

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
from Source.Physics.Rheology import CrossWLF
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers

H, L = 1.0e-3, 20.0e-3                 # semiapertura y longitud [m]
NX, NY = 80, 16
T_IN = T_WALL = 230.0                  # °C
RHO, CP, K = 750.0, 2700.0, 0.17       # fundido: densidad [kg/m3], calor específico [J/(kg K)], conductividad [W/(m K)]
Q = 1.0e-4                             # caudal por unidad de profundidad en la mitad de la rendija [m2/s] (U media 100 mm/s)

melt = CrossWLF(n=0.28, tau_star=2.5e4, D1=5.0e13, D2=263.15, A1=30.0, A2=51.6)

mesh = Mesh()
mesh.generate2DMesh((L, H), (NX, NY), ElementType.quadrilateral)
xy = np.array([node.coor for node in mesh.NL])
inlet = mesh.boundary_edges(lambda x, y: x < 1e-12)
outlet = mesh.boundary_edges(lambda x, y: x > L - 1e-12)
axis = mesh.boundary_edges(lambda x, y: y < 1e-12)
wall = mesh.boundary_edges(lambda x, y: y > H - 1e-12)


def inlet_profile(q):
    u_max = 1.5 * q / H                                    # parábola de semiapertura H con caudal q
    return lambda x, y: (u_max * (1 - (y / H) ** 2), 0.0)


def configure(model, kind):
    model.solverConfiguration(Study=StudyType.steady_state, Type=kind, Method=SolverType.direct,
                              Solver=DirectSolvers.PARDISO)


def flow_bcs(fluid, q):
    fluid.addBC_VelocityEdges(inlet, inlet_profile(q))
    fluid.addBC_SymmetryEdges(axis)
    fluid.addBC_NoSlipEdges(wall)
    fluid.addBC_OutletEdges(outlet, 0.0)


def inlet_pressure(model):
    nodes = np.unique(np.ravel(inlet))
    return model.sol['p'][nodes].mean()


def bulk_temperature(model, x):

    """Temperatura de mezcla (ponderada con el caudal) en la sección x: integral(u T dy) / integral(u dy)."""

    column = np.abs(xy[:, 0] - x) < 1e-12
    order = np.argsort(xy[column, 1])
    y, u, T = xy[column, 1][order], model.sol['u'][column, 0][order], model.sol['T'][column][order]
    return np.trapezoid(u * T, y) / np.trapezoid(u, y)


# ----- 1. Reología -----
gamma = np.logspace(-1, 4, 200)
mu0_230 = float(melt.zero_shear(T_IN))
print(f"Viscosidad a corte nulo a {T_IN:.0f} °C: {mu0_230:.0f} Pa·s; transición a gamma = tau*/mu0 = "
      f"{melt.tau_star / mu0_230:.1f} 1/s")

# ----- 2. Isotérmico: newtoniano frente a Cross-WLF -----
model = Model(name='rendija', dim=2, mesh=mesh, mat=material('PP', miu=mu0_230, rho=RHO, Cp=CP, k=K), psc=[ff, ht])
fluid, heat = model.physics.fluid, model.physics.heat
flow_bcs(fluid, Q)
heat.addBC_TemperatureEdges(inlet, T_IN)
heat.addBC_TemperatureEdges(wall, T_WALL)
model.physics.setVelocity('u')                          # el calor viaja con la velocidad calculada
model.physics.setStabilization('SUPG')                  # Péclet alto: el fundido conduce poco el calor
model.physics.initField('T', T_IN)

configure(model, ProblemType.nonlinear)
start = time.perf_counter()
model.solve()                                           # newtoniano (mu0): punto de partida
newtonian = dict(u=model.sol['u'].copy(), p=model.sol['p'].copy(), dp=inlet_pressure(model))
print(f"Newtoniano (mu0): caída de presión {newtonian['dp'] / 1e5:.2f} bar ({time.perf_counter() - start:.1f} s)")

fluid.setViscosityLaw(melt)                             # la T sigue en 230 °C: isotérmico efectivo
start = time.perf_counter()
model.solve()
isothermal = dict(u=model.sol['u'].copy(), p=model.sol['p'].copy(), dp=inlet_pressure(model),
                  mu=model.sol['viscosity'].copy(), rate=model.sol['shear_rate'].copy())
print(f"Cross-WLF isotérmico: caída de presión {isothermal['dp'] / 1e5:.3f} bar, "
      f"{model.newton_iterations} iteraciones ({time.perf_counter() - start:.1f} s)")

# ----- 3. No isotérmico con disipación viscosa -----
fluid.setViscousHeating(True)
start = time.perf_counter()
model.solve()
thermal = dict(u=model.sol['u'].copy(), p=model.sol['p'].copy(), T=model.sol['T'].copy(), dp=inlet_pressure(model),
               mu=model.sol['viscosity'].copy(), rate=model.sol['shear_rate'].copy())
q_in, q_out = -fluid.flowRate(inlet), fluid.flowRate(outlet)
print(f"Con calentamiento viscoso: caída de presión {thermal['dp'] / 1e5:.3f} bar, T máx {thermal['T'].max():.2f} °C, "
      f"{model.newton_iterations} iteraciones ({time.perf_counter() - start:.1f} s)")
print(f"Caudal: entrada {q_in:.6e}, salida {q_out:.6e} m2/s (diferencia {abs(q_in - q_out):.1e})")
print(f"Balance de energía: aumento adiabático dP/(rho Cp) = {thermal['dp'] / (RHO * CP):.3f} K; aumento de la temperatura "
      f"de mezcla a la salida = {bulk_temperature(model, L) - T_IN:.3f} K (el resto sale por las paredes a {T_WALL:.0f} °C)")

files = model.exportVTK(os.path.join('resultados_vtk', 'rendija_polimero'))
print("Archivos para ParaView:", files)

# ----- 4. Caudal frente a caída de presión (no isotérmico) -----
rates, drops = [], []
for q in (0.1 * Q, 0.25 * Q, 0.5 * Q, Q, 2 * Q):
    fluid.addBC_VelocityEdges(inlet, inlet_profile(q))   # reemplaza el perfil de entrada
    model.solve()                                        # parte de la solución anterior (continuación en caudal)
    rates.append(q)
    drops.append(inlet_pressure(model))
    print(f"   Q = {q:.2e} m2/s -> caída {drops[-1] / 1e5:.3f} bar, T máx {model.sol['T'].max():.2f} °C, "
          f"{model.newton_iterations} iteraciones")
slope = np.polyfit(np.log(rates), np.log(drops), 1)[0]
local = np.diff(np.log(drops)) / np.diff(np.log(rates))
print(f"Pendiente log(dP)/log(Q) = {slope:.2f} en promedio; por tramos {np.round(local, 2)} "
      f"(1 para un fluido newtoniano; tiende a n = {melt.n} a alto corte)")

# ----- Gráficos -----
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
for T in (210.0, 230.0, 250.0):
    ax.loglog(gamma, melt.viscosity(gamma, T), label=f'{T:.0f} °C')
ax.set_xlabel('tasa de corte [1/s]'); ax.set_ylabel('viscosidad [Pa·s]'); ax.legend(); ax.set_title('Cross-WLF')
ax.grid(True, which='both', alpha=0.3)

ax = axes[0, 1]
column = np.abs(xy[:, 0] - 0.75 * L) < 1e-12
order = np.argsort(xy[column, 1])
y_mm = xy[column, 1][order] * 1e3
for data, label in ((newtonian, 'newtoniano (mu0)'), (isothermal, 'Cross-WLF isotérmico'), (thermal, 'Cross-WLF + calentamiento')):
    ax.plot(data['u'][column, 0][order] * 1e3, y_mm, label=label)
ax.set_xlabel('u [mm/s]'); ax.set_ylabel('y [mm]'); ax.legend(); ax.set_title('Perfil de velocidad en x = 15 mm')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
triang = mtri.Triangulation(xy[:, 0] * 1e3, xy[:, 1] * 1e3)
cs = ax.tricontourf(triang, thermal['T'] - T_IN, levels=30, cmap='inferno')
fig.colorbar(cs, ax=ax, label='T - 230 °C [K]')
ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_title('Temperatura por disipación viscosa (Q nominal)')

ax = axes[1, 1]
ax.loglog(np.array(rates) * 1e6, np.array(drops) / 1e5, 'o-', label=f'FEM-sim (pendiente {slope:.2f})')
ax.loglog(np.array(rates) * 1e6, drops[0] / 1e5 * np.array(rates) / rates[0], 'k--', lw=0.8, label='newtoniano (pendiente 1)')
ax.set_xlabel('caudal [mm²/s]'); ax.set_ylabel('caída de presión [bar]'); ax.legend(); ax.set_title('Curva del dado')
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.show()
