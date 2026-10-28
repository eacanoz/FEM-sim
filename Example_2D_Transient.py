## Ejemplo 2D: enfriamiento transitorio no lineal de una placa, con estudio de convergencia en el tiempo
#
# Placa ceramica de 0.2 m x 0.1 m con un punto caliente inicial (gaussiana de 300 C sobre 25 C).
#   - borde izquierdo: temperatura fija de 25 C (Dirichlet)
#   - borde derecho: radiacion al ambiente (no lineal, Newton/linealizacion en cada paso)
#   - borde superior: conveccion al aire (h = 15 W/m2K)
#   - borde inferior: aislado
#
# Nota: Crank-Nicolson solo muestra su orden 2 con pasos pequenos; con pasos grandes oscila (no es L-estable).
#
# El script resuelve el problema con el theta-metodo (Euler implicito y Crank-Nicolson) para varios
# pasos de tiempo, mide el error frente a una solucion de referencia con un paso muy pequeno y muestra:
#   1. Tabla de convergencia en el tiempo (orden esperado: 1 para Euler y 2 para Crank-Nicolson).
#   2. Instantaneas del campo de temperatura.
#   3. Evolucion temporal (maximo, punto de control y media) y grafico de convergencia log-log.

import time

import numpy as np
import matplotlib.pyplot as plt

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers

# ----- Parametros del problema -----

Lx, Ly = 0.2, 0.1                   # Dimensiones de la placa [m]
nx, ny = 40, 20                     # Elementos Q4 en cada direccion
k, rho, Cp = 5.0, 2500.0, 800.0     # Conductividad [W/mK], densidad [kg/m3], calor especifico [J/kgK]
T_amb, T_hot = 25.0, 300.0          # Temperaturas [C]
total_time = 3000.0                 # Tiempo total [s]
probe = (0.1, 0.05)                 # Punto de control (centro de la placa) [m]


def initial_temperature(x, y):
    return T_amb + (T_hot - T_amb) * np.exp(-((x - 0.1) ** 2 + (y - 0.05) ** 2) / (2 * 0.02 ** 2))


def solve_plate(theta, dt):

    """Solves the transient problem. Returns the model and the wall-clock time of solve()."""

    mesh = Mesh()
    mesh.generate2DMesh((Lx, Ly), (nx, ny), ElementType.quadrilateral)

    model = Model(name='Placa_transitoria', dim=2, mesh=mesh,
                  mat=material('Ceramica', k=k, rho=rho, Cp=Cp), psc=ht)
    model.physics.Convection = False
    model.physics.initField('T', initial_temperature)

    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x < 1e-12), T_amb)
    model.physics.addBC_RadiationEdges(mesh.boundary_edges(lambda x, y: x > Lx - 1e-12), 0.8, T_amb)
    model.physics.addBC_ConvectionEdges(mesh.boundary_edges(lambda x, y: y > Ly - 1e-12), 15.0, T_amb)

    model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear,
                              Method=SolverType.direct, Solver=DirectSolvers.PARDISO,
                              timeDisc=theta, timeStep=dt, totalTime=total_time)

    start = time.time()
    model.solve()

    return model, time.time() - start


# ----- Solucion de referencia (Crank-Nicolson con paso muy pequeno) -----

reference, _ = solve_plate(0.5, total_time / 1920)
T_ref = np.asarray(reference.sol['T'])[:, -1]

# ----- Convergencia en el tiempo -----

steps = [20, 40, 80, 160, 320]
dts = np.array([total_time / n for n in steps])
errors = {1.0: [], 0.5: []}
runtimes = {1.0: [], 0.5: []}

for theta in (1.0, 0.5):
    for dt in dts:
        model, elapsed = solve_plate(theta, dt)
        errors[theta].append(np.max(np.abs(np.asarray(model.sol['T'])[:, -1] - T_ref)))
        runtimes[theta].append(elapsed)

print(f"Placa {Lx} m x {Ly} m, malla {nx}x{ny} Q4 ({reference.mesh.getNoN()} nodos), t = {total_time:.0f} s")
print(f"{'dt [s]':>9} {'Pasos':>6} | {'Euler: error':>13} {'orden':>6} | {'Crank-N.: error':>15} {'orden':>6} | {'t calc [s]':>10}")
print("-" * 82)
for i, (dt, n) in enumerate(zip(dts, steps)):
    rate_e = "-" if i == 0 else f"{np.log2(errors[1.0][i - 1] / errors[1.0][i]):.2f}"
    rate_c = "-" if i == 0 else f"{np.log2(errors[0.5][i - 1] / errors[0.5][i]):.2f}"
    print(f"{dt:>9.2f} {n:>6d} | {errors[1.0][i]:>13.3e} {rate_e:>6} | {errors[0.5][i]:>15.3e} {rate_c:>6} | {runtimes[0.5][i]:>10.3f}")

print("\nNota: Euler implicito es robusto pero de primer orden. Crank-Nicolson es de segundo orden, pero no es")
print("L-estable: con pasos grandes los modos rigidos oscilan (errores y 'ordenes' irregulares al principio) y")
print("solo muestra orden 2 cuando el paso es suficientemente pequeno.")

# ----- Resultados de la corrida base (Crank-Nicolson, 60 pasos) -----

model, elapsed = solve_plate(0.5, total_time / 60)
T = np.asarray(model.sol['T'])
times = model.timeVector
xy = np.array([node.coor for node in model.mesh.NL])
probe_node = int(np.argmin((xy[:, 0] - probe[0]) ** 2 + (xy[:, 1] - probe[1]) ** 2))

print(f"\nCorrida base (Crank-Nicolson, dt = {total_time / 60:.1f} s): {elapsed:.3f} s")
print(f"  T maxima: {T[:, 0].max():.1f} C -> {T[:, -1].max():.1f} C;  T en el centro: {T[probe_node, 0]:.1f} C -> {T[probe_node, -1]:.1f} C")

# ----- Grafico 1: instantaneas -----

snapshots = [0, 3, 12, 60]
fig1, axes = plt.subplots(1, len(snapshots), figsize=(15, 3.6), sharey=True)
for ax, step in zip(axes, snapshots):
    contour = model.postProcess2D(step=step, ax=ax)
    ax.set_title(f't = {times[step]:.0f} s  (Tmax = {T[:, step].max():.0f} C)')
    if ax is not axes[0]:
        ax.set_ylabel('')
fig1.tight_layout()

# ----- Grafico 2: evolucion temporal y convergencia -----

fig2, (ax_t, ax_c) = plt.subplots(1, 2, figsize=(11, 4.5))

ax_t.plot(times, T.max(axis=0), label='Maximo')
ax_t.plot(times, T[probe_node], label=f'Centro {probe}')
ax_t.plot(times, T.mean(axis=0), label='Media nodal')
ax_t.set_xlabel('t [s]')
ax_t.set_ylabel('Temperatura [C]')
ax_t.set_title('Evolucion de la temperatura')
ax_t.grid(True, alpha=0.3)
ax_t.legend()

ax_c.loglog(dts, errors[1.0], 'o-', label='Euler implicito (theta = 1)')
ax_c.loglog(dts, errors[0.5], 's-', label='Crank-Nicolson (theta = 0.5)')
ax_c.loglog(dts, errors[1.0][0] * (dts / dts[0]), 'k--', alpha=0.6, label='Pendiente 1')
ax_c.loglog(dts, errors[0.5][0] * (dts / dts[0]) ** 2, 'k:', alpha=0.6, label='Pendiente 2')
ax_c.set_xlabel('Paso de tiempo dt [s]')
ax_c.set_ylabel('Error maximo frente a la referencia [C]')
ax_c.set_title('Convergencia en el tiempo')
ax_c.grid(True, which='both', alpha=0.3)
ax_c.legend()

fig2.tight_layout()
plt.show()
