## Ejemplo 2D: reaccion exotermica A -> C acoplada con transferencia de calor
#
# Modelo con dos fisicas que comparten malla y campos: Model(psc=[ht, mt]).
# Los campos son T (temperatura, C), A (reactivo) y C (producto). La velocidad de reaccion sigue la ley de
# Arrhenius con la temperatura del propio modelo y cada mol que reacciona libera Q = -dH de calor:
#
#     r = k0 exp(-E_R / (T + 273.15)) A      -k lap(T) = Q r      -D lap(A) = -r      -D lap(C) = +r
#
# Placa de 1 x 0.5 (unidades adimensionales):
#   - borde izquierdo: se alimenta A (A = 1, C = 0) y se mantiene T = 25 C
#   - bordes superior e inferior: enfriados por conveccion (h = 4, ambiente a 25 C)
#   - borde derecho: aislado para el calor y las especies
#
# El script muestra:
#   1. Verificacion: un reactor adiabatico uniforme frente a la solucion de la EDO equivalente (scipy).
#   2. Barrido del calor de reaccion Q: la retroalimentacion termica eleva la temperatura y la conversion.
#   3. Campos de temperatura y de producto, y arranque transitorio hasta el estacionario.

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
from Source.Physics.MassTransfer import mt
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers

# ----- Parametros (adimensionales) -----

k_cond, rho_cp, D = 2.0, 5.0, 1.0   # Conductividad, capacidad calorifica volumetrica y difusividad
k0, E_R = 3.0e4, 2000.0             # Arrhenius: factor de frecuencia [1/s] y E/R [K]
T_amb = 25.0                        # Temperatura ambiente y de alimentacion [C]
Lx, Ly, nx, ny = 1.0, 0.5, 40, 20   # Placa y malla Q4


def build_plate(Q):

    """Plate reactor with heat of reaction Q = -dH (Q = 0 switches the heat release off)."""

    mesh = Mesh()
    mesh.generate2DMesh((Lx, Ly), (nx, ny), ElementType.quadrilateral)

    model = Model(name='Reactor_placa', dim=2, mesh=mesh,
                  mat=material('Medio', k=k_cond, rho=rho_cp, Cp=1.0), psc=[ht, mt])
    heat, mass = model.physics.heat, model.physics.mass
    heat.Convection = False
    mass.Convection = False

    for species in ('A', 'C'):
        mass.setChemSpecies(species, species)
        mass.setDiffusivity(species, D)
    mass.addReaction({'A': -1, 'C': 1}, k0=k0, E_R=E_R, T='T', order={'A': 1}, dH=-Q if Q else None)

    feed = mesh.boundary_edges(lambda x, y: x < 1e-12)
    walls = mesh.boundary_edges(lambda x, y: y < 1e-12 or y > Ly - 1e-12)

    heat.addBC_TemperatureEdges(feed, T_amb)
    heat.addBC_ConvectionEdges(walls, 4.0, T_amb)
    mass.addBC_ConcentrationEdges(feed, 'A', 1.0)
    mass.addBC_ConcentrationEdges(feed, 'C', 0.0)

    model.physics.initField('T', T_amb)
    model.physics.initField('A', 0.5)
    model.physics.initField('C', 0.0)

    return model


def solve_steady(model):
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear,
                              Method=SolverType.direct, Solver=DirectSolvers.PARDISO)
    start = time.time()
    model.solve()
    return time.time() - start


def conversion_rate(model):

    """Integral of the reaction rate over the plate (mol/s per unit depth)."""

    disc = model._geometry
    A = np.asarray(model.sol['A'])[disc.conn]
    T = np.asarray(model.sol['T'])[disc.conn]
    A_g = np.einsum('gi,ei->eg', disc.N, A)
    T_g = np.einsum('gi,ei->eg', disc.N, T)
    return np.sum(disc.detJw * k0 * np.exp(-E_R / (T_g + 273.15)) * A_g)


# ----- 1. Verificacion: reactor adiabatico uniforme frente a la EDO -----

Q_ad, t_end = 400.0, 10.0
mesh_ad = Mesh()
mesh_ad.generate2DMesh((1.0, 1.0), (4, 4), ElementType.quadrilateral)
adiabatic = Model(name='Adiabatico', dim=2, mesh=mesh_ad, mat=material('Medio', k=k_cond, rho=rho_cp, Cp=1.0), psc=[ht, mt])
for species in ('A', 'C'):
    adiabatic.physics.mass.setChemSpecies(species, species)
    adiabatic.physics.mass.setDiffusivity(species, D)
adiabatic.physics.mass.addReaction({'A': -1, 'C': 1}, k0=300.0, E_R=E_R, T='T', order={'A': 1}, dH=-Q_ad)
adiabatic.physics.initField('T', T_amb)
adiabatic.physics.initField('A', 1.0)
adiabatic.physics.initField('C', 0.0)
adiabatic.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, Method=SolverType.direct,
                              Solver=DirectSolvers.PARDISO, timeDisc=0.5, timeStep=0.02, totalTime=t_end)
adiabatic.solve()


def ode(t, y):
    r = 300.0 * np.exp(-E_R / (y[0] + 273.15)) * y[1]
    return [Q_ad * r / rho_cp, -r, r]


reference = solve_ivp(ode, [0, t_end], [T_amb, 1.0, 0.0], method='LSODA', rtol=1e-12, atol=1e-13, dense_output=True)
T_fem = np.asarray(adiabatic.sol['T'])[0]
T_ref = reference.sol(adiabatic.timeVector)[0]

print("1. Reactor adiabatico uniforme (Crank-Nicolson, dt = 0.02) frente a la EDO equivalente")
print(f"   T final: FEM {T_fem[-1]:.4f} C, EDO {T_ref[-1]:.4f} C, error maximo en el tiempo {np.max(np.abs(T_fem - T_ref)):.2e} C")

# ----- 2. Barrido del calor de reaccion -----

print("\n2. Placa reactiva en estado estacionario (Newton desde un estado inicial uniforme)")
print(f"   {'Q':>6} {'T max [C]':>10} {'Reaccion total':>15} {'A salida (x=Lx)':>16} {'t calc [s]':>11}")
print("   " + "-" * 64)

sweep = [0.0, 20.0, 40.0, 80.0, 120.0]
results = {}
for Q in sweep:
    model = build_plate(Q)
    elapsed = solve_steady(model)
    results[Q] = model
    T = np.asarray(model.sol['T'])
    A = np.asarray(model.sol['A'])
    xy = np.array([n.coor for n in model.mesh.NL])
    A_out = A[xy[:, 0] > Lx - 1e-12].mean()
    print(f"   {Q:>6.0f} {T.max():>10.2f} {conversion_rate(model):>15.4f} {A_out:>16.4f} {elapsed:>11.3f}")

# ----- 3. Arranque transitorio hasta el estacionario (Q = 80) -----

Q_run = 80.0
startup = build_plate(Q_run)
startup.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, Method=SolverType.direct,
                            Solver=DirectSolvers.PARDISO, timeDisc=1.0, timeStep=0.02, totalTime=1.5)
start = time.time()
startup.solve()
T_hist = np.asarray(startup.sol['T'])
steady_T = np.asarray(results[Q_run].sol['T'])
print(f"\n3. Arranque transitorio (Euler implicito, 75 pasos, {time.time() - start:.2f} s): "
      f"T max final {T_hist[:, -1].max():.2f} C, diferencia con el estacionario {np.max(np.abs(T_hist[:, -1] - steady_T)):.2e} C")

# ----- Graficos -----

fig1, axes = plt.subplots(1, 2, figsize=(12, 4.2))
results[Q_run].postProcess2D('T', ax=axes[0])
axes[0].set_title(f'Temperatura [C] (Q = {Q_run:.0f})')
results[Q_run].postProcess2D('C', ax=axes[1])
axes[1].set_title('Producto C')
fig1.tight_layout()

fig2, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15, 4.2))

ax_a.plot(adiabatic.timeVector, T_ref, 'k-', label='EDO (scipy)')
ax_a.plot(adiabatic.timeVector[::25], T_fem[::25], 'o', label='FEM Crank-Nicolson')
ax_a.set_xlabel('t')
ax_a.set_ylabel('Temperatura [C]')
ax_a.set_title('Verificacion: reactor adiabatico')
ax_a.legend()
ax_a.grid(True, alpha=0.3)

ax_b.plot(sweep, [np.asarray(results[q].sol['T']).max() for q in sweep], 'o-', label='T maxima')
ax_b.set_xlabel('Calor de reaccion Q')
ax_b.set_ylabel('T maxima [C]')
ax_b.set_title('Retroalimentacion termica')
ax_b2 = ax_b.twinx()
ax_b2.plot(sweep, [conversion_rate(results[q]) for q in sweep], 's--', color='tab:red', label='Reaccion total')
ax_b2.set_ylabel('Reaccion total', color='tab:red')
ax_b.grid(True, alpha=0.3)

ax_c.plot(startup.timeVector, T_hist.max(axis=0), label='T maxima')
ax_c.axhline(steady_T.max(), color='k', ls='--', alpha=0.6, label='Estacionario')
ax_c.set_xlabel('t')
ax_c.set_ylabel('Temperatura [C]')
ax_c.set_title(f'Arranque transitorio (Q = {Q_run:.0f})')
ax_c.legend()
ax_c.grid(True, alpha=0.3)

fig2.tight_layout()
plt.show()
