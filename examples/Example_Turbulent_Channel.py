## Ejemplo: canal turbulento con Spalart-Allmaras (Re_tau = 180, 395 y 590)
#
# Flujo turbulento completamente desarrollado entre placas planas. Por simetría se resuelve medio canal: pared en
# y = 0 y plano de simetría en y = h = 1. Una fuerza de cuerpo unitaria hace las veces del gradiente de presión, de modo
# que el esfuerzo en la pared es 1, u_tau = 1 y nu = 1 / Re_tau: la velocidad sale directamente en unidades de pared
# (u+ = u / u_tau) y y+ = y Re_tau.
#
# Modelo: Model(psc=[ff, sa]): RANS con la viscosidad efectiva mu + mu_t, donde mu_t = rho nu~ f_v1 y nu~ obedece la
# ecuación de Spalart-Allmaras (SA-neg). Todo se resuelve junto con Newton. La malla se agrupa hacia la pared para que
# el primer nodo quede en y+ < 1 (sin funciones de pared).
#
# Comparaciones:
#   - ley de pared de Reichardt (subcapa viscosa, zona de transición y zona logarítmica);
#   - velocidad en el centro de la DNS de Moser, Kim y Mansour (1999): U_c+ ~ 18.3, 20.1 y 21.4;
#   - coeficiente de fricción de la correlación de Dean, C_f = 0.073 Re_m^(-1/4), Re_m = 2 h U_b / nu.

import time

import numpy as np
import matplotlib.pyplot as plt

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.Turbulence import sa
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers

CASES = {180.0: (48, 2.5, 18.3), 395.0: (64, 3.0, 20.1), 590.0: (80, 3.2, 21.4)}     # ny, clustering, DNS U_c+


def reichardt(y_plus):
    return np.log(1 + 0.41 * y_plus) / 0.41 + 7.8 * (1 - np.exp(-y_plus / 11) - y_plus / 11 * np.exp(-y_plus / 3))


def solve_channel(re_tau, ny, beta):

    mesh = Mesh()
    mesh.generate2DMesh((0.1, 1.0), (2, ny), ElementType.quadrilateral)
    for node in mesh.NL:
        x, e = node.coor
        node.coor = (x, 1.0 + np.tanh(beta * (e - 1.0)) / np.tanh(beta))
    xy = np.array([node.coor for node in mesh.NL])

    model = Model(name=f'canal_{re_tau:.0f}', dim=2, mesh=mesh, mat=material('fluido', miu=1.0 / re_tau, rho=1.0),
                  psc=[ff, sa])
    fluid, turbulence = model.physics.fluid, model.physics.turbulence
    fluid.setInertia(True)
    fluid.setBodyForce((1.0, 0.0))
    wall = mesh.boundary_edges(lambda x, y: y < 1e-9)
    fluid.addBC_NoSlipEdges(wall)
    fluid.addBC_SymmetryEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9))
    fluid.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x < 1e-9))
    fluid.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x > 0.1 - 1e-9))
    turbulence.setWalls(wall)
    turbulence.coupleFlow()

    # initial guess: law of the wall and a mixing-length nu~
    fluid.var['u'].updateField(np.c_[reichardt(xy[:, 1] * re_tau), np.zeros(len(xy))])
    turbulence.var['nut'].updateField(0.41 * xy[:, 1] * (1 - xy[:, 1]) + 1e-3)

    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, Method=SolverType.direct,
                              Solver=DirectSolvers.PARDISO)
    model.solve()

    column = np.abs(xy[:, 0]) < 1e-12
    order = np.argsort(xy[column, 1])
    y = xy[column, 1][order]
    return model, y, model.sol['u'][column, 0][order], model.sol['eddy_viscosity'][column][order]


results = {}
print(f"{'Re_tau':>6} | {'y+ 1er nodo':>11} | {'U_c+':>6} {'DNS':>5} {'dif':>6} | {'U_b+':>6} | {'C_f':>8} {'Dean':>8} {'dif':>6} | Newton")
for re_tau, (ny, beta, dns) in CASES.items():
    start = time.perf_counter()
    model, y, u, mu_t = solve_channel(re_tau, ny, beta)
    u_bulk = np.trapezoid(u, y)
    cf = 2.0 / u_bulk ** 2                                  # tau_w / (rho U_b^2 / 2), tau_w = 1
    dean = 0.073 * (2.0 * u_bulk * re_tau) ** -0.25
    results[re_tau] = (y, u, mu_t)
    print(f"{re_tau:6.0f} | {y[1] * re_tau:11.2f} | {u.max():6.2f} {dns:5.1f} {u.max() / dns - 1:+6.1%} | {u_bulk:6.2f} | "
          f"{cf:8.5f} {dean:8.5f} {cf / dean - 1:+6.1%} | {model.newton_iterations} ({time.perf_counter() - start:.1f} s)")

# ----- Gráficos -----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
y_plus = np.logspace(-1, np.log10(600), 200)
ax = axes[0]
ax.semilogx(y_plus, reichardt(y_plus), 'k--', lw=1, label='ley de pared (Reichardt)')
ax.semilogx(y_plus[y_plus < 12], y_plus[y_plus < 12], 'k:', lw=1, label='u+ = y+')
for re_tau, (y, u, _) in results.items():
    ax.semilogx(y[1:] * re_tau, u[1:], label=f'SA, Re_tau = {re_tau:.0f}')
ax.set_xlabel('y+'); ax.set_ylabel('u+'); ax.legend(); ax.grid(True, which='both', alpha=0.3)
ax.set_title('Perfil de velocidad en unidades de pared')

ax = axes[1]
for re_tau, (y, _, mu_t) in results.items():
    ax.plot(y, mu_t * re_tau, label=f'Re_tau = {re_tau:.0f}')
ax.set_xlabel('y / h'); ax.set_ylabel('nu_t / nu'); ax.legend(); ax.grid(True, alpha=0.3)
ax.set_title('Viscosidad turbulenta')

plt.tight_layout()
plt.show()
