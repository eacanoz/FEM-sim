## Ejemplo: reómetro capilar simulado (flujo axisimétrico de un fundido de polímero)
#
# Un pistón empuja el fundido por un barril de 9.5 mm de diámetro hacia un capilar de 1 mm de diámetro y L/D = 16,
# con entrada plana (180°). Es el ensayo estándar para medir la viscosidad de polímeros a alta tasa de corte.
# La geometría es de revolución: se resuelve el plano meridiano (x = r, y = z, eje en x = 0) con Model(axisymmetric=True).
#
# Para cada caudal se procesa la "medición" como en el laboratorio:
#   tasa de corte aparente     gamma_a = 4 Q / (pi R^3)
#   esfuerzo en la pared       tau_w = (R / 2) |dp/dz|, con el gradiente del tramo desarrollado del capilar
#   pérdida de entrada         dp_e = caída total - (tramos desarrollados de barril y capilar)  (corrección de Bagley)
#   corrección de Rabinowitsch gamma_w = gamma_a (3 n' + 1) / (4 n'),  n' = d ln(tau_w) / d ln(gamma_a)
#   viscosidad verdadera       eta = tau_w / gamma_w
# y se compara la viscosidad "medida" con la ley Cross que se dio al modelo: el reómetro simulado debe recuperarla.

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import gmsh

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.Rheology import Cross
from Source.enums import StudyType, ProblemType, SolverType, DirectSolvers

RB, LB = 4.75e-3, 8.0e-3          # radio y longitud del barril [m]
RC, LC = 0.5e-3, 16.0e-3          # radio y longitud del capilar [m] (L/D = 16)
melt = Cross(mu0=945.0, tau_star=2.5e4, n=0.28)          # fundido tipo PP a 230 °C (ilustrativo)
GAMMA_A = (20.0, 60.0, 200.0, 600.0, 2000.0)             # tasas de corte aparentes del ensayo [1/s]


def rheometer_mesh(lc_barrel=0.35e-3, lc_cap=0.06e-3, lc_corner=0.015e-3):

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add('capilar')
        geo = gmsh.model.geo
        pts = [(0, 0, lc_barrel), (RB, 0, lc_barrel), (RB, LB, lc_barrel), (RC, LB, lc_corner),
               (RC, LB + LC, lc_cap), (0, LB + LC, lc_cap), (0, LB, lc_cap)]
        p = [geo.addPoint(x, y, 0, s) for x, y, s in pts]
        lines = [geo.addLine(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]
        geo.addPlaneSurface([geo.addCurveLoop(lines)])
        geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [lines[0]], name='inlet')
        gmsh.model.addPhysicalGroup(1, [lines[1], lines[2], lines[3]], name='wall')
        gmsh.model.addPhysicalGroup(1, [lines[4]], name='outlet')
        gmsh.model.addPhysicalGroup(1, [lines[5], lines[6]], name='axis')
        gmsh.model.mesh.generate(2)
        return Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()


mesh = rheometer_mesh()
xy = np.array([node.coor for node in mesh.NL])
print(f"Malla: {mesh.getNoN()} nodos, {mesh.getNoE()} triángulos")

model = Model(name='reometro', dim=2, mesh=mesh, mat=material('PP', miu=melt.mu0), psc=ff, axisymmetric=True)
fluid = model.physics
fluid.addBC_SymmetryEdges(mesh.boundary_edges('axis'))
fluid.addBC_NoSlipEdges(mesh.boundary_edges('wall'))
fluid.addBC_OutletEdges(mesh.boundary_edges('outlet'), 0.0)
print(model)


def set_flow_rate(q):
    u_mean = q / (np.pi * RB ** 2)                       # pistón: perfil desarrollado newtoniano en el barril
    fluid.addBC_VelocityEdges(mesh.boundary_edges('inlet'), lambda r, z: (0.0, 2 * u_mean * (1 - (r / RB) ** 2)))


def axial_gradient(z0, z1):
    on_axis = xy[:, 0] < 1e-12
    sel = on_axis & (xy[:, 1] > z0) & (xy[:, 1] < z1)
    return -np.polyfit(xy[sel, 1], model.sol['p'][sel], 1)[0]


# ----- Arranque newtoniano y paso a la ley Cross -----
set_flow_rate(GAMMA_A[0] * np.pi * RC ** 3 / 4)
model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear, Method=SolverType.direct,
                          Solver=DirectSolvers.PARDISO)
model.solve()
fluid.setViscosityLaw(melt)
model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, Method=SolverType.direct,
                          Solver=DirectSolvers.PARDISO)

rows = []
for gamma_a in GAMMA_A:
    q_nominal = gamma_a * np.pi * RC ** 3 / 4
    set_flow_rate(q_nominal)
    start = time.perf_counter()
    model.solve()                                        # parte de la solución anterior (continuación en caudal)

    q = fluid.flowRate(mesh.boundary_edges('outlet'))    # caudal discreto [m3/s]
    gamma_app = 4 * q / (np.pi * RC ** 3)
    inlet_nodes = np.unique(np.ravel(mesh.boundary_edges('inlet')))
    dp_total = model.sol['p'][inlet_nodes].mean()
    grad_cap = axial_gradient(LB + 0.4 * LC, LB + 0.9 * LC)
    grad_bar = axial_gradient(0.1 * LB, 0.4 * LB)
    dp_entrance = dp_total - grad_cap * LC - grad_bar * LB
    tau_w = RC / 2 * grad_cap
    rows.append((gamma_app, tau_w, dp_total, dp_entrance, model.newton_iterations, time.perf_counter() - start))

rows = np.array(rows)
gamma_app, tau_w, dp_total, dp_entrance = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]

# Rabinowitsch: n' local de la curva log(tau_w) - log(gamma_a)
n_prime = np.gradient(np.log(tau_w), np.log(gamma_app))
gamma_w = gamma_app * (3 * n_prime + 1) / (4 * n_prime)
eta_measured = tau_w / gamma_w
eta_law = melt.viscosity(gamma_w)

print(f"\n{'gamma_a':>9} {'tau_w [kPa]':>11} {'dp total':>9} {'dp entrada':>11} {'n_prime':>7} {'gamma_w':>9} "
      f"{'eta med.':>9} {'eta ley':>9} {'error':>7} {'Newton':>6}")
for k in range(len(rows)):
    print(f"{gamma_app[k]:9.1f} {tau_w[k] / 1e3:11.2f} {dp_total[k] / 1e5:7.2f} b {dp_entrance[k] / 1e5:9.2f} b "
          f"{n_prime[k]:7.3f} {gamma_w[k]:9.1f} {eta_measured[k]:9.2f} {eta_law[k]:9.2f} "
          f"{eta_measured[k] / eta_law[k] - 1:7.1%} {int(rows[k, 4]):6d}")
print(f"Pérdida de entrada / caída total: {np.round(dp_entrance / dp_total, 3)}")

files = model.exportVTK(os.path.join('resultados_vtk', 'reometro_capilar'))
print("Archivos para ParaView:", files)

# ----- Gráficos (último caudal) -----
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3)

triang = mtri.Triangulation(xy[:, 0] * 1e3, xy[:, 1] * 1e3, [e.getNodesId() for e in mesh.EL])
window = dict(xlim=(0, 2.5), ylim=(LB * 1e3 - 2.0, LB * 1e3 + 3.0))

ax = fig.add_subplot(gs[:, 0])
cs = ax.tricontourf(triang, model.sol['p'] / 1e5, levels=40, cmap='viridis')
fig.colorbar(cs, ax=ax, label='p [bar]')
ax.set_aspect('equal'); ax.set_xlabel('r [mm]'); ax.set_ylabel('z [mm]'); ax.set_title('Presión (último caudal)')

ax = fig.add_subplot(gs[0, 1])
cs = ax.tricontourf(triang, np.linalg.norm(model.sol['u'], axis=1) * 1e3, levels=40, cmap='magma')
fig.colorbar(cs, ax=ax, label='|u| [mm/s]')
ax.set(**window); ax.set_aspect('equal'); ax.set_xlabel('r [mm]'); ax.set_ylabel('z [mm]')
ax.set_title('Velocidad en la entrada del capilar')

ax = fig.add_subplot(gs[1, 1])
cs = ax.tricontourf(triang, np.log10(model.sol['viscosity']), levels=40, cmap='cividis')
fig.colorbar(cs, ax=ax, label='log10 viscosidad [Pa·s]')
ax.set(**window); ax.set_aspect('equal'); ax.set_xlabel('r [mm]'); ax.set_ylabel('z [mm]')
ax.set_title('Viscosidad local')

ax = fig.add_subplot(gs[0, 2])
g = np.logspace(0, 4.5, 200)
ax.loglog(g, melt.viscosity(g), 'k-', label='ley Cross del modelo')
ax.loglog(gamma_app, tau_w / gamma_app, 's', mfc='none', label='aparente (tau_w / gamma_a)')
ax.loglog(gamma_w, eta_measured, 'o', label='corregida (Rabinowitsch)')
ax.set_xlabel('tasa de corte [1/s]'); ax.set_ylabel('viscosidad [Pa·s]'); ax.legend(); ax.set_title('Curva medida')
ax.grid(True, which='both', alpha=0.3)

ax = fig.add_subplot(gs[1, 2])
ax.semilogx(gamma_app, dp_total / 1e5, 'o-', label='caída total')
ax.semilogx(gamma_app, dp_entrance / 1e5, 's-', label='pérdida de entrada (Bagley)')
ax.set_xlabel('tasa de corte aparente [1/s]'); ax.set_ylabel('presión [bar]'); ax.legend(); ax.set_title('Presiones')
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.show()
