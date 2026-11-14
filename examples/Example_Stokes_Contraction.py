## Ejemplo: contraccion plana 4:1 en flujo reptante (Stokes)
#
# Caso clasico del procesamiento de polimeros (entrada a una boquilla o a un dado): un canal de semialtura H = 4
# se reduce bruscamente a uno de semialtura h = 1. Por simetria se resuelve la mitad superior.
#
# - Entrada (x = -L1): perfil de Poiseuille desarrollado con el caudal Q.
# - Salida (x = L2): p = 0 y flujo desarrollado.  - Eje (y = 0): simetria.  - Paredes: no deslizamiento.
#
# Resultados:
# 1. Conservacion: caudal entrante = caudal saliente (exacto con PSPG).
# 2. Perdida de presion en la entrada: la caida total menos la de los dos tramos desarrollados (Poiseuille),
#    y la correccion de Couette adimensional C = dP_ent / (2 tau_w), con tau_w el esfuerzo en la pared aguas abajo.
#    Se calcula en dos mallas para ver su convergencia.
# 3. Vortice de esquina (Moffatt): zona de recirculacion en la esquina saliente de la contraccion.
#
# Fluido newtoniano: mu = 1000 Pa s (orden de magnitud de un polimero fundido). Los no newtonianos llegan en la
# Etapa 12 del plan (docs/PLAN_FLUIDOS.md).

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
from Source.enums import StudyType, ProblemType, SolverType, DirectSolvers

H, h = 4.0e-3, 1.0e-3          # semialturas aguas arriba y aguas abajo [m]
L1, L2 = 16 * h, 16 * h        # longitudes de los tramos [m]
MU = 1000.0                    # viscosidad [Pa s]
Q = 1.0e-6                     # caudal por unidad de profundidad en la mitad del canal [m2/s]


def contraction_mesh(lc, refine=0.15):

    """Triangulos de gmsh con la malla refinada (lc * refine) en la esquina reentrante (0, h)."""

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add('contraction')
        geo = gmsh.model.geo
        pts = [(-L1, 0, lc), (L2, 0, lc), (L2, h, lc), (0, h, lc * refine), (0, H, lc), (-L1, H, lc)]
        p = [geo.addPoint(x, y, 0, size) for x, y, size in pts]
        lines = [geo.addLine(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]
        geo.addPlaneSurface([geo.addCurveLoop(lines)])
        geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [lines[0]], name='axis')
        gmsh.model.addPhysicalGroup(1, [lines[1]], name='outlet')
        gmsh.model.addPhysicalGroup(1, [lines[2], lines[3], lines[4]], name='walls')
        gmsh.model.addPhysicalGroup(1, [lines[5]], name='inlet')
        gmsh.model.mesh.generate(2)
        return Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()


def solve(lc):

    mesh = contraction_mesh(lc)
    model = Model(name='contraccion', dim=2, mesh=mesh, mat=material('fundido', miu=MU), psc=ff)

    U_in = 1.5 * Q / H                                     # velocidad maxima del perfil de entrada
    model.physics.addBC_OutletEdges(mesh.boundary_edges('outlet'), 0.0)
    model.physics.addBC_SymmetryEdges(mesh.boundary_edges('axis'))
    model.physics.addBC_VelocityEdges(mesh.boundary_edges('inlet'), lambda x, y: (U_in * (1 - (y / H) ** 2), 0.0))
    model.physics.addBC_NoSlipEdges(mesh.boundary_edges('walls'))
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear, Method=SolverType.direct,
                              Solver=DirectSolvers.PARDISO)
    model.solve()

    return model, mesh


def entrance_loss(model, mesh, q):

    """q: caudal discreto (el perfil de entrada interpolado lleva un caudal algo menor que Q, O(lc^2))."""

    inlet = np.unique(np.ravel(mesh.boundary_edges('inlet')))
    dp_total = model.sol['p'][inlet].mean()

    grad_up, grad_down = 3 * MU * q / H ** 3, 3 * MU * q / h ** 3       # |dp/dx| desarrollado: 2 mu U / b^2
    dp_ent = dp_total - grad_up * L1 - grad_down * L2
    tau_w = 3 * MU * q / h ** 2                                          # 2 mu U_max / h, aguas abajo

    return dp_total, dp_ent, dp_ent / (2 * tau_w)


results = []
for lc in (h / 4, h / 8, h / 16):
    start = time.perf_counter()
    model, mesh = solve(lc)
    elapsed = time.perf_counter() - start

    q_in = -model.physics.flowRate(mesh.boundary_edges('inlet'))
    q_out = model.physics.flowRate(mesh.boundary_edges('outlet'))
    dp_total, dp_ent, couette = entrance_loss(model, mesh, q_out)
    results.append((model, mesh, q_out))

    print(f"lc = h/{h / lc:.0f}: {mesh.getNoN()} nodos, {elapsed:.1f} s")
    print(f"   caudal entrante {q_in:.6e}, saliente {q_out:.6e} m2/s (diferencia {abs(q_in - q_out):.1e})")
    print(f"   caida total {dp_total / 1e5:.4f} bar; perdida de entrada {dp_ent / 1e5:.4f} bar; "
          f"correccion de Couette C = {couette:.4f}")

model, mesh, q_h = results[-1]
xy = np.array([n.coor for n in mesh.NL])
u = model.sol['u']

corner = (xy[:, 0] < 0) & (xy[:, 0] > -H) & (xy[:, 1] > h)
reverse = corner & (u[:, 0] < -1e-6 * np.abs(u[:, 0]).max())
if np.any(reverse):
    print(f"Vortice de esquina: flujo invertido hasta {-xy[reverse, 0].min() * 1e3:.2f} mm aguas arriba de la "
          f"contraccion ({-xy[reverse, 0].min() / (2 * H):.3f} veces el ancho aguas arriba)")

files = model.exportVTK(os.path.join('resultados_vtk', 'contraccion'))
print("Archivos para ParaView:", files)

# ----- Graficos -----
triang = mesh.triangulation(scale=1e3)

fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

cs = axes[0].tricontourf(triang, model.sol['p'] / 1e5, levels=40, cmap='viridis')
fig.colorbar(cs, ax=axes[0], label='p [bar]')
axes[0].set_title('Presion')

speed = np.linalg.norm(u, axis=1)
cs = axes[1].tricontourf(triang, speed * 1e3, levels=40, cmap='magma')
fig.colorbar(cs, ax=axes[1], label='|u| [mm/s]')
gx, gy = np.meshgrid(np.linspace(-L1, L2, 400) * 1e3, np.linspace(0, H, 120) * 1e3)
ux = mtri.LinearTriInterpolator(triang, u[:, 0])(gx, gy)
uy = mtri.LinearTriInterpolator(triang, u[:, 1])(gx, gy)
axes[1].streamplot(gx, gy, ux, uy, color='w', density=1.6, linewidth=0.6)
axes[1].set_title('Velocidad y lineas de corriente')

for ax in axes[:2]:
    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')

for model_k, mesh_k, _ in results:
    xy_k = np.array([n.coor for n in mesh_k.NL])
    on_axis = np.abs(xy_k[:, 1]) < 1e-12
    order = np.argsort(xy_k[on_axis, 0])
    axes[2].plot(xy_k[on_axis, 0][order] * 1e3, model_k.sol['p'][on_axis][order] / 1e5, label=f'{mesh_k.getNoN()} nodos')
x_line = np.array([-L1, 0.0, L2])
axes[2].plot(x_line * 1e3, np.array([3 * MU * q_h / H ** 3 * L1 + 3 * MU * q_h / h ** 3 * L2, 3 * MU * q_h / h ** 3 * L2, 0]) / 1e5,
             'k--', label='Poiseuille por tramos (sin perdida de entrada)')
axes[2].set_xlabel('x [mm]')
axes[2].set_ylabel('p en el eje [bar]')
axes[2].legend()
axes[2].grid(True)

plt.show()
