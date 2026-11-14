## Ejemplo: flujo de Darcy en un medio poroso heterogeneo y transporte de una especie
#
# Un canal de 2 x 1 m con una inclusion circular de baja permeabilidad (100 veces menor) en el centro.
# 1. Darcy: presion fija a la entrada (x = 0) y a la salida (x = 2), paredes superior e inferior impermeables.
#    La velocidad u = -(kappa/mu) grad p rodea la inclusion.
# 2. Transporte: una especie entra por una franja de la pared izquierda (0.4 < y < 0.6) y la velocidad de Darcy
#    la arrastra (conveccion-difusion estacionaria con SUPG). La pluma se abre al rodear la inclusion.
# 3. Comprobacion: el caudal que entra es igual al que sale (conservacion de masa del campo de velocidad).
#
# La velocidad se pasa de un modelo al otro en los puntos de Gauss (misma malla):
#     transport.physics.setVelocity(flow.physics.velocity())

import os

import numpy as np
import matplotlib.pyplot as plt

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.Darcy import darcy
from Source.Physics.MassTransfer import mt
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers

LX, LY = 2.0, 1.0
NX, NY = 80, 40
KAPPA, MU, DP = 1e-10, 1e-3, 1e4          # permeabilidad [m2], viscosidad [Pa s], caida de presion [Pa]
D = 5e-6                                  # dispersion hidrodinamica ~ alfa |u|, alfa = 1 cm [m2/s]


def permeability(x, y):
    inside = (x - 1.0) ** 2 + (y - 0.5) ** 2 < 0.2 ** 2
    return np.where(inside, KAPPA / 100.0, KAPPA)


def configure(model):
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear, Method=SolverType.direct,
                              Solver=DirectSolvers.PARDISO)


mesh = Mesh()
mesh.generate2DMesh((LX, LY), (NX, NY), ElementType.quadrilateral)

inlet = mesh.boundary_edges(lambda x, y: x < 1e-9)
outlet = mesh.boundary_edges(lambda x, y: x > LX - 1e-9)

# ----- 1. Darcy -----
flow = Model(name='darcy', dim=2, mesh=mesh, mat=material('arenisca', kappa=KAPPA, miu=MU), psc=darcy)
flow.physics.setPermeability(permeability)
flow.physics.addBC_PressureEdges(inlet, DP)
flow.physics.addBC_PressureEdges(outlet, 0.0)
configure(flow)
flow.solve()

u_nodes = flow.sol['u']
print(f"Velocidad media sin inclusion: {KAPPA / MU * DP / LX:.3e} m/s; maxima en la malla: "
      f"{np.max(np.linalg.norm(u_nodes, axis=1)):.3e} m/s")

# ----- 3. Conservacion: caudal por la entrada y por la salida -----
disc = flow.geometry
u_gp = flow.physics.velocity()


def flow_rate(edges):

    """Caudal saliente integral(u . n) por las aristas, con la velocidad del elemento que posee cada arista."""

    facets = disc.facets(edges)
    p = flow.sol['p']
    grad = np.einsum('egia,ei->ega', disc.B, p[disc.conn]).mean(axis=1)[facets.elems]      # (nf, dim)
    k = permeability(*disc.x_gp.mean(axis=1).T)[facets.elems]
    u = -(k / MU)[:, None] * grad
    return float(np.sum(np.einsum('fa,fqa->fq', u, facets.normals) * facets.detJw))


q_in, q_out = flow_rate(inlet), flow_rate(outlet)
print(f"Caudal entrante {-q_in:.6e} m2/s, saliente {q_out:.6e} m2/s, diferencia relativa {abs(q_in + q_out) / q_out:.1e}")

# ----- 2. Transporte de una especie -----
transport = Model(name='pluma', dim=2, mesh=mesh, mat=material('agua'), psc=mt)
transport.physics.setChemSpecies('A', 'trazador')
transport.physics.setDiffusivity('A', D)
transport.physics.setVelocity(u_gp)
transport.physics.setStabilization('SUPG')
transport.physics.addBC_ConcentrationEdges(mesh.boundary_edges(lambda x, y: x < 1e-9 and 0.4 < y < 0.6), 'A', 1.0)
transport.physics.addBC_ConcentrationEdges(mesh.boundary_edges(lambda x, y: x < 1e-9 and not 0.4 < y < 0.6), 'A', 0.0)
transport.physics.initField('A', 0.0)
configure(transport)
transport.solve()

c = np.asarray(transport.sol['A'])
Pe = KAPPA / MU * DP / LX * (LX / NX) / D
print(f"Peclet de malla ~ {Pe:.1f}; concentracion en [{c.min():.3f}, {c.max():.3f}]")

files = flow.exportVTK(os.path.join('resultados_vtk', 'darcy')) + transport.exportVTK(os.path.join('resultados_vtk', 'pluma'))
print("Archivos para ParaView:", files)

# ----- Graficos -----
xy = np.array([node.coor for node in mesh.NL])
triang = mesh.triangulation()

fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)

cs = axes[0].tricontourf(triang, flow.sol['p'], levels=30)
fig.colorbar(cs, ax=axes[0], label='p [Pa]')
axes[0].set_title('Presion')

speed = np.linalg.norm(u_nodes, axis=1)
cs = axes[1].tricontourf(triang, speed, levels=30, cmap='viridis')
fig.colorbar(cs, ax=axes[1], label='|u| [m/s]')
step = 4
sel = (np.arange(len(xy)) % (NX + 1) % step == 0) & (np.arange(len(xy)) // (NX + 1) % step == 0)
axes[1].quiver(xy[sel, 0], xy[sel, 1], u_nodes[sel, 0], u_nodes[sel, 1], color='w')
axes[1].set_title('Velocidad de Darcy')

cs = axes[2].tricontourf(triang, c, levels=np.linspace(0, 1, 21), cmap='magma')
fig.colorbar(cs, ax=axes[2], label='c [-]')
axes[2].set_title('Pluma del trazador')

for ax in axes:
    ax.add_patch(plt.Circle((1.0, 0.5), 0.2, fill=False, ls='--', color='k'))
    ax.set_aspect('equal')

plt.show()
