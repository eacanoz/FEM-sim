## Ejemplo 2D: conduccion estacionaria con fuente, verificacion y convergencia de malla
#
# Problema: -k * lap(T) = Q(x, y) en el cuadrado [0, 1] x [0, 1], con T = 0 en el contorno.
# Solucion exacta (para verificar):  T(x, y) = sin(pi x) sin(pi y)
#                                    Q(x, y) = 2 pi^2 k sin(pi x) sin(pi y)
#
# El script resuelve el problema con mallas de 4, 8, 16 y 32 elementos por lado,
# calcula el error L2 frente a la solucion exacta, muestra la tabla de convergencia
# (el orden esperado para Q4 es 2) y dibuja dos graficos:
#   1. Campo de temperatura y error de la malla mas fina.
#   2. Error L2 frente al tamano de elemento h (escala log-log), con la pendiente 2 de referencia.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
from Source.Simulation.VectorizedAssembler import LinearGeometry
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers

# ----- Parametros del problema -----

k = 1.0          # Conductividad [W/(m K)]
L = 1.0          # Lado del cuadrado [m]
mesh_sizes = [4, 8, 16, 32]   # Elementos por lado en cada malla


def exact(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def source(x, y):
    return 2 * np.pi ** 2 * k * exact(x, y)


def solve_poisson(n):

    """Solves the problem on an n x n mesh. Returns the model, the mesh and the L2 error."""

    # Malla de cuadrilateros Q4
    mesh = Mesh()
    mesh.generate2DMesh((L, L), (n, n), ElementType.quadrilateral)

    # Modelo de conduccion de calor
    model = Model(name=f'Poisson_{n}x{n}', dim=2, mesh=mesh,
                  mat=material('Material', k=k, rho=1, Cp=1), psc=ht)
    model.physics.Convection = False
    model.physics.F_const = source  # fuente Q(x, y), evaluada en los puntos de Gauss

    # Temperatura nula en todo el contorno
    contorno = mesh.boundary_edges(lambda x, y: True)
    model.physics.addBC_TemperatureEdges(contorno, 0.0)

    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear,
                              Method=SolverType.direct, Solver=DirectSolvers.PARDISO)
    model.solve()

    # Error L2 con la misma cuadratura de Gauss 2x2 del ensamblador
    geo = LinearGeometry(mesh)
    T_nodal = np.asarray(model.sol['T'])
    T_gauss = np.einsum('gi,ei->eg', geo.N, T_nodal[geo.conn])
    x_gauss, y_gauss = geo.x_gp[..., 0], geo.x_gp[..., 1]
    error_L2 = np.sqrt(np.sum(geo.detJw * (T_gauss - exact(x_gauss, y_gauss)) ** 2))

    return model, mesh, error_L2


# ----- Convergencia de malla -----

print(f"{'Elementos':>12} {'h [m]':>10} {'Nodos':>8} {'Error L2':>14} {'Orden':>8}")
print("-" * 58)

results = []
for n in mesh_sizes:
    model, mesh, err = solve_poisson(n)
    h = L / n
    results.append((n, h, mesh.getNoN(), err, model))

    if len(results) == 1:
        orden = "-"
    else:
        orden = f"{np.log(results[-2][3] / err) / np.log(results[-2][1] / h):.2f}"

    print(f"{n}x{n:<10} {h:>10.4f} {mesh.getNoN():>8d} {err:>14.4e} {orden:>8}")

errores = np.array([r[3] for r in results])
tamanos = np.array([r[1] for r in results])
orden_medio = np.polyfit(np.log(tamanos), np.log(errores), 1)[0]
print("-" * 58)
print(f"Orden de convergencia medio (ajuste log-log): {orden_medio:.3f}  (esperado: 2)")


# ----- Graficos -----

n_fina, h_fina, _, err_fina, model_fina = results[-1]

# Grafico 1: campo de temperatura y error puntual en la malla mas fina
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

xy = np.array([node.coor[:2] for node in model_fina.mesh.NL])
triangulos = []
for element in model_fina.mesh.EL:
    a, b, c, d = element.getNodesId()
    triangulos += [[a, b, c], [a, c, d]]

triang =mtri.Triangulation(xy[:, 0], xy[:, 1], triangulos)

T_nodal = np.asarray(model_fina.sol['T'])
error_nodal = T_nodal - exact(xy[:, 0], xy[:, 1])

campo = ax1.tricontourf(triang, T_nodal, levels=30, cmap='inferno')
fig1.colorbar(campo, ax=ax1, label='T [K]')
ax1.set_title(f'Temperatura (malla {n_fina}x{n_fina})')

error_plot = ax2.tricontourf(triang, error_nodal, levels=30, cmap='RdBu_r')
fig1.colorbar(error_plot, ax=ax2, label='T_h - T_exacta')
ax2.set_title('Error nodal')

for ax in (ax1, ax2):
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

fig1.tight_layout()

# Grafico 2: convergencia log-log
fig2, ax = plt.subplots(figsize=(6, 5))

ax.loglog(tamanos, errores, 'o-', label='Error L2 (Q4)')

# Pendiente 2 de referencia, anclada en el primer punto
ref = errores[0] * (tamanos / tamanos[0]) ** 2
ax.loglog(tamanos, ref, 'k--', label='Pendiente 2 (referencia)')

for h, e in zip(tamanos, errores):
    ax.annotate(f'{e:.2e}', (h, e), textcoords='offset points', xytext=(6, 6), fontsize=8)

ax.set_xlabel('h = tamano del elemento [m]')
ax.set_ylabel('Error L2')
ax.set_title(f'Convergencia de malla (orden medio = {orden_medio:.2f})')
ax.grid(True, which='both', alpha=0.3)
ax.legend()
fig2.tight_layout()

plt.show()
