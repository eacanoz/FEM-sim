## Ejemplo: escalado de solvers y salida para ParaView
#
# Poisson 2D (-lap T = 1 en el cuadrado unitario, T = 0 en el contorno) con mallas Q4 cada vez mas finas.
# Se compara el tiempo total (ensamblado + solucion) y las iteraciones de:
#   - directo PARDISO
#   - CG sin precondicionador (las iteraciones crecen con 1/h)
#   - CG + AMG (pyamg: las iteraciones casi no dependen de la malla)
# Luego se muestra la reutilizacion de la factorizacion en un transitorio lineal y se exporta el
# resultado a VTK (abra resultados_vtk/transitorio.pvd en ParaView).

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers, IterativeSolvers, Preconditioners


def poisson(n):
    mesh = Mesh()
    mesh.generate2DMesh((1.0, 1.0), (n, n), ElementType.quadrilateral)
    model = Model(name='poisson', dim=2, mesh=mesh, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
    model.physics.Convection = False
    model.physics.F_const = 1.0
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(), 0.0)
    model.physics.initField('T', 0.0)
    return model


def run(n, label):
    model = poisson(n)
    if label == 'PARDISO':
        model.solverConfiguration(Method=SolverType.direct, Solver=DirectSolvers.PARDISO)
    elif label == 'CG':
        model.solverConfiguration(Method=SolverType.iterative, Solver=IterativeSolvers.CG, prec=None)
    else:
        model.solverConfiguration(Method=SolverType.iterative, Solver=IterativeSolvers.CG, prec=Preconditioners.AMG)

    start = time.time()
    model.solve()
    elapsed = time.time() - start
    iterations = model.solver_iterations[0] if model.solver_iterations else 0
    return np.asarray(model.sol['T']), elapsed, iterations


print("1. Poisson 2D: tiempo total [s] (ensamblado + solucion) e iteraciones")
print(f"   {'n':>5} {'incognitas':>11} | {'PARDISO':>9} | {'CG':>9} {'it':>6} | {'CG+AMG':>9} {'it':>5} | {'dif. AMG-directo':>17}")
print("   " + "-" * 82)

sizes = (50, 100, 200, 400)
rows = []
for n in sizes:
    T_direct, t_direct, _ = run(n, 'PARDISO')
    T_amg, t_amg, it_amg = run(n, 'AMG')
    if n <= 200:
        _, t_cg, it_cg = run(n, 'CG')
    else:
        t_cg, it_cg = float('nan'), 0
    rows.append((n, t_direct, t_cg, it_cg, t_amg, it_amg))
    print(f"   {n:>5} {(n + 1) ** 2:>11} | {t_direct:>9.3f} | {t_cg:>9.3f} {it_cg:>6} | {t_amg:>9.3f} {it_amg:>5} | "
          f"{np.max(np.abs(T_amg - T_direct)):>17.2e}")

# ----- 2. Reutilizacion de la factorizacion -----

print("\n2. Transitorio lineal (200 pasos) en una malla de 100 x 100: una sola factorizacion")
model = poisson(100)
model.physics.initField('T', lambda x, y: 100.0 * x * (1 - x) * y * (1 - y))
model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.linear, timeDisc=1.0, timeStep=0.001, totalTime=0.2)
start = time.time()
model.solve()
print(f"   {time.time() - start:.2f} s, factorizaciones: {model._factorizer.n_factorizations}")

files = model.exportVTK(os.path.join('resultados_vtk', 'transitorio'), every=20)
print(f"   exportado a ParaView: {len(files) - 1} pasos en {files[-1]}")

# ----- grafico -----

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
dofs = [(n + 1) ** 2 for n, *_ in rows]
ax1.loglog(dofs, [r[1] for r in rows], 'o-', label='PARDISO')
ax1.loglog(dofs, [r[4] for r in rows], 's-', label='CG + AMG')
ax1.loglog(dofs[:3], [r[2] for r in rows][:3], '^--', label='CG')
ax1.set_xlabel('incognitas')
ax1.set_ylabel('tiempo total [s]')
ax1.legend()
ax1.grid(True, which='both', alpha=0.3)

ax2.semilogx(dofs, [r[5] for r in rows], 's-', label='CG + AMG')
ax2.semilogx(dofs[:3], [r[3] for r in rows][:3], '^--', label='CG')
ax2.set_xlabel('incognitas')
ax2.set_ylabel('iteraciones')
ax2.legend()
ax2.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()
