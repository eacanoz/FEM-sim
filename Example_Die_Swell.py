## Ejemplo: hinchamiento del extrudado a la salida de un dado (superficie libre, Source/Physics/FreeSurface.py)
#
# Dado plano de semialtura H = 1 y largo 10 H, extrudado de 20 H, flujo reptante con perfil parabolico a la entrada. La superficie
# libre se calcula por el metodo de la linea de corriente (Tanner): en cada iteracion se resuelve el flujo con traccion nula en la
# superficie, se integra dh/dx = v/u desde el borde del dado y se mueve la malla.
#
#   - newtoniano: B = h(final) / H frente a 1.190 +- 0.002 (Tanner 2002; Test/manual/die_swell.py lo converge en malla: 1.188, con
#     una extrapolacion a 1.186, dentro de los calculos publicados 1.184-1.200)
#   - Giesekus (alpha = 0.2, beta = 1/9) con continuacion en Wi = lam U / H: un poco de elasticidad reduce el hinchamiento (minimo
#     cerca de Wi = 0.25) y luego lo aumenta casi linealmente. Sin tabla de referencia: la tendencia es la publicada
#
# Cada iteracion de la superficie se imprime, para ver el avance.
#
# Parametros: python Example_Die_Swell.py [tamano en el borde, 0.05 grueso] [Wi maximo] [paso de Wi].

import logging
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import gmsh

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.Viscoelastic import ve
from Source.Physics.FreeSurface import FreeSurface
from Source.enums import StudyType, ProblemType
from Source.logger_config import sim_logger

sim_logger.setLevel(logging.WARNING)             # the progress is that of each iteration of the surface

H, L1, L2, U = 1.0, 10.0, 20.0, 1.0
LIP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
WI_MAX = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
STEP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
ALPHA, BETA = 0.2, 1.0 / 9.0


def extrusion_mesh():

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        geo = gmsh.model.geo
        far = 6 * LIP
        pts = [(-L1, 0, far), (L2, 0, far), (L2, H, far), (0, H, LIP), (-L1, H, far)]
        tags = [geo.addPoint(x, y, 0, c) for x, y, c in pts]
        lines = [geo.addLine(tags[i], tags[(i + 1) % 5]) for i in range(5)]
        geo.addPlaneSurface([geo.addCurveLoop(lines)])
        geo.synchronize()
        for name, group in (('axis', [lines[0]]), ('outlet', [lines[1]]), ('surface', [lines[2]]), ('wall', [lines[3]]), ('inlet', [lines[4]])):
            gmsh.model.addPhysicalGroup(1, group, name=name)
        field = gmsh.model.mesh.field
        dist = field.add('Distance')
        field.setNumbers(dist, 'PointsList', [tags[3]])
        thr = field.add('Threshold')
        field.setNumber(thr, 'InField', dist)
        field.setNumber(thr, 'SizeMin', LIP)
        field.setNumber(thr, 'SizeMax', far)
        field.setNumber(thr, 'DistMin', 0.02)
        field.setNumber(thr, 'DistMax', 2.0)
        field.setAsBackgroundMesh(thr)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
        return Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()


def conditions(mesh, flow):

    flow.addBC_VelocityEdges(mesh.boundary_edges('inlet'), lambda x, y: (1.5 * U * (1 - (y / H) ** 2), 0.0))
    flow.addBC_NoSlipEdges(mesh.boundary_edges('wall'))
    flow.addBC_SymmetryEdges(mesh.boundary_edges('axis'))
    flow.addBC_OutletEdges(mesh.boundary_edges('outlet'), 0.0)


shapes = {}

# ----- newtoniano -----
mesh = extrusion_mesh()
model = Model(name='hinchamiento', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.0), psc=ff)
conditions(mesh, model.physics)
model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear)
print(f"Newtoniano: {mesh.getNoN()} nodos, {model.layout.ndof} incognitas", flush=True)
surface = FreeSurface(model, mesh.boundary_edges('surface'), lip=0.0)
surface.solve()
shapes['newtoniano'] = surface.height()
B_newtonian = surface.swell()
print(f"Newtoniano: B = {surface.swell():.4f} (referencia 1.190 +- 0.002)\n", flush=True)

# ----- Giesekus, con continuacion en Wi -----
mesh = extrusion_mesh()
wi = 0.01
model = Model(name='hinchamiento', dim=2, mesh=mesh, mat=material('X', miu=BETA, rho=1.0), psc=[ff, ve])
flow, polymer = model.physics.fluid, model.physics.viscoelastic
conditions(mesh, flow)
polymer.setModes([(1 - BETA, wi * H / U, ALPHA)])
polymer.setDEVSS(1 - BETA)
polymer.addBC_OutletEdges(mesh.boundary_edges('outlet'))
polymer.addBC_InflowEdges(mesh.boundary_edges('inlet'), lambda x, y: [[0.0, -3.0 * U * y / H ** 2], [0.0, 0.0]])
model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
print(f"Giesekus (alpha {ALPHA}, beta {BETA:.3f}): {model.layout.ndof} incognitas", flush=True)
surface = FreeSurface(model, mesh.boundary_edges('surface'), lip=0.0)
rows = []
while wi <= WI_MAX + 1e-9:
    print(f"Wi = {wi:.2f}", flush=True)
    start = time.perf_counter()
    if not surface.solve():
        print(f"Wi = {wi:.2f}: la superficie no converge", flush=True)
        break
    rows.append((wi, surface.swell()))
    shapes[f'Wi = {wi:.2f}'] = surface.height()
    print(f"Wi = {wi:.2f}: B = {surface.swell():.4f} ({time.perf_counter() - start:.0f} s)\n", flush=True)
    surface.history = []
    new = STEP if wi < 0.05 else wi + STEP
    polymer.setWeissenberg(new / wi)
    wi = new

print("Wi, B:", ", ".join(f"{w:.2f} {b:.4f}" for w, b in rows))

fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for label, (s, h) in shapes.items():
    axes[0].plot(s, h, label=label)
axes[0].set_xlim(0, 8); axes[0].set_xlabel('x / H'); axes[0].set_ylabel('h / H'); axes[0].legend(); axes[0].grid(True)
axes[0].set_title('Superficie libre')
if rows:
    rows = np.array(rows)
    axes[1].plot(rows[:, 0], rows[:, 1], 'o-')
    axes[1].axhline(B_newtonian, color='k', ls='--', lw=0.8, label='newtoniano')
    axes[1].set_xlabel('Wi'); axes[1].set_ylabel('B'); axes[1].legend(); axes[1].grid(True)
    axes[1].set_title(f'Giesekus, alpha = {ALPHA}, beta = {BETA:.3f}')
plt.show()
