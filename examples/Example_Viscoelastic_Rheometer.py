## Ejemplo: reometro capilar con un fundido viscoelastico multimodo: cuanto aporta la elasticidad a la perdida de entrada
#
# Barril de 9.5 mm y capilar de 1 mm de diametro con L/D = 16 (la geometria de Example_Capillary_Rheometer.py). El fundido es un
# espectro de 4 modos Giesekus, ilustrativo, con un 5 % de solvente. Se compara con un fluido inelastico que tiene EXACTAMENTE
# su misma curva de viscosidad (ve.inelasticEquivalent): lo que difiere entre ambos es la elasticidad. El esfuerzo en la pared debe
# coincidir, y lo hace: es una comprobacion independiente de los esfuerzos del polimero.
#
#   perdida de entrada = caida total - tramos desarrollados de barril y capilar  (correccion de Bagley)
#
# El modo mas largo fija el limite: sin ayuda, falla hacia 50 1/s (Wi = 50) con b casi singular en la pared junto a la esquina
# reentrante. Una difusion artificial pequena alrededor de la esquina (setDiffusion) lleva el calculo a tasas de proceso. Cambia algo
# la perdida de entrada a tasas bajas: -7 % con s = 4 y -4 % con s = 2 a 9 1/s, frente a sin difusion. Los resultados elasticos
# estan sujetos a esa incertidumbre, del orden del 4 %. Sin superficie libre: el hinchamiento es la estimacion de Tanner.
#
# Parametros: python Example_Viscoelastic_Rheometer.py [s, tamano de malla: 2 es fina, 4 gruesa] [gamma_a maximo].

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
from Source.enums import StudyType, ProblemType

RB, LB, RC, LC = 4.75e-3, 8.0e-3, 0.5e-3, 16.0e-3
MODES = [(100.0, 1e-3, 0.3), (300.0, 1e-2, 0.3), (300.0, 1e-1, 0.3), (245.0, 1.0, 0.3)]     # mu [Pa s], lam [s], alpha
TOTAL = sum(m[0] for m in MODES)
SOLVENT = 0.05 * TOTAL
SIZE = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
GAMMA_MAX = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0


def rheometer_mesh():

    lc_barrel, lc_cap, lc_corner = 0.35e-3 * SIZE, 0.06e-3 * SIZE, 0.015e-3 * SIZE
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        geo = gmsh.model.geo
        points = [(0, 0, lc_barrel), (RB, 0, lc_barrel), (RB, LB, lc_barrel), (RC, LB, lc_corner), (RC, LB + LC, lc_cap),
                  (0, LB + LC, lc_cap), (0, LB, lc_cap)]
        tags = [geo.addPoint(x, y, 0, c) for x, y, c in points]
        lines = [geo.addLine(tags[i], tags[(i + 1) % len(tags)]) for i in range(len(tags))]
        geo.addPlaneSurface([geo.addCurveLoop(lines)])
        geo.synchronize()
        for name, group in (('inlet', lines[:1]), ('wall', lines[1:4]), ('outlet', lines[4:5]), ('axis', lines[5:])):
            gmsh.model.addPhysicalGroup(1, group, name=name)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
        return Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()


mesh = rheometer_mesh()
xy = np.array([node.coor for node in mesh.NL])
inlet, outlet = mesh.boundary_edges('inlet'), mesh.boundary_edges('outlet')
inlet_nodes = np.unique(np.ravel(inlet))
axis = xy[:, 0] < 1e-12


def set_flow(flow, q):
    u_mean = q / (np.pi * RB ** 2)
    flow.addBC_VelocityEdges(inlet, lambda r, z: (0.0, 2 * u_mean * (1 - (r / RB) ** 2)))
    return u_mean


def measure(model, flow):

    def gradient(z0, z1):
        selected = axis & (xy[:, 1] > z0) & (xy[:, 1] < z1)
        return -np.polyfit(xy[selected, 1], model.sol['p'][selected], 1)[0]

    gamma = 4 * flow.flowRate(outlet) / (np.pi * RC ** 3)
    total = model.sol['p'][inlet_nodes].mean()
    capillary, barrel = gradient(LB + 0.4 * LC, LB + 0.9 * LC), gradient(0.1 * LB, 0.4 * LB)
    return gamma, total, total - capillary * LC - barrel * LB, RC / 2 * capillary


# ----- Viscoelastico, con continuacion en el caudal -----
model = Model(name='reometro', dim=2, mesh=mesh, mat=material('fundido', miu=SOLVENT), psc=[ff, ve], axisymmetric=True)
flow, polymer = model.physics.fluid, model.physics.viscoelastic
polymer.setModes(MODES)
polymer.setDEVSS(TOTAL)
polymer.setDiffusion(1.0, lambda x, y: np.exp(-((x - RC) ** 2 + (y - LB) ** 2) / (0.3 * RC) ** 2))
flow.addBC_SymmetryEdges(mesh.boundary_edges('axis'))
flow.addBC_NoSlipEdges(mesh.boundary_edges('wall'))
flow.addBC_OutletEdges(outlet, 0.0)
polymer.addBC_OutletEdges(outlet)

state = {'u_mean': set_flow(flow, np.pi * RC ** 3 / 4)}
polymer.addBC_InflowEdges(inlet, lambda r, z: [[0.0, 0.0], [-4 * state['u_mean'] * r / RB ** 2, 0.0]])
print(f"Malla: {mesh.getNoN()} nodos, {model.layout.ndof} incognitas")

elastic = []
gamma = 1.0
print(f"{'gamma_a':>8} {'caida [bar]':>11} {'entrada [bar]':>13} {'tau_w [kPa]':>11} {'Tanner':>7} {'Newton':>6} {'s':>6}")
while gamma <= GAMMA_MAX:
    state['u_mean'] = set_flow(flow, gamma * np.pi * RC ** 3 / 4)
    polymer.setWeissenberg(1.0)                          # recalcula los valores de entrada de b
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
    start = time.perf_counter()
    model.solve()
    if not model.nonlinear_converged:
        print(f"gamma_a = {gamma:.0f}: no converge. Limite de esta malla.")
        break

    row = measure(model, flow)
    section = (np.abs(xy[:, 0] - RC) < 1e-12) & (xy[:, 1] > LB + 0.5 * LC) & (xy[:, 1] < LB + 0.9 * LC)
    stress = model.sol['polymer_stress'][section].mean(axis=0)
    tanner = 0.1 + (1 + 0.5 * ((stress[2] - stress[0]) / (2 * row[3])) ** 2) ** (1.0 / 6.0)
    elastic.append(row + (tanner,))
    print(f"{row[0]:8.1f} {row[1] / 1e5:11.3f} {row[2] / 1e5:13.3f} {row[3] / 1e3:11.2f} {tanner:7.3f} {model.newton_iterations:6d} "
          f"{time.perf_counter() - start:6.1f}", flush=True)
    gamma *= 3.0 if gamma < 8 else 1.6

elastic = np.array(elastic)

# ----- Inelastico, con la misma curva de viscosidad -----
reference = Model(name='inelastico', dim=2, mesh=mesh, mat=material('fundido', miu=TOTAL), psc=ff, axisymmetric=True)
fluid = reference.physics
fluid.addBC_SymmetryEdges(mesh.boundary_edges('axis'))
fluid.addBC_NoSlipEdges(mesh.boundary_edges('wall'))
fluid.addBC_OutletEdges(outlet, 0.0)
set_flow(fluid, np.pi * RC ** 3 / 4)
reference.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear)
reference.solve()
fluid.setViscosityLaw(polymer.inelasticEquivalent(SOLVENT))
reference.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)

print(f"\n{'gamma_a':>8} {'entrada inelastica':>18} {'elastica':>9} {'razon':>6} {'tau_w: inel. / elast.':>22}")
inelastic = []
for row in elastic:
    set_flow(fluid, row[0] * np.pi * RC ** 3 / 4)
    reference.solve()
    if not reference.nonlinear_converged:
        raise RuntimeError(f"El fluido inelastico no converge a gamma_a = {row[0]:.1f}.")
    other = measure(reference, fluid)
    inelastic.append(other)
    print(f"{row[0]:8.1f} {other[2] / 1e5:18.3f} {row[2] / 1e5:9.3f} {row[2] / other[2]:6.2f} {other[3] / row[3]:22.4f}", flush=True)
inelastic = np.array(inelastic)

# ----- Graficos -----
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
axes[0].loglog(elastic[:, 0], elastic[:, 2] / 1e5, 'o-', label='viscoelastico')
axes[0].loglog(inelastic[:, 0], inelastic[:, 2] / 1e5, 's--', label='inelastico (misma viscosidad)')
axes[0].set_xlabel('tasa de corte aparente [1/s]'); axes[0].set_ylabel('perdida de entrada [bar]')
axes[0].grid(True, which='both', alpha=0.3); axes[0].legend()
axes[1].semilogx(elastic[:, 0], elastic[:, 2] / inelastic[:, 2], 'o-')
axes[1].axhline(1.0, color='k', lw=0.8)
axes[1].set_xlabel('tasa de corte aparente [1/s]'); axes[1].set_ylabel('razon de perdidas: elastica / inelastica')
axes[1].grid(True, which='both', alpha=0.3)
plt.show()
