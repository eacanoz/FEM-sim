## Ejemplo: reometro capilar no isotermo con un fundido viscoelastico multimodo
#
# La geometria y el espectro de Example_Viscoelastic_Rheometer.py (barril de 9.5 mm, capilar de 1 mm con L/D = 16, 4 modos Giesekus,
# 5 % de solvente), ahora con calor: el fundido entra a 220 C, las paredes estan a 220 C, y los tiempos de relajacion y el solvente
# se desplazan con T (Arrhenius, E/R = 5000 K). El calentamiento es el trabajo de todo el esfuerzo (polimero y solvente). Para cada
# tasa aparente se compara con el calculo isotermo a 220 C:
#
#   caida total, perdida de entrada (Bagley), aumento maximo de T y T media de mezcla a la salida
#
# Propiedades termicas de un PP tipico (rho 900 kg/m3, Cp 2000 J/kg K, k 0.2 W/m K). El espectro es ilustrativo. Cada iteracion
# de los resolvedores se imprime con el tiempo, para ver el avance y detectar un estancamiento.
#
# Parametros: python Example_Viscoelastic_NonIsothermal.py [s, tamano de malla: 2 fina, 4 gruesa] [gamma_a maximo].

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
from Source.Physics.HeatTransfer import ht
from Source.Physics.Viscoelastic import ve
from Source.Physics.Rheology import Arrhenius, Newtonian, arrhenius_shift
from Source.enums import StudyType, ProblemType
from Source.logger_config import sim_logger

RB, LB, RC, LC = 4.75e-3, 8.0e-3, 0.5e-3, 16.0e-3
MODES = [(100.0, 1e-3, 0.3), (300.0, 1e-2, 0.3), (300.0, 1e-1, 0.3), (245.0, 1.0, 0.3)]     # mu [Pa s], lam [s], alpha a T_REF
TOTAL = sum(m[0] for m in MODES)
SOLVENT = 0.05 * TOTAL
T_REF = T_IN = T_WALL = 220.0
E_R = 5000.0
RHO, CP, K = 900.0, 2000.0, 0.2
SIZE = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
GAMMA_MAX = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0
START = time.perf_counter()


class Progress(logging.Handler):

    """Una linea por iteracion de Newton o del pseudo-transitorio, con el tiempo transcurrido: un estancamiento se ve enseguida."""

    tag = ''

    def emit(self, record):
        message = record.getMessage()
        if message.startswith('|') and message[1:8].strip().isdigit() or 'did not' in message or 'stopped' in message:
            print(f"      [{time.perf_counter() - START:7.0f} s] {self.tag} {message.strip()[:80]}", flush=True)


PROGRESS = Progress()
for handler in list(sim_logger.handlers):
    sim_logger.removeHandler(handler)
sim_logger.addHandler(PROGRESS)


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
inlet, outlet, wall = mesh.boundary_edges('inlet'), mesh.boundary_edges('outlet'), mesh.boundary_edges('wall')
inlet_nodes = np.unique(np.ravel(inlet))
axis = xy[:, 0] < 1e-12
row_out = np.unique(np.ravel(outlet))
row_out = row_out[np.argsort(xy[row_out, 0])]


def build(thermal):

    psc = [ff, ht, ve] if thermal else [ff, ve]
    model = Model(name='reometro', dim=2, mesh=mesh, mat=material('PP', miu=SOLVENT, rho=RHO, Cp=CP, k=K), psc=psc, axisymmetric=True)
    flow, polymer = model.physics.fluid, model.physics.viscoelastic
    polymer.setModes(MODES)
    polymer.setDEVSS(TOTAL)
    polymer.setDiffusion(1.0, lambda x, y: np.exp(-((x - RC) ** 2 + (y - LB) ** 2) / (0.3 * RC) ** 2))
    flow.addBC_SymmetryEdges(mesh.boundary_edges('axis'))
    flow.addBC_NoSlipEdges(wall)
    flow.addBC_OutletEdges(outlet, 0.0)
    polymer.addBC_OutletEdges(outlet)
    if thermal:
        heat = model.physics.heat
        heat.setVelocity('u')
        heat.setStabilization('SUPG')
        heat.addBC_TemperatureEdges(inlet, T_IN)
        heat.addBC_TemperatureEdges(wall, T_WALL)
        heat.var['T'].updateField(np.full(mesh.getNoN(), T_IN))
        flow.setViscosityLaw(Arrhenius(Newtonian(SOLVENT), E_R, T_REF))
        flow.setViscousHeating()
        polymer.setTemperatureShift(arrhenius_shift(E_R, T_REF))
        polymer.setViscousHeating()
    state = {'u_mean': 0.0}
    polymer.addBC_InflowEdges(inlet, lambda r, z: [[0.0, 0.0], [-4 * state['u_mean'] * r / RB ** 2, 0.0]], T_IN)
    return model, flow, polymer, state


def set_rate(flow, polymer, state, gamma):

    state['u_mean'] = gamma * np.pi * RC ** 3 / 4 / (np.pi * RB ** 2)
    flow.addBC_VelocityEdges(inlet, lambda r, z: (0.0, 2 * state['u_mean'] * (1 - (r / RB) ** 2)))
    polymer.setWeissenberg(1.0)                                   # recalcula los valores de entrada de b


def measure(model, flow, thermal):

    def gradient(z0, z1):
        selected = axis & (xy[:, 1] > z0) & (xy[:, 1] < z1)
        return -np.polyfit(xy[selected, 1], model.sol['p'][selected], 1)[0]

    total = model.sol['p'][inlet_nodes].mean()
    capillary, barrel = gradient(LB + 0.4 * LC, LB + 0.9 * LC), gradient(0.1 * LB, 0.4 * LB)
    entrance = total - capillary * LC - barrel * LB
    if not thermal:
        return total, entrance, 0.0, T_IN
    T = np.asarray(model.sol['T'])
    w = np.asarray(model.sol['u'])[row_out, 1] * xy[row_out, 0]
    r = xy[row_out, 0]
    mixing = np.trapezoid(w * T[row_out], r) / np.trapezoid(w, r)
    return total, entrance, T.max() - T_IN, mixing


results = {}
for thermal in (False, True):
    label = 'no isotermo' if thermal else 'isotermo'
    model, flow, polymer, state = build(thermal)
    print(f"\n{label}: {mesh.getNoN()} nodos, {model.layout.ndof} incognitas", flush=True)
    print(f"{'gamma_a':>8} {'caida [bar]':>11} {'entrada [bar]':>13} {'dT max [K]':>10} {'T mezcla':>9} {'Newton':>6} {'s':>6}", flush=True)
    rows = []
    gamma = 1.0
    while gamma <= GAMMA_MAX:
        set_rate(flow, polymer, state, gamma)
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
        PROGRESS.tag = f"{label} {gamma:.0f} 1/s"
        start = time.perf_counter()
        model.solve()
        if not model.nonlinear_converged:
            print(f"gamma_a = {gamma:.0f}: no converge. Limite de esta malla.", flush=True)
            break
        total, entrance, dT, mixing = measure(model, flow, thermal)
        rows.append((gamma, total, entrance, dT, mixing))
        print(f"{gamma:8.1f} {total / 1e5:11.3f} {entrance / 1e5:13.3f} {dT:10.2f} {mixing:9.2f} {model.newton_iterations:6d} "
              f"{time.perf_counter() - start:6.1f}", flush=True)
        gamma *= 3.0 if gamma < 8 else 1.6
    results[thermal] = np.array(rows)

iso, hot = results[False], results[True]
n = min(len(iso), len(hot))
print(f"\n{'gamma_a':>8} {'caida no iso / iso':>18} {'entrada no iso / iso':>20} {'dT max [K]':>10}")
for a, b in zip(iso[:n], hot[:n]):
    print(f"{a[0]:8.1f} {b[1] / a[1]:18.3f} {b[2] / a[2]:20.3f} {b[3]:10.2f}")

if n:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].loglog(iso[:, 0], iso[:, 1] / 1e5, 'o-', label='isotermo')
    axes[0].loglog(hot[:, 0], hot[:, 1] / 1e5, 's-', label='no isotermo')
    axes[0].set_xlabel('gamma_a [1/s]'); axes[0].set_ylabel('caida total [bar]'); axes[0].legend(); axes[0].grid(True, which='both')
    axes[1].semilogx(hot[:, 0], hot[:, 3], 's-')
    axes[1].set_xlabel('gamma_a [1/s]'); axes[1].set_ylabel('aumento maximo de T [K]'); axes[1].grid(True, which='both')
    plt.show()
