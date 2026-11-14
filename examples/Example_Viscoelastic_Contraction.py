## Ejemplo: contraccion plana 4:1 de un fluido viscoelastico (entrada a un dado)
#
# Un canal de semialtura H = 4 h se reduce a uno de semialtura h, en flujo reptante; por simetria, la mitad superior. El fluido
# es Oldroyd-B con beta = 1/9 (el de los fluidos de Boger) o Giesekus, con Taylor-Hood, raiz de la conformacion y DEVSS-G.
#
# Para cada De = lam U / h, con U la velocidad media aguas abajo y continuacion desde el De anterior, da:
#   - la correccion de Couette C = dP_entrada / (2 tau_w), con tau_w = 3 mu U / h el esfuerzo newtoniano en la pared aguas abajo;
#   - la longitud del vortice de esquina X_R, medida desde la contraccion y dividida por h, como en Alves et al. (2003);
#     Test/manual/contraction_alves.py la compara con sus tablas en tres mallas;
#   - la estimacion de Tanner del hinchamiento del extrudado en la seccion desarrollada. Es solo una estimacion: el
#     hinchamiento real necesita superficie libre, que este codigo no tiene.
#
# Parametros: python Example_Viscoelastic_Contraction.py [divisiones de h] [De maximo] [alpha] [incremento de De].

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

H, h, L1, L2 = 4.0, 1.0, 20.0, 20.0
U = 1.0
MU, BETA = 1.0, 1.0 / 9.0
DIVISIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
DE_MAX = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
STEP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1


def contraction_mesh(lc, refine=0.15):

    """Triangulos cuadraticos con la malla refinada en la esquina reentrante (0, h) y una zona fina alrededor de ella."""

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        geo = gmsh.model.geo
        far = 4.0 * lc
        points = [(-L1, 0, far), (L2, 0, far), (L2, h, far), (0, h, lc * refine), (0, H, lc), (-L1, H, far)]
        tags = [geo.addPoint(x, y, 0, size) for x, y, size in points]
        lines = [geo.addLine(tags[i], tags[(i + 1) % 6]) for i in range(6)]
        geo.addPlaneSurface([geo.addCurveLoop(lines)])
        geo.synchronize()
        for name, group in (('axis', lines[:1]), ('outlet', lines[1:2]), ('walls', lines[2:5]), ('inlet', lines[5:])):
            gmsh.model.addPhysicalGroup(1, group, name=name)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
        return Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()


lc_wall = h / DIVISIONS
mesh = contraction_mesh(lc_wall)
xy = np.array([n.coor for n in mesh.NL])
mu_s, mu_p = BETA * MU, (1 - BETA) * MU

de = 0.1
model = Model(name='contraccion', dim=2, mesh=mesh, mat=material('X', miu=mu_s), psc=[ff, ve])
flow, polymer = model.physics.fluid, model.physics.viscoelastic
polymer.setModes([(mu_p, de * h / U, ALPHA)])
polymer.setDEVSS(mu_p)

inlet, outlet = mesh.boundary_edges('inlet'), mesh.boundary_edges('outlet')
U_in = 1.5 * U * h / H                                       # velocidad maxima de la parabola de entrada: mismo caudal
flow.addBC_VelocityEdges(inlet, lambda x, y: (U_in * (1 - (y / H) ** 2), 0.0))
flow.addBC_NoSlipEdges(mesh.boundary_edges('walls'))
flow.addBC_SymmetryEdges(mesh.boundary_edges('axis'))
flow.addBC_OutletEdges(outlet, 0.0)
polymer.addBC_OutletEdges(outlet)
polymer.addBC_InflowEdges(inlet, lambda x, y: [[0.0, -2.0 * U_in * y / H ** 2], [0.0, 0.0]])
print(f"Malla: {mesh.getNoN()} nodos, {model.layout.ndof} incognitas; beta = {BETA:.3f}, alpha = {ALPHA}")


def entrance_correction():

    nodes = np.unique(np.ravel(inlet))
    q = flow.flowRate(outlet)
    dp = model.sol['p'][nodes].mean()
    return (dp - 3 * MU * q / H ** 3 * L1 - 3 * MU * q / h ** 3 * L2) / (2 * 3 * MU * q / h ** 2)


wall = np.flatnonzero((np.abs(xy[:, 1] - H) < 1e-9) & (xy[:, 0] < -1e-9))
wall = wall[np.argsort(xy[wall, 0])]


def corner_vortex():

    """Donde cambia de signo du/dy (la vorticidad en la pared aguas arriba), proyectado a los nodos de la pared: X_R = distancia a la
    contraccion / h."""

    disc = model.geometry
    shear = disc.project(disc.gradient(model.sol['u'])[:, :, 0, 1])[wall]
    x = xy[wall, 0]
    reversed_flow = np.nonzero(shear > 0.0)[0]
    if not len(reversed_flow) or reversed_flow[0] == 0:
        return 0.0
    k = reversed_flow[0]
    return -(x[k - 1] + (x[k] - x[k - 1]) * (0.0 - shear[k - 1]) / (shear[k] - shear[k - 1])) / h


def tanner_swell():

    """Estimacion de Tanner en la pared del tramo desarrollado, flujo plano: (1 + (N1 / (2 tau_w))^2 / 2)^(1/4)."""

    section = (np.abs(xy[:, 0] - 0.5 * L2) < 1.5 * h) & (xy[:, 1] > h - 1e-9)
    s = model.sol['polymer_stress'][section].mean(axis=0)
    return (1.0 + 0.5 * ((s[0] - s[2]) / (2.0 * (s[1] + mu_s * -3.0 * U / h))) ** 2) ** (1.0 / 4.0)


rows = []
print(f"{'De':>5} {'C':>8} {'X_R':>7} {'Tanner':>7} {'Newton':>6} {'s':>7}")
while de <= DE_MAX + 1e-9:
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
    start = time.perf_counter()
    model.solve()
    if not model.nonlinear_converged:
        print(f"De = {de:.2f}: no converge ({model.newton_iterations} iteraciones). Limite de esta malla.")
        break
    rows.append((de, entrance_correction(), corner_vortex(), tanner_swell()))
    print(f"{de:5.2f} {rows[-1][1]:8.4f} {rows[-1][2]:7.3f} {rows[-1][3]:7.3f} {model.newton_iterations:6d} {time.perf_counter() - start:7.1f}", flush=True)
    polymer.setWeissenberg((de + STEP) / de)
    de += STEP

rows = np.array(rows)
if len(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(rows[:, 0], rows[:, 1], 'o-')
    axes[0].set_xlabel('De'); axes[0].set_ylabel('Correccion de Couette C'); axes[0].grid(True)
    axes[1].plot(rows[:, 0], rows[:, 2], 's-')
    axes[1].set_xlabel('De'); axes[1].set_ylabel('Vortice de esquina X_R / h'); axes[1].grid(True)
    plt.show()
