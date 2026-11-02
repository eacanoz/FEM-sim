## Ejemplo: desprendimiento de vórtices detrás de un cilindro (banco de pruebas de Schäfer y Turek, caso 2D-2)
#
# Canal de 2.2 x 0.41 con un cilindro de diámetro D = 0.1 centrado en (0.2, 0.2), ligeramente asimétrico en altura.
# Entrada parabólica con U_max = 1.5 (velocidad media U = 1), salida libre (p = 0), paredes y cilindro sin
# deslizamiento, rho = 1 y mu = 1e-3: Re = U D / nu = 100. El flujo arranca desde el reposo; la pequeña asimetría
# del canal y de la malla rompe la simetría y aparece la calle de vórtices de von Kármán.
#
# Navier-Stokes transitorio con BDF2 (segundo orden, L-estable) y Newton en cada paso. Las fuerzas sobre el cilindro
# se calculan por reacciones (ff.boundaryForce) y se comparan con los intervalos de referencia:
#     C_D,max = 3.22 - 3.24,   C_L,max = 0.99 - 1.01,   St = D f / U = 0.295 - 0.305
# (M. Schäfer y S. Turek, "Benchmark computations of laminar flow around a cylinder", 1996).
#
# Tiempo de cálculo: varios minutos (malla de unos miles de nodos, cientos de pasos de tiempo).

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import gmsh

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.enums import StudyType, ProblemType, SolverType, DirectSolvers

L, H, XC, YC, D = 2.2, 0.41, 0.2, 0.2, 0.1
U_MAX, MU, RHO = 1.5, 1e-3, 1.0
U_MEAN = 2.0 / 3.0 * U_MAX
T_END, DT = float(os.environ.get("CYLINDER_T_END", 8.0)), 0.01
T_FORCES = 2.0                      # the forces are evaluated over the last T_FORCES seconds (periodic regime)


def channel_mesh(lc_far=0.025, lc_cyl=0.006):

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add('cilindro')
        occ = gmsh.model.occ
        box = occ.addRectangle(0, 0, 0, L, H)
        disk = occ.addDisk(XC, YC, 0, D / 2, D / 2)
        fluid, _ = occ.cut([(2, box)], [(2, disk)])
        occ.synchronize()

        groups = {'inlet': [], 'outlet': [], 'walls': [], 'cylinder': []}
        for _, tag in gmsh.model.getBoundary(fluid, oriented=False):
            x, y, _ = occ.getCenterOfMass(1, tag)
            if abs(x) < 1e-9:
                groups['inlet'].append(tag)
            elif abs(x - L) < 1e-9:
                groups['outlet'].append(tag)
            elif abs(y) < 1e-9 or abs(y - H) < 1e-9:
                groups['walls'].append(tag)
            else:
                groups['cylinder'].append(tag)
        for name, tags in groups.items():
            gmsh.model.addPhysicalGroup(1, tags, name=name)

        field = gmsh.model.mesh.field
        dist = field.add('Distance')
        field.setNumbers(dist, 'CurvesList', groups['cylinder'])
        thr = field.add('Threshold')
        field.setNumber(thr, 'InField', dist)
        field.setNumber(thr, 'SizeMin', lc_cyl)
        field.setNumber(thr, 'SizeMax', lc_far)
        field.setNumber(thr, 'DistMin', 0.01)
        field.setNumber(thr, 'DistMax', 0.25)
        field.setAsBackgroundMesh(thr)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)

        gmsh.model.mesh.generate(2)
        return Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()


mesh = channel_mesh()
xy = np.array([node.coor for node in mesh.NL])
cylinder = mesh.boundary_edges('cylinder')

model = Model(name='cilindro', dim=2, mesh=mesh, mat=material('fluido', miu=MU, rho=RHO), psc=ff)
flow = model.physics
flow.setInertia(True)
flow.addBC_VelocityEdges(mesh.boundary_edges('inlet'), lambda x, y: (4 * U_MAX * y * (H - y) / H ** 2, 0.0))
flow.addBC_NoSlipEdges(mesh.boundary_edges('walls'))
flow.addBC_NoSlipEdges(cylinder)
flow.addBC_OutletEdges(mesh.boundary_edges('outlet'), 0.0, fully_developed=False)
flow.initField('u', 0.0)
print(f"Malla: {mesh.getNoN()} nodos, {mesh.getNoE()} triángulos, {model.layout.ndof} incógnitas; "
      f"Re = {RHO * U_MEAN * D / MU:.0f}")

model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, Method=SolverType.direct,
                          Solver=DirectSolvers.PARDISO, timeStep=DT, totalTime=T_END, timeScheme='BDF2')
start = time.perf_counter()
model.solve()
elapsed = time.perf_counter() - start
print(f"{len(model.step_iterations)} pasos en {elapsed / 60:.1f} min; Newton por paso: "
      f"{np.mean(model.step_iterations):.1f} de media, {max(model.step_iterations)} como máximo")

# ----- Fuerzas sobre el cilindro en el régimen periódico -----
times = np.asarray(model.timeVector)
U_hist, P_hist = model.sol['u'], model.sol['p']
layout = model.layout


def state(k):
    return layout.join({'u': U_hist[..., k], 'p': P_hist[:, k]})


steps = np.nonzero(times >= T_END - T_FORCES - 1e-9)[0]
drag, lift = [], []
for k in steps:
    flow.var['u'].updateField(U_hist[..., k])
    flow.var['p'].updateField(P_hist[:, k])
    rate = (1.5 * state(k) - 2.0 * state(k - 1) + 0.5 * state(k - 2)) / DT          # BDF2 derivative
    fx, fy = flow.boundaryForce(cylinder, rate=rate)
    drag.append(2 * fx / (RHO * U_MEAN ** 2 * D))
    lift.append(2 * fy / (RHO * U_MEAN ** 2 * D))
drag, lift, t_forces = np.array(drag), np.array(lift), times[steps]

# Strouhal: period from the zero crossings (upwards) of the lift
up = np.nonzero((lift[:-1] < 0) & (lift[1:] >= 0))[0]
crossings = t_forces[up] - lift[up] * (t_forces[up + 1] - t_forces[up]) / (lift[up + 1] - lift[up])
strouhal = D / (np.mean(np.diff(crossings)) * U_MEAN) if len(crossings) > 1 else np.nan

print(f"C_D,max = {drag.max():.4f}   (referencia 3.22 - 3.24)")
print(f"C_L,max = {lift.max():.4f}   (referencia 0.99 - 1.01)")
print(f"St      = {strouhal:.4f}   (referencia 0.295 - 0.305)")

files = model.exportVTK(os.path.join('resultados_vtk', 'cilindro'), every=10)
print(f"Serie para ParaView: {files[-1]} ({len(files) - 1} instantes)")

# ----- Gráficos -----
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2)
triang = mesh.triangulation()
G = model.geometry
omega = G.project(G.gradient(U_hist[..., -1])[..., 1, 0] - G.gradient(U_hist[..., -1])[..., 0, 1])

ax = fig.add_subplot(gs[0, :])
lim = np.percentile(np.abs(omega), 99)
cs = ax.tricontourf(triang, np.clip(omega, -lim, lim), levels=41, cmap='RdBu_r')
fig.colorbar(cs, ax=ax, label='vorticidad')
ax.add_patch(plt.Circle((XC, YC), D / 2, color='0.3'))
ax.set_aspect('equal'); ax.set_title(f'Calle de vórtices de von Kármán, t = {times[-1]:.2f}')

ax = fig.add_subplot(gs[1, 0])
ax.plot(t_forces, drag, label='C_D')
ax.axhspan(3.22, 3.24, color='tab:blue', alpha=0.15, label='C_D,max referencia')
ax.set_xlabel('t'); ax.set_ylabel('C_D'); ax.legend(); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
ax.plot(t_forces, lift, color='tab:orange', label='C_L')
ax.axhspan(0.99, 1.01, color='tab:orange', alpha=0.15, label='C_L,max referencia')
ax.set_xlabel('t'); ax.set_ylabel('C_L'); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
