## Ejemplo: geometria no rectangular con gmsh y triangulos T3
#
# 1. Cuarto de anillo (r1 = 1, r2 = 2) con T = 100 C en el radio interior y 20 C en el exterior.
#    La solucion estacionaria es logaritmica, T = T1 + (T2 - T1) ln(r/r1) / ln(r2/r1): se mide el error
#    al refinar la malla.
# 2. Placa cuadrada con un orificio circular caliente, enfriada por conveccion en el contorno exterior.
#
# Las geometrias se construyen con la API de gmsh, se mallan en memoria y se importan con
# Mesh.from_gmsh_model(); los grupos fisicos (inner, outer, ...) se usan por nombre en las condiciones
# de contorno. Un archivo .msh se importa igual con Mesh.read_msh('archivo.msh').

import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import gmsh

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
from Source.enums import StudyType, ProblemType, SolverType, DirectSolvers

R1, R2, T1, T2 = 1.0, 2.0, 100.0, 20.0


def mesh_with_gmsh(build):

    """Runs build(gmsh.model.geo) -> None inside a gmsh session and returns the imported Mesh."""

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add('geometry')
        build(gmsh.model)
        gmsh.model.mesh.generate(2)
        return Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()


def quarter_annulus(lc):

    def build(model):
        geo = model.geo
        c = geo.addPoint(0, 0, 0, lc)
        p = [geo.addPoint(R1, 0, 0, lc), geo.addPoint(R2, 0, 0, lc), geo.addPoint(0, R2, 0, lc), geo.addPoint(0, R1, 0, lc)]
        lines = [geo.addLine(p[0], p[1]), geo.addCircleArc(p[1], c, p[2]), geo.addLine(p[2], p[3]), geo.addCircleArc(p[3], c, p[0])]
        surface = geo.addPlaneSurface([geo.addCurveLoop(lines)])
        geo.synchronize()
        model.addPhysicalGroup(1, [lines[3]], name='inner')
        model.addPhysicalGroup(1, [lines[1]], name='outer')
        model.addPhysicalGroup(2, [surface], name='plate')

    return mesh_with_gmsh(build)


def plate_with_hole(lc, side=2.0, radius=0.4):

    def build(model):
        occ = model.occ
        plate = occ.addRectangle(-side / 2, -side / 2, 0, side, side)
        hole = occ.addDisk(0, 0, 0, radius, radius)
        cut, _ = occ.cut([(2, plate)], [(2, hole)])
        occ.synchronize()
        surface = cut[0][1]

        boundary = model.getBoundary([(2, surface)], oriented=False)
        outer, inner = [], []
        for _, tag in boundary:
            x, y, _ = model.occ.getCenterOfMass(1, tag)
            (inner if np.hypot(x, y) < 1e-6 else outer).append(tag)   # the circle's center of mass is the origin

        model.addPhysicalGroup(1, inner, name='hole')
        model.addPhysicalGroup(1, outer, name='outside')
        model.addPhysicalGroup(2, [surface], name='plate')
        gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
        gmsh.option.setNumber("Mesh.MeshSizeMin", lc / 3)

    return mesh_with_gmsh(build)


def solve(mesh, setup):
    model = Model(name='gmsh', dim=2, mesh=mesh, mat=material('X', k=3.0, rho=1.0, Cp=1.0), psc=ht)
    model.physics.Convection = False
    setup(model.physics, mesh)
    model.physics.initField('T', 50.0)
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, Method=SolverType.direct,
                              Solver=DirectSolvers.PARDISO)
    start = time.time()
    model.solve()
    return model, time.time() - start


def plot_mesh_and_field(ax, model, title):
    xy = np.array([n.coor for n in model.mesh.NL])
    tris = np.array([e.getNodesId() for e in model.mesh.EL])
    triang = mtri.Triangulation(xy[:, 0], xy[:, 1], tris)
    contour = ax.tricontourf(triang, np.asarray(model.sol['T']), levels=30, cmap='inferno')
    ax.triplot(triang, color='white', lw=0.25, alpha=0.5)
    ax.figure.colorbar(contour, ax=ax, label='T [C]')
    ax.set_aspect('equal')
    ax.set_title(title)


# ----- 1. Cuarto de anillo: convergencia frente a la solucion logaritmica -----

def annulus_setup(physics, mesh):
    physics.addBC_TemperatureEdges(mesh.boundary_edges('inner'), T1)
    physics.addBC_TemperatureEdges(mesh.boundary_edges('outer'), T2)


print("1. Cuarto de anillo (T3) frente a T = T1 + (T2 - T1) ln(r/r1)/ln(r2/r1)")
print(f"   {'lc':>7} {'elementos':>10} {'error max [C]':>14} {'orden':>7} {'t calc [s]':>11}")
print("   " + "-" * 55)

previous, models = None, {}
for lc in (0.4, 0.2, 0.1, 0.05):
    mesh = quarter_annulus(lc)
    model, elapsed = solve(mesh, annulus_setup)
    xy = np.array([n.coor for n in mesh.NL])
    r = np.hypot(xy[:, 0], xy[:, 1])
    error = np.max(np.abs(np.asarray(model.sol['T']) - (T1 + (T2 - T1) * np.log(r / R1) / np.log(R2 / R1))))
    order = '' if previous is None else f"{np.log2(previous / error):.2f}"
    print(f"   {lc:>7.3f} {mesh.getNoE():>10d} {error:>14.4e} {order:>7} {elapsed:>11.3f}")
    previous, models[lc] = error, model

print("   (el orden se degrada algo por la aproximacion poligonal del arco con elementos de lados rectos)")

# ----- 2. Placa con orificio caliente -----

def plate_setup(physics, mesh):
    physics.addBC_TemperatureEdges(mesh.boundary_edges('hole'), 150.0)
    physics.addBC_ConvectionEdges(mesh.boundary_edges('outside'), 10.0, 25.0)


mesh_hole = plate_with_hole(0.12)
model_hole, elapsed = solve(mesh_hole, plate_setup)
T_hole = np.asarray(model_hole.sol['T'])
print(f"\n2. Placa con orificio ({mesh_hole.getNoE()} triangulos, {elapsed:.3f} s): T entre {T_hole.min():.2f} y {T_hole.max():.2f} C")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
plot_mesh_and_field(axes[0], models[0.2], 'Cuarto de anillo, lc = 0.2')
plot_mesh_and_field(axes[1], models[0.05], 'Cuarto de anillo, lc = 0.05')
plot_mesh_and_field(axes[2], model_hole, 'Placa con orificio caliente')
fig.tight_layout()
plt.show()
