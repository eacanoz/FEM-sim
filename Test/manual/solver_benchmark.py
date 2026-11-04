"""Compares the linear solvers on the scenarios of the project and reports which is the fastest valid one for each.

Run from the project root:  python Test/manual/solver_benchmark.py [spd] [transient] [convection] [stokes] [cavity]
No argument runs everything. Valid: converged and within 1e-6 of the first direct solution (relative to its size).
"""
import sys, os, logging
sys.path.insert(0, os.getcwd())
import numpy as np
import gmsh
from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
from Source.Physics.FluidFlow import ff
from Source.Simulation.SolverSelection import CANDIDATES, benchmark, format_table
from Source.enums import ElementType, StudyType, ProblemType
from Source.logger_config import sim_logger

sim_logger.setLevel(logging.ERROR)

QUAD = ElementType.quadrilateral
STEADY = dict(Study=StudyType.steady_state, Type=ProblemType.linear, tol=1e-10)


def poisson(n):
    def build():
        mesh = Mesh()
        mesh.generate2DMesh((1.0, 1.0), (n, n), QUAD)
        model = Model(name='poisson', dim=2, mesh=mesh, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
        model.physics.Convection = False
        model.physics.F_const = 1.0
        model.physics.addBC_TemperatureEdges(mesh.boundary_edges(), 0.0)
        model.physics.initField('T', 0.0)
        return model
    return build


def transient_poisson(n):
    build = poisson(n)
    def make():
        model = build()
        model.physics.initField('T', lambda x, y: 100.0 * x * (1 - x) * y * (1 - y))
        return model
    return make


def convection(n, Pe):
    def build():
        mesh = Mesh()
        mesh.generate2DMesh((1.0, 1.0), (n, n), QUAD)
        model = Model(name='conv', dim=2, mesh=mesh, mat=material('X', k=1.0 / Pe, rho=1.0, Cp=1.0), psc=ht)
        heat = model.physics
        heat.setVelocity((1.0, 0.5))
        heat.setStabilization('SUPG')
        heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x < 1e-9 or y < 1e-9), 0.0)
        heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x > 1 - 1e-9 or y > 1 - 1e-9), 1.0)
        heat.initField('T', 0.0)
        return model
    return build


def stokes_contraction(lc):
    def build():
        h, H, L = 1.0, 4.0, 12.0
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            geo = gmsh.model.geo
            pts = [(-L, 0, lc), (L, 0, lc), (L, h, lc), (0, h, lc * 0.2), (0, H, lc), (-L, H, lc)]
            p = [geo.addPoint(x, y, 0, s) for x, y, s in pts]
            lines = [geo.addLine(p[i], p[(i + 1) % 6]) for i in range(6)]
            geo.addPlaneSurface([geo.addCurveLoop(lines)])
            geo.synchronize()
            for name, group in (('axis', lines[:1]), ('outlet', lines[1:2]), ('walls', lines[2:5]), ('inlet', lines[5:])):
                gmsh.model.addPhysicalGroup(1, group, name=name)
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.setOrder(2)
            mesh = Mesh.from_gmsh_model()
        finally:
            gmsh.finalize()
        model = Model(name='stokes', dim=2, mesh=mesh, mat=material('X', miu=1.0), psc=ff)
        flow = model.physics
        flow.addBC_VelocityEdges(mesh.boundary_edges('inlet'), lambda x, y: (1.5 * h / H * (1 - (y / H) ** 2), 0.0))
        flow.addBC_NoSlipEdges(mesh.boundary_edges('walls'))
        flow.addBC_SymmetryEdges(mesh.boundary_edges('axis'))
        flow.addBC_OutletEdges(mesh.boundary_edges('outlet'), 0.0)
        return model
    return build


def cavity(n, Re):
    def build():
        mesh = Mesh()
        mesh.generate2DMesh((1.0, 1.0), (n, n), QUAD)
        model = Model(name='cavity', dim=2, mesh=mesh, mat=material('f', miu=1.0 / Re, rho=1.0), psc=ff)
        model.physics.setInertia(True)
        model.physics.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), (1.0, 0.0))
        model.physics.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or x < 1e-9 or x > 1 - 1e-9))
        return model
    return build


def run(title, build, base, kind, field):
    only = os.environ.get('BENCH_ONLY')                      # comma-separated labels to run
    candidates = [c for c in CANDIDATES[kind] if not only or c.label in only.split(',')]
    results = benchmark(build, base, candidates, field, repeat=1 if only else 2)
    print(format_table(title, results), flush=True)
    print()


SIZES = tuple(int(s) for s in os.environ.get('BENCH_SIZES', '100,200,400').split(','))   # Poisson divisions
wanted = set(sys.argv[1:]) or {'spd', 'transient', 'convection', 'stokes', 'cavity'}

if 'spd' in wanted:
    for n in SIZES:
        run(f"Poisson 2D Q4, {(n + 1) ** 2} unknowns", poisson(n), STEADY, 'spd', 'T')

if 'transient' in wanted:
    base = dict(Study=StudyType.transient, Type=ProblemType.linear, timeDisc=1.0, timeStep=0.001, totalTime=0.1, tol=1e-10)
    for n in (50, 150):
        run(f"Linear transient, 100 steps, {(n + 1) ** 2} unknowns", transient_poisson(n), base, 'spd', 'T')

if 'convection' in wanted:
    for n, Pe in ((100, 50.0), (200, 1000.0)):
        run(f"Convection-diffusion SUPG, Pe = {Pe:g}, {(n + 1) ** 2} unknowns", convection(n, Pe), STEADY, 'general', 'T')

if 'stokes' in wanted:
    for lc in (0.5, 0.25, 0.15):
        run(f"Stokes contraction Taylor-Hood, lc = {lc}", stokes_contraction(lc), STEADY, 'flow', 'u')

if 'cavity' in wanted:
    base = dict(Study=StudyType.steady_state, Type=ProblemType.nonlinear, tol=1e-8)
    for n, Re in ((32, 400.0), (64, 400.0)):
        run(f"Navier-Stokes cavity Q4, Re = {Re:g}, {n}x{n} (Newton)", cavity(n, Re), base, 'flow', 'u')
