"""Mixing elbow of example_mixing_elbow.ipynb, water, SA and heat on curved T6 with a boundary-layer mesh. No wall functions: y+ must be about 1.
Exact checks: mass balance, and the mean outlet temperature from the energy balance, (U1 D1 T1 + U2 D2 T2) / (U1 D1 + U2 D2) = 28.57 C.
Args: tag lc_mm y1_mm layers [supg]. Environment: ELBOW_JUNCTION=h_mm refines the union of the side inlet (box x 310-360, y 35-90 mm),
ELBOW_HEAT_GALERKIN=1 leaves the heat without SUPG, ELBOW_SAVE=path keeps the fields, ELBOW_MESH_ONLY=1 stops after the mesh."""
import sys, os, time, logging
sys.path.insert(0, os.getcwd())
import numpy as np
import gmsh
import jax.numpy as jnp
from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.HeatTransfer import ht
from Source.Physics.Turbulence import sa
from Source.enums import StudyType, ProblemType
from Source.logger_config import sim_logger


class Progress(logging.Handler):

    def __init__(self):
        super().__init__()
        self.count = 0

    def emit(self, record):
        message = record.getMessage()
        if 'rejected' in message and self.count < 3:
            print(f"[{sys.argv[1]}] {message[:80]}", flush=True)
        if message.startswith('|') and '||R||' in message:
            self.count += 1
            if self.count % 25 == 0:
                print(f"[{sys.argv[1]}] {message[:80]}", flush=True)


for handler in list(sim_logger.handlers):
    sim_logger.removeHandler(handler)
sim_logger.addHandler(Progress())

tag, lc, y1, layers = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
supg = len(sys.argv) > 5 and sys.argv[5] == 'supg'
MU, RHO, CP, K = 1e-3, 1000.0, 4180.0, 0.6
U1, T1, D1, U2, T2, D2 = 0.4, 20.0, 0.100, 1.2, 40.0, 0.025
PR_T, CV1 = 0.85, 7.1
T_MIX = (U1 * D1 * T1 + U2 * D2 * T2) / (U1 * D1 + U2 * D2)


def elbow_mesh():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("mixing_elbow")
        geo = gmsh.model.geo
        mm = 1e-3
        cy, R_out = 200.0, 200.0
        y_low, y_hi = cy - np.sqrt(R_out ** 2 - 125.0 ** 2), cy - np.sqrt(R_out ** 2 - 150.0 ** 2)
        fine = 0.4 * lc
        P = lambda x, y, h=lc: geo.addPoint(x * mm, y * mm, 0, h * mm)
        p1, p2, p3, p4, p5, p6 = P(0, 0), P(200, 0), P(325, y_low, fine), P(325, -25, fine), P(350, -25, fine), P(350, y_hi, fine)
        p7, p8, p9, p10, p11, p12, c = P(400, 200), P(400, 400), P(300, 400), P(300, 200), P(200, 100), P(0, 100), P(200, cy)
        bottom, arc_lo, side_l, in2, side_r = (geo.addLine(p1, p2), geo.addCircleArc(p2, c, p3), geo.addLine(p3, p4), geo.addLine(p4, p5),
                                               geo.addLine(p5, p6))
        trans, right, out, left, arc_up, top, in1 = (geo.addCircleArc(p6, c, p7), geo.addLine(p7, p8), geo.addLine(p8, p9), geo.addLine(p9, p10),
                                                     geo.addCircleArc(p10, c, p11), geo.addLine(p11, p12), geo.addLine(p12, p1))
        walls = [bottom, arc_lo, side_l, side_r, trans, right, left, arc_up, top]
        surface = geo.addPlaneSurface([geo.addCurveLoop([bottom, arc_lo, side_l, in2, side_r, trans, right, out, left, arc_up, top, in1])])
        geo.synchronize()
        gm = gmsh.model
        gm.addPhysicalGroup(1, [in1], name='inlet_main')
        gm.addPhysicalGroup(1, [in2], name='inlet_side')
        gm.addPhysicalGroup(1, [out], name='outlet')
        gm.addPhysicalGroup(1, walls, name='walls')
        gm.addPhysicalGroup(2, [surface], name='fluid')
        if layers > 0:
            ratio = 1.3
            field = gm.mesh.field.add('BoundaryLayer')
            gm.mesh.field.setNumbers(field, 'CurvesList', walls)
            gm.mesh.field.setNumber(field, 'Size', y1 * mm)
            gm.mesh.field.setNumber(field, 'Ratio', ratio)
            gm.mesh.field.setNumber(field, 'Thickness', y1 * mm * (ratio ** layers - 1.0) / (ratio - 1.0))
            gm.mesh.field.setAsBoundaryLayer(field)
        if os.environ.get('ELBOW_JUNCTION'):
            box = gm.mesh.field.add('Box')
            gm.mesh.field.setNumber(box, 'VIn', float(os.environ['ELBOW_JUNCTION']) * mm)
            gm.mesh.field.setNumber(box, 'VOut', lc * mm)
            for name, value in (('XMin', 310), ('XMax', 360), ('YMin', 35), ('YMax', 90)):
                gm.mesh.field.setNumber(box, name, value * mm)
            gm.mesh.field.setNumber(box, 'Thickness', 15 * mm)
            gm.mesh.field.setAsBackgroundMesh(box)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gm.mesh.generate(2)
        gm.mesh.setOrder(2)
        return Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()


mesh = elbow_mesh()
xy = np.array([n.coor for n in mesh.NL])
print(f"[{tag}] {mesh.getNoE()} elements, {len(xy)} nodes", flush=True)
if os.environ.get('ELBOW_MESH_ONLY'):
    sys.exit(0)

model = Model(name='elbow', dim=2, mesh=mesh, mat=material('water', miu=MU, rho=RHO, Cp=CP, k=K), psc=[ff, ht, sa])
fl, heat, tu = model.physics.fluid, model.physics.heat, model.physics.turbulence
fl.setInertia(True)
if supg:
    fl.setStabilization('SUPG')
heat.setVelocity('u')
if not os.environ.get('ELBOW_HEAT_GALERKIN'):
    heat.setStabilization('SUPG')
inlet_main, inlet_side = mesh.boundary_edges('inlet_main'), mesh.boundary_edges('inlet_side')
outlet, walls = mesh.boundary_edges('outlet'), mesh.boundary_edges('walls')
fl.addBC_VelocityEdges(inlet_main, (U1, 0.0))
fl.addBC_VelocityEdges(inlet_side, (0.0, U2))
fl.addBC_NoSlipEdges(walls)
fl.addBC_OutletEdges(outlet)
heat.addBC_TemperatureEdges(inlet_main, T1)
heat.addBC_TemperatureEdges(inlet_side, T2)
nu = MU / RHO
tu.setWalls(walls)
tu.addBC_TurbulenceEdges(inlet_main, 3.0 * nu)
tu.addBC_TurbulenceEdges(inlet_side, 3.0 * nu)
tu.coupleFlow()


def turbulent_conductivity(nut):
    chi = jnp.maximum(nut, 0.0) / nu
    return K + RHO * CP * jnp.maximum(nut, 0.0) * chi ** 3 / (chi ** 3 + CV1 ** 3) / PR_T


if not os.environ.get('ELBOW_NO_LAW'):
    heat.setConductivityLaw(turbulent_conductivity, depends_on=('nut',))
n = len(xy)
tu.var['nut'].updateField(np.full(n, 3.0 * nu))
heat.var['T'].updateField(np.full(n, T_MIX))
# A Stokes flow of the same mesh, as the start of the march: u = 0 with an imposed inflow breaks continuity, the pressure of a pseudo-time
# step then grows as 1 / CFL, and lowering the CFL makes the residual larger. The march needs a divergence-free start.
start = Model(name='start', dim=2, mesh=mesh, mat=material('water', miu=MU, rho=RHO), psc=ff)
start.physics.addBC_VelocityEdges(inlet_main, (U1, 0.0))
start.physics.addBC_VelocityEdges(inlet_side, (0.0, U2))
start.physics.addBC_NoSlipEdges(walls)
start.physics.addBC_OutletEdges(outlet)
start.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear)
start.solve()
fl.var['u'].updateField(start.sol['u'])
fl.var['p'].updateField(start.sol['p'])
print(f"[{tag}] dofs {model.layout.ndof}", flush=True)
model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, globalization='pseudo-transient', cfl0=1.0, nonlinearMaxIter=800)
t = time.time(); model.solve()
print(f"[{tag}] {time.time() - t:.0f} s, its {model.newton_iterations}, conv {model.nonlinear_converged}", flush=True)

u, T = model.sol['u'], np.asarray(model.sol['T'])
q_main, q_side, q_out = fl.flowRate(inlet_main), fl.flowRate(inlet_side), fl.flowRate(outlet)
T_MIX = (q_main * T1 + q_side * T2) / (q_main + q_side)            # the no-slip corners take a little of the nominal inflow
print(f"[{tag}] flow rates {q_main:.5f} {q_side:.5f} {q_out:.5f} (exact {U1 * D1}, {U2 * D2}), imbalance {abs(q_main + q_side + q_out):.1e}", flush=True)
row = np.unique(np.ravel(outlet))
o = row[np.argsort(xy[row, 0])]
h = xy[o[2::2], 0] - xy[o[:-2:2], 0]                               # the outlet row alternates vertex and midside: 3-point rule of each quadratic edge
simpson = lambda f: np.sum(h / 6.0 * (f[:-2:2] + 4.0 * f[1::2] + f[2::2]))
flux = simpson(u[o, 1] * T[o]) / simpson(u[o, 1])
print(f"[{tag}] mean outlet T {flux:.3f} (energy balance {T_MIX:.3f}); T in [{T.min():.2f}, {T.max():.2f}], bounds [{T1}, {T2}]; outlet T in [{T[o].min():.2f}, {T[o].max():.2f}]", flush=True)
nearest = np.unique(mesh.plot_triangles()[np.any(np.isin(mesh.plot_triangles(), np.unique(np.ravel(walls))), axis=1)])
d = np.asarray(tu.var['wall_distance'].values, float)
speed = np.linalg.norm(u, axis=1)
off = nearest[(d[nearest] > 1e-12)]
print(f"[{tag}] first node off the wall: d {d[off].min():.2e} to {d[off].max():.2e} m; nodal y+ estimate {np.median(np.sqrt(nu * speed[off] / d[off]) * d[off] / nu):.2f}", flush=True)
print(f"[{tag}] nut/nu max {np.asarray(model.sol['nut']).max() / nu:.0f}", flush=True)
if os.environ.get('ELBOW_SAVE'):                                  # the fields, to study them without solving again
    disc = model.geometry
    np.savez(os.environ['ELBOW_SAVE'], xy=xy, conn=disc.conn, u=u, p=np.asarray(model.sol['p']), T=T, nut=np.asarray(model.sol['nut']), d=d,
             gradient=disc.gradient(np.asarray(u, dtype=float)), x_gp=disc.x_gp, detJw=disc.detJw,
             wall_edges=np.array(walls), wall_tags=np.array([list(e) for e in walls]), outlet=row)
