"""Backward-facing step of Driver & Seegmiller (1985), Re_H = 36000, expansion 8H -> 9H (NASA TMR, tmbwg.github.io/turbmodels/backstep_val.html).
Reference: experiment x_r/H = 6.26 +- 0.10; SST at TMR about 6.50, WIND SST 6.43. Viscous top wall, as in TMR.
Args: tag order scale(mesh multiplier) [sa]. Environment: STEP_MESH_ONLY=1 prints the spacing and stops."""
import sys, os, time, logging
sys.path.insert(0, os.getcwd())
import numpy as np
import gmsh
from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.Turbulence import sst, sa
from Source.enums import StudyType, ProblemType
from Source.logger_config import sim_logger


class Progress(logging.Handler):

    """Every 25th line of the pseudo-transient march, to follow a run of an hour."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def emit(self, record):
        message = record.getMessage()
        if message.startswith('|') and '||R||' in message:
            self.count += 1
            if self.count % 25 == 0:
                print(f"[{sys.argv[1]}] {message[:80]}", flush=True)


for handler in list(sim_logger.handlers):
    sim_logger.removeHandler(handler)
sim_logger.addHandler(Progress())

tag, order, scale = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
kind = sa if len(sys.argv) > 4 and sys.argv[4] == 'sa' else sst
Re = 36000.0
U_IN = 0.96                                                       # 1 at the edge of the layer at x = -4, which the walls accelerate
L, X, T = 100.0, 40.0, 9.0                                        # inlet at x = -L, outlet at x = X, top at y = T


def mesh_step():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add('step')
        g = gmsh.model.geo
        nu = int(45 * scale); nd = int(90 * scale); nt = int(70 * scale); nb = int(30 * scale)
        pts = {'a': (-L, 1), 'b': (0, 1), 'c': (X, 1), 'd': (-L, T), 'e': (0, T), 'f': (X, T), 'g': (0, 0), 'h': (X, 0)}
        p = {k: g.addPoint(x, y, 0, 1.0) for k, (x, y) in pts.items()}
        line = lambda a, b: g.addLine(p[a], p[b])
        ab, bc, de, ef, ad, be, cf, gh, gb, hc = (line('a', 'b'), line('b', 'c'), line('d', 'e'), line('e', 'f'), line('a', 'd'),
                                                  line('b', 'e'), line('c', 'f'), line('g', 'h'), line('g', 'b'), line('h', 'c'))
        s1 = g.addPlaneSurface([g.addCurveLoop([ab, be, -de, -ad])])
        s2 = g.addPlaneSurface([g.addCurveLoop([bc, cf, -ef, -be])])
        s3 = g.addPlaneSurface([g.addCurveLoop([gh, hc, -bc, -gb])])
        g.synchronize()
        m = gmsh.model.mesh
        m.setTransfiniteCurve(ab, nu + 1, 'Progression', 0.94)    # towards the step
        m.setTransfiniteCurve(de, nu + 1, 'Progression', 0.94)
        for c in (bc, ef, gh):
            m.setTransfiniteCurve(c, nd + 1, 'Progression', 1.03)  # away from the step
        for c in (ad, be, cf):
            m.setTransfiniteCurve(c, nt + 1, 'Bump', 0.0013)       # to the floor and the top
        for c in (gb, hc):
            m.setTransfiniteCurve(c, nb + 1, 'Bump', 0.006)        # to both ends
        for s in (s1, s2, s3):
            m.setTransfiniteSurface(s)
            m.setRecombine(2, s)
        gmsh.model.addPhysicalGroup(1, [ad], name='inlet')
        gmsh.model.addPhysicalGroup(1, [cf, hc], name='outlet')
        gmsh.model.addPhysicalGroup(1, [de, ef], name='top')
        gmsh.model.addPhysicalGroup(1, [ab, gb, gh], name='floor')
        gmsh.model.addPhysicalGroup(2, [s1, s2, s3], name='fluid')
        m.generate(2)
        if order == 2:
            m.setOrder(2)
        return Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()


mesh = mesh_step()
xy = np.array([n.coor for n in mesh.NL])
print(f"[{tag}] {mesh.getNoE()} elements, {len(xy)} nodes, smallest y over floor {xy[xy[:, 1] > 1e-12, 1].min():.2e}", flush=True)
above = np.unique(np.round(xy[np.abs(xy[:, 0] + L) < 1e-9, 1], 10))
print(f"[{tag}] first spacing upstream: floor {above[1] - above[0]:.2e}, top {above[-1] - above[-2]:.2e}", flush=True)
if os.environ.get('STEP_MESH_ONLY'):
    sys.exit(0)
model = Model(name='step', dim=2, mesh=mesh, mat=material('X', miu=1.0 / Re, rho=1.0), psc=[ff, kind])
fl, tu = model.physics.fluid, model.physics.turbulence
fl.setInertia(True)
floor = mesh.boundary_edges('floor')
wall = [e for e in floor if xy[list(e), 0].min() > -L + 3.0]
slip = [e for e in floor if xy[list(e), 0].min() <= -L + 3.0]
top = [e for e in mesh.boundary_edges('top') if xy[list(e), 0].min() > -L + 3.0]
fl.addBC_SymmetryEdges([e for e in mesh.boundary_edges('top') if xy[list(e), 0].min() <= -L + 3.0])
fl.addBC_NoSlipEdges(wall)
fl.addBC_SymmetryEdges(slip)
fl.addBC_NoSlipEdges(top)
fl.addBC_VelocityEdges(mesh.boundary_edges('inlet'), (U_IN, 0.0))
fl.addBC_OutletEdges(mesh.boundary_edges('outlet'))
tu.setWalls(wall + top)
n = len(xy)
fl.var['u'].updateField(np.c_[np.where(xy[:, 1] > 1e-12, U_IN, 0.0), np.zeros(n)])
nu = 1.0 / Re
if kind is sst:
    k_in, ratio = 1.5 * (6.1e-4) ** 2, 0.009                      # TMR freestream values
    omega_in = k_in / (ratio * nu)
    tu.addBC_TurbulenceEdges(mesh.boundary_edges('inlet'), k_in, omega_in)
    tu.var['k'].updateField(np.full(n, k_in))
    tu.var['ln_omega'].updateField(np.full(n, np.log(omega_in)))
else:
    tu.addBC_TurbulenceEdges(mesh.boundary_edges('inlet'), 3.0 * nu)
    tu.var['nut'].updateField(np.full(n, 3.0 * nu))
tu.coupleFlow()
print(f"[{tag}] dofs {model.layout.ndof}", flush=True)
model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, globalization='pseudo-transient', cfl0=1.0, nonlinearMaxIter=1000)
t = time.time(); model.solve()
u = model.sol['u']
print(f"[{tag}] {time.time() - t:.0f} s, its {model.newton_iterations}, conv {model.nonlinear_converged}", flush=True)

# reattachment: sign of u at the first row over the floor, downstream of the step
down = xy[:, 0] > 1e-9
y1 = xy[down & (xy[:, 1] > 1e-12), 1].min()
row = down & (np.abs(xy[:, 1] - y1) < 1e-9)
o = np.argsort(xy[row, 0]); xr, ur = xy[row, 0][o], u[row, 0][o]
cross = [xr[i] - ur[i] * (xr[i + 1] - xr[i]) / (ur[i + 1] - ur[i]) for i in range(len(xr) - 1) if ur[i] < 0 <= ur[i + 1]]
print(f"[{tag}] x_r / H: {[round(float(c), 3) for c in cross]} (experiment 6.26 +- 0.10; TMR SST about 6.50), min u row {ur.min():.3f}", flush=True)

# boundary layer upstream
xc = xy[np.argmin(np.abs(xy[:, 0] + 4.0)), 0]
col = np.abs(xy[:, 0] - xc) < 1e-9
o = np.argsort(xy[col, 1]); yy, uu = xy[col, 1][o] - 1.0, u[col, 0][o]
Ue = uu[(yy > 2.0) & (yy < 6.0)].max(); ratio_u = uu / Ue
half = yy < 4.0                                                   # the floor's layer, not the top's
theta = np.trapezoid((ratio_u * (1 - ratio_u))[half], yy[half])
delta = yy[np.argmax(ratio_u > 0.99)]
cf = 2 * nu * uu[1] / yy[1]
print(f"[{tag}] effective Re_H (Ue at x = -4) {Ue * Re:.0f} (36000)", flush=True)
print(f"[{tag}] x {xc:.2f}: Re_theta {theta * Re:.0f} (5000), delta99 {delta:.2f} H (1.5), C_f {cf:.5f}, y+ {yy[1] * np.sqrt(cf / 2) * Re:.2f}, Ue {Ue:.3f}", flush=True)
if kind is sst:
    kk = model.sol['k']
    print(f"[{tag}] zero k free nodes {int(np.sum((kk <= 0) & (xy[:, 1] > 1e-12)))}", flush=True)
