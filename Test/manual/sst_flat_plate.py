"""SST flat plate: what closes the residual? Args: tag order nx ny Re option(none|supg|floor|fine). Monitors C_f along the iterations."""
import sys, os, time, logging
sys.path.insert(0, os.getcwd())
import numpy as np
from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.Turbulence import sst, sa
from Source.enums import ElementType, StudyType, ProblemType
from Source.logger_config import sim_logger

tag, order, nx, ny, Re = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
option = sys.argv[6] if len(sys.argv) > 6 else 'none'
kind = sa if option == 'sa' else sst
history = []
class H(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if m.startswith('|') and ('||R||' in m or 'rejected' in m):
            history.append(m)
sim_logger.addHandler(H())
for h in list(sim_logger.handlers):
    if not isinstance(h, H):
        sim_logger.removeHandler(h)
L, H, x0 = 2.0, 1.0, -0.33
nu = 1.0 / Re
m = Mesh(); m.generate2DMesh((1.0, 1.0), (nx, ny), ElementType.quadrilateral, order=order)
beta_y, beta_x = 10.0, 3.0
for node in m.NL:
    a, b = node.coor
    y = H * (np.exp(beta_y * b) - 1) / (np.exp(beta_y) - 1)
    s = np.sinh(beta_x * (a - 0.25)) / np.sinh(beta_x * 0.75)
    x = L * s if s >= 0 else -x0 * s / (np.sinh(beta_x * 0.25) / np.sinh(beta_x * 0.75))
    node.coor = (x, y)
xy = np.array([n.coor for n in m.NL])
model = Model(name='plate', dim=2, mesh=m, mat=material('X', miu=nu, rho=1.0), psc=[ff, kind])
fl, tu = model.physics.fluid, model.physics.turbulence
fl.setInertia(True)
if option == 'supg':
    fl.setStabilization('SUPG')
xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
wall = m.boundary_edges(lambda x, y: y < 1e-12 and x > -1e-12)
inlet = m.boundary_edges(lambda x, y: x < xmin + 1e-9)
fl.addBC_VelocityEdges(inlet, (1.0, 0.0))
fl.addBC_NoSlipEdges(wall)
fl.addBC_SymmetryEdges(m.boundary_edges(lambda x, y: y < 1e-12 and x < 1e-12))
fl.addBC_SymmetryEdges(m.boundary_edges(lambda x, y: y > H - 1e-9))
fl.addBC_OutletEdges(m.boundary_edges(lambda x, y: x > xmax - 1e-9))
tu.setWalls(wall)
n = len(xy)
fl.var['u'].updateField(np.c_[np.ones(n), np.zeros(n)])
if kind is sst:
    k_in, ratio = 1e-6, 1.0
    tu.addBC_TurbulenceEdges(inlet, k_in, k_in / (ratio * nu))
    tu.var['k'].updateField(np.full(n, k_in)); tu.var['ln_omega'].updateField(np.full(n, np.log(k_in / (ratio * nu))))
else:
    tu.addBC_TurbulenceEdges(inlet, 3.0 * nu)
    tu.var['nut'].updateField(np.full(n, 3.0 * nu))
tu.coupleFlow()
print(f"[{tag}] order {order} {nx}x{ny} Re {Re:.0e} {option}: dofs {model.layout.ndof}", flush=True)
model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, globalization='pseudo-transient', cfl0=1.0)
t = time.time(); model.solve()
u = model.sol['u'][:, 0]
print(f"[{tag}] {time.time() - t:.0f} s, its {model.newton_iterations}, conv {model.nonlinear_converged}", flush=True)
for xq in (0.25, 0.5, 0.97, 1.5):
    xc = xy[np.argmin(np.abs(xy[:, 0] - xq)), 0]; col = np.abs(xy[:, 0] - xc) < 1e-9
    o = np.argsort(xy[col, 1]); y1, u1 = xy[col, 1][o][1], u[col][o][1]
    cf = 2 * nu * u1 / y1; rex = Re * xc
    print(f"[{tag}] x {xc:.3f}: C_f {cf:.5f}, Schultz-Grunow {0.37 / np.log10(rex) ** 2.584:.5f} ({100 * (cf / (0.37 / np.log10(rex) ** 2.584) - 1):+.1f} %), y+ {y1 * np.sqrt(cf / 2) * Re:.2f}", flush=True)
if kind is sst:
    kk = model.sol['k']
    print(f"[{tag}] k min {kk.min():.2e} zero-clipped free {int(np.sum((kk == 0) & (xy[:, 1] > 1e-12)))}", flush=True)

for i in list(range(0, 20, 4)) + list(range(20, len(history), 40)) + list(range(max(0, len(history) - 6), len(history))):
    print(f"[{tag}] {history[i][:85]}", flush=True)
print(f"[{tag}] rejected {sum('rejected' in h for h in history)}")
system = model._system(); state = model._state()
J, R = system.residual_and_tangent(state)
J, R = system.hold_clipped(J, R, state)
scale = np.maximum(np.abs(J.diagonal()), 1.0) * system.field_scales(state)
r = R / scale
for name in model.layout.names:
    dofs = model.layout.dofs(name)
    rr = np.abs(r[dofs]).reshape(len(xy), -1).max(axis=1)
    b = np.argsort(rr)[-3:][::-1]
    print(f"[{tag}] {name:14s} norm {np.linalg.norm(r[dofs]):.2e}, largest " + '; '.join(f"({xy[i,0]:.3f},{xy[i,1]:.2e}) {rr[i]:.1e}" for i in b), flush=True)
