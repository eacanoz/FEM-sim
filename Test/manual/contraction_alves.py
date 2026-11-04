"""4:1 planar contraction, Oldroyd-B with beta = 1/9, against Alves, Oliveira and Pinho (2003), JNNFM 110, 45-75: corner vortex X_R / h and the largest
centreline velocity, as tabulated by Kaufmann et al., arXiv:2112.06829 (Tables 6 and 8, 'Alves et al. [4] (Tab. 2)'). Wi = lam U / h, U the
mean velocity downstream, h its half-height. The Couette correction C is printed without a reference (not in those tables).
Args: divisions of h, L1 = L2 (in h), Wi max, Wi step. Environment: CONTRACTION_MESH_ONLY=1 stops after the mesh, CONTRACTION_SAVE=path
keeps u, p and the polymer stress of every Wi."""
import sys, os, time, logging
sys.path.insert(0, os.getcwd())
import numpy as np
import gmsh
from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.Viscoelastic import ve
from Source.enums import StudyType, ProblemType
from Source.logger_config import sim_logger

sim_logger.setLevel(logging.WARNING)

ALVES_XR = {0.0: 1.500, 0.5: 1.452, 1.0: 1.373, 1.5: 1.279, 2.0: 1.181, 2.5: 1.077, 3.0: 0.973, 4.0: 0.788, 5.0: 0.638}
ALVES_UMAX = {0.0: 1.501, 0.5: 1.511, 1.0: 1.525, 1.5: 1.537, 2.0: 1.546, 2.5: 1.554, 3.0: 1.562}

h, H, U, MU, BETA = 1.0, 4.0, 1.0, 1.0, 1.0 / 9.0
DIVISIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
L1 = L2 = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
WI_MAX = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
STEP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
tag = f"h/{DIVISIONS} L{L1:g}"


def contraction_mesh(lc, refine=0.15):
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
print(f"[{tag}] {mesh.getNoN()} nodes, corner element {lc_wall * 0.15:.4f} h", flush=True)
if os.environ.get('CONTRACTION_MESH_ONLY'):
    sys.exit(0)

wi = 0.01                                                    # the Newtonian limit of the tables, Wi = 0
model = Model(name='contraction', dim=2, mesh=mesh, mat=material('X', miu=mu_s), psc=[ff, ve])
flow, polymer = model.physics.fluid, model.physics.viscoelastic
polymer.setModes([(mu_p, wi * h / U, 0.0)])
polymer.setDEVSS(mu_p)
inlet, outlet = mesh.boundary_edges('inlet'), mesh.boundary_edges('outlet')
U_in = 1.5 * U * h / H
flow.addBC_VelocityEdges(inlet, lambda x, y: (U_in * (1 - (y / H) ** 2), 0.0))
flow.addBC_NoSlipEdges(mesh.boundary_edges('walls'))
flow.addBC_SymmetryEdges(mesh.boundary_edges('axis'))
flow.addBC_OutletEdges(outlet, 0.0)
polymer.addBC_OutletEdges(outlet)
polymer.addBC_InflowEdges(inlet, lambda x, y: [[0.0, -2.0 * U_in * y / H ** 2], [0.0, 0.0]])
print(f"[{tag}] {model.layout.ndof} unknowns", flush=True)
axis = np.flatnonzero(np.abs(xy[:, 1]) < 1e-9)


def couette_correction():
    nodes = np.unique(np.ravel(inlet))
    q = flow.flowRate(outlet)
    return (model.sol['p'][nodes].mean() - 3 * MU * q / H ** 3 * L1 - 3 * MU * q / h ** 3 * L2) / (2 * 3 * MU * q / h ** 2)


wall = np.flatnonzero((np.abs(xy[:, 1] - H) < 1e-9) & (xy[:, 0] < -1e-9))
wall = wall[np.argsort(xy[wall, 0])]


def corner_vortex(u):
    """Reattachment on the upstream wall y = H, where the wall vorticity du/dy changes sign, as a distance to the contraction plane
    over h: du/dy projected to the nodes (vertices and midsides), the first node from upstream with reversed flow, and a linear zero."""
    disc = model.geometry
    s = disc.project(disc.gradient(np.asarray(u))[:, :, 0, 1])[wall]
    x = xy[wall, 0]
    reversed_flow = np.nonzero(s > 0.0)[0]
    if not len(reversed_flow) or reversed_flow[0] == 0:
        return 0.0
    k = reversed_flow[0]
    return -(x[k - 1] + (x[k] - x[k - 1]) * (0.0 - s[k - 1]) / (s[k] - s[k - 1])) / h


print(f"[{tag}] {'Wi':>5} {'X_R/h':>7} {'Alves':>6} {'diff':>7} {'u_max':>7} {'Alves':>6} {'diff':>8} {'C':>8} {'Newton':>6} {'s':>6}", flush=True)
total = time.perf_counter()
saved = {}
step, last, good = STEP, None, model._state()
while wi <= WI_MAX + 1e-9:
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
    start = time.perf_counter()
    model.solve()
    if not model.nonlinear_converged:
        print(f"[{tag}] Wi = {wi:.3f}: does not converge ({model.newton_iterations} iterations)", flush=True)
        if last is None or step < STEP / 8 + 1e-12:              # adaptive continuation: back to the last solution, half the step
            break
        step *= 0.5
        for name, values in model.layout.split(good).items():
            model.physics.var[name].updateField(values)
        polymer.setWeissenberg((last + step) / wi)
        wi = last + step
        continue
    last, good = wi, model._state()
    key = round(wi, 4) if wi > 0.05 else 0.0
    xr, umax = corner_vortex(model.sol['u']), np.asarray(model.sol['u'])[axis, 0].max()
    if os.environ.get('CONTRACTION_SAVE'):
        saved[f'u_{key:.2f}'], saved[f'p_{key:.2f}'] = np.asarray(model.sol['u']), np.asarray(model.sol['p'])
        saved[f'tau_{key:.2f}'] = np.asarray(model.sol['polymer_stress'])
        np.savez(os.environ['CONTRACTION_SAVE'], xy=xy, conn=model.geometry.conn, **saved)
    if key in ALVES_XR or abs(key % 0.5) < 1e-9 or abs(key % 0.5 - 0.5) < 1e-9:
        ref_x, ref_u = ALVES_XR.get(key), ALVES_UMAX.get(key)
        fx = f"{ref_x:6.3f} {100 * (xr / ref_x - 1):+6.2f}%" if ref_x else f"{'':6s} {'':7s}"
        fu = f"{ref_u:6.3f} {100 * (umax / ref_u - 1):+7.3f}%" if ref_u else f"{'':6s} {'':8s}"
        print(f"[{tag}] {key:5.2f} {xr:7.3f} {fx} {umax:7.4f} {fu} {couette_correction():8.4f} {model.newton_iterations:6d} {time.perf_counter() - start:6.0f}", flush=True)
    new = 0.1 if wi < 0.05 else wi + step
    polymer.setWeissenberg(new / wi)
    wi = new
print(f"[{tag}] last converged Wi {last}; total {time.perf_counter() - total:.0f} s", flush=True)
