"""Viscoelastic die swell, plane, by the streamline method: Giesekus (or Oldroyd-B, alpha = 0) with beta = mu_s / mu, continuation in
Wi = lam U / H (U the mean velocity in the die, H its half-height). No tabulated reference was found: what is checked is the Newtonian
limit (Wi -> 0 against die_swell.py on the same mesh), the convergence of each free-surface iteration and the trend with Wi.
Args: lip size, far size, alpha, beta, Wi max, Wi step. Environment: SWELL_SAVE=path keeps the swell of each Wi (npz)."""
import sys, os, time, logging
sys.path.insert(0, os.getcwd())
import numpy as np
import gmsh
from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.Viscoelastic import ve
from Source.Physics.FreeSurface import FreeSurface
from Source.enums import StudyType, ProblemType
from Source.logger_config import sim_logger

sim_logger.setLevel(logging.WARNING)
lip_size, far, alpha, beta, wi_max, step = (float(a) for a in sys.argv[1:7])
H, L1, L2, U, MU = 1.0, 10.0, 20.0, 1.0, 1.0


def mesh():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        geo = gmsh.model.geo
        pts = [(-L1, 0, far), (L2, 0, far), (L2, H, far), (0, H, lip_size), (-L1, H, far)]
        tags = [geo.addPoint(x, y, 0, c) for x, y, c in pts]
        lines = [geo.addLine(tags[i], tags[(i + 1) % 5]) for i in range(5)]
        geo.addPlaneSurface([geo.addCurveLoop(lines)])
        geo.synchronize()
        for name, group in (('axis', [lines[0]]), ('outlet', [lines[1]]), ('surface', [lines[2]]), ('wall', [lines[3]]), ('inlet', [lines[4]])):
            gmsh.model.addPhysicalGroup(1, group, name=name)
        field = gmsh.model.mesh.field
        dist = field.add('Distance')
        field.setNumbers(dist, 'PointsList', [tags[3]])
        thr = field.add('Threshold')
        field.setNumber(thr, 'InField', dist)
        field.setNumber(thr, 'SizeMin', lip_size)
        field.setNumber(thr, 'SizeMax', far)
        field.setNumber(thr, 'DistMin', 0.02)
        field.setNumber(thr, 'DistMax', 2.0)
        field.setAsBackgroundMesh(thr)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
        return Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()


m = mesh()
wi = float(os.environ.get("SWELL_WI0", 0.01))
model = Model(name='die swell ve', dim=2, mesh=m, mat=material('X', miu=beta * MU, rho=1.0), psc=[ff, ve])
flow, polymer = model.physics.fluid, model.physics.viscoelastic
polymer.setModes([((1 - beta) * MU, wi * H / U, alpha)])
polymer.setDEVSS((1 - beta) * MU)
inlet, outlet = m.boundary_edges('inlet'), m.boundary_edges('outlet')
flow.addBC_VelocityEdges(inlet, lambda x, y: (1.5 * U * (1 - (y / H) ** 2), 0.0))
flow.addBC_NoSlipEdges(m.boundary_edges('wall'))
flow.addBC_SymmetryEdges(m.boundary_edges('axis'))
flow.addBC_OutletEdges(outlet, 0.0)
polymer.addBC_OutletEdges(outlet)
polymer.addBC_InflowEdges(inlet, lambda x, y: [[0.0, -3.0 * U * y / H ** 2], [0.0, 0.0]])
model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
tag = f"Giesekus alpha {alpha} beta {beta:.3f} lip {lip_size}"
print(f"[{tag}] {m.getNoN()} nodes, {model.layout.ndof} unknowns", flush=True)

surface = FreeSurface(model, m.boundary_edges('surface'), lip=0.0, relaxation=1.0)
rows = []
start = time.perf_counter()
while wi <= wi_max + 1e-9:
    t = time.perf_counter()
    print(f"[{tag}] Wi = {wi:.3f}", flush=True)
    ok = surface.solve(max_iterations=60, tol=1e-6)
    if not ok:
        print(f"[{tag}] Wi = {wi:.3f}: the free surface does not converge (last swell {surface.swell():.5f})", flush=True)
        break
    rows.append((wi, surface.swell(), len(surface.history)))
    print(f"[{tag}] Wi = {wi:.3f}: swell {surface.swell():.5f} in {len(surface.history)} iterations of the surface, "
          f"{time.perf_counter() - t:.0f} s (total {time.perf_counter() - start:.0f} s)", flush=True)
    if os.environ.get('SWELL_SAVE'):
        np.savez(os.environ['SWELL_SAVE'], rows=np.array(rows))
    surface.history = []
    new = step if wi < 0.05 else wi + step
    polymer.setWeissenberg(new / wi)
    wi = new
print(f"[{tag}] Wi, swell:", ", ".join(f"{w:.2f} {s:.4f}" for w, s, _ in rows), flush=True)
