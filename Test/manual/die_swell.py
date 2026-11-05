"""Newtonian die swell in creeping flow by the streamline method (Source/Physics/FreeSurface.py), plane or axisymmetric.
References: plane 1.190 +- 0.002 (Tanner 2002, extrapolated; other computations 1.184 to 1.200, as tabulated in arXiv:1408.5167);
axisymmetric about 1.127 +- 0.003 (as quoted from Mitsoulis et al. 2012; Allan 1977 gave 1.132).
Die of half-height (radius) H = 1, length L1, extrudate length L2, parabolic inflow, the mesh refined at the lip of the die.
Args: lip size, far size, L1, L2, ['axi']."""
import sys, os, time
sys.path.insert(0, os.getcwd())
import numpy as np
import gmsh
from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.FreeSurface import FreeSurface
from Source.enums import StudyType, ProblemType
from Source.logger_config import sim_logger
import logging

sim_logger.setLevel(logging.WARNING)
lip_size, far, L1, L2 = (float(a) for a in sys.argv[1:5])
axi = len(sys.argv) > 5 and sys.argv[5] == 'axi'
H = 1.0


def mesh():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        geo = gmsh.model.geo
        if not axi:            # x along the flow, y across: (-L1, 0) to (L2, H)
            pts = [(-L1, 0, far), (L2, 0, far), (L2, H, far), (0, H, lip_size), (-L1, H, far)]
        else:                  # x = r, y = z: die z in (-L1, 0), axis r = 0
            pts = [(0, -L1, far), (0, L2, far), (H, L2, far), (H, 0, lip_size), (H, -L1, far)]
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
model = Model(name='die swell', dim=2, mesh=m, mat=material('X', miu=1.0, rho=1.0), psc=ff, axisymmetric=axi)
flow = model.physics
inflow = (lambda x, y: (0.0, 2.0 * (1 - x ** 2))) if axi else (lambda x, y: (1.5 * (1 - y ** 2), 0.0))
flow.addBC_VelocityEdges(m.boundary_edges('inlet'), inflow)
flow.addBC_NoSlipEdges(m.boundary_edges('wall'))
flow.addBC_SymmetryEdges(m.boundary_edges('axis'))
flow.addBC_OutletEdges(m.boundary_edges('outlet'), 0.0)
model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear)
print(f"[{'axi' if axi else 'plane'} lip {lip_size} far {far} L1 {L1} L2 {L2}] {m.getNoN()} nodes, {model.layout.ndof} unknowns", flush=True)
surface = FreeSurface(model, m.boundary_edges('surface'), lip=0.0, along=1 if axi else 0, relaxation=1.0)
t = time.perf_counter()
ok = surface.solve(max_iterations=60, tol=1e-6)
outlet = m.boundary_edges('outlet')
q_in, q_out = -flow.flowRate(m.boundary_edges('inlet')), flow.flowRate(outlet)
print(f"[{'axi' if axi else 'plane'} lip {lip_size}] converged {ok} in {len(surface.history)} iterations, {time.perf_counter() - t:.0f} s: "
      f"swell {surface.swell():.5f}; flow in {q_in:.8f} out {q_out:.8f}, through the free surface {flow.flowRate(m.boundary_edges('surface')) / q_in:+.1e} Q; "
      f"|u.n|/|u| max {surface.streamline()[1]:.1e}", flush=True)
