"""Free surface of an extrudate (Source/Physics/FreeSurface.py): the streamline method on coarse meshes, with progress output.

Run from the project root:  python -u Test/test_free_surface.py
"""

import logging
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

import gmsh
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.FreeSurface import FreeSurface
from Source.enums import StudyType, ProblemType
from Source.logger_config import sim_logger

sim_logger.setLevel(logging.WARNING)
START = time.perf_counter()


def note(text):
    print(f"   [{time.perf_counter() - START:6.1f} s] {text}", flush=True)


def extrusion_mesh(lip, far, L1, L2, axisymmetric=False, H=1.0):

    """Die of half-height (radius) H and length L1, extrudate of length L2, refined at the lip. Plane: x along the flow;
    axisymmetric: x = r, y = z."""

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        geo = gmsh.model.geo
        pts = ([(0, -L1, far), (0, L2, far), (H, L2, far), (H, 0, lip), (H, -L1, far)] if axisymmetric else
               [(-L1, 0, far), (L2, 0, far), (L2, H, far), (0, H, lip), (-L1, H, far)])
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
        field.setNumber(thr, 'SizeMin', lip)
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


def newtonian(axisymmetric, slip=False, lip=0.05, far=0.3):
    mesh = extrusion_mesh(lip, far, 10.0, 20.0, axisymmetric)
    model = Model(name='swell', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.0), psc=ff, axisymmetric=axisymmetric)
    flow = model.physics
    if slip:                                                       # plug flow in a die without friction: nothing to swell
        inflow = (lambda x, y: (0.0, 1.0)) if axisymmetric else (lambda x, y: (1.0, 0.0))
        flow.addBC_SymmetryEdges(mesh.boundary_edges('wall'))
    else:
        inflow = (lambda x, y: (0.0, 2.0 * (1 - x ** 2))) if axisymmetric else (lambda x, y: (1.5 * (1 - y ** 2), 0.0))
        flow.addBC_NoSlipEdges(mesh.boundary_edges('wall'))
    flow.addBC_VelocityEdges(mesh.boundary_edges('inlet'), inflow)
    flow.addBC_SymmetryEdges(mesh.boundary_edges('axis'))
    flow.addBC_OutletEdges(mesh.boundary_edges('outlet'), 0.0)
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear)
    surface = FreeSurface(model, mesh.boundary_edges('surface'), lip=0.0, along=1 if axisymmetric else 0)
    converged = surface.solve(max_iterations=40, tol=1e-6)
    leak = flow.flowRate(mesh.boundary_edges('surface')) / -flow.flowRate(mesh.boundary_edges('inlet'))
    return surface, converged, leak, mesh


def test_plug_flow_does_not_swell():

    """A die without friction carries a plug flow, which leaves it unchanged: the surface stays at H to round-off."""

    for axisymmetric in (False, True):
        surface, converged, leak, _ = newtonian(axisymmetric, slip=True)
        s, h = surface.height()
        note(f"{'axisymmetric' if axisymmetric else 'plane'} plug flow: swell {surface.swell():.12f}, largest |h - H| {np.max(np.abs(h - 1)):.1e}")
        assert converged and np.max(np.abs(h - 1.0)) < 1e-9


def test_newtonian_swell_on_a_coarse_mesh():

    """Plane and axisymmetric Newtonian swell on a coarse mesh (refined at the lip to 0.05 H): within 1 % of the references (0.4 and 0.6 % here)
    (plane 1.190 +- 0.002, Tanner 2002; axisymmetric about 1.127, as quoted from Mitsoulis et al. 2012), the iteration converged in
    fewer than 20 steps, and what leaves through the surface below 3e-4 of the flow. Test/manual/die_swell.py converges it in the mesh."""

    for axisymmetric, reference in ((False, 1.190), (True, 1.127)):
        surface, converged, leak, mesh = newtonian(axisymmetric)
        note(f"{'axisymmetric' if axisymmetric else 'plane'}: {mesh.getNoN()} nodes, swell {surface.swell():.5f} (reference {reference}), "
             f"{len(surface.history)} iterations, through the surface {leak:+.1e} Q")
        assert converged and len(surface.history) < 20
        assert abs(surface.swell() / reference - 1) < 0.01
        assert abs(leak) < 3e-4


def test_the_surface_must_start_straight_at_the_lip():

    mesh = extrusion_mesh(0.2, 0.5, 2.0, 2.0)
    model = Model(name='swell', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.0), psc=ff)
    for lip in (0.5, -0.5):
        try:
            FreeSurface(model, mesh.boundary_edges('surface'), lip=lip)
        except ValueError:
            continue
        raise AssertionError(f"lip = {lip} accepted")


if __name__ == "__main__":
    tests = [test_plug_flow_does_not_swell, test_newtonian_swell_on_a_coarse_mesh, test_the_surface_must_start_straight_at_the_lip]
    failed = 0
    for t in tests:
        note(f"start {t.__name__}")
        try:
            t()
            print(f"PASS  {t.__name__}", flush=True)
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__} {e}", flush=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
