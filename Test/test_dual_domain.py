"""Thin-walled solids as two skins (Source/Pre_processing/DualDomain.py) and their Hele-Shaw filling (HeleShawFilling.from_dual_domain):
skins and edges of a plate read back from a STEP file and its exact thickness; a wedge (exact on the flat side, h cos(angle) on the
inclined one); a plate with a rib (the rib at its width, the base under it at its own thickness); the filling of a solid strip against
the exact pressure, gated on both skins and on one (mass to round-off, both skins at one pressure and one front); the non-isothermal
filling against the midplane of the same triangles; the packing refused. Each test prints its progress.

Run from the project root:  python -u Test/test_dual_domain.py
"""

import os
import sys
import tempfile
import time

import gmsh
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.DualDomain import DualDomain
from Source.Pre_processing.Mesh import Mesh
from Source.Physics.HeleShaw import HeleShawFilling
from Source.Physics.Rheology import CrossWLF, TaitPVT

START = time.perf_counter()
L, W, H, Q, MU = 0.1, 0.02, 0.002, 1e-6, 1000.0
LAW = CrossWLF(n=0.2619, tau_star=1.778e4, D1=4.378e17, D2=266.7, A1=39.22, A2=51.6, gamma_min=1e-2)
THERMAL = dict(T_melt=230.0, T_mold=50.0, rho=970.0, cp=2000.0, k=0.22, htc=5000.0, T_noflow=180.75, cells=10)


def note(text):
    print(f"   [{time.perf_counter() - START:6.1f} s] {text}", flush=True)


def newtonian(g2):
    return MU + 0 * g2


def solid(build, size):

    """DualDomain of the solid that build(occ) makes, its surface meshed with triangles of `size`."""

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber('Mesh.MeshSizeMax', size)
        gmsh.model.mesh.generate(2)
        return DualDomain.from_gmsh_model()
    finally:
        gmsh.finalize()


def centroids(dd):
    return dd.xyz[dd.conn].mean(axis=1)


def test_plate_from_step():

    """A plate 100 x 20 x 2 mm written to STEP and read back: two skins (its large faces) and four edges (the rim), thickness 2 mm at
    every skin triangle (1e-12), every node linked to the opposite skin with barycentric weights in [0, 1] adding up to 1."""

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    path = os.path.join(tempfile.mkdtemp(), 'plate.step')
    try:
        gmsh.model.occ.addBox(0, 0, 0, 100.0, 20.0, 2.0)                  # STEP in mm
        gmsh.model.occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()
    dd = DualDomain.from_step(path, size=3.0, curvature=0)
    err = np.max(np.abs(dd.thickness / 2.0 - 1))
    w = dd.link_weights
    note(f"{len(dd.conn)} skin triangles, {len(dd.xyz)} nodes; skins {dd.skins}, edges {dd.edges}; thickness {err:.1e}; "
         f"{len(dd.link_source)} links, {len(dd.unjoined)} nodes without one")
    assert len(dd.skins) == 2 and len(dd.edges) == 4 and err < 1e-12 and np.all(dd.measured)
    assert len(dd.unjoined) == 0 and len(np.unique(dd.link_source)) == len(dd.xyz)
    assert np.all(w > -1e-9) and np.all(w < 1 + 1e-9) and np.allclose(w.sum(axis=1), 1.0)


def test_wedge_thickness():

    """A wedge 1 to 3 mm thick over 100 mm (flat bottom, inclined top, slope 0.02): the bottom skin measures h(x) exactly (1e-12), the
    top skin along its own normal h cos(angle), within angle^2 / 2 = 2e-4 of h(x)."""

    def build(occ):
        pts = [occ.addPoint(*p) for p in [(0, 0, 0), (L, 0, 0), (L, 0, 0.003), (0, 0, 0.001)]]
        lines = [occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        occ.extrude([(2, occ.addPlaneSurface([occ.addCurveLoop(lines)]))], 0, W, 0)

    dd = solid(build, 2e-3)
    c = centroids(dd)
    exact = 0.001 + 0.002 * c[:, 0] / L
    bottom = c[:, 2] < 1e-9
    e_bottom, e_top = np.max(np.abs(dd.thickness[bottom] / exact[bottom] - 1)), np.max(np.abs(dd.thickness[~bottom] / exact[~bottom] - 1))
    note(f"skins {dd.skins}, edges {dd.edges}; bottom {e_bottom:.1e}, top {e_top:.2e} (angle^2 / 2 = 2.0e-4)")
    assert len(dd.skins) == 2 and e_bottom < 1e-12 and e_top < 2.1e-4


def test_rib_thickness():

    """A plate 2 mm thick with a rib 1.2 mm wide and 12 mm high: the rib sides are skins at 1.2 mm, the base at 2 mm also under the rib
    (whose rays run up the rib: taken from the neighbours), the top of the rib and the rim are edges; only the nodes along the root of
    the rib (the corner of base and rib) have no link."""

    def build(occ):
        a = occ.addBox(0, 0, 0, 0.1, 0.04, 0.002)
        b = occ.addBox(0.02, 0.0194, 0.002, 0.06, 0.0012, 0.012)
        occ.fuse([(3, a)], [(3, b)])

    dd = solid(build, 2e-3)
    c = centroids(dd)
    rib = c[:, 2] > 0.002 + 1e-9
    e_rib = np.max(np.abs(dd.thickness[rib] / 0.0012 - 1))
    e_base = np.max(np.abs(dd.thickness[~rib] / 0.002 - 1))
    root = dd.xyz[dd.unjoined]
    note(f"skins {len(dd.skins)}, edges {len(dd.edges)}; rib {e_rib:.1e}, base {e_base:.1e} ({np.sum(~dd.measured)} triangles from their "
         f"neighbours); {len(dd.unjoined)} nodes without a link, at z = {np.unique(np.round(root[:, 2] * 1e3, 6))} mm")
    assert len(dd.skins) == 4 and e_rib < 1e-12 and e_base < 1e-12
    assert np.all(np.abs(root[:, 2] - 0.002) < 1e-9) and np.all(np.abs(np.abs(root[:, 1] - 0.02) - 0.0006) < 1e-9)


def test_strip_filling():

    """A solid strip 100 x 20 x 2 mm, Newtonian, gated on the rim of both skins and of the bottom skin only (the melt crosses the wall
    by the links): the melt volume grows as Q t to round-off, both skins carry the same pressure (1e-3 of the gate pressure) and the
    same front, the gate pressure at half the fill is within 2 % of 12 mu Q x_f / (h^3 W) (the front known to one control volume)."""

    dd = solid(lambda occ: occ.addBox(0, 0, 0, L, W, H), 4e-3)
    slope = 12 * MU * Q / (H ** 3 * W)
    for name, sel in (('both skins', lambda x, y, z: x < 1e-9), ('bottom skin', lambda x, y, z: x < 1e-9 and z < 1e-9)):
        fill = HeleShawFilling.from_dual_domain(dd, newtonian)
        fill.addGate(dd.mesh.boundary_facets(sel), Q)
        v0 = fill.melt_volume()
        fill.fill(0.5 * L * W * H / Q, progress=False)
        growth = abs(fill.melt_volume() - v0 - Q * fill.time) / (Q * fill.time)
        fill.solve_pressure()
        xf = fill.melt_volume() / (H * W)
        err = fill.p[fill._gate_nodes()].mean() / (slope * xf) - 1
        bottom = fill.xy[:, 2] < 1e-9
        # the same pressure across the wall: node by node against the other skin at the same (x, y)
        top_xy = {tuple(np.round(p[:2], 9)): k for k, p in zip(np.flatnonzero(~bottom), fill.xy[~bottom])}
        pairs = [(i, top_xy[tuple(np.round(fill.xy[i, :2], 9))]) for i in np.flatnonzero(bottom) if tuple(np.round(fill.xy[i, :2], 9)) in top_xy]
        dp = max(abs(fill.p[i] - fill.p[j]) for i, j in pairs) / fill.p.max()
        front_b = fill.xy[bottom & (fill.fill_factor >= 1), 0].max()
        front_t = fill.xy[~bottom & (fill.fill_factor >= 1), 0].max()
        note(f"{name}: {len(fill.xy)} nodes, {len(fill.history)} steps; melt volume against Q t {growth:.1e}; gate pressure {err:+.2%}; "
             f"pressure across the wall {dp:.1e} ({len(pairs)} pairs); filled to x = {front_b * 1e3:.1f} / {front_t * 1e3:.1f} mm")
        assert growth < 1e-12 and abs(err) < 0.02 and dp < 1e-3 and abs(front_b - front_t) < 4.1e-3


def test_strip_thermal():

    """The solid strip with POLYFORT into a cold mold (frozen layer, viscous heating), filled in 1 s in batches: against the midplane
    made of the same triangles as one skin, the gate pressure at the end is within 2 %, the frozen fraction and the mean temperature
    of the skins equal the midplane node by node within 0.1 (one cell of ten) and 2 K, mass to 1e-10."""

    dd = solid(lambda occ: occ.addBox(0, 0, 0, L, W, H), 4e-3)
    q = L * W * H / 1.0
    bottom = np.all(dd.xyz[dd.conn][:, :, 2] < 1e-9, axis=1)
    used = np.unique(dd.conn[bottom])
    renumber = np.full(len(dd.xyz), -1)
    renumber[used] = np.arange(len(used))
    mid = HeleShawFilling(Mesh.from_arrays(dd.xyz[used][:, :2], renumber[dd.conn[bottom]]), H, LAW, T=230.0)
    mid.setThermal(**THERMAL)
    mid.addGate(mid.mesh.boundary_edges(lambda x, y: x < 1e-12), q)
    mid.fill(progress=False, batch=0.1)
    fill = HeleShawFilling.from_dual_domain(dd, LAW, T=230.0)
    fill.setThermal(**THERMAL)
    fill.addGate(dd.mesh.boundary_facets(lambda x, y, z: x < 1e-9), q)
    v0 = fill.melt_volume()
    fill.fill(progress=False, batch=0.1)
    growth = abs(fill.melt_volume() - v0 - q * fill.time) / (q * fill.time)
    nodes = np.flatnonzero(fill.xy[:, 2] < 1e-9)                         # the bottom skin: the same nodes as the midplane
    where = {tuple(np.round(p, 9)): k for k, p in enumerate(mid.xy)}
    match = np.array([where[tuple(np.round(fill.xy[i, :2], 9))] for i in nodes])
    d_frozen = np.max(np.abs(fill.frozen_fraction()[nodes] - mid.frozen_fraction()[match]))
    d_T = np.max(np.abs(fill.mean_temperature()[nodes] - mid.mean_temperature()[match]))
    dp = fill.history[-1][2] / mid.history[-1][2] - 1
    note(f"dual {len(fill.xy)} nodes, midplane {len(mid.xy)}: gate pressure {fill.history[-1][2] / 1e6:.3f} against {mid.history[-1][2] / 1e6:.3f} "
         f"MPa ({dp:+.2%}); frozen fraction up to {d_frozen:.3f} apart, mean temperature {d_T:.2f} K; mass {growth:.1e}")
    assert abs(dp) < 0.02 and d_frozen < 0.1 + 1e-9 and d_T < 2.0 and growth < 1e-10


def test_packing_refused():

    """The links are in the incompressible filling only: the compressible filling and the packing of a dual domain refuse to run."""

    dd = solid(lambda occ: occ.addBox(0, 0, 0, L, W, H), 5e-3)
    fill = HeleShawFilling.from_dual_domain(dd, newtonian)
    fill.addGate(dd.mesh.boundary_facets(lambda x, y, z: x < 1e-9), Q)
    fill.setDensity(TaitPVT(1.006e-3, 4.936e-7, 7.989e7, 3.53e-3, 9.073e-4, 1.49e-7, 2.601e8, 2.353e-3, 453.9, 1.709e-7, 1.009e-4,
                            4.788e-2, 1.553e-8).density)
    try:
        fill.fill(progress=False)
    except NotImplementedError as e:
        note(f"refused: {e}")
        return
    raise AssertionError("the compressible filling of a dual domain ran")


if __name__ == "__main__":
    tests = [test_plate_from_step, test_wedge_thickness, test_rib_thickness, test_strip_filling, test_strip_thermal, test_packing_refused]
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
