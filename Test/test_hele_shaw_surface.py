"""Checks for Hele-Shaw mold filling on a midplane surface in space (phase 6 of docs/PLAN_3D.md): a folded part as plates joined at
edges, each with its gap.

The Discretization works on 2D elements with three coordinates (tangential gradients, the measure from the metric J^T J, edge normals
in the plane of each element). A fold is an isometry, so a folded strip must fill exactly as the flat one; a T joint of three plates has
an exact solution for a Newtonian melt.

Run from the project root:  python Test/test_hele_shaw_surface.py
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh, Node, Element
from Source.Physics.HeleShaw import HeleShawFilling
from Source.Physics.Rheology import TaitPVT
from Source.Simulation.Discretization import Discretization
from Source.enums import ElementType

START = time.perf_counter()
MU = 1000.0
L, W, H, Q = 0.1, 0.02, 2e-3, 1e-6
TAIT = TaitPVT(1.006e-3, 4.936e-7, 7.989e7, 3.53e-3, 9.073e-4, 1.49e-7, 2.601e8, 2.353e-3, 453.9, 1.709e-7, 1.009e-4, 4.788e-2,
               1.553e-8)                                            # POLYFORT FPP 30 GFC K1079, as Test/test_hele_shaw_packing.py


def note(text):
    print(f"   [{time.perf_counter() - START:6.1f} s] {text}", flush=True)


def newtonian(g2, T=None):
    return MU + 0 * g2


def power_law(g2, T=None):
    return 1e4 * jnp.maximum(g2, 1e-12) ** ((0.35 - 1) / 2)


def fold(mesh, at, angle):

    """Folds a planar mesh about the line x = at (parallel to y) by `angle` out of the plane: x > at goes to
    (at + (x - at) cos angle, y, (x - at) sin angle). An isometry: lengths and areas are kept."""

    for node in mesh.NL:
        x, y = node.coor[:2]
        s = x - at
        node.coor = (x, y, 0.0) if s <= 0 else (at + s * np.cos(angle), y, s * np.sin(angle))
    return mesh


def strip_mesh(nx, ny, folded, angle=np.pi / 2, element=ElementType.quadrilateral):
    mesh = Mesh()
    mesh.generate2DMesh((L, W), (nx, ny), element, order=1)
    return fold(mesh, 0.4 * L, angle) if folded else mesh


def arc_length(xyz):
    """The coordinate along the strip, flat or folded at 0.4 L (90 degrees)."""
    return np.where(xyz[:, 2] > 1e-12, 0.4 * L + xyz[:, 2], xyz[:, 0]) if xyz.shape[1] == 3 else xyz[:, 0]


# ----- the geometry of a surface -----

def test_surface_discretization():
    """On a strip folded twice (by 90 and by 45 degrees) of T3 and Q4: total area, the gradient of a field linear in the arc length is
    tangential and of unit size, the boundary edges and their length, and the normals of the edges in the plane of their element,
    pointing out. A planar mesh whose nodes carry z = 0 is still a plain 2D one."""

    for element in (ElementType.quadrilateral, ElementType.triangle):
        mesh = Mesh()
        mesh.generate2DMesh((L, W), (10, 2), element)
        for node in mesh.NL:                                         # two folds: at x = 0.03 by 90 degrees, at 0.07 by 45 more
            x, y = node.coor
            if x <= 0.03:
                c = (x, y, 0.0)
            elif x <= 0.07:
                c = (0.03, y, x - 0.03)
            else:
                s = x - 0.07
                c = (0.03 - s * np.sin(np.pi / 4), y, 0.04 + s * np.cos(np.pi / 4))
            node.coor = c
        disc = Discretization(mesh)
        assert disc.manifold and disc.dim == 3
        assert np.isclose(disc.detJw.sum(), L * W)
        arc = np.array([n.coor for n in mesh.NL])
        s = np.where(arc[:, 2] < 1e-12, arc[:, 0], np.where(arc[:, 0] > 0.03 - 1e-12, 0.03 + arc[:, 2],
                                                            0.07 + np.hypot(arc[:, 0] - 0.03, arc[:, 2] - 0.04)))
        G = disc.gradient(s)
        assert np.allclose(np.linalg.norm(G, axis=-1), 1.0), element
        edges = mesh.boundary_facets()
        facets = disc.facets(edges)
        assert np.isclose(facets.detJw.sum(), 2 * (L + W)), element
        side = disc.facets(mesh.boundary_facets(lambda x, y, z: y > W - 1e-12))
        assert np.allclose(side.normals, [0.0, 1.0, 0.0]), element
        end = disc.facets(mesh.boundary_facets(lambda x, y, z: z > 0.04 + 1e-9 and y < W - 1e-12 and y > 1e-12))
        assert np.allclose(end.normals, [-np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)]), element

    flat = Mesh()
    flat.generate2DMesh((L, W), (4, 2), ElementType.quadrilateral)
    for node in flat.NL:
        node.coor = tuple(node.coor) + (0.0,)
    assert not Discretization(flat).manifold and Discretization(flat).dim == 2


def t_joint(dx, LA=0.03, LB=0.06, LC=0.06, Wj=0.01):

    """Three plates of quadrilaterals sharing the edge x = 0, z = 0: A (-LA < x < 0) and B (0 < x < LB) in the plane z = 0, C (0 < z < LC)
    at x = 0, normal to them."""

    ny = max(2, int(round(Wj / dx)))
    ys = np.linspace(0.0, Wj, ny + 1)
    mesh = Mesh()
    mesh.NL, mesh.EL = [], []
    index = {}

    def node(c):
        key = tuple(np.round(c, 12))
        if key not in index:
            index[key] = len(mesh.NL)
            mesh.NL.append(Node(id=len(mesh.NL), coor=tuple(float(v) for v in c)))
        return mesh.NL[index[key]]

    def plate(points):
        for i in range(len(points) - 1):
            for j in range(ny):
                corners = [points[i](ys[j]), points[i + 1](ys[j]), points[i + 1](ys[j + 1]), points[i](ys[j + 1])]
                mesh.EL.append(Element(id=len(mesh.EL), nodes=[node(c) for c in corners]))

    plate([lambda y, x=x: (x, y, 0.0) for x in np.linspace(-LA, 0.0, int(round(LA / dx)) + 1)])
    plate([lambda y, x=x: (x, y, 0.0) for x in np.linspace(0.0, LB, int(round(LB / dx)) + 1)])
    plate([lambda y, z=z: (0.0, y, z) for z in np.linspace(0.0, LC, int(round(LC / dx)) + 1)])
    mesh.PD, mesh.element_type, mesh.meshType = 3, 'quad4', ElementType.quadrilateral
    mesh.NoN, mesh.NoE = len(mesh.NL), len(mesh.EL)
    return mesh


def test_t_joint_topology_and_gmsh_surfaces():
    """The joint edge of a T is shared by three elements, so it is not boundary: the boundary is the outline of the three plates.
    A box without lid from gmsh (5 faces, surface=True) is read with its three coordinates and its groups, with the area of its faces."""

    mesh = t_joint(5e-3)
    disc = Discretization(mesh)
    length = disc.facets(mesh.boundary_facets()).detJw.sum()
    assert np.isclose(length, 2 * (0.03 + 0.06 + 0.06) + 3 * 0.01), length
    assert not any(abs(c[0]) < 1e-12 and abs(c[2]) < 1e-12 for f in mesh.boundary_facets() for c in [mesh.NL[f[0]].coor, mesh.NL[f[1]].coor]
                   if abs(mesh.NL[f[0]].coor[0]) < 1e-12 and abs(mesh.NL[f[1]].coor[0]) < 1e-12 and abs(mesh.NL[f[0]].coor[2]) < 1e-12
                   and abs(mesh.NL[f[1]].coor[2]) < 1e-12)

    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        volume = gmsh.model.occ.addBox(0, 0, 0, 0.1, 0.06, 0.02)
        gmsh.model.occ.synchronize()
        faces = [tag for _, tag in gmsh.model.getEntities(2)]
        lid = [s for s in faces if abs(gmsh.model.occ.getCenterOfMass(2, s)[2] - 0.02) < 1e-12]
        gmsh.model.occ.remove([(3, volume)])                         # the faces stay; the lid can then go
        gmsh.model.occ.remove([(2, s) for s in lid])
        gmsh.model.occ.synchronize()
        faces = [tag for _, tag in gmsh.model.getEntities(2)]
        rim = [c for c in gmsh.model.getBoundary([(2, s) for s in faces], combined=True, oriented=False)]
        gmsh.model.addPhysicalGroup(1, [abs(t) for _, t in rim], name='rim')
        gmsh.model.addPhysicalGroup(2, faces, name='part')
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.005)
        gmsh.model.mesh.generate(2)
        box = Mesh.from_gmsh_model(surface=True)
    finally:
        gmsh.finalize()
    d = Discretization(box)
    assert d.manifold and box.element_type == 'tri3'
    area = 0.1 * 0.06 + 2 * 0.1 * 0.02 + 2 * 0.06 * 0.02
    assert np.isclose(d.detJw.sum(), area, rtol=1e-12)
    assert np.isclose(d.facets(box.boundary_facets('rim')).detJw.sum(), 2 * (0.1 + 0.06))
    assert len(box.boundary_facets('rim')) == len(box.boundary_facets())
    note(f"box without lid: {box.getNoE()} triangles, area {d.detJw.sum():.6f} m2")


# ----- filling on a folded strip -----

def fill_strip(folded, law, t_fraction=0.75, element=ElementType.quadrilateral, thermal=None, pack=None):
    mesh = strip_mesh(40, 4, folded, element=element)
    fill = HeleShawFilling(mesh, H, law, T=230.0 if thermal else None)
    if thermal:
        fill.setThermal(**thermal)
    if pack:
        fill.setDensity(TAIT.density)
    fill.addGate(mesh.boundary_facets(lambda *c: c[0] < 1e-12), Q)
    fill.setFilled(lambda *c: (np.asarray(c[0]) <= 0.005 + 1e-12) & (np.asarray(c[2] if len(c) > 2 else 0 * c[0]) < 1e-12))
    if pack:
        fill.fill(progress=False, switchover=0.98)
        history = np.array(fill.pack(0.8 * fill.history[-1][2], 0.3, t_cool=0.2, progress=False))
    else:
        history = np.array(fill.fill(t_fraction * L * W * H / Q, progress=False))
    return fill, history


def compare_strips(label, **options):
    flat, h_flat = fill_strip(False, **options)
    folded, h_folded = fill_strip(True, **options)
    scale = np.maximum(np.max(np.abs(h_flat), axis=0), 1e-300)
    diff = np.max(np.abs(h_flat - h_folded) / scale)
    s_flat, s_folded = arc_length(flat.xy), arc_length(folded.xy)
    assert np.allclose(s_flat, s_folded)
    d_fill = np.nanmax(np.abs(flat.fill_time - folded.fill_time)) if np.any(np.isfinite(flat.fill_time)) else 0.0
    extra = ''
    # The packing iterates Newton to 1e-9 in every step and the march carries those differences: equal to that level, not to round-off
    tolerance = 1e-6 if options.get('pack') else 1e-10
    if flat.thermal is not None:
        extra = f", temperature {np.max(np.abs(flat.T - folded.T)):.1e} K"
        assert np.max(np.abs(flat.T - folded.T)) < (1e-3 if options.get('pack') else 1e-8)
    if options.get('pack'):
        extra += f", shrinkage {np.max(np.abs(flat.shrinkage() - folded.shrinkage())):.1e}"
        assert np.max(np.abs(flat.shrinkage() - folded.shrinkage())) < 1e-6
    note(f"{label}: folded - flat, history {diff:.1e}, fill times {d_fill:.1e} s{extra}")
    assert diff < tolerance and d_fill < (1e-5 if options.get("pack") else 1e-10)
    return flat, folded


def test_folded_strip_fills_as_the_flat_one():
    """A strip folded by 90 degrees is an isometry of the flat strip: the same history (time, volume, gate pressure), the same fill times
    of the nodes. Newtonian and power law n = 0.35 on Q4, Newtonian on T3."""

    compare_strips('Newtonian Q4', law=newtonian)
    compare_strips('power law Q4', law=power_law)
    compare_strips('Newtonian T3', law=newtonian, element=ElementType.triangle)


def test_folded_strip_thermal_and_packing():
    """The same with the temperature across the gap (cold mold, frozen layer, viscous heating) and then with the Tait melt filled
    compressibly, packed and cooled: temperature fields, histories and volumetric shrinkage of the folded strip equal those of the flat
    one (the filling to round-off; the packing to the tolerance of its Newton iterations, measured 2e-7 in the history, 3e-5 K, 2e-8 in
    the shrinkage)."""

    thermal = dict(T_melt=230.0, T_mold=50.0, rho=970.0, cp=2000.0, k=0.2, htc=5000.0, T_noflow=150.0)
    law = lambda g2, T: MU * jnp.exp(-0.02 * (T - 230.0)) + 0 * g2
    compare_strips('non-isothermal', law=law, thermal=thermal)
    compare_strips('Tait fill + pack + cool', law=law, thermal=thermal, pack=True)


# ----- a T joint -----

def test_t_joint_against_the_exact_split():
    """A plate A (gap 3 mm) feeds a joint where plate B (2 mm) continues in its plane and plate C (1 mm) rises at 90 degrees. For a Newtonian
    melt both branches fill from the joint with x_i dx_i/dt = h_i^2 p_j / (12 mu): x_i = h_i s(t), so their volumes keep the ratio
    (h_B / h_C)^2 = 4, and s = Q t' / (W sum h_i^2), p_j = 12 mu s ds/dt. Mass to round-off; the volume ratio and the gate pressure
    converge at first order (one control volume of the front), measured 4.137, 4.067, 4.033 and +5.4, +2.2, +1.1 %."""

    LA, LB, Wj, hA, hB, hC, Qj = 0.03, 0.06, 0.01, 3e-3, 2e-3, 1e-3, 1e-6
    results = []
    for dx in (2e-3, 1e-3):
        mesh = t_joint(dx, LA=LA, LB=LB, Wj=Wj)
        gap = lambda x, y, z: np.where(x < -1e-12, hA, np.where(z > 1e-12, hC, hB))
        fill = HeleShawFilling(mesh, gap, newtonian)
        fill.addGate(mesh.boundary_facets(lambda x, y, z: x < -LA + 1e-12), Qj)
        fill.setFilled(lambda x, y, z: x <= -LA + 1e-12)
        t_joint_arrival = LA * Wj * hA / Qj
        history = np.array(fill.fill(t_joint_arrival + 0.5 * LB * Wj * (hB + hC * hC / hB) / Qj, progress=False))
        xyz = fill.xy
        in_b = (xyz[:, 2] < 1e-12) & (xyz[:, 0] > 1e-12)
        in_c = xyz[:, 2] > 1e-12
        ratio = (fill.volume[in_b] @ fill.fill_factor[in_b]) / (fill.volume[in_c] @ fill.fill_factor[in_c])
        t = history[-1, 0]
        s, ds = Qj * (t - t_joint_arrival) / (Wj * (hB ** 2 + hC ** 2)), Qj / (Wj * (hB ** 2 + hC ** 2))
        p_exact = 12 * MU * s * ds + 12 * MU * Qj * LA / (Wj * hA ** 3)
        mass = abs(history[-1, 1] - history[0, 1] - Qj * t) / (Qj * t)
        results.append((ratio, history[-1, 2] / p_exact - 1, mass))
        note(f"T joint, cell {dx * 1e3:.0f} mm ({mesh.getNoE()} elements): volume ratio B / C {ratio:.3f} (exact 4), gate pressure "
             f"{results[-1][1]:+.2%}, mass {mass:.1e}")

    (r1, p1, m1), (r2, p2, m2) = results
    assert m1 < 1e-12 and m2 < 1e-12
    assert abs(r2 - 4) < 0.1 and 1.6 < (r1 - 4) / (r2 - 4) < 2.5
    assert abs(p2) < 0.03 and 1.6 < p1 / p2 < 3.0


def main():
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    failures = 0

    for name, fn in tests:
        start = time.perf_counter()
        try:
            fn()
            print(f"PASS  {name} ({time.perf_counter() - start:.1f} s)")
        except Exception as exc:
            failures += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}: {exc!r}")

    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
