"""Mold filling in the Hele-Shaw approximation (Source/Physics/HeleShaw.py): the gap fluidity, and the filling of a strip and of a
centre-gated disk against their exact solutions, on coarse meshes. Each test prints its progress.

Run from the project root:  python -u Test/test_hele_shaw.py
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

import gmsh
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.Physics.HeleShaw import GapFluidity, HeleShawFilling
from Source.Physics.Rheology import Cross
from Source.enums import ElementType

START = time.perf_counter()
MU, K, N = 1000.0, 1e4, 0.35


def note(text):
    print(f"   [{time.perf_counter() - START:6.1f} s] {text}", flush=True)


def newtonian(g2):
    return MU + 0 * g2


def power_law(g2):
    return K * (g2 + 1e-6) ** ((N - 1) / 2)


def test_gap_fluidity():

    """S = integral z^2 / eta over the gap: Newtonian 2 b^3 / (3 mu), power law 2 n / (2n + 1) b^(2+1/n) K^(-1/n) G^(1/n - 1), exactly;
    and a Rheology law (Cross) through its own call."""

    b, G = 1e-3, np.array([1e4, 1e6, 1e8])
    assert np.allclose(GapFluidity(newtonian)(b, G), 2 * b ** 3 / (3 * MU), rtol=1e-13)
    exact = 2 * N / (2 * N + 1) * b ** (2 + 1 / N) * K ** (-1 / N) * G ** (1 / N - 1)
    pure = lambda g2: K * g2 ** ((N - 1) / 2)                        # without the regularization of power_law: at G = 1e4 the rate is 1e-9 1/s
    with np.errstate(divide='ignore'):
        err = np.max(np.abs(GapFluidity(pure, layers=16)(b, G) / exact - 1))
    note(f"power law with 16 layers: {err:.1e}")
    assert err < 1e-10
    cross = GapFluidity(Cross(4565.0, 1.778e4, 0.2619), T=230.0)
    assert np.all(np.diff(cross(b, G) * G) > 0)                          # the flow rate grows with the gradient


def strip(law, nx):
    L, W, h, Q, x0 = 0.1, 0.02, 2e-3, 1e-6, 0.005
    mesh = Mesh()
    mesh.generate2DMesh((L, W), (nx, max(2, nx // 10)), ElementType.quadrilateral, order=1)
    fill = HeleShawFilling(mesh, h, law)
    fill.addGate(mesh.boundary_edges(lambda x, y: x < 1e-12), Q)
    fill.setFilled(lambda x, y: x <= x0 + 1e-12)
    history = np.array(fill.fill((0.08 - x0) * h * W / Q, progress=False))
    t, volume, p_gate = history[:, 0], history[:, 1], history[:, 2]
    if law is newtonian:
        exact = lambda xf: 12 * MU * Q * xf / (h ** 3 * W)
    else:
        G = K * (Q / W / (2 * N / (2 * N + 1) * (h / 2) ** (2 + 1 / N))) ** N
        exact = lambda xf: G * xf
    growth = np.max(np.abs(volume - volume[0] - Q * t)) / (Q * t[-1])
    return growth, p_gate[-1] / exact(volume[-1] / (h * W)) - 1


def test_strip():

    """Constant flow rate into a strip: the melt volume grows as Q t to round-off (the control volumes conserve mass), and the gate pressure
    approaches the exact one of the filled length at first order in the size of a control volume (the front is known to one of them)."""

    for law, name in ((newtonian, 'Newtonian'), (power_law, 'power law n = 0.35')):
        errors = []
        for nx in (40, 80):
            growth, err = strip(law, nx)
            errors.append(err)
            note(f"{name}, {nx} elements: melt volume against Q t {growth:.1e}, gate pressure {err:+.2%}")
            assert growth < 1e-12
        assert abs(errors[1]) < 0.01 and abs(errors[0] / errors[1] - 2) < 0.1


def test_centre_gated_disk():

    """A centre-gated disk (a quarter, by symmetry), the curved front: mass to round-off, the front within one control volume of the exact
    radius, the gate pressure 6 mu Q / (pi h^3) ln(r_f / r0) within 0.5 % at the end."""

    r0, R, h, Q, r1, size = 2e-3, 30e-3, 2e-3, 4e-6, 4e-3, 1.5e-3
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        geo = gmsh.model.geo
        c = geo.addPoint(0, 0, 0, size)
        a1, a2, b1, b2 = (geo.addPoint(x, y, 0, size) for x, y in ((r0, 0), (0, r0), (R, 0), (0, R)))
        inner, outer = geo.addCircleArc(a1, c, a2), geo.addCircleArc(b1, c, b2)
        geo.addPlaneSurface([geo.addCurveLoop([geo.addLine(a1, b1), outer, geo.addLine(b2, a2), -inner])])
        geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [inner], name='gate')
        gmsh.model.mesh.generate(2)
        mesh = Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()
    fill = HeleShawFilling(mesh, h, newtonian)
    fill.addGate(mesh.boundary_edges('gate'), Q / 4)
    fill.setFilled(lambda x, y: np.hypot(x, y) <= r1)
    history = np.array(fill.fill(np.pi * h * (0.025 ** 2 - r1 ** 2) / Q, progress=False))
    t, volume, p_gate = history[:, 0], history[:, 1], history[:, 2]
    growth = np.max(np.abs(volume - volume[0] - Q / 4 * t)) / (Q / 4 * t[-1])
    rf = np.sqrt(4 * volume[-1] / (np.pi * h) + r0 ** 2)
    err = p_gate[-1] / (6 * MU * Q / (np.pi * h ** 3) * np.log(rf / r0)) - 1
    r = np.hypot(fill.xy[:, 0], fill.xy[:, 1])
    front = (fill.fill_factor > 0) & (fill.fill_factor < 1)
    note(f"{len(fill.xy)} nodes, {len(t)} steps: melt volume against Q t {growth:.1e}; front nodes at {r[front].min() * 1e3:.2f} to "
         f"{r[front].max() * 1e3:.2f} mm (from the volume {rf * 1e3:.2f}); gate pressure {err:+.2%}")
    assert growth < 1e-12 and abs(err) < 0.005
    assert r[front].min() > rf - 2 * size and r[front].max() < rf + 2 * size


if __name__ == "__main__":
    tests = [test_gap_fluidity, test_strip, test_centre_gated_disk]
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
