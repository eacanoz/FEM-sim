"""Hele-Shaw filling with several nodes per step (fill(batch=...)) and the faster assembly: the fixed-pattern assembly against the plain
one, the centre-gated disk filled in batches (mass to round-off, the exact gate pressure, the fill times of the node-by-node filling), the
energy kept exactly by the overflow of the nodes that fill inside a step (adiabatic), the thermal filling against the node-by-node one,
and the compressible filling refusing batches (node by node only). Each test prints its progress.

Run from the project root:  python -u Test/test_hele_shaw_batch.py
"""

import os
import sys
import time

import gmsh
import numpy as np
import jax.numpy as jnp
import scipy.sparse as sp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.Physics.HeleShaw import HeleShawFilling, _scatter
from Source.Physics.Rheology import Cross
from Source.Simulation.Discretization import Discretization
from Source.enums import ElementType

START = time.perf_counter()
MU = 1000.0
CROSS = Cross(4565.0, 1.778e4, 0.2619)


def note(text):
    print(f"   [{time.perf_counter() - START:6.1f} s] {text}", flush=True)


def newtonian(g2, T=None):
    return MU + 0 * g2


def quarter_disk(r0=2e-3, R=30e-3, size=1.5e-3):
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
        return Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()


def test_assembly():

    """The fixed-pattern assembly equals the plain COO sum (triangles and quadrilaterals), also after a matrix of it had its zeros
    eliminated in place (the pattern is not shared)."""

    for et in (ElementType.triangle, ElementType.quadrilateral):
        mesh = Mesh()
        mesh.generate2DMesh((1.0, 0.5), (11, 7), et, order=1)
        disc = Discretization(mesh)
        conn, nn = disc.conn, disc.conn.shape[1]
        rng = np.random.default_rng(1)
        for _ in range(2):
            Ke = rng.random((len(conn), nn, nn))
            Ke[::3] = 0.0
            plain = sp.csr_matrix((Ke.ravel(), (np.repeat(conn, nn, axis=1).ravel(), np.tile(conn, (1, nn)).ravel())),
                                  shape=(disc.n_nodes,) * 2)
            A = _scatter(disc, Ke)
            err = abs(A - plain).max()
            A.eliminate_zeros()
            note(f"{et.name}: {A.shape[0]} nodes, against the plain assembly {err:.1e}")
            assert err < 1e-14


def test_disk_batches():

    """Centre-gated quarter disk, Newtonian: filled in batches of 20 % of the front, the melt volume grows as Q t to round-off, the
    gate pressure is within 1 % of 6 mu Q / (pi h^3) ln(r_f / r0) (0.5 % node by node), the steps are at least 3 times fewer, and the fill times are those
    of the node-by-node filling within 1.5 % of the fill time (rms) and one step (max)."""

    r0, h, Q, r1 = 2e-3, 2e-3, 4e-6, 4e-3
    mesh = quarter_disk(r0)
    t_end = np.pi * h * (0.025 ** 2 - r1 ** 2) / Q
    results = {}
    for batch in (0.0, 0.2):
        fill = HeleShawFilling(mesh, h, newtonian)
        fill.addGate(mesh.boundary_edges('gate'), Q / 4)
        fill.setFilled(lambda x, y: np.hypot(x, y) <= r1)
        history = np.array(fill.fill(t_end, progress=False, batch=batch))
        t, volume, p_gate = history[:, 0], history[:, 1], history[:, 2]
        growth = abs(fill.melt_volume() - volume[0] - Q / 4 * fill.time) / (Q / 4 * fill.time)
        rf = np.sqrt(4 * fill.melt_volume() / (np.pi * h) + r0 ** 2)
        fill.solve_pressure()
        p = fill.p[fill._gate_nodes()].mean()
        err = p / (6 * MU * Q / (np.pi * h ** 3) * np.log(rf / r0)) - 1
        results[batch] = (fill.fill_time.copy(), len(t))
        note(f"batch {batch}: {len(fill.xy)} nodes, {len(t)} steps; melt volume against Q t {growth:.1e}; gate pressure {err:+.2%}")
        assert growth < 1e-12 and abs(err) < (0.005 if batch == 0 else 0.01)
    (f0, n0), (f1, n1) = results[0.0], results[0.2]
    both = ~np.isnan(f0) & ~np.isnan(f1)
    rms, worst = np.sqrt(np.mean((f1 - f0)[both] ** 2)), np.max(np.abs(f1 - f0)[both])
    note(f"fill times against node by node: rms {rms / t_end:.2%}, max {worst / t_end:.2%} of the fill time; {n0 / n1:.1f} times fewer steps")
    assert n0 >= 3 * n1 and rms < 0.015 * t_end and worst < 0.06 * t_end


def test_energy_overflow():

    """Adiabatic walls, no viscous heating, no freezing: melt at 150 °C already in the cavity, 230 °C injected. The heat content
    sum V f integral T dz grows by Q t T_melt to round-off, node by node and in batches (the overflow carries its heat), and every
    temperature stays between 150 and 230 °C."""

    r0, h, Q, r1 = 2e-3, 2e-3, 4e-6, 6e-3
    mesh = quarter_disk(r0, R=20e-3, size=2e-3)
    for batch in (0.0, 0.3):
        fill = HeleShawFilling(mesh, h, CROSS)
        fill.setThermal(230.0, 50.0, 970.0, 2000.0, 0.2, htc=0.0, viscous_heating=False, cells=6)
        fill.addGate(mesh.boundary_edges('gate'), Q / 4)
        fill.setFilled(lambda x, y: np.hypot(x, y) <= r1)
        fill.T[:] = 150.0
        dz = fill.thermal['dz']
        heat = lambda: float(np.sum(fill.volume * fill.fill_factor * (fill.T @ dz)))
        e0 = heat()
        fill.fill(progress=False, batch=batch)
        err = abs(heat() - e0 - Q / 4 * fill.time * 230.0) / (Q / 4 * fill.time * 230.0)
        melt = fill.fill_factor > 0
        note(f"batch {batch}: {len(fill.history)} steps; heat content against the injected {err:.1e}; T from {fill.T[melt].min():.4f} to "
             f"{fill.T[melt].max():.4f} °C")
        assert err < 1e-10 and fill.T[melt].min() > 150.0 - 1e-8 and fill.T[melt].max() < 230.0 + 1e-8


def test_thermal_batches():

    """Cross melt into a cold mold (frozen layer, viscous heating), quarter disk: in batches of 10 % of the front the gate pressure at the
    end is within 1.5 % of the node-by-node filling, the fill times within 1 % (rms) of the fill time, the mean temperatures within 5 °C,
    the mass to round-off, with at least 2 times fewer steps."""

    r0, h, Q = 2e-3, 2e-3, 4e-6
    mesh = quarter_disk(r0, R=25e-3, size=1.5e-3)
    out = {}
    for batch in (0.0, 0.1):
        fill = HeleShawFilling(mesh, h, CROSS)
        fill.setThermal(230.0, 50.0, 970.0, 2000.0, 0.2, htc=5000.0, T_noflow=150.0, cells=10)
        fill.addGate(mesh.boundary_edges('gate'), Q / 4)
        start = time.perf_counter()
        fill.fill(progress=False, batch=batch)
        growth = abs(fill.melt_volume() - Q / 4 * fill.time - fill.volume[fill._gate_nodes()].sum()) / (Q / 4 * fill.time)
        out[batch] = (fill.history[-1][2], fill.fill_time.copy(), fill.mean_temperature(), len(fill.history), fill.time)
        note(f"batch {batch}: {len(fill.xy)} nodes, {len(fill.history)} steps, {time.perf_counter() - start:.1f} s; gate pressure "
             f"{fill.history[-1][2] / 1e6:.4f} MPa, mean frozen fraction {fill.frozen_fraction().mean():.3f}; mass {growth:.1e}")
        assert growth < 1e-12
    (p0, f0, T0, n0, t0), (p1, f1, T1, n1, _) = out[0.0], out[0.1]
    rms = np.sqrt(np.nanmean((f1 - f0) ** 2))
    note(f"in batches: gate pressure {p1 / p0 - 1:+.2%}, fill times rms {rms / t0:.2%} of the fill time, mean temperature up to "
         f"{np.max(np.abs(T1 - T0)):.2f} °C apart; {n0 / n1:.1f} times fewer steps")
    assert abs(p1 / p0 - 1) < 0.015 and rms < 0.01 * t0 and np.max(np.abs(T1 - T0)) < 5.0 and n0 >= 2 * n1


def test_compressible_refuses_batches():

    """Batches are for the incompressible filling: the compressible one refuses them (with a nearly incompressible melt they gave a gate
    pressure 2e5 times too high: a node that fills early inside the step cannot pass its excess on without compressing)."""

    mesh = Mesh()
    mesh.generate2DMesh((0.1, 0.02), (10, 2), ElementType.quadrilateral, order=1)
    fill = HeleShawFilling(mesh, 2e-3, newtonian)
    fill.addGate(mesh.boundary_edges(lambda x, y: x < 1e-12), 1e-6)
    fill.setDensity(lambda p, T: 900.0 * jnp.exp(1e-14 * (p - 1e5)))
    try:
        fill.fill(progress=False, batch=0.2)
    except ValueError as e:
        note(f"refused: {e}")
        return
    raise AssertionError("the compressible filling took batch")


if __name__ == "__main__":
    tests = [test_assembly, test_disk_batches, test_energy_overflow, test_thermal_batches, test_compressible_refuses_batches]
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
