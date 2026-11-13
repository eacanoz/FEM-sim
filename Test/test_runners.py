"""Feed systems for the Hele-Shaw filling (Source/Physics/Runners.py, HeleShawFilling.addRunners): the fluidity of a circular tube
(Newtonian and power law exact); a runner that fills before its cavity, with the inlet pressure of the power law at the front position;
two symmetric cavities that fill together; a family mold (two strips of different length) against an independent integration of the
balance of flow rates, with two diameters of one branch (the imbalance changes as predicted); the heat that the gates bring; the
compressible filling refused. Each test prints its progress.

Run from the project root:  python -u Test/test_runners.py
"""

import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.Physics.HeleShaw import HeleShawFilling
from Source.Physics.Rheology import Cross
from Source.Physics.Runners import RunnerSystem, TubeFluidity
from Source.enums import ElementType

START = time.perf_counter()
MU, K, N = 1000.0, 5000.0, 0.35
H = 2e-3


def note(text):
    print(f"   [{time.perf_counter() - START:6.1f} s] {text}", flush=True)


def newtonian(g2):
    return MU + 0 * g2


def power_law(g2):
    return K * (g2 + 1e-30) ** ((N - 1) / 2)


def strips(lengths, W=0.01, dx=2.5e-3, gap=0.03):

    """Strips of the given lengths along x from x = 0, W wide, one above the other (y offset `gap`), in one mesh; returns the mesh and,
    per strip, the nodes of its edge at x = 0 and its node indices."""

    xy, conn, edges, members = [], [], [], []
    for k, L in enumerate(lengths):
        nx, ny = int(round(L / dx)), max(2, int(round(W / dx)))
        x, y = np.linspace(0, L, nx + 1), k * gap + np.linspace(0, W, ny + 1)
        base = len(xy)
        xy += [(xi, yj) for yj in y for xi in x]
        idx = lambda i, j: base + j * (nx + 1) + i
        conn += [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)] for j in range(ny) for i in range(nx)]
        edges.append([idx(0, j) for j in range(ny + 1)])
        members.append(np.arange(base, len(xy)))
    return Mesh.from_arrays(np.array(xy), np.array(conn), 'quad4'), edges, members


def test_tube_fluidity():

    """S(R, G) of a tube: Newtonian pi R^4 / (8 mu) to round-off, power law pi n / (3n + 1) R^3 (R G / 2K)^(1/n) to 1e-12 (G >= 1e4,
    where the regularization of the test law does not show), and a Cross melt with the flow rate growing with G."""

    R, G = 2e-3, np.logspace(4, 9, 6)
    e_newt = np.max(np.abs(TubeFluidity(newtonian)(np.full_like(G, R), G) / (np.pi * R ** 4 / (8 * MU)) - 1))
    exact = np.pi * N / (3 * N + 1) * R ** 3 * (R * G / (2 * K)) ** (1 / N)
    e_pow = np.max(np.abs(TubeFluidity(power_law)(np.full_like(G, R), G) * G / exact - 1))
    cross = TubeFluidity(Cross(4565.0, 1.778e4, 0.2619), T=230.0)
    rising = np.all(np.diff(cross(np.full_like(G, R), G) * G) > 0)
    note(f"Newtonian {e_newt:.1e}, power law {e_pow:.1e}, Cross flow rate rising: {rising}")
    assert e_newt < 1e-14 and e_pow < 1e-12 and rising


def test_runner_front():

    """A runner 100 mm long, 4 mm across (40 elements), then a strip 50 x 10 x 2 mm through a pin gate, power law: the melt volume grows
    as Q t to round-off, the runner fills first (no cavity node before it), and while it fills the inlet pressure is G(Q) times the front
    position, within one element of the runner (G = 2K / R (Q (3n + 1) / (pi n R^3))^n, exact)."""

    R, Q, Lr = 2e-3, 1e-6, 0.1
    mesh, _, _ = strips([0.05])
    fill = HeleShawFilling(mesh, H, power_law)
    rs = RunnerSystem()
    a = rs.addNode((-Lr, 0.005, 0.0))
    b = rs.addSegment(a, (0.0, 0.005, 0.0), 2 * R, divisions=40)
    rs.setInlet(a, Q)
    rs.addGate(b, (0.0, 0.005), 1.5e-3, 1e-3)
    fill.addRunners(rs)
    v0 = fill.melt_volume()
    h = np.array(fill.fill(progress=False))
    growth = abs(fill.melt_volume() - v0 - Q * fill.time) / (Q * fill.time)
    t_runner = np.nanmax(rs.fill_time)
    first_cavity = np.nanmin(fill.fill_time)
    G = 2 * K / R * (Q * (3 * N + 1) / (np.pi * N * R ** 3)) ** N
    during = (h[:, 0] > 0) & (h[:, 0] < t_runner)
    x_front = h[during, 1] / (np.pi * R ** 2)                               # melt in the runner / its section
    off = np.max(np.abs(h[during, 2] / G - x_front)) / (Lr / 40)
    note(f"{len(h)} steps; mass {growth:.1e}; runner full at {t_runner:.4f} s, first cavity node at {first_cavity:.4f} s; inlet pressure / G "
         f"against the front position: within {off:.2f} elements")
    assert growth < 1e-12 and first_cavity >= t_runner - 1e-12 and off <= 1.0


def test_symmetric_cavities():

    """Two equal strips 60 x 10 x 2 mm fed by a sprue and two equal branches (Newtonian): both cavities hold the same melt at every
    step (1e-9 of the cavity) and fill at the same time, node by node to 1e-9 of the fill time."""

    Q = 2e-6
    mesh, _, members = strips([0.06, 0.06])
    fill = HeleShawFilling(mesh, H, newtonian)
    rs = RunnerSystem()
    inlet = rs.addNode((-0.03, 0.02, 0.0))
    joint = rs.addSegment(inlet, (-0.02, 0.02, 0.0), 5e-3, divisions=4)
    e1 = rs.addSegment(joint, (-0.02, 0.005, 0.0), 4e-3, divisions=6)
    e1 = rs.addSegment(e1, (0.0, 0.005, 0.0), 4e-3, divisions=8)
    e2 = rs.addSegment(joint, (-0.02, 0.035, 0.0), 4e-3, divisions=6)
    e2 = rs.addSegment(e2, (0.0, 0.035, 0.0), 4e-3, divisions=8)
    rs.setInlet(inlet, Q)
    rs.addGate(e1, (0.0, 0.005), 1.5e-3, 1e-3)
    rs.addGate(e2, (0.0, 0.035), 1.5e-3, 1e-3)
    fill.addRunners(rs)
    split = []
    fill.fill(progress=False, on_step=lambda f: split.append(f.fill_factor[members[0]] @ f.volume[members[0]]
                                                              - f.fill_factor[members[1]] @ f.volume[members[1]]))
    V = fill.volume[members[0]].sum()
    t0, t1 = fill.fill_time[members[0]], fill.fill_time[members[1]]
    note(f"{len(fill.history)} steps; melt in cavity 1 - cavity 2 up to {np.max(np.abs(split)) / V:.1e} of a cavity; fill times apart "
         f"{np.max(np.abs(t0 - t1)) / fill.time:.1e} of the fill time")
    assert np.max(np.abs(split)) < 1e-9 * V and np.max(np.abs(t0 - t1)) < 1e-9 * fill.time


def family(D2):

    """A family mold, Newtonian: strips of 40 and 80 mm (10 x 2 mm), branches of 30 mm, 4 mm and D2 across, fan gates along the whole
    edge of each strip (a gate tube 6 mm across and 0.1 mm long to every node of it: the strip then fills as in 1D). Returns the time
    each cavity is full in the simulation and by the balance of flow rates p = Q_i (R_branch_i(V_i) + R_cavity_i(V_i)) from the moment
    the junction fills (solve_ivp)."""

    Q, W, Lb, D1 = 2e-6, 0.01, 0.03, 4e-3
    lengths = [0.04, 0.08]
    mesh, edges, members = strips(lengths, W=W)
    fill = HeleShawFilling(mesh, H, newtonian)
    rs = RunnerSystem()
    inlet = rs.addNode((-0.05, 0.02, 0.0))
    joint = rs.addSegment(inlet, (-0.03, 0.02, 0.0), 6e-3, divisions=4)
    ends = []
    for k, (D, edge) in enumerate(zip((D1, D2), edges)):
        y = mesh.NL[edge[len(edge) // 2]].coor[1]
        bend = rs.addSegment(joint, (-0.03, y, 0.0), D, divisions=2)
        ends.append(rs.addSegment(bend, (-0.03 + Lb, y, 0.0), D, divisions=12))
    rs.setInlet(inlet, Q)
    for end, edge in zip(ends, edges):
        for node in edge:
            rs.addGate(end, mesh.NL[node].coor, 6e-3, 1e-4)
    fill.addRunners(rs)
    fill.fill(progress=False)
    t_sim = [float(np.max(fill.fill_time[m])) for m in members]
    # the balance: branch i from the junction, length l_i (the bend and the straight part), then its strip
    X = np.array(rs.xyz)
    t_joint = rs.fill_time[joint]
    l_branch = [abs(X[joint, 1] - mesh.NL[e[len(e) // 2]].coor[1]) + Lb for e in edges]
    D = (D1, D2)
    # the junction node holds half of the first element of each branch (control volumes): the branch beyond it is that much shorter
    first = [abs(X[joint, 1] - mesh.NL[e[len(e) // 2]].coor[1]) / 2 for e in edges]
    Vb = [np.pi * (d / 2) ** 2 * (l - f / 2) for d, l, f in zip(D, l_branch, first)]
    Vc = [W * H * L for L in lengths]

    def resistance(i, V):
        in_branch = min(V, Vb[i]) / (np.pi * (D[i] / 2) ** 2)
        in_cavity = max(V - Vb[i], 0.0) / (W * H)
        return 8 * MU * in_branch / (np.pi * (D[i] / 2) ** 4) + 12 * MU * in_cavity / (H ** 3 * W) + 1e-3

    def rates(t, V):
        open_ = [V[i] < Vb[i] + Vc[i] for i in range(2)]
        g = np.array([1.0 / resistance(i, V[i]) if open_[i] else 0.0 for i in range(2)])
        return Q * g / g.sum() if g.sum() > 0 else 0.0 * g

    events = [lambda t, V, i=i: V[i] - (Vb[i] + Vc[i]) for i in range(2)]
    t_ode = [np.nan, np.nan]
    V, t = np.zeros(2), t_joint
    for _ in range(2):
        for e in events:
            e.terminal = True
        sol = solve_ivp(rates, (t, t + 100.0), V, events=events, rtol=1e-10, atol=1e-16, max_step=0.01)
        t, V = sol.t[-1], sol.y[:, -1]
        for i in range(2):
            if np.isnan(t_ode[i]) and V[i] >= Vb[i] + Vc[i] - 1e-15:
                t_ode[i] = t
                V[i] = Vb[i] + Vc[i]
        events = [ev for i, ev in enumerate(events) if np.isnan(t_ode[i])]
    return t_sim, t_ode


def test_family_mold():

    """The family mold with the second branch 3 and 5 mm across: the time each cavity is full agrees with the balance of flow rates
    within 1 % of the fill time, and the imbalance (the long strip later) within 1 %; the wider branch cuts it to a third, as the
    balance says."""

    for D2 in (3e-3, 5e-3):
        t_sim, t_ode = family(D2)
        T = max(t_sim)
        err = max(abs(a - b) for a, b in zip(t_sim, t_ode)) / T
        imb_sim, imb_ode = t_sim[1] - t_sim[0], t_ode[1] - t_ode[0]
        note(f"branch 2 of {D2 * 1e3:.0f} mm: cavities full at {t_sim[0]:.3f} / {t_sim[1]:.3f} s, balance {t_ode[0]:.3f} / {t_ode[1]:.3f} s "
             f"({err:.2%} of the fill time); imbalance {imb_sim:+.3f} s against {imb_ode:+.3f} s")
        assert err < 0.01 and abs(imb_sim - imb_ode) < 0.01 * T and np.sign(imb_sim) == np.sign(imb_ode)


def test_heat_through_the_gates():

    """With temperature (adiabatic walls, no heating, no freezing) the melt enters the cavity at the injection temperature through the
    gates: every filled cell starts at 150 °C, and at the end the heat content of the cavity is 230 °C times its melt (1e-10)."""

    mesh, _, _ = strips([0.04])
    fill = HeleShawFilling(mesh, H, Cross(4565.0, 1.778e4, 0.2619))
    fill.setThermal(230.0, 50.0, 970.0, 2000.0, 0.2, htc=0.0, viscous_heating=False, cells=6)
    fill.T[:] = 150.0
    rs = RunnerSystem()
    a = rs.addNode((-0.03, 0.005, 0.0))
    b = rs.addSegment(a, (0.0, 0.005, 0.0), 4e-3, divisions=10)
    rs.setInlet(a, 1e-6)
    rs.addGate(b, (0.0, 0.005), 1.5e-3, 1e-3)
    fill.addRunners(rs)
    fill.fill(progress=False)
    heat = float(np.sum(fill.volume * fill.fill_factor * (fill.T @ fill.thermal['dz'])))
    err = abs(heat / (230.0 * fill.cavity_volume_filled()) - 1)
    note(f"{len(fill.history)} steps; heat content against 230 °C times the melt {err:.1e}; T from {fill.T.min():.3f} to {fill.T.max():.3f} °C")
    assert err < 1e-10


def test_compressible_refused():

    """The feed system is in the incompressible filling only: the compressible filling refuses it."""

    import jax.numpy as jnp
    mesh, _, _ = strips([0.02])
    fill = HeleShawFilling(mesh, H, newtonian)
    rs = RunnerSystem()
    a = rs.addNode((-0.01, 0.005, 0.0))
    rs.setInlet(a, 1e-6)
    rs.addGate(rs.addSegment(a, (0.0, 0.005, 0.0), 4e-3, 2), (0.0, 0.005), 1.5e-3, 1e-3)
    fill.addRunners(rs)
    fill.setDensity(lambda p, T: 900.0 * jnp.exp(1e-14 * (p - 1e5)))
    try:
        fill.fill(progress=False)
    except NotImplementedError as e:
        note(f"refused: {e}")
        return
    raise AssertionError("the compressible filling took the feed system")


if __name__ == "__main__":
    tests = [test_tube_fluidity, test_runner_front, test_symmetric_cavities, test_family_mold, test_heat_through_the_gates,
             test_compressible_refused]
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
