"""Balancing a family mold with the feed system of the Hele-Shaw filling (Source/Physics/Runners.py): two plates of different size,
2 mm thick, POLYFORT FPP 30 GFC K1079 (Cross-WLF) at 230 °C into a mold at 50 °C, fed by a sprue and two branches with a pin gate at
the middle of the short side of each plate. With equal branches the small plate fills first; its branch is then narrowed (the usual
way: restrict the cavity that fills early) by the secant method on its diameter until both plates are full at the same time. Widening
the branch of the large plate hardly helps: the gates and the plates hold most of the resistance, and a wide branch only balances by
the time it takes to fill itself. The melt in the network is at the injection temperature (a hot runner).

Result (770 nodes, about 20 s per filling, 5 fillings): with both branches of 4 mm the plates are full at 0.454 and 0.582 s (the large
one 0.128 s later); with 2.4 mm, 0.073 s earlier; the secant settles at 2.77 mm (both at 0.568 s). The inlet pressure printed is that of
the last step, which jumps as the last nodes fill (19.7 to 24.4 MPa for nearly the same filling): compare fill times, not it. A hot
runner and no packing: a cold runner and the packing through the gates would move the balance. No measurements to compare with.

Run from the project root:  python -u Example_Family_Mold.py
"""

import time

import numpy as np

from Source.Pre_processing.Mesh import Mesh
from Source.Physics.HeleShaw import HeleShawFilling
from Source.Physics.Rheology import CrossWLF
from Source.Physics.Runners import RunnerSystem

LAW = CrossWLF(n=0.2619, tau_star=1.778e4, D1=4.378e17, D2=266.7, A1=39.22, A2=51.6, gamma_min=1e-2)
H, DX = 2e-3, 3e-3
PLATES = [(0.06, 0.04, 0.00), (0.10, 0.04, 0.06)]            # length, width, y of the lower edge
SPRUE_D, BRANCH_D, GATE_D, GATE_L = 6e-3, 4e-3, 1.5e-3, 1e-3
Q = 25e-6                                                    # 25 cm3/s


def plates():
    xy, conn, members = [], [], []
    for L, W, y0 in PLATES:
        nx, ny = int(round(L / DX)), int(round(W / DX))
        x, y = np.linspace(0, L, nx + 1), y0 + np.linspace(0, W, ny + 1)
        base = len(xy)
        xy += [(xi, yj) for yj in y for xi in x]
        idx = lambda i, j: base + j * (nx + 1) + i
        conn += [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)] for j in range(ny) for i in range(nx)]
        members.append(np.arange(base, len(xy)))
    return Mesh.from_arrays(np.array(xy), np.array(conn), 'quad4'), members


def run(D1, mesh, members):

    """Fills with the branch of the small plate D1 across (the other BRANCH_D); returns when each plate is full and the filling."""

    fill = HeleShawFilling(mesh, H, LAW, T=230.0)
    fill.setThermal(230.0, 50.0, 970.0, 2000.0, 0.22, htc=5000.0, T_noflow=453.9 - 273.15, cells=12)
    rs = RunnerSystem()
    inlet = rs.addNode((-0.04, 0.05, 0.0))
    joint = rs.addSegment(inlet, (-0.02, 0.05, 0.0), SPRUE_D, divisions=4)
    for (L, W, y0), D in zip(PLATES, (D1, BRANCH_D)):
        y = y0 + W / 2
        bend = rs.addSegment(joint, (-0.02, y, 0.0), D, divisions=4)
        end = rs.addSegment(bend, (0.0, y, 0.0), D, divisions=6)
        rs.addGate(end, (0.0, y), GATE_D, GATE_L)
    rs.setInlet(inlet, Q)
    fill.addRunners(rs)
    fill.fill(progress=False, batch=0.1)
    return [float(np.max(fill.fill_time[m])) for m in members], fill


def report(D1, mesh, members, history):
    start = time.perf_counter()
    full, fill = run(D1, mesh, members)
    history.append((D1, full[1] - full[0]))
    print(f"branch of the small plate {D1 * 1e3:.3f} mm: plates full at {full[0]:.3f} and {full[1]:.3f} s (imbalance {full[1] - full[0]:+.3f} s), "
          f"inlet pressure {fill.history[-1][2] / 1e6:.2f} MPa ({time.perf_counter() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    mesh, members = plates()
    print(f"{mesh.getNoN()} nodes; plates {PLATES[0][0] * 1e3:.0f} and {PLATES[1][0] * 1e3:.0f} mm long, 40 mm wide, 2 mm thick; "
          f"Q = {Q * 1e6:.0f} cm3/s; branches of {BRANCH_D * 1e3:.0f} mm, pin gates of {GATE_D * 1e3:.1f} mm")
    history = []
    for D1 in (BRANCH_D, 0.6 * BRANCH_D):
        report(D1, mesh, members, history)
    for _ in range(6):                                                   # secant on the imbalance
        (d0, f0), (d1, f1) = history[-2], history[-1]
        if abs(f1) < 2e-3 or f1 == f0:
            break
        report(max(d1 - f1 * (d1 - d0) / (f1 - f0), 0.5e-3), mesh, members, history)
    print(f"balanced with the branch of the small plate {history[-1][0] * 1e3:.2f} mm across (the other {BRANCH_D * 1e3:.1f} mm)")
