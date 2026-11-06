"""Hele-Shaw filling of a strip at constant flow rate against the exact solution: front x_f = x0 + Q t / (h W), and the gate pressure of the
filled length, Newtonian 12 mu Q x_f / (h^3 W), power law G x_f with G from q = 2 n / (2n + 1) b^(2+1/n) (G / K)^(1/n).
Args: law ('newtonian' or 'power'), elements along, reinit every."""
import sys, os, time
sys.path.insert(0, os.getcwd())
import numpy as np
from Source.Pre_processing.Mesh import Mesh
from Source.Physics.HeleShaw import HeleShawFilling
from Source.enums import ElementType

law = sys.argv[1] if len(sys.argv) > 1 else 'newtonian'
nx = int(sys.argv[2]) if len(sys.argv) > 2 else 80
reinit = int(sys.argv[3]) if len(sys.argv) > 3 else 5
L, W, h, Q, mu, x0 = 0.1, 0.02, 2e-3, 1e-6, 1000.0, 0.005
K, n = 1e4, 0.35
b = h / 2

mesh = Mesh()
mesh.generate2DMesh((L, W), (nx, max(2, nx // 10)), ElementType.quadrilateral, order=1)
viscosity = (lambda g2: mu + 0 * g2) if law == 'newtonian' else (lambda g2: K * (g2 + 1e-6) ** ((n - 1) / 2))
fill = HeleShawFilling(mesh, h, viscosity)
fill.addGate(mesh.boundary_edges(lambda x, y: x < 1e-12), Q)
fill.setFilled(lambda x, y: x <= x0 + 1e-12)

if law == 'newtonian':
    gate_pressure = lambda xf: 12 * mu * Q * xf / (h ** 3 * W)
else:
    q = Q / W
    G = K * (q / (2 * n / (2 * n + 1) * b ** (2 + 1 / n))) ** n
    gate_pressure = lambda xf: G * xf

start = time.perf_counter()
t_end = (0.08 - x0) * h * W / Q
history = np.array(fill.fill(t_end, progress=True))
t, volume, p_gate = history[:, 0], history[:, 1], history[:, 2]
growth = np.max(np.abs(volume - volume[0] - Q * t)) / (Q * t[-1])
xf_volume = volume / (h * W)
# the pressure is solved with p = 0 on the first nodes not filled: its exact value is that of the length up to them
print(f"[{law} nx {nx}] {len(t)} steps, {time.perf_counter() - start:.0f} s; melt volume against Q t: {growth:.1e}; gate pressure at the end "
      f"{p_gate[-1] / 1e6:.4f} MPa, exact for the melt volume {gate_pressure(xf_volume[-1]) / 1e6:.4f} ({p_gate[-1] / gate_pressure(xf_volume[-1]) - 1:+.2%}), "
      f"largest over the filling {np.max(np.abs(p_gate[5:] / gate_pressure(xf_volume[5:]) - 1)):.2%} (after the first 5 steps)", flush=True)
