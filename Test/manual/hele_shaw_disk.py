"""Hele-Shaw filling of a centre-gated disk against the exact solution, on a quarter of an annulus (symmetry: no flux through the straight sides):
front r_f = sqrt(r1^2 + Q t / (pi h)), gate pressure Newtonian 6 mu Q / (pi h^3) ln(r_f / r0); power law G1 (r_f^(1-n) - r0^(1-n)) / (1 - n) with
G1 = K (Q / (2 pi c b^(2+1/n)))^n, c = 2n / (2n + 1). Q is the flow rate of the whole disk. Args: law, element size [mm], reinit every."""
import sys, os, time
sys.path.insert(0, os.getcwd())
import numpy as np
import gmsh
from Source.Pre_processing.Mesh import Mesh
from Source.Physics.HeleShaw import HeleShawFilling

law = sys.argv[1] if len(sys.argv) > 1 else 'newtonian'
size = float(sys.argv[2]) * 1e-3 if len(sys.argv) > 2 else 1e-3
reinit = int(sys.argv[3]) if len(sys.argv) > 3 else 5
r0, R, h, Q, mu, r1 = 2e-3, 50e-3, 2e-3, 4e-6, 1000.0, 4e-3
K, n = 1e4, 0.35
b = h / 2

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
try:
    geo = gmsh.model.geo
    c = geo.addPoint(0, 0, 0, size)
    a1, a2 = geo.addPoint(r0, 0, 0, size), geo.addPoint(0, r0, 0, size)
    b1, b2 = geo.addPoint(R, 0, 0, size), geo.addPoint(0, R, 0, size)
    inner, outer = geo.addCircleArc(a1, c, a2), geo.addCircleArc(b1, c, b2)
    s1, s2 = geo.addLine(a1, b1), geo.addLine(b2, a2)
    geo.addPlaneSurface([geo.addCurveLoop([s1, outer, s2, -inner])])
    geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [inner], name='gate')
    gmsh.model.addPhysicalGroup(1, [outer], name='vent')
    gmsh.model.mesh.generate(2)
    mesh = Mesh.from_gmsh_model()
finally:
    gmsh.finalize()

viscosity = (lambda g2: mu + 0 * g2) if law == 'newtonian' else (lambda g2: K * (g2 + 1e-6) ** ((n - 1) / 2))
fill = HeleShawFilling(mesh, h, viscosity)
fill.addGate(mesh.boundary_edges('gate'), Q / 4)
fill.setFilled(lambda x, y: np.hypot(x, y) <= r1)
if law == 'newtonian':
    gate_pressure = lambda rf: 6 * mu * Q / (np.pi * h ** 3) * np.log(rf / r0)
else:
    cn = 2 * n / (2 * n + 1)
    G1 = K * (Q / (2 * np.pi * cn * b ** (2 + 1 / n))) ** n
    gate_pressure = lambda rf: G1 * (rf ** (1 - n) - r0 ** (1 - n)) / (1 - n)

print(f"[disk {law} size {size * 1e3:g} mm] {len(fill.xy)} nodes", flush=True)
start = time.perf_counter()
t_end = np.pi * h * (0.04 ** 2 - r1 ** 2) / Q
history = np.array(fill.fill(t_end, progress=True))
t, volume, p_gate = history[:, 0], history[:, 1], history[:, 2]
growth = np.max(np.abs(volume - volume[0] - Q / 4 * t)) / (Q / 4 * t[-1])
rf_volume = np.sqrt(4 * volume / (np.pi * h) + r0 ** 2)           # the quarter of the annulus from r0
r = np.hypot(fill.xy[:, 0], fill.xy[:, 1])
front = (fill.fill_factor > 0) & (fill.fill_factor < 1)
rel = p_gate[5:] / gate_pressure(rf_volume[5:]) - 1
print(f"[disk {law} size {size * 1e3:g}] {len(t)} steps, {time.perf_counter() - start:.0f} s; melt volume against Q t: {growth:.1e}; front nodes at "
      f"r = {r[front].min() * 1e3:.2f} to {r[front].max() * 1e3:.2f} mm (from the volume {rf_volume[-1] * 1e3:.2f}); gate pressure at the end "
      f"{p_gate[-1] / 1e6:.4f} MPa, exact for the melt volume {gate_pressure(rf_volume[-1]) / 1e6:.4f} ({rel[-1]:+.2%}), largest after 5 steps "
      f"{np.max(np.abs(rel)):.2%}", flush=True)
