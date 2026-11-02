"""Do the T overshoots of the elbow coincide with a high y+? Real y+ from the wall shear, per wall, and overshoot nodes crossed with it."""
import sys
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial import cKDTree

D = np.load(sys.argv[1])
xy, conn, u, T, d, G, x_gp = D['xy'], D['conn'], D['u'], D['T'], D['d'], D['gradient'], D['x_gp']
walls = D['wall_tags']
NU, MU, RHO = 1e-6, 1e-3, 1000.0
T1, T2 = 20.0, 40.0
mm = xy * 1e3


def segment(c):
    x, y = c
    r = np.hypot(x - 200, y - 200)
    if abs(y) < 1e-3 and x <= 200: return 'bottom'
    if abs(y - 100) < 1e-3 and x <= 200: return 'top'
    if abs(x - 300) < 1e-3 and y >= 200: return 'left'
    if abs(x - 400) < 1e-3 and y >= 200: return 'right'
    if abs(x - 325) < 1e-3 and y < 50: return 'side_l'
    if abs(x - 350) < 1e-3 and y < 70: return 'side_r'
    if r < 150: return 'arc_inner'
    return 'arc_outer' if x <= 325 else 'arc_trans'


owner = {}
for e, nodes in enumerate(conn):
    for n in nodes:
        owner.setdefault(int(n), []).append(e)

rows = []
for edge in walls:
    candidates = set(owner[int(edge[0])])
    for n in edge[1:]:
        candidates &= set(owner[int(n)])
    e = next(iter(candidates))
    ends = edge[np.argsort(np.hypot(*(mm[edge] - mm[edge].mean(axis=0)).T))[-2:]]
    t = xy[ends[1]] - xy[ends[0]]
    length = np.linalg.norm(t); t = t / length
    n = np.array([-t[1], t[0]])
    centre = xy[edge].mean(axis=0)
    g = np.argmin(np.linalg.norm(x_gp[e] - centre, axis=1))
    S = G[e, g] + G[e, g].T
    tau = MU * abs(t @ S @ n)
    ut = np.sqrt(tau / RHO)
    off = conn[e][d[conn[e]] > 1e-12]
    y_first = d[off].min()
    vertices = conn[e][:3]
    v_off = vertices[d[vertices] > 1e-12]
    y_vertex = d[v_off].min() if v_off.size else y_first
    rows.append((segment(centre * 1e3), centre[0], centre[1], ut, y_first, y_vertex, y_first * ut / NU, y_vertex * ut / NU, length))
seg = np.array([r[0] for r in rows])
A = np.array([r[1:] for r in rows], dtype=float)
cx, cy, ut, y_first, y_vertex, yp, ypv, length = A.T
print(f"{len(rows)} wall facets. y+ of the first node: median {np.median(yp):.2f}, 95 % {np.percentile(yp, 95):.2f}, max {yp.max():.2f}; of the first vertex: median {np.median(ypv):.2f}, max {ypv.max():.2f}")
print(f"{'wall':10s} {'n':>5s} {'y1 mm min..max':>18s} {'y+ med':>7s} {'p95':>7s} {'max':>7s} {'y+v med':>8s} {'max':>7s} {'share y+>5':>10s} {'u_tau med':>9s} {'facet mm':>9s}")
for s in ('bottom', 'top', 'arc_inner', 'arc_outer', 'side_l', 'side_r', 'arc_trans', 'right', 'left'):
    m = seg == s
    if not m.any():
        continue
    print(f"{s:10s} {m.sum():5d} {y_first[m].min() * 1e3:8.3f}..{y_first[m].max() * 1e3:8.3f} {np.median(yp[m]):7.2f} {np.percentile(yp[m], 95):7.2f} {yp[m].max():7.2f} {np.median(ypv[m]):8.2f} {ypv[m].max():7.2f} {100 * np.mean(yp[m] > 5):9.0f}% {np.median(ut[m]):9.4f} {np.median(length[m]) * 1e3:9.2f}")

excess = np.maximum(T1 - T, 0.0) + np.maximum(T - T2, 0.0)
over = excess > 1e-6
inside = (mm[:, 0] > 310) & (mm[:, 0] < 360) & (mm[:, 1] > 35) & (mm[:, 1] < 90)
print(f"\nIn the union box: {inside.sum()} nodes, T in [{T[inside].min():.2f}, {T[inside].max():.2f}], over 0.5 C: {(np.maximum(T1 - T, 0) + np.maximum(T - T2, 0))[inside].__gt__(0.5).sum()}; outside: T in [{T[~inside].min():.2f}, {T[~inside].max():.2f}], over 0.5 C: {(np.maximum(T1 - T, 0) + np.maximum(T - T2, 0))[~inside].__gt__(0.5).sum()}")
print(f"\nT in [{T.min():.2f}, {T.max():.2f}]. Nodes out of bounds: {over.sum()} of {len(T)} ({100 * over.mean():.1f} %); by more than 0.5 C: {(excess > 0.5).sum()}, 1 C: {(excess > 1).sum()}, 2 C: {(excess > 2).sum()}")
vertex_nodes = np.unique(conn[:, :3])
is_vertex = np.isin(np.arange(len(T)), vertex_nodes)
print(f"of the > 0.5 C: vertices {(excess[is_vertex] > 0.5).sum()}, midside {(excess[~is_vertex] > 0.5).sum()}; below 20: {(T < 19.5).sum()}, above 40: {(T > 40.5).sum()}")

centres = np.c_[cx, cy]
tree = cKDTree(centres)
_, nearest = tree.query(xy)
for threshold in (0.5, 2.0):
    bad = excess > threshold
    if bad.any():
        print(f"\nexcess > {threshold} C by nearest wall, with distance to it:")
        for s in np.unique(seg[nearest[bad]]):
            m = bad & (seg[nearest] == s)
            print(f"  {s:10s} {m.sum():5d} nodes, largest {excess[m].max():6.2f} C, d {np.percentile(d[m], 10) * 1e3:7.3f}..{np.percentile(d[m], 90) * 1e3:7.3f} mm, y+ of that wall there {np.median(yp[nearest[m]]):6.2f}")
bad = excess > 0.5
print("\nexcess > 0.5 C by distance to the wall:")
for lo, hi in ((0, 0.1), (0.1, 0.5), (0.5, 2), (2, 5), (5, 20), (20, 200)):
    m = (d * 1e3 >= lo) & (d * 1e3 < hi)
    print(f"  d {lo:5.1f}-{hi:5.1f} mm: {m.sum():6d} nodes, {(bad & m).sum():5d} out ({100 * (bad & m).sum() / max(m.sum(), 1):5.1f} %), largest {excess[m].max() if m.any() else 0:6.2f} C")

order = np.argsort(excess)[::-1][:15]
print("\nworst nodes:   x mm     y mm    T      d mm   nearest wall  y+ there  vertex |u| m/s")
for i in order:
    print(f"            {mm[i, 0]:7.1f}  {mm[i, 1]:7.1f}  {T[i]:6.2f}  {d[i] * 1e3:6.2f}  {seg[nearest[i]]:12s} {yp[nearest[i]]:7.2f}  {str(is_vertex[i]):5s} {np.linalg.norm(u[i]):.3f}")

print("\ncrossing, facet by facet: largest excess within 5 mm of each facet, against its y+")
tree_nodes = cKDTree(xy)
local = np.array([excess[tree_nodes.query_ball_point(c, 5e-3)].max() for c in centres])
rho, p = spearmanr(yp, local)
print(f"  Spearman y+ of the first node vs local excess: {rho:+.3f} (p = {p:.1e}, n = {len(yp)})")
rho, p = spearmanr(ypv, local)
print(f"  Spearman y+ of the first vertex vs local excess: {rho:+.3f} (p = {p:.1e})")
for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 11), (11, 30), (30, 1e9)):
    m = (yp >= lo) & (yp < hi)
    if m.any():
        print(f"  y+ {lo:4.0f}-{min(hi, 999):4.0f}: {m.sum():5d} facets, median local excess {np.median(local[m]):6.3f} C, share > 0.5 C {100 * np.mean(local[m] > 0.5):5.1f} %, largest {local[m].max():6.2f} C")
for s in np.unique(seg):
    m = seg == s
    if m.sum() > 3 and np.ptp(yp[m]) > 0:
        rho, _ = spearmanr(yp[m], local[m])
        print(f"  within {s:10s}: Spearman {rho:+.3f}, median local excess {np.median(local[m]):6.3f} C")
