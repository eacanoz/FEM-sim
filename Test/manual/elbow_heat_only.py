"""Heat alone on the saved flow of the mixing elbow: constant properties, so T does not feed back and the problem is linear.
The saved u and nut are kept exactly when refining: every T6 splits in 4 and the new nodes take the parent's quadratic values.
Variants: galerkin; A the code as it is (tau without k, residual without diffusion); B tau with k(x), residual without diffusion;
C tau with k(x), residual with div(k grad T). Args: npz levels variants"""
import sys, os, time
sys.path.insert(0, os.getcwd())
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from Source.Pre_processing.Mesh import Mesh, Node, Element
from Source.enums import ElementType
from Source.Simulation.Discretization import Discretization
import Source.Simulation.Operators as ops

MU, RHO, CP, K = 1e-3, 1000.0, 4180.0, 0.6
NU, PR_T, CV1, T1, T2 = MU / RHO, 0.85, 7.1, 20.0, 40.0
REF = np.array([[0, 0], [1, 0], [0, 1], [.5, 0], [.5, .5], [0, .5]])


def p2(r, s):
    a, b, c = 1 - r - s, r, s
    return np.array([a * (2 * a - 1), b * (2 * b - 1), c * (2 * c - 1), 4 * a * b, 4 * b * c, 4 * c * a])


def refine(xy, conn, fields):
    """Each T6 into 4; the geometry and the fields stay the parent's quadratics."""
    children = [(0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)]          # vertices of the children, as local nodes of the parent
    weights = [[p2(*(REF[c[i]] + REF[c[j]]) / 2) for i, j in ((0, 1), (1, 2), (2, 0))] for c in children]
    new, values, out = {}, [], []
    n = len(xy)
    data = np.c_[xy, fields]
    for nodes in conn:
        local = data[nodes]
        for c, w in zip(children, weights):
            mids = []
            for (i, j), wk in zip(((0, 1), (1, 2), (2, 0)), w):
                key = tuple(sorted((int(nodes[c[i]]), int(nodes[c[j]]))))
                if key not in new:
                    new[key] = n + len(values)
                    values.append(wk @ local)
                mids.append(new[key])
            out.append([nodes[c[0]], nodes[c[1]], nodes[c[2]], *mids])
    data = np.r_[data, np.array(values)]
    return data[:, :2], np.array(out), data[:, 2:]


def discretization(xy, conn):
    mesh = Mesh()
    mesh.PD, mesh.element_type, mesh.meshType = 2, 'tri6', ElementType.triangle
    mesh.NL = [Node(id=i, coor=tuple(c)) for i, c in enumerate(xy)]
    mesh.EL = [Element(id=e, nodes=[mesh.NL[i] for i in row]) for e, row in enumerate(conn)]
    mesh.NoN, mesh.NoE = len(mesh.NL), len(mesh.EL)
    return Discretization(mesh)


def conductivity(nut):
    nut = np.maximum(nut, 0.0)
    chi = nut / NU
    return K + RHO * CP * nut * chi ** 3 / (chi ** 3 + CV1 ** 3) / PR_T


def solve(disc, xy, u, nut, variant):
    v, c = disc.interpolate(u), RHO * CP
    k = conductivity(disc.interpolate(nut))
    Ke = ops.convection(disc, c, v) + ops.diffusion(disc, k)
    if variant != 'galerkin':
        tau = ops.supg_tau(disc, c, v, None if variant == 'A' else k)
        Ke = Ke + ops.supg_terms(disc, c, v, tau)[0]
        if variant == 'C':
            a = c * v
            wa = np.einsum('egia,ega->egi', disc.B, a)
            grad_k = disc.gradient(conductivity(nut))
            div_flux = k[..., None] * disc.laplacian + np.einsum('egja,ega->egj', disc.B, grad_k)
            Ke = Ke - np.einsum('eg,egi,egj->eij', disc.detJw * tau, wa, div_flux)
    conn = disc.conn
    rows = np.repeat(conn, conn.shape[1], axis=1).ravel()
    cols = np.tile(conn, (1, conn.shape[1])).ravel()
    A = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(len(xy), len(xy)))
    tol = 1e-9
    main, side = np.abs(xy[:, 0]) < tol, np.abs(xy[:, 1] + 0.025) < tol
    fixed = main | side
    T = np.zeros(len(xy))
    T[main], T[side] = T1, T2
    free = np.flatnonzero(~fixed)
    T[free] = spla.splu((A[free][:, free]).tocsc()).solve(-(A[free][:, np.flatnonzero(fixed)] @ T[fixed]))
    return T


def report(tag, disc, xy, u, T):
    Tg = disc.interpolate(T)
    excess = np.maximum(T1 - Tg, 0) + np.maximum(Tg - T2, 0)
    nodal = np.maximum(T1 - T, 0) + np.maximum(T - T2, 0)
    mm = xy * 1e3
    box = (mm[:, 0] > 310) & (mm[:, 0] < 360) & (mm[:, 1] > 35) & (mm[:, 1] < 90)
    area = disc.detJw.sum()
    # the same flux for every level: integral of u_y T over the outlet row with the exact 3-point rule of the quadratic edge
    row = np.flatnonzero(np.abs(xy[:, 1] - 0.4) < 1e-9)
    row = row[np.argsort(xy[row, 0])]
    x = xy[row, 0]
    assert len(row) % 2 == 1
    h = x[2::2] - x[:-2:2]
    f = u[row, 1] * T[row]; g = u[row, 1]
    bulk = np.sum(h / 6 * (f[:-2:2] + 4 * f[1::2] + f[2::2])) / np.sum(h / 6 * (g[:-2:2] + 4 * g[1::2] + g[2::2]))
    i = np.argmax(nodal)
    print(f"{tag:22s} {len(xy):7d} T [{T.min():6.2f},{T.max():6.2f}] box [{T[box].min():6.2f},{T[box].max():6.2f}] out [{T[~box].min():6.2f},{T[~box].max():6.2f}]"
          f" area>0.5C {1e6 * disc.detJw[excess > 0.5].sum():8.1f} mm2 ({100 * disc.detJw[excess > 0.5].sum() / area:5.2f} %) int(excess) {1e6 * (disc.detJw * excess).sum():8.1f} C mm2"
          f" outlet {bulk:.3f} worst ({mm[i, 0]:.1f},{mm[i, 1]:.1f})", flush=True)
    corners = np.array([[325.0, 200 - np.sqrt(200.0 ** 2 - 125.0 ** 2)], [350.0, 200 - np.sqrt(200.0 ** 2 - 150.0 ** 2)]])
    r = np.min(np.linalg.norm(mm[:, None] - corners[None], axis=2), axis=1)
    print("    largest excess farther than r from the corners of the union: " + ", ".join(f"r>{q:g} mm {nodal[r > q].max():.2f} C" for q in (0, 1, 2, 5, 10)))
    # energy: outlet flux - inlet flux = integral of T div(u), because the convective form does not see a discrete div(u) that is not zero
    du = np.einsum('egaa->eg', disc.gradient(u))
    source = (disc.detJw * Tg * du).sum()
    flow = np.sum(h / 6 * (g[:-2:2] + 4 * g[1::2] + g[2::2]))
    print(f"    integral T div(u) / Q_out = {source / flow:+.4f} C; integral div(u) / Q_out = {(disc.detJw * du).sum() / flow:+.2e}; L1 of div(u) / Q_out = {(disc.detJw * np.abs(du)).sum() / flow:.3f}", flush=True)
    return T


D = np.load(sys.argv[1])
levels, variants = int(sys.argv[2]), sys.argv[3].split(',')
xy, conn, fields = D['xy'], D['conn'], np.c_[D['u'], D['nut']]
for level in range(levels + 1):
    if level:
        xy, conn, fields = refine(xy, conn, fields)
    disc = discretization(xy, conn)
    u, nut = fields[:, :2], fields[:, 2]
    if level == 0:
        print(f"divergence of the saved flow, L2: {np.sqrt((disc.detJw * np.einsum('egaa->eg', disc.gradient(u)) ** 2).sum()):.3e}")
    for variant in variants:
        t = time.time()
        T = solve(disc, xy, u, nut, variant)
        report(f"L{level} {variant} ({time.time() - t:.0f} s)", disc, xy, u, T)
        if level == 0 and variant == 'A':
            print(f"    against the saved coupled T: max |diff| {np.abs(T - D['T']).max():.2e} C, saved in [{D['T'].min():.2f},{D['T'].max():.2f}]", flush=True)
