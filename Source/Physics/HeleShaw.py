# Mold filling of a thin cavity in the Hele-Shaw approximation (2.5D, midplane): the mesh is the planform of the part, of gap h(x, y)
# (full thickness), symmetric about its midplane. Per unit area the gap carries the flow rate q = -S grad p, with the fluidity
#
#     S = integral over the gap of z^2 / eta dz = 2 integral_0^b z^2 / eta(gamma_dot(z)) dz,   b = h / 2,
#
# and the local shear rate solves gamma_dot eta(gamma_dot) = z |grad p| (generalized Newtonian, the stress grows linearly across the gap).
# The pressure satisfies div(S grad p) = 0 in the melt.
#
# The front, by control volumes with a fill factor (the FAN method of Wang & Hieber, the usual one of injection-molding software): node i
# owns the volume V_i = integral N_i h dA and is filled to f_i in [0, 1]. The pressure is solved on the filled nodes, with p = 0 on the
# nodes not yet filled that touch the melt (the front, vented: the air leaves freely); the reaction of the system at those nodes is the flow
# rate they receive, V_i df_i / dt = R_i, and each step lasts until the next node fills. The mass is conserved exactly by construction (the
# reactions add up to the injected flow rate); the front is known to one control volume, and the pressure converges with the mesh.
# A level set carried by the velocity of a melt-air mixture was tried first: the flow rate of the elements cut by the front and of the air
# next to it is not reliable (a disk front came out 1.5 % slow or 4.6 % fast depending on how the velocity was reconstructed).
#
# Linear elements (T3, Q4). Isothermal in this stage. SI units: gap [m], pressure [Pa], flow rate [m3/s].
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from Source.Simulation.Discretization import Discretization
import Source.Simulation.Operators as ops


def _scatter(disc, Ke):
    conn = disc.conn
    n = disc.n_nodes
    rows = np.repeat(conn, conn.shape[1], axis=1).ravel()
    cols = np.tile(conn, (1, conn.shape[1])).ravel()
    return sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(n, n))


def _scatter_vector(disc, fe, nodes):
    out = np.zeros(disc.n_nodes)
    np.add.at(out, nodes.ravel(), fe.ravel())
    return out


class GapFluidity:

    """S(|grad p|) of a gap of half-thickness b for a viscosity law eta(gamma_dot^2) (a callable, or a law of Source.Physics.Rheology called
    with a fixed T): Gauss-Legendre with `layers` points across the half gap, the local shear rate from gamma_dot eta(gamma_dot) = z G by
    Newton on log(gamma_dot)."""

    def __init__(self, viscosity, T=None, layers=16):
        from Source.Physics.Rheology import ViscosityLaw
        if isinstance(viscosity, ViscosityLaw):
            self.eta = lambda g2: np.asarray(viscosity(g2, T if T is not None else 0.0), dtype=float)
        else:
            self.eta = lambda g2: np.asarray(viscosity(g2), dtype=float)
        x, w = np.polynomial.legendre.leggauss(layers)
        self.s, self.ws = 0.5 * (x + 1.0), 0.5 * w                       # points and weights on (0, 1)

    def shear_rate(self, tau):

        """gamma_dot with gamma_dot eta(gamma_dot) = tau (arrays), Newton on log gamma_dot from the Newtonian guess at rest."""

        tau = np.maximum(np.asarray(tau, dtype=float), 1e-300)
        with np.errstate(divide='ignore', invalid='ignore'):
            rest = self.eta(np.zeros_like(tau))
        rest = np.where(np.isfinite(rest) & (rest > 0), rest, self.eta(np.ones_like(tau)))    # a power law has no rest viscosity
        g = tau / rest
        for _ in range(60):
            g = np.maximum(g, 1e-300)
            eta = self.eta(g ** 2)
            h = 1e-6 * g
            slope = ((g + h) * self.eta((g + h) ** 2) - (g - h) * self.eta((g - h) ** 2)) / (2 * h)   # d(tau)/d(gamma_dot)
            step = (g * eta - tau) / np.maximum(slope * g, 1e-300)
            g = g * np.exp(-np.clip(step, -5.0, 5.0))
            if np.max(np.abs(step)) < 1e-12:
                break
        return g

    def table(self, b, G_min=1e-3, G_max=1e12, points=1201):

        """log G and log q_m(G) = log(S(b, G) G) on a log grid, for interpolation: q_m grows monotonically with G."""

        logG = np.linspace(np.log(G_min), np.log(G_max), points)
        G = np.exp(logG)
        return logG, np.log(self(np.full_like(G, b), G) * G)

    def __call__(self, b, G):

        """S for half gaps b and pressure gradients G (same shape)."""

        b, G = np.broadcast_arrays(np.asarray(b, dtype=float), np.asarray(G, dtype=float))
        z = b[..., None] * self.s                                              # (..., layers)
        gamma = self.shear_rate(z * G[..., None])
        eta = self.eta(gamma ** 2)
        return 2.0 * b * np.sum(self.ws * z ** 2 / eta, axis=-1)


class HeleShawFilling:

    def __init__(self, mesh, gap, viscosity, T=None, layers=16):

        """mesh: the planform (2D, linear elements). gap: full thickness h [m], a number or f(x, y). viscosity: eta(gamma_dot^2) or a Rheology
        law at the fixed temperature T."""

        self.mesh = mesh
        self.disc = Discretization(mesh)
        if self.disc.element.linear_subspace:
            raise ValueError("Hele-Shaw usa elementos lineales (T3, Q4): los volumenes de control necesitan pesos positivos.")
        self.xy = np.array([n.coor[:2] for n in mesh.NL], dtype=float)
        self.fluidity = GapFluidity(viscosity, T, layers)
        gp = self.disc.x_gp
        self.h = np.broadcast_to(np.asarray(gap(gp[..., 0], gp[..., 1]) if callable(gap) else gap, dtype=float), self.disc.detJw.shape)
        self.volume = _scatter_vector(self.disc, np.einsum('eg,gi->ei', self.disc.detJw * self.h, self.disc.N), self.disc.conn)
        self.fill_factor = np.zeros(len(self.xy))
        self.fill_time = np.full(len(self.xy), np.nan)                  # when each node filled: the isochrones of the filling
        self.p = np.zeros(len(self.xy))
        self.gates = []
        self.time, self.history = 0.0, []

    def addGate(self, edges, Q):

        """Inflow Q [m3/s] spread evenly (per unit length) over the boundary edges; their nodes start filled."""

        facets = self.disc.facets(edges)
        self.gates.append((facets, float(Q)))
        self.fill_factor[np.unique(facets.nodes)] = 1.0
        self.fill_time[np.unique(facets.nodes)] = 0.0

    def setFilled(self, region):

        """Nodes where region(x, y) is True start filled (an initial short shot)."""

        inside = np.asarray(region(self.xy[:, 0], self.xy[:, 1]), dtype=bool)
        self.fill_factor[inside] = 1.0
        self.fill_time[inside] = 0.0

    # ----- the pressure of the melt -----

    def _tables(self):
        if getattr(self, '_table', None) is None:
            b_e = 0.5 * self.h.mean(axis=1)
            keys = np.round(b_e / b_e.max(), 9)
            unique, index = np.unique(keys, return_inverse=True)
            if len(unique) > 200:
                raise ValueError("Mas de 200 espesores distintos: la tabla de la fluidez por espesor no esta pensada para un espesor continuo.")
            self._table = ([self.fluidity.table(float(k * b_e.max())) for k in unique], index)
        return self._table

    def _element_fluidity(self, G, derivative=False):

        """S_e(G_e) of each element from the table of its gap, G_e the mean |grad p| of the element; with derivative, dS_e / dG_g at each point."""

        tables, index = self._tables()
        w = self.disc.detJw
        W = w.sum(axis=1)
        G_e = np.maximum(np.sum(w * G, axis=1) / W, 1e-300)
        logG = np.log(G_e)
        S, slope = np.empty_like(G_e), np.empty_like(G_e)
        inside = np.ones_like(G_e, dtype=bool)
        for k, (lg, lq) in enumerate(tables):
            m = index == k
            # outside the table, the fluidity of its end: an element whose nodes are all at p = 0 (just reached by the melt) has G_e = 0, and
            # q / G with q from the end of the table would give S ~ 1e297, which pins its pressure
            lgm = np.clip(logG[m], lg[0], lg[-1])
            inside[m] = (logG[m] > lg[0]) & (logG[m] < lg[-1])
            S[m] = np.exp(np.interp(lgm, lg, lq) - lgm)
            i = np.clip(np.searchsorted(lg, lgm) - 1, 0, len(lg) - 2)
            slope[m] = (lq[i + 1] - lq[i]) / (lg[i + 1] - lg[i])                # d log q / d log G
        Sg = np.broadcast_to(S[:, None], w.shape)
        if not derivative:
            return Sg, None
        dS_e = np.where(inside, S * (slope - 1.0) / G_e, 0.0)                 # dS / dG = (q' - S) / G
        return Sg, (dS_e / W)[:, None] * w

    def solve_pressure(self, tol=1e-10, max_iterations=50):

        """div(S grad p) = 0 over the elements with a filled node, p unknown on the filled nodes and 0 on the others; Newton with the full
        tangent of S(|grad p|) (Picard does not converge for a strongly thinning melt), started by damped Picard from rest. Returns the
        reactions (the flow rate each node receives) and the Newton iterations."""

        disc = self.disc
        filled = self.fill_factor >= 1.0
        active = np.any(filled[disc.conn], axis=1)                              # elements that carry melt
        F = np.zeros(len(self.xy))
        for facets, Q in self.gates:
            F += _scatter_vector(disc, ops.facet_source(facets, Q / facets.detJw.sum()), facets.nodes)
        free = np.flatnonzero(filled)
        mask = active[:, None].astype(float)

        def stiffness(S):
            return _scatter(disc, ops.diffusion(disc, S * mask))

        p = np.where(filled, self.p, 0.0)
        B = disc.B
        if not np.any(p[free]):
            S = self.fluidity(0.5 * self.h, np.full_like(self.h, 1e6))
            for _ in range(30):                                                 # damped Picard: Newton alone does not start from rest
                A = stiffness(S)
                p[free] = spla.spsolve(A[free][:, free].tocsc(), F[free])
                S_new, _ = self._element_fluidity(np.linalg.norm(disc.gradient(p), axis=-1))
                change = np.max(np.abs(np.log(S_new[active, 0] / S[active, 0])))
                S = np.sqrt(S * S_new)
                if change < 1e-3:
                    break
        for k in range(1, max_iterations + 1):
            gp = disc.gradient(p)
            G = np.linalg.norm(gp, axis=-1)
            S, dS = self._element_fluidity(G, derivative=True)
            A = stiffness(S)
            R = A @ p - F
            a = np.einsum('eg,egia,ega->ei', disc.detJw, B, gp) * active[:, None]
            c = np.einsum('eg,ega,egja->ej', dS / np.maximum(G, 1e-300), gp, B)
            J = A + _scatter(disc, np.einsum('ei,ej->eij', a, c))
            dp = np.zeros_like(p)
            dp[free] = spla.spsolve(J[free][:, free].tocsc(), -R[free])
            p = p + dp
            if np.max(np.abs(dp)) <= tol * max(np.max(np.abs(p)), 1e-300):
                break
        self.p = p
        S, _ = self._element_fluidity(np.linalg.norm(disc.gradient(p), axis=-1))
        self.S = S * mask
        reaction = F - stiffness(S) @ p                                          # flow rate into each node not filled
        reaction[filled] = 0.0
        return np.maximum(reaction, 0.0), k

    # ----- the filling -----

    def melt_volume(self):
        return float(self.fill_factor @ self.volume)

    def fill(self, t_end=np.inf, progress=True, every=50, on_step=None):

        """Fills node by node until t_end or until every node is filled. Keeps self.history of (t, melt volume, mean gate pressure, Newton
        iterations, filled nodes) and returns it; self.fill_time holds when each node filled. on_step(self) is called after each step (to take
        snapshots for an animation)."""

        gate_nodes = np.unique(np.concatenate([f.nodes.ravel() for f, _ in self.gates]))
        start, step = time.perf_counter(), 0
        while self.time < t_end and not np.all(self.fill_factor >= 1.0):
            reaction, iterations = self.solve_pressure()
            self.history.append((self.time, self.melt_volume(), float(self.p[gate_nodes].mean()), iterations,
                                 int(np.sum(self.fill_factor >= 1.0))))
            receiving = reaction > 0
            if not np.any(receiving):
                raise RuntimeError("Ningun nodo recibe fundido: la cavidad esta cerrada sin venteo o la entrada no tiene caudal.")
            remaining = (1.0 - self.fill_factor[receiving]) * self.volume[receiving] / reaction[receiving]
            dt = min(float(remaining.min()), t_end - self.time)
            f = self.fill_factor[receiving] + dt * reaction[receiving] / self.volume[receiving]
            self.fill_factor[receiving] = np.where(f > 1.0 - 1e-12, 1.0, f)           # the node that set dt is full, to round-off
            self.time += dt
            step += 1
            new = (self.fill_factor >= 1.0) & np.isnan(self.fill_time)
            self.fill_time[new] = self.time
            if on_step is not None:
                on_step(self)
            if progress and step % every == 0:
                print(f"   llenado: paso {step}, t = {self.time:.4g} s, {100 * np.mean(self.fill_factor >= 1):.1f} % de los nodos llenos, p de entrada "
                      f"{self.history[-1][2] / 1e6:.4g} MPa, {iterations} it. de Newton ({time.perf_counter() - start:.0f} s)", flush=True)
        return self.history
