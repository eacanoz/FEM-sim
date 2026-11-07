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
# Temperature (setThermal): each node carries a profile T(z) in finite-volume cells across the half gap (symmetric at the midplane,
# the mold wall at z = b by a heat transfer coefficient or a fixed temperature). Per step, split:
#   - in-plane convection, implicit upwind per cell, conservative: the flow between two nodes is that of the FE fluxes, K_ij (p_i - p_j),
#     shared among the cells by the velocity profile u(z) of each element; the gate brings melt at T_melt. The melt of a node fills its
#     cells evenly (f V dz_k), so a cell that receives more than its share passes the excess to the cells that receive less, at its own
#     temperature (the vertical flow): at the front this is the fountain flow, the hot core melt that reaches the wall. Energy is
#     conserved exactly by the convection;
#   - conduction across the gap with viscous heating eta gamma_dot^2 = z G gamma_dot, implicit (a tridiagonal system per node).
# The fluidity then integrates z^2 / eta(gamma_dot, T(z)) cell by cell (two Gauss points per cell); cells colder than the no-flow
# temperature are frozen (no flow, no heating): the frozen layer narrows the channel. The shear rate per point by Newton in a jax kernel.
#
# Linear elements (T3, Q4). SI units: gap [m], pressure [Pa], flow rate [m3/s]; temperature in the units of the viscosity law (°C).
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


def _thermal_kernel(law, edges, T_noflow):

    """A jitted f(b, G, Tc, guess) -> S, dS/dG, the flow of each cell (integral of u over it), the heating of each cell (mean eta
    gamma_dot^2), the shear rates, for elements of half gap b, mean |grad p| G and cell temperatures Tc (nel, ncells), from the shear rates
    `guess` (nel, ncells, 2) where positive (the previous call: Newton then takes a few steps). edges: the cell edges on (0, 1)."""

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    x, w = np.polynomial.legendre.leggauss(2)
    lo, d = edges[:-1], np.diff(edges)
    zq = lo[:, None] + d[:, None] * 0.5 * (x + 1.0)                       # (ncells, 2) on (0, 1)
    wq = d[:, None] * 0.5 * w
    frozen_below = -np.inf if T_noflow is None else float(T_noflow)

    def kernel(b, G, Tc, guess):
        z = b[:, None, None] * zq
        wz = b[:, None, None] * wq
        T = jnp.broadcast_to(Tc[:, :, None], z.shape)
        tau = z * G[:, None, None]
        melt = T >= frozen_below
        Ts = jnp.where(melt, T, jnp.maximum(frozen_below, -1e300))           # a frozen cell: its viscosity at the no-flow temperature
        tau_s = jnp.maximum(tau, 1e-300)

        def stress(g):
            return g * law(g ** 2, Ts)

        g0 = jnp.where(guess > 0, guess, tau_s / law(jnp.zeros_like(tau), Ts))      # the shear rates of the previous call, if any

        def newton(state):
            g, _, i = state
            s, ds = jax.jvp(stress, (g,), (g,))                                 # ds = g dtau/dg = dtau/dlog g
            step = jnp.where(tau > 0, jnp.log(s / tau_s) * s / ds, 0.0)
            return g * jnp.exp(-jnp.clip(step, -5.0, 5.0)), jnp.max(jnp.abs(step)), i + 1

        g, _, _ = jax.lax.while_loop(lambda s: (s[1] > 1e-13) & (s[2] < 60), newton, (g0, jnp.inf, 0))
        g = jnp.where((tau > 0) & melt, g, 0.0)
        eta, deta = jax.jvp(lambda gg: law(gg ** 2, Ts), (g,), (jnp.ones_like(g),))
        slope = eta + g * deta                                                   # dtau / dgamma_dot
        S = 2.0 * jnp.sum(jnp.where(melt, wz * z ** 2 / eta, 0.0), axis=(1, 2))
        dq = 2.0 * jnp.sum(jnp.where(melt, wz * z ** 2 / slope, 0.0), axis=(1, 2))
        dS = jnp.where(G > 1e-300, (dq - S) / jnp.maximum(G, 1e-300), 0.0)
        through = jnp.sum(wz * g, axis=2)                                         # integral of gamma_dot over each cell
        above = jnp.cumsum(through[:, ::-1], axis=1)[:, ::-1] - through           # u at the top of each cell (u = 0 at the wall)
        flow = b[:, None] * d * above + jnp.sum(wz * (z - b[:, None, None] * lo[:, None]) * g, axis=2)
        heat = jnp.sum(wz * tau * g, axis=2) / (b[:, None] * d)
        return S, dS, flow, heat, g

    return jax.jit(kernel)


def _tridiagonal(lower, diag, upper, rhs):

    """Thomas algorithm along the last axis, vectorized over the others."""

    n = diag.shape[-1]
    c, d = np.empty_like(diag), np.empty_like(rhs)
    c[..., 0] = upper[..., 0] / diag[..., 0]
    d[..., 0] = rhs[..., 0] / diag[..., 0]
    for k in range(1, n):
        m = diag[..., k] - lower[..., k] * c[..., k - 1]
        c[..., k] = upper[..., k] / m if k < n - 1 else 0.0
        d[..., k] = (rhs[..., k] - lower[..., k] * d[..., k - 1]) / m
    x = np.empty_like(rhs)
    x[..., -1] = d[..., -1]
    for k in range(n - 2, -1, -1):
        x[..., k] = d[..., k] - c[..., k] * x[..., k + 1]
    return x


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
        self.law = viscosity
        self.thermal = None
        self.short_shot = None                                             # with temperature: when the flow froze before filling

    def setThermal(self, T_melt, T_mold, rho, cp, k, htc=None, T_noflow=None, cells=20, viscous_heating=True, edges=None):

        """Temperature across the gap: melt injected at T_melt, mold wall at T_mold through the heat transfer coefficient htc [W/m2K]
        (None: the wall at T_mold; 0: adiabatic). rho [kg/m3], cp [J/kgK], k [W/mK] of the melt, constant. T_noflow: cells colder than it
        are frozen. cells: uniform cells across the half gap, or their edges on (0, 1) (refined towards the wall at 1). The viscosity law
        must then take the temperature: a Rheology law, or a callable eta(gamma_dot^2, T) written with jax.numpy."""

        from Source.Physics.Rheology import ViscosityLaw
        law = self.law
        if isinstance(law, ViscosityLaw):
            fn = lambda g2, T: law(g2, T)
        else:
            fn = law
        edges = np.linspace(0.0, 1.0, cells + 1) if edges is None else np.asarray(edges, dtype=float)
        disc = self.disc
        area = _scatter_vector(disc, np.einsum('eg,gi->ei', disc.detJw, disc.N), disc.conn)
        weights = np.einsum('eg,gi->ei', disc.detJw * self.h, disc.N)                      # integral N_i h over each element
        self.thermal = dict(T_melt=float(T_melt), T_mold=float(T_mold), rhocp=float(rho) * float(cp), k=float(k),
                            htc=None if htc is None else float(htc), T_noflow=T_noflow, heating=bool(viscous_heating),
                            edges=edges, dz=np.diff(edges), kernel=_thermal_kernel(fn, edges, T_noflow),
                            b_node=self.volume / (2.0 * area), weights=weights)
        self.T = np.full((len(self.xy), len(edges) - 1), float(T_melt))
        return self

    def frozen_fraction(self):

        """Fraction of the gap below the no-flow temperature at each node."""

        th = self.thermal
        if th is None or th['T_noflow'] is None:
            return np.zeros(len(self.xy))
        return (self.T < th['T_noflow']) @ th['dz']

    def mean_temperature(self):

        """Mean temperature across the gap at each node."""

        return self.T @ self.thermal['dz']

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

        w = self.disc.detJw
        W = w.sum(axis=1)
        if self.thermal is not None:
            G_e = np.sum(w * G, axis=1) / W
            Tc = self.T[self.disc.conn].mean(axis=1)
            guess = self.thermal.get('guess')
            if guess is None:
                guess = np.zeros(Tc.shape + (2,))
            S, dS_e, flow, heat, g = (np.asarray(a) for a in self.thermal['kernel'](0.5 * self.h.mean(axis=1), G_e, Tc, guess))
            self.thermal['guess'] = g
            self._layers = (flow, heat, G_e)
            Sg = np.broadcast_to(S[:, None], w.shape)
            return Sg, (dS_e / W)[:, None] * w if derivative else None
        tables, index = self._tables()
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

        F = np.zeros(len(self.xy))
        for facets, Q in self.gates:
            F += _scatter_vector(disc, ops.facet_source(facets, Q / facets.detJw.sum()), facets.nodes)
        filled = self.fill_factor >= 1.0
        active = np.any(filled[disc.conn], axis=1)                              # elements that carry melt
        if self.thermal is not None and self.thermal['T_noflow'] is not None:
            # an element frozen through the whole gap carries nothing: filled nodes surrounded by such elements drop out of the unknowns
            active &= np.any(self.T[disc.conn].mean(axis=1) >= self.thermal['T_noflow'], axis=1)
            # and so do pockets of melt that the frozen elements cut off from the front (no node at p = 0): their pressure is undefined
            from scipy.sparse.csgraph import connected_components
            nn = disc.conn.shape[1]
            graph = _scatter(disc, np.broadcast_to(active[:, None, None] * 1.0, (len(active), nn, nn)))
            graph.eliminate_zeros()                                              # csgraph takes stored zeros as edges
            _, label = connected_components(graph, directed=False)
            touched = _scatter_vector(disc, np.repeat(active[:, None], disc.conn.shape[1], axis=1).astype(float), disc.conn) > 0
            vented = np.unique(label[touched & ~filled])
            filled = filled & touched & np.isin(label, vented)
            if not np.any(filled & (F > 0)):                                   # the gate is cut off: a short shot
                return np.zeros(len(self.xy)), 0
        free = np.flatnonzero(filled)
        mask = active[:, None].astype(float)

        def stiffness(S):
            return _scatter(disc, ops.diffusion(disc, S * mask))

        p = np.where(filled, self.p, 0.0)
        B = disc.B
        if not np.any(p[free]):
            if self.thermal is None:
                S = self.fluidity(0.5 * self.h, np.full_like(self.h, 1e6))
            else:
                S, _ = self._element_fluidity(np.full_like(self.h, 1e6))
            for _ in range(30):                                                 # damped Picard: Newton alone does not start from rest
                A = stiffness(S)
                p[free] = spla.spsolve(A[free][:, free].tocsc(), F[free])
                S_new, _ = self._element_fluidity(np.linalg.norm(disc.gradient(p), axis=-1))
                with np.errstate(divide='ignore', invalid='ignore'):
                    change = np.nanmax(np.abs(np.log(S_new[active, 0] / S[active, 0])))
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
            if not np.all(np.isfinite(dp)):
                raise RuntimeError(f"Tiro corto a t = {self.time:.4g} s: la presion no tiene solucion (el fundido se congelo y cerro el paso).")
            p = p + dp
            if np.max(np.abs(dp)) <= tol * max(np.max(np.abs(p)), 1e-300):
                break
        self.p = p
        S, _ = self._element_fluidity(np.linalg.norm(disc.gradient(p), axis=-1))
        self.S = S * mask
        self._step = (ops.diffusion(disc, S * mask), F, active)
        reaction = F - stiffness(S) @ p                                          # flow rate into each node not filled
        reaction[self.fill_factor >= 1.0] = 0.0
        return np.maximum(reaction, 0.0), k

    # ----- the temperature -----

    def _advance_temperature(self, dt, f_old):

        """One step of the temperature over dt, after the fill factors of the step: convection in the plane (implicit upwind per cell), then
        conduction across the gap with the viscous heating (implicit)."""

        th, disc = self.thermal, self.disc
        Ke, F, active = self._step
        flow, heat, _ = self._layers
        conn, n, dz = disc.conn, len(self.xy), th['dz']
        nc = len(dz)
        total = flow.sum(axis=1, keepdims=True)
        phi = np.where(total > 0, flow / np.where(total > 0, total, 1.0), dz)              # share of the flow of each cell, per element
        phi = phi * active[:, None]
        w = th['weights'] * active[:, None]                                                 # (nel, nn)
        wsum = _scatter_vector(disc, w, conn)
        node_phi = np.zeros((n, nc))
        np.add.at(node_phi, conn.ravel(), (w[:, :, None] * phi[:, None, :]).reshape(-1, nc))
        node_phi = np.where(wsum[:, None] > 0, node_phi / np.where(wsum > 0, wsum, 1.0)[:, None], dz)
        # flow from node j into node i through each element: K_ij (p_i - p_j)
        pe = self.p[conn]
        Wf = Ke * (pe[:, :, None] - pe[:, None, :])
        layer = np.arange(nc)
        rows = (conn[:, :, None, None] + n * layer).repeat(conn.shape[1], axis=2)
        cols = (conn[:, None, :, None] + n * layer).repeat(conn.shape[1], axis=1)
        data = Wf[:, :, :, None] * phi[:, None, None, :]
        M = sp.csr_matrix((data.ravel(), (rows.ravel(), cols.ravel())), shape=(n * nc, n * nc))
        M.sum_duplicates()
        M.data = np.maximum(M.data, 0.0)
        M.setdiag(0.0)
        M.eliminate_zeros()
        gate = (F[:, None] * node_phi).T.ravel()                                            # melt entering at the gates, per cell
        # cell order k * n + i. Per cell, with c the melt volume, the vertical excess m = in - out - dc/dt leaves at the cell temperature
        # and the cells short of melt (m < 0) receive the mixing-cup temperature of the excess of their node:
        #     c_old (T - T_old) / dt + sum_in q (T - T_up) + m- (T - T_mix) = 0
        content = ((f_old * self.volume)[None, :] * dz[:, None]).ravel() / dt
        growth = (((self.fill_factor - f_old) * self.volume)[None, :] * dz[:, None]).ravel() / dt
        inflow = np.asarray(M.sum(axis=1)).ravel() + gate
        excess = inflow - np.asarray(M.sum(axis=0)).ravel() - growth
        through = (inflow.reshape(nc, n).sum(axis=0) + content.reshape(nc, n).sum(axis=0))
        excess = np.where(np.abs(excess) > 1e-12 * np.tile(through, nc), excess, 0.0)
        give, take = np.maximum(excess, 0.0).reshape(nc, n), np.maximum(-excess, 0.0).reshape(nc, n)
        given = give.sum(axis=0)
        take = np.where(given > 0, take, 0.0)
        mixer = []
        nodes = np.flatnonzero(given > 0)
        if len(nodes):
            share = give[:, nodes] / given[nodes]                                            # (nc, m)
            r = (nodes[None, None, :] + n * layer[:, None, None]) + 0 * layer[None, :, None]
            c = (nodes[None, None, :] + n * layer[None, :, None]) + 0 * layer[:, None, None]
            v = take[:, nodes][:, None, :] * share[None, :, :]
            mixer = [sp.csr_matrix((v.ravel(), (r.ravel(), c.ravel())), shape=(n * nc, n * nc))]
        diag = content + inflow + take.ravel()
        old = self.T.T.ravel()
        idle = diag <= 0
        diag = np.where(idle, 1.0, diag)
        rhs = np.where(idle, old, content * old + gate * th['T_melt'])
        A = sp.diags(diag) - M
        for X in mixer:
            A = A - X
        self.T = spla.spsolve(A.tocsc(), rhs).reshape(nc, n).T.copy()
        # conduction across the gap, with the heating, where there is melt
        melt = self.fill_factor > 0
        b = th['b_node'][melt]
        h = b[:, None] * dz
        k, rc = th['k'], th['rhocp']
        a = k / (0.5 * (h[:, 1:] + h[:, :-1]))                                              # between cell centres
        wall = 1.0 / (0.5 * h[:, -1] / k + (0.0 if th['htc'] is None else (np.inf if th['htc'] == 0 else 1.0 / th['htc'])))
        lower = np.zeros_like(h); upper = np.zeros_like(h)
        lower[:, 1:] = -a; upper[:, :-1] = -a
        diagz = rc * h / dt
        diagz[:, 1:] += a; diagz[:, :-1] += a
        diagz[:, -1] += wall
        rhs = rc * h / dt * self.T[melt]
        rhs[:, -1] += wall * th['T_mold']
        if th['heating']:
            # the heat of each element goes to the melt it holds, sum_i w_ei f_i, so that it is all kept
            held = np.sum(w * self.fill_factor[conn], axis=1)
            scale = np.where(held > 0, w.sum(axis=1) / np.where(held > 0, held, 1.0), 0.0)
            node_heat = np.zeros((n, nc))
            np.add.at(node_heat, conn.ravel(), ((w * scale[:, None])[:, :, None] * heat[:, None, :]).reshape(-1, nc))
            node_heat = node_heat / self.volume[:, None]
            rhs += h * node_heat[melt]
        self.T[melt] = _tridiagonal(lower, diagz, upper, rhs)

    # ----- the filling -----

    def melt_volume(self):
        return float(self.fill_factor @ self.volume)

    def fill(self, t_end=np.inf, progress=True, every=50, on_step=None):

        """Fills node by node until t_end or until every node is filled. Keeps self.history of (t, melt volume, mean gate pressure, Newton
        iterations, filled nodes) and returns it; self.fill_time holds when each node filled; with setThermal, self.T the temperature of each
        cell across the gap at each node. on_step(self) is called after each step (to take
        snapshots for an animation)."""

        gate_nodes = np.unique(np.concatenate([f.nodes.ravel() for f, _ in self.gates]))
        start, step = time.perf_counter(), 0
        while self.time < t_end and not np.all(self.fill_factor >= 1.0):
            reaction, iterations = self.solve_pressure()
            self.history.append((self.time, self.melt_volume(), float(self.p[gate_nodes].mean()), iterations,
                                 int(np.sum(self.fill_factor >= 1.0))))
            receiving = reaction > 0
            if not np.any(receiving):
                if self.thermal is not None:
                    self.short_shot = self.time
                    print(f"   tiro corto a t = {self.time:.4g} s: ningun nodo recibe fundido (la capa congelada cerro el paso)", flush=True)
                    break
                raise RuntimeError("Ningun nodo recibe fundido: la cavidad esta cerrada sin venteo o la entrada no tiene caudal.")
            remaining = (1.0 - self.fill_factor[receiving]) * self.volume[receiving] / reaction[receiving]
            dt = min(float(remaining.min()), t_end - self.time)
            f_old = self.fill_factor.copy()
            f = self.fill_factor[receiving] + dt * reaction[receiving] / self.volume[receiving]
            self.fill_factor[receiving] = np.where(f > 1.0 - 1e-12, 1.0, f)           # the node that set dt is full, to round-off
            if self.thermal is not None and dt > 0:
                self._advance_temperature(dt, f_old)
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
