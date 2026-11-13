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
# Packing (setDensity, pack): after the filling the melt is compressible, density rho(p, T) (the Tait law), and each node keeps its
# mass m_i. Per step, implicit: V_i rho_i(p) - m_i_old + dt (K_rho p)_i = 0, with K_rho the stiffness of rho_e S_e (mass fluxes) and
# rho_i the mean over the cells of the gap; the gate held at the packing pressure (its reaction is the mass that enters), then closed
# for the cooling. The masses are kept exactly. Each node has its own equation through the compressibility, so frozen nodes and
# pockets keep a pressure (that of their mass in their volume); the pressure does not drop below ambient (the part leaves the wall:
# the node keeps its mass). When a node freezes through the gap its specific volume is recorded: the volumetric shrinkage is
# 1 - v(ambient, T_room) / v_frozen.
#
# With setDensity the filling is compressible too, by the same step: the gate is then a flow rate (the mass Q rho(p, T_melt) of the melt
# at the gate pressure), and a node is full when its mass reaches V rho(ambient, T): the nodes short of it stay at ambient pressure and
# receive melt (the front, with the fill factor m / (V rho(ambient, T))). Each step lasts until the next node fills, as in the
# incompressible filling, with the flow rates of the previous step. With temperature, the compression heats the melt, beta T Dp/Dt
# (beta = -(1/rho) d(rho)/dT from the state law): dp/dt at the node, and u . grad p from the flow that reaches each cell.
#
# Crystallization (setCrystallization): each cell carries the Nakamura time tau = integral K(T) dt, transported with the melt like the
# temperature (fresh melt at tau = 0); X = 1 - exp(-tau^n). The latent heat rho L dX/dt enters the conduction across the gap (a few
# iterations with the rate at the mean temperature of the step); cells with X >= X_noflow stop flowing; with a pvT law the density
# follows X, (1 - X) v_melt + X v_solid, instead of the transition temperature, and the shrinkage uses the crystallinity reached.
#
# The planform may also be a surface in space, the midplane of a folded part (plates joined at edges, T joints included): T3/Q4 with three
# coordinates (Mesh.from_gmsh_model(surface=True)); the Discretization then gives tangential gradients and the measure of the surface, and
# nothing else changes (Test/test_hele_shaw_surface.py, Example_Filling_Box_3D.py).
#
# Or the two skins of a solid (dual domain, from_dual_domain with Source/Pre_processing/DualDomain.py, a STEP part): each skin element
# carries half of its wall (share = 0.5 scales the measure), and links join every filled skin node i to the opposite point of the other
# skin, a flow c (p_i - sum w_k p_k) with the weights of the filled nodes of the opposite triangle (an outlet c p_i into an empty one):
# the melt crosses the wall freely and the front advances on both skins together; the flow of the links enters the convection of the
# temperature. Incompressible filling only (Test/test_dual_domain.py, Example_Filling_Box_STEP.py).
#
# A feed system (addRunners, Source/Physics/Runners.py): sprue, runners and gates as circular tubes; one pressure for the cavity and the
# network, Newton with the tangent of each tube and a line search; the network fills by the same control volumes and its melt reaches
# the cavity through the gates at the injection temperature (a hot runner). Incompressible filling only (Test/test_runners.py).
# Linear elements (T3, Q4). SI units: gap [m], pressure [Pa], flow rate [m3/s]; temperature in the units of the viscosity law (°C).
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from Source.Simulation.Discretization import Discretization
from Source.Physics.Physics import KELVIN_OFFSET
import Source.Simulation.Operators as ops


def _scatter(disc, Ke):

    """Assembles element matrices Ke (nel, nn, nn) into a CSR (n, n). The pattern and where each element entry lands in it are built once
    per discretization; then the assembly is a bincount into the data (the matrix keeps explicit zeros of the pattern)."""

    pattern = getattr(disc, '_hele_shaw_pattern', None)
    if pattern is None:
        conn = disc.conn
        n, nn = disc.n_nodes, conn.shape[1]
        rows = np.repeat(conn, nn, axis=1).ravel()
        cols = np.tile(conn, (1, nn)).ravel()
        A = sp.csr_matrix((np.arange(1, len(rows) + 1, dtype=float), (rows, cols)), shape=(n, n))     # duplicates summed: only the pattern
        A.sort_indices()
        # the slot of each element entry in the CSR data: the rank of its (row, col) key, row-major as the sorted CSR
        order = np.lexsort((cols, rows))
        sorted_keys = rows[order] * n + cols[order]
        unique_keys = np.unique(sorted_keys)
        slot = np.empty(len(rows), dtype=np.int64)
        slot[order] = np.searchsorted(unique_keys, sorted_keys)
        pattern = (A.indptr.copy(), A.indices.copy(), slot, len(unique_keys), n)
        disc._hele_shaw_pattern = pattern
    indptr, indices, slot, nnz, n = pattern
    data = np.bincount(slot, weights=np.ascontiguousarray(Ke, dtype=float).ravel(), minlength=nnz)
    return sp.csr_matrix((data, indices.copy(), indptr.copy()), shape=(n, n))      # copies: eliminate_zeros works in place


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

    def kernel(b, G, Tc, guess, solid):
        z = b[:, None, None] * zq
        wz = b[:, None, None] * wq
        T = jnp.broadcast_to(Tc[:, :, None], z.shape)
        tau = z * G[:, None, None]
        melt = (T >= frozen_below) & ~solid[:, :, None]                         # solid: crystallized beyond X_noflow
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


def _convection_solve(A, rhs, rtol=1e-13):

    """The implicit upwind convection of the cells, A x = rhs (one or more columns): A is diagonally dominant (an M-matrix, the diagonal
    the content plus the inflow), so BiCGSTAB with the diagonal converges in a few iterations, several times faster than SuperLU (whose
    fill-in grows with the front); SuperLU if it does not converge."""

    A = A.tocsr()
    d = A.diagonal()
    # rows scaled by the diagonal: the residual is then in the units of the unknowns in every cell (a cell nearly empty of melt has a
    # tiny row, which a tolerance on the unscaled residual leaves unresolved)
    As = sp.diags(1.0 / d) @ A
    columns = rhs[:, None] if rhs.ndim == 1 else rhs
    out = np.empty_like(columns, dtype=float)
    for j in range(columns.shape[1]):
        b = columns[:, j]
        bs = b / d
        x, info = spla.bicgstab(As, bs, x0=bs, rtol=rtol, atol=0.0, maxiter=500)
        if info != 0 or not np.all(np.isfinite(x)):
            x = spla.spsolve(A.tocsc(), b)
        out[:, j] = x
    return out[:, 0] if rhs.ndim == 1 else out


def _kth(values, batch):

    """The k-th smallest of values, k a fraction batch of their number (at least 1: batch 0 gives the smallest)."""

    k = max(1, int(np.ceil(batch * len(values))))
    return float(np.partition(values, k - 1)[k - 1])


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

    def __init__(self, mesh, gap, viscosity, T=None, layers=16, share=1.0):

        """mesh: the planform (2D, linear elements), or the midplane of a folded part as a surface in space (T3/Q4 with three coordinates:
        Mesh.from_gmsh_model(surface=True); plates joined at edges, T joints included). gap: full thickness h [m], a number,
        f(x, y[, z]) or an array (nel,) per element. viscosity: eta(gamma_dot^2) or a Rheology law at the fixed temperature T.
        share: the fraction of its gap that each element carries (0.5 on the two skins of a dual domain: from_dual_domain)."""

        self.mesh = mesh
        self.disc = Discretization(mesh)
        if self.disc.element.linear_subspace:
            raise ValueError("Hele-Shaw usa elementos lineales (T3, Q4): los volumenes de control necesitan pesos positivos.")
        if share != 1.0:
            self.disc.detJw = self.disc.detJw * float(share)            # volume, flow and heat of each element: its share of the gap
        self.share = float(share)
        self.connectors, self.coupling = None, 0.0
        self.runners = None
        self.xy = self.disc.node_xy                                     # (x, y) of a planform, (x, y, z) of a surface in space
        self.fluidity = GapFluidity(viscosity, T, layers)
        gp = self.disc.x_gp
        h = gap(*[gp[..., a] for a in range(gp.shape[-1])]) if callable(gap) else np.asarray(gap, dtype=float)
        if np.ndim(h) == 1 and len(h) == len(self.disc.conn):
            h = h[:, None]                                              # per element
        self.h = np.broadcast_to(np.asarray(h, dtype=float), self.disc.detJw.shape)
        self.volume = _scatter_vector(self.disc, np.einsum('eg,gi->ei', self.disc.detJw * self.h, self.disc.N), self.disc.conn)
        self.fill_factor = np.zeros(len(self.xy))
        self.fill_time = np.full(len(self.xy), np.nan)                  # when each node filled: the isochrones of the filling
        self.p = np.zeros(len(self.xy))
        self.gates = []
        self.time, self.history = 0.0, []
        self.law = viscosity
        self.T_fixed = T
        self.density = None
        self.thermal = None
        self.short_shot = None                                             # with temperature: when the flow froze before filling

    def addRunners(self, system):

        """Feeds the cavity through a network of runners and gates (Source/Physics/Runners.py, a RunnerSystem): the flow rate enters at its
        inlet, the network fills first and the melt reaches the cavity nodes of its gates. Replaces addGate. The network is at the injection
        temperature (T_melt of setThermal, called before this, or the fixed T): call setThermal first. Incompressible filling only."""

        from Source.Physics.Runners import TubeFluidity
        if self.gates:
            raise ValueError("Con una red de alimentacion la entrada es la de la red: sin addGate.")
        T = self.thermal['T_melt'] if self.thermal is not None else self.T_fixed
        from Source.Physics.Rheology import ViscosityLaw
        law = self.law
        if self.thermal is not None and not isinstance(law, ViscosityLaw):
            fn = law
            law = lambda g2: np.asarray(fn(g2, T), dtype=float)           # a callable eta(gamma_dot^2, T) at the injection temperature
        system.bind(self.xy, TubeFluidity(law, T))
        self.runners = system
        return self

    @classmethod
    def from_dual_domain(cls, dual, viscosity, T=None, layers=16, coupling=20.0):

        """The filling of a thin-walled solid on its two skins (Source/Pre_processing/DualDomain.py): each skin triangle carries half of
        its wall thickness (from its mold wall to the midplane), and each link of a skin node to the opposite point of the other skin
        carries the flow c (p_i - sum w_k p_k), c = `coupling` times the largest in-plane diagonal of its nodes, so that the melt crosses
        the wall freely and the front advances on both skins together. Gates on the skin mesh (dual.mesh)."""

        fill = cls(dual.mesh, dual.thickness, viscosity, T, layers, share=0.5)
        V = dual.connectors.tocsr()
        fill.connectors = V
        fill._link_graph = (abs(V).T @ abs(V)).tocsr()                  # node pairs that a link joins (and the diagonal)
        fill._link_graph.setdiag(0.0)
        fill._link_graph.eliminate_zeros()
        fill.coupling = float(coupling)
        fill.dual = dual
        return fill

    def _connector_laplacian(self, D):

        """The stiffness of the links, V^T diag(c) V (V the rows e_i - sum w_k e_k), c = coupling times the largest in-plane diagonal D
        of the stiffness among the nodes of the link (the conductance of a node to its neighbours): it follows the fluidity, also its frozen
        layer, without a derivative of its own. Rows add up to zero: mass is kept."""

        V = self._links
        A = abs(V).tocsr()
        largest = np.zeros(V.shape[0])
        np.maximum.at(largest, np.repeat(np.arange(V.shape[0]), np.diff(A.indptr)), D[A.indices])
        c = self.coupling * largest
        self._link_conductance = c
        return (V.T @ sp.diags(c) @ V).tocsr()

    def _set_links(self, filled):

        """The links of this step, from the filled nodes only (the opposite skin gives the links of the other way):
            - to a triangle with filled nodes: the opposite point by those nodes (weights renormalized), all the link among filled nodes;
            - to a triangle with none: all its weights, an outlet c p_i into its empty nodes (how the melt crosses to an empty skin).
        A link from an empty node, or one into the empty nodes of a partly filled triangle, could take melt out of empty nodes: the
        penalty is not an M-matrix and leaves filled nodes near the front slightly below zero (a negative inflow, 12 % of the mass created
        in a strip with coupling 100)."""

        d = self.dual
        tf = filled[d.link_nodes]
        some = np.any(tf, axis=1)
        use = filled[d.link_source]
        w = np.where(some[:, None], d.link_weights * tf, d.link_weights)
        total = w.sum(axis=1)
        use &= total > 1e-12
        w = w[use] / total[use, None]
        src, tri = d.link_source[use], d.link_nodes[use]
        m = len(src)
        rows = np.repeat(np.arange(m), 4)
        cols = np.column_stack([src, tri]).ravel()
        vals = np.column_stack([np.ones(m), -w]).ravel()
        self._links = sp.csr_matrix((vals, (rows, cols)), shape=(m, len(self.xy)))
        self._link_parts = (src, tri, w)

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
        self.thermal = dict(T_melt=float(T_melt), T_mold=float(T_mold), rhocp=float(rho) * float(cp), rho=float(rho), k=float(k),
                            htc=None if htc is None else float(htc), T_noflow=T_noflow, heating=bool(viscous_heating),
                            edges=edges, dz=np.diff(edges), kernel=_thermal_kernel(fn, edges, T_noflow),
                            b_node=self.volume / (2.0 * area), weights=weights)
        self.T = np.full((len(self.xy), len(edges) - 1), float(T_melt))
        return self

    def setDensity(self, density, ambient=1e5, compression_heating=True, crystallinity=False):

        """Compressible melt for the filling and the packing: density(p_abs, T) [kg/m3] written with jax.numpy (TaitPVT(...).density),
        p_abs = the pressure of the model + ambient [Pa]. compression_heating: beta T Dp/Dt in the energy (with setThermal); beta from
        the law, or for a TaitPVT density its own expansion without the jump of the transition. crystallinity: the law is
        density(p_abs, T, X) with the relative crystallinity X of setCrystallization (TaitPVT(...).density_crystallinity)."""

        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        dz = jnp.asarray(self.thermal['dz'] if self.thermal is not None else np.ones(1))

        law = density if crystallinity else (lambda p, T, X: density(p, T))

        def mean(p, T, X):
            return jnp.sum(dz * law(p[:, None] + ambient, T, X), axis=1)

        own = None if crystallinity else getattr(getattr(density, '__self__', None), 'expansion', None)   # TaitPVT(...).density

        def expansion(p, T, X):                                                    # beta = -d log(rho) / dT per cell, at fixed X
            P = jnp.broadcast_to(p[:, None] + ambient, T.shape)
            if own is not None:
                return own(P, T)
            return -jax.jvp(lambda TT: jnp.log(law(P, TT, X)), (T,), (jnp.ones_like(T),))[1]

        def gate(p, T):
            return jax.jvp(lambda q: law(q + ambient, T, jnp.zeros_like(T)), (p,), (jnp.ones_like(p),))

        self.density = dict(law=law, ambient=float(ambient), heating=bool(compression_heating),
                            nodal=jax.jit(lambda p, T, X: jax.jvp(lambda q: mean(q, T, X), (p,), (jnp.ones_like(p),))),
                            beta=jax.jit(expansion), gate=jax.jit(gate))
        return self

    def setCrystallization(self, kinetics, latent_heat, X_noflow=None, pvt=None, ambient=1e5, compression_heating=True):

        """Quiescent crystallization (needs setThermal): kinetics a Crystallization.NakamuraHoffman; latent_heat [J/kg] released at X = 1
        (the latent heat of the crystal times its maximum absolute crystallinity times the polymer fraction of a filled grade); cells with
        X >= X_noflow stop flowing (None: only T_noflow); pvt: a TaitPVT whose density then follows X (setDensity is called with it)."""

        if self.thermal is None:
            raise ValueError("La cristalizacion necesita setThermal.")
        self.crystal = dict(kinetics=kinetics, latent=float(latent_heat), X_noflow=None if X_noflow is None else float(X_noflow))
        self.tau = np.zeros_like(self.T)
        if pvt is not None:
            self.setDensity(pvt.density_crystallinity, ambient, compression_heating, crystallinity=True)
        return self

    def crystallinity(self):

        """Relative crystallinity of each cell across the gap at each node (zeros without crystallization)."""

        if getattr(self, 'crystal', None) is None:
            return np.zeros_like(self._temperature_cells())
        return self.crystal['kinetics'].crystallinity(self.tau)

    def _crystallinity_cells(self):
        if getattr(self, 'crystal', None) is None:
            return np.zeros_like(self._temperature_cells())
        ahead, rate = getattr(self, '_ahead', 0.0), getattr(self, '_tau_rate', None)
        tau = self.tau if (rate is None or ahead == 0.0) else self.tau + ahead * rate
        return self.crystal['kinetics'].crystallinity(tau)

    def _freezes(self):
        return self.thermal is not None and (self.thermal['T_noflow'] is not None or
                                             (getattr(self, 'crystal', None) is not None and self.crystal['X_noflow'] is not None))

    def _frozen_cells(self, T, X):
        frozen = np.zeros(T.shape, dtype=bool)
        if self.thermal['T_noflow'] is not None:
            frozen |= T < self.thermal['T_noflow']
        if getattr(self, 'crystal', None) is not None and self.crystal['X_noflow'] is not None:
            frozen |= X >= self.crystal['X_noflow']
        return frozen

    def _element_frozen(self):

        """Frozen cells of each element (nel, ncells): by the mean temperature and crystallinity of its nodes."""

        conn = self.disc.conn
        return self._frozen_cells(self.T[conn].mean(axis=1), self.crystallinity()[conn].mean(axis=1))

    def full_density(self):

        """The density of each node at ambient pressure: a node is full when its mass reaches V times it."""

        return self.nodal_density(np.zeros(len(self.xy)))[0]

    def _temperature_cells(self):
        if self.thermal is not None:
            # inside a compressible step, the temperature predicted for its end (the rate of the last step): the cooling of the step and
            # the melt that makes up for it then come in the same step, instead of a deficit left to the next, much shorter one
            ahead, rate = getattr(self, '_ahead', 0.0), getattr(self, '_T_rate', None)
            return self.T if (rate is None or ahead == 0.0) else self.T + ahead * rate
        return np.full((len(self.xy), 1), float(self.T_fixed if self.T_fixed is not None else 0.0))

    def nodal_density(self, p=None):

        """Mean density across the gap at each node, and its derivative in p."""

        rho, drho = self.density['nodal'](self.p if p is None else p, self._temperature_cells(), self._crystallinity_cells())
        return np.asarray(rho), np.asarray(drho)

    def frozen_fraction(self):

        """Fraction of the gap below the no-flow temperature at each node."""

        if not self._freezes():
            return np.zeros(len(self.xy))
        return self._frozen_cells(self.T, self.crystallinity()) @ self.thermal['dz']

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

        """Nodes where region(x, y[, z]) is True start filled (an initial short shot)."""

        inside = np.asarray(region(*self.xy.T), dtype=bool)
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

    def _element_fluidity(self, G, derivative=False, active=None):

        """S_e(G_e) of each element from the table of its gap, G_e the mean |grad p| of the element; with derivative, dS_e / dG_g at each point.
        active (with temperature): only these elements are evaluated (the others get S = 0 and no flow): the kernel then runs on a subset
        padded to nel / 2^k elements, so that jax compiles at most a few shapes."""

        w = self.disc.detJw
        W = w.sum(axis=1)
        if self.thermal is not None:
            nel = len(w)
            G_e = np.sum(w * G, axis=1) / W
            Tc = self.T[self.disc.conn].mean(axis=1)
            guess = self.thermal.get('guess')
            if guess is None:
                guess = np.zeros(Tc.shape + (2,))
            solid = self._element_frozen() if getattr(self, 'crystal', None) is not None else np.zeros(Tc.shape, dtype=bool)
            b = 0.5 * self.h.mean(axis=1)
            idx = None if active is None else np.flatnonzero(active)
            if idx is not None and len(idx) and len(idx) <= nel // 2:
                size = nel
                while size // 2 >= len(idx) and size > 64:
                    size //= 2
                pick = np.concatenate([idx, np.full(size - len(idx), idx[0])])
                out = [np.asarray(a) for a in self.thermal['kernel'](b[pick], G_e[pick], Tc[pick], guess[pick], solid[pick])]
                k = len(idx)
                S, dS_e = np.zeros(nel), np.zeros(nel)
                flow, heat = np.zeros((nel,) + out[2].shape[1:]), np.zeros((nel,) + out[3].shape[1:])
                g = guess.copy()
                S[idx], dS_e[idx], flow[idx], heat[idx], g[idx] = (a[:k] for a in out)
            else:
                S, dS_e, flow, heat, g = (np.asarray(a) for a in self.thermal['kernel'](b, G_e, Tc, guess, solid))
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
        if self.runners is not None:
            return self._solve_with_runners(tol, max_iterations)

        F = np.zeros(len(self.xy))
        for facets, Q in self.gates:
            F += _scatter_vector(disc, ops.facet_source(facets, Q / facets.detJw.sum()), facets.nodes)
        filled = self.fill_factor >= 1.0
        active = np.any(filled[disc.conn], axis=1)                              # elements that carry melt
        if self._freezes():
            # an element frozen through the whole gap carries nothing: filled nodes surrounded by such elements drop out of the unknowns
            active &= ~np.all(self._element_frozen(), axis=1)
            # and so do pockets of melt that the frozen elements cut off from the front (no node at p = 0): their pressure is undefined
            from scipy.sparse.csgraph import connected_components
            nn = disc.conn.shape[1]
            graph = _scatter(disc, np.broadcast_to(active[:, None, None] * 1.0, (len(active), nn, nn)))
            touched = _scatter_vector(disc, np.repeat(active[:, None], disc.conn.shape[1], axis=1).astype(float), disc.conn) > 0
            if self.connectors is not None:                                      # the skins of a dual domain: joined across the wall
                t = sp.diags(touched.astype(float))
                graph = graph + t @ self._link_graph @ t
            graph.eliminate_zeros()                                              # csgraph takes stored zeros as edges
            _, label = connected_components(graph, directed=False)
            vented = np.unique(label[touched & ~filled])
            filled = filled & touched & np.isin(label, vented)
            if not np.any(filled & (F > 0)):                                   # the gate is cut off: a short shot
                return np.zeros(len(self.xy)), 0
        free = np.flatnonzero(filled)
        mask = active[:, None].astype(float)
        if self.connectors is not None:
            self._set_links(self.fill_factor >= 1.0)

        def stiffness(S):
            K = _scatter(disc, ops.diffusion(disc, S * mask))
            return K if self.connectors is None else K + self._connector_laplacian(K.diagonal())

        p = np.where(filled, self.p, 0.0)
        B = disc.B
        if not np.any(p[free]):
            if self.thermal is None:
                S = self.fluidity(0.5 * self.h, np.full_like(self.h, 1e6))
            else:
                S, _ = self._element_fluidity(np.full_like(self.h, 1e6), active=active)
            for _ in range(30):                                                 # damped Picard: Newton alone does not start from rest
                A = stiffness(S)
                p[free] = spla.spsolve(A[free][:, free].tocsc(), F[free])
                S_new, _ = self._element_fluidity(np.linalg.norm(disc.gradient(p), axis=-1), active=active)
                with np.errstate(divide='ignore', invalid='ignore'):
                    change = np.nanmax(np.abs(np.log(S_new[active, 0] / S[active, 0])))
                S = np.sqrt(S * S_new)
                if change < 1e-3:
                    break
        for k in range(1, max_iterations + 1):
            gp = disc.gradient(p)
            G = np.linalg.norm(gp, axis=-1)
            S, dS = self._element_fluidity(G, derivative=True, active=active)
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
        S, _ = self._element_fluidity(np.linalg.norm(disc.gradient(p), axis=-1), active=active)
        self.S = S * mask
        self._step = (ops.diffusion(disc, S * mask), F, active)
        reaction = F - stiffness(S) @ p                                          # flow rate into each node not filled
        if self.connectors is not None:                                           # the flow of the links, for the temperature
            self._link_flow(p)
        reaction[self.fill_factor >= 1.0] = 0.0
        return np.maximum(reaction, 0.0), k

    def _solve_with_runners(self, tol, max_iterations):

        """solve_pressure with the feed system: one Newton over the filled nodes of the cavity and of the network (cavity first), the
        tubes with their exact tangent. Sets self.p, self.runners.p, the gate inflow of each cavity node (self._step, for the temperature)
        and self._runner_reaction; returns the reactions of the cavity nodes and the Newton iterations."""

        disc, R = self.disc, self.runners
        n, nr = len(self.xy), R.n
        N = n + nr
        filled_c, filled_r = self.fill_factor >= 1.0, R.fill_factor >= 1.0
        active = np.any(filled_c[disc.conn], axis=1)
        if self._freezes():
            active &= ~np.all(self._element_frozen(), axis=1)
        mask = active[:, None].astype(float)
        if self.connectors is not None:
            self._set_links(filled_c)
        F = np.zeros(N)
        F[n + R.inlet[0]] = R.inlet[1]
        free = np.flatnonzero(np.concatenate([filled_c, filled_r]))
        p = np.concatenate([np.where(filled_c, self.p, 0.0), np.where(filled_r, R.p, 0.0)])

        def cavity(S):
            K = _scatter(disc, ops.diffusion(disc, S * mask))
            if self.connectors is not None:
                K = K + self._connector_laplacian(K.diagonal())
            return sp.block_diag((K, sp.csr_matrix((nr, nr))), format='csr')

        def tubes(p):
            return R.assemble(p[:n], p[n:], n, filled_c, filled_r)

        B = disc.B
        if not np.any(p[free]):                                         # from rest: damped Picard on the secant conductances
            if self.thermal is None:
                S = self.fluidity(0.5 * self.h, np.full_like(self.h, 1e6))
            else:
                S, _ = self._element_fluidity(np.full_like(self.h, 1e6), active=active)
            for _ in range(60):
                Kt, _, _ = tubes(p)
                A = cavity(S) + Kt
                p_new = np.zeros(N)
                p_new[free] = spla.spsolve(A[free][:, free].tocsc(), F[free])
                S_new, _ = self._element_fluidity(np.linalg.norm(disc.gradient(p_new[:n]), axis=-1), active=active)
                change = np.max(np.abs(p_new - p)) / max(np.max(np.abs(p_new)), 1e-300)
                p = p_new if not np.any(p) else 0.5 * (p + p_new)
                S = np.sqrt(S * S_new)
                if change < 1e-3:
                    break
        def state(p):
            gp = disc.gradient(p[:n])
            G = np.linalg.norm(gp, axis=-1)
            S, dS = self._element_fluidity(G, derivative=True, active=active)
            Kt, Jt, _ = tubes(p)
            Rv = (cavity(S) + Kt) @ p - F
            return gp, G, S, dS, Kt, Jt, Rv, np.linalg.norm(Rv[free])

        # Newton with backtracking: q ~ dp^(1/n) of a shear-thinning melt in a tube is strongly convex, and from a node just filled (at
        # zero) the full step overshoots by orders of magnitude; halve it until the residual falls (the accepted state is reused)
        current = state(p)
        noise = 1e-12 * np.linalg.norm(F)                                  # a residual at round-off: no backtracking below it
        for k in range(1, max_iterations + 1):
            gp, G, S, dS, Kt, Jt, Rv, norm = current
            if norm <= noise and k > 1:
                break
            a = np.einsum('eg,egia,ega->ei', disc.detJw, B, gp) * active[:, None]
            c = np.einsum('eg,ega,egja->ej', dS / np.maximum(G, 1e-300), gp, B)
            J = cavity(S) + sp.block_diag((_scatter(disc, np.einsum('ei,ej->eij', a, c)), sp.csr_matrix((nr, nr))), format='csr') + Jt
            dp = np.zeros(N)
            dp[free] = spla.spsolve(J[free][:, free].tocsc(), -Rv[free])
            if not np.all(np.isfinite(dp)):
                raise RuntimeError(f"Tiro corto a t = {self.time:.4g} s: la presion no tiene solucion (el fundido se congelo y cerro el paso).")
            alpha = 1.0
            for _ in range(40):
                trial = state(p + alpha * dp)
                if trial[-1] <= max(norm, noise) or alpha < 1e-9:
                    break
                alpha *= 0.5
            p, current = p + alpha * dp, trial
            if alpha == 1.0 and np.max(np.abs(dp)) <= tol * max(np.max(np.abs(p)), 1e-300):
                break
        self.p, R.p = p[:n].copy(), p[n:].copy()
        S, _ = self._element_fluidity(np.linalg.norm(disc.gradient(self.p), axis=-1), active=active)
        self.S = S * mask
        Kt, _, q = tubes(p)
        reaction = F - (cavity(S) + Kt) @ p
        tube_in = -(Kt @ p)[:n]                                         # what the gates bring to each cavity node
        self._step = (ops.diffusion(disc, S * mask), np.maximum(tube_in, 0.0), active)
        self._tube_flow = q
        if self.connectors is not None:
            self._link_flow(self.p)
        r_c, r_r = reaction[:n], reaction[n:]
        r_c[filled_c] = 0.0
        r_r[filled_r] = 0.0
        self._runner_reaction = np.maximum(r_r, 0.0)
        return np.maximum(r_c, 0.0), k

    def _link_flow(self, p):
        q = self._link_conductance * (self._links @ p)                        # from the source node to the opposite point
        src, tri, w = self._link_parts
        out, back = q > 0, q < 0
        rows = np.concatenate([tri[out].ravel(), np.repeat(src[back], 3)])
        cols = np.concatenate([np.repeat(src[out], 3), tri[back].ravel()])
        vals = np.concatenate([(q[out, None] * w[out]).ravel(), (-q[back, None] * w[back]).ravel()])
        keep = vals > 0
        n = len(p)
        self._connector_flow = sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, n))   # [j, i] from i into j

    # ----- the temperature -----

    def _advance_temperature(self, dt, old, new, p_old=None, transfer=None):

        """One step of the temperature over dt: convection in the plane (implicit upwind per cell), then conduction across the gap with the
        viscous heating (implicit). old, new: the content of each node at the start and the end of the step, in the units of the fluxes of
        self._step (melt volume f V in the filling, mass in the packing). transfer: extra flow rates between nodes (a sparse (n, n), entry
        [j, i] the rate from i into j), the overflow of the nodes that filled before the end of a step; they carry the cells of i by its
        shares of the flow."""

        th, disc = self.thermal, self.disc
        T_start = self.T.copy()
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
        if transfer is not None and transfer.nnz:
            tr = transfer.tocoo()
            rows_t = (tr.row[None, :] + n * layer[:, None]).ravel()
            cols_t = (tr.col[None, :] + n * layer[:, None]).ravel()
            data_t = (tr.data[None, :] * node_phi[tr.col].T).ravel()
            M = (M + sp.csr_matrix((data_t, (rows_t, cols_t)), shape=(n * nc, n * nc))).tocsr()
        gate =(np.maximum(F, 0.0)[:, None] * node_phi).T.ravel()                         # melt entering at the gates, per cell
        back = (np.maximum(-F, 0.0)[:, None] * node_phi).T.ravel()                         # and leaving (packing, falling gate pressure)
        # cell order k * n + i. Per cell, with c the melt volume, the vertical excess m = in - out - dc/dt leaves at the cell temperature
        # and the cells short of melt (m < 0) receive the mixing-cup temperature of the excess of their node:
        #     c_old (T - T_old) / dt + sum_in q (T - T_up) + m- (T - T_mix) = 0
        content = (old[None, :] * dz[:, None]).ravel() / dt
        growth = ((new - old)[None, :] * dz[:, None]).ravel() / dt
        inflow = np.asarray(M.sum(axis=1)).ravel() + gate
        excess = inflow - np.asarray(M.sum(axis=0)).ravel() - back - growth
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
        crystal = getattr(self, 'crystal', None)
        if crystal is None:
            self.T = _convection_solve(A, rhs).reshape(nc, n).T.copy()
        else:                                                                       # the Nakamura time rides with the melt (fresh: 0)
            tau_start = self.tau.copy()
            old_tau = self.tau.T.ravel()
            both = _convection_solve(A, np.column_stack([rhs, np.where(idle, old_tau, content * old_tau)]))
            self.T = both[:, 0].reshape(nc, n).T.copy()
            self.tau = np.maximum(both[:, 1].reshape(nc, n).T, 0.0)
        compression = None
        if p_old is not None and self.density is not None and self.density['heating']:
            # Dp/Dt per cell: the change at the node, and the pressure difference of the melt that arrives (u . grad p)
            P = np.tile(self.p, nc)
            dp_in = np.asarray(M.sum(axis=1)).ravel() * P - M @ P                        # sum_j q_ij (p_i - p_j), mass flows
            mass_cells = (new[None, :] * dz[:, None]).ravel()
            DpDt = np.tile((self.p - p_old) / dt, nc) + np.where(mass_cells > 0, dp_in / np.maximum(mass_cells, 1e-300), 0.0)
            beta = np.asarray(self.density['beta'](self.p, self.T, self.crystallinity()))
            compression = (beta * (self.T + KELVIN_OFFSET)) * DpDt.reshape(nc, n).T          # W/m3, at the start of the conduction
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
        if compression is not None:
            rhs += h * compression[melt]
        if crystal is None:
            self.T[melt] = _tridiagonal(lower, diagz, upper, rhs)
        else:
            # latent heat: tau grows with the rate at the mean temperature of the step, fixed point on that temperature
            kin, rho_L = crystal['kinetics'], th['rho'] * crystal['latent']
            T_conv, tau_conv = self.T[melt], self.tau[melt]
            X_conv = kin.crystallinity(tau_conv)
            T_new = T_conv
            for _ in range(30):
                tau_new = tau_conv + dt * kin.rate(0.5 * (T_conv + T_new))
                T_next = _tridiagonal(lower, diagz, upper, rhs + h * rho_L * (kin.crystallinity(tau_new) - X_conv) / dt)
                change = np.max(np.abs(T_next - T_new)) if T_next.size else 0.0
                T_new = T_next if change < 5.0 else 0.5 * (T_new + T_next)       # damped while the release is strong
                if change < 1e-8:
                    break
            tau_new = tau_conv + dt * kin.rate(0.5 * (T_conv + T_new))
            T_final = _tridiagonal(lower, diagz, upper, rhs + h * rho_L * (kin.crystallinity(tau_new) - X_conv) / dt)
            self.T[melt], self.tau[melt] = T_final, tau_new
            self._tau_rate = (self.tau - tau_start) / dt
        self._T_rate = (self.T - T_start) / dt

    # ----- the filling -----

    # ----- the packing -----

    def _open_elements(self):
        active = np.ones(len(self.disc.conn), dtype=bool)
        if self._freezes():
            active = ~np.all(self._element_frozen(), axis=1)
        return active

    def _gate_nodes(self):
        if not self.gates:
            return np.zeros(0, dtype=int)
        return np.unique(np.concatenate([f.nodes.ravel() for f, _ in self.gates]))

    def _gate_rates(self):

        """The volumetric flow rate of the gates at each node [m3/s]."""

        q = np.zeros(len(self.xy))
        for facets, Q in self.gates:
            q += _scatter_vector(self.disc, ops.facet_source(facets, Q / facets.detJw.sum()), facets.nodes)
        return q

    def _gate_mass(self, p):

        """Mass inflow of the gates at each node, q_i rho(p_i, T_melt), and its derivative in p."""

        q = self._gate_rates()
        T_in = self.thermal['T_melt'] if self.thermal is not None else float(self.T_fixed if self.T_fixed is not None else 0.0)
        rho, drho = self.density['gate'](p, np.full_like(p, T_in))
        return q * np.asarray(rho), q * np.asarray(drho)

    def _pack_step(self, dt, p_gate, inject=False, front=None, tol=1e-9, max_iterations=60):

        """One implicit step of the compressible melt: the gate at p_gate, or injecting its flow rate (inject), or closed. Nodes short of
        their full mass stay at ambient pressure and take what flows in. Returns the new masses, the gate inflow per node, the iterations
        and the nodes at ambient pressure. front: the nodes taken at ambient pressure to start with (default: those short of their mass at
        ambient pressure); the active set corrects it. Newton with the tangent of the fluidity and of the compressibility, rho_e lagged."""

        self._ahead = dt
        try:
            return self._pack_solve(dt, p_gate, inject, front, tol, max_iterations)
        finally:
            self._ahead = 0.0

    def _pack_solve(self, dt, p_gate, inject, front, tol, max_iterations):
        disc, n = self.disc, len(self.xy)
        gate_nodes = self._gate_nodes()
        fixed = np.zeros(n, dtype=bool)
        p = self.p.copy()
        if p_gate is not None:
            fixed[gate_nodes] = True
            p[gate_nodes] = p_gate
        m_old = self.mass
        if front is None:
            front = m_old < self.volume * self.full_density()                       # short of melt: at ambient pressure
        clamped = ~fixed & front
        p[clamped] = 0.0
        open_e = self._open_elements()
        F, dF = np.zeros(n), np.zeros(n)
        for _ in range(20):                                                         # active set of the nodes at ambient pressure
            for k in range(1, max_iterations + 1):
                gp = disc.gradient(p)
                G = np.linalg.norm(gp, axis=-1)
                S, dS = self._element_fluidity(G, derivative=True)
                rho, drho = self.nodal_density(p)
                rho_e = rho[disc.conn].mean(axis=1)
                Ke = ops.diffusion(disc, S * (rho_e * open_e)[:, None])
                K = _scatter(disc, Ke)
                if inject:
                    F, dF = self._gate_mass(p)
                r = self.volume * rho - m_old + dt * (K @ p) - dt * F
                # Newton in the fluidity too (a fixed point on S does not converge for a strongly thinning melt), rho_e lagged
                a = np.einsum('eg,egia,ega->ei', disc.detJw, disc.B, gp) * (rho_e * open_e)[:, None]
                c = np.einsum('eg,ega,egja->ej', dS / np.maximum(G, 1e-300), gp, disc.B)
                J = (sp.diags(self.volume * drho - dt * dF) + dt * (K + _scatter(disc, np.einsum('ei,ej->eij', a, c)))).tocsr()
                free = np.flatnonzero(~fixed & ~clamped)
                dp = np.zeros(n)
                if len(free):
                    dp[free] = spla.spsolve(J[free][:, free].tocsc(), -r[free])
                p_new = np.maximum(p + dp, -0.5 * self.density['ambient'])            # the state law has no meaning at p_abs < 0
                dp, p = p_new - p, p_new
                if np.max(np.abs(dp)) <= tol * max(np.max(np.abs(p)), self.density['ambient']):
                    break
            low = ~fixed & ~clamped & (p < 0.0)
            # a clamped node that its neighbours would overfill comes back
            flux = dt * (K @ p - F)
            back = clamped & (m_old - flux > self.volume * self.nodal_density(np.where(clamped, 0.0, p))[0])
            if not np.any(low) and not np.any(back):
                break
            clamped = (clamped | low) & ~back
            p[clamped] = 0.0
        S, _ = self._element_fluidity(np.linalg.norm(disc.gradient(p), axis=-1))   # the state of the step, for the temperature
        rho, _ = self.nodal_density(p)
        rho_e = rho[disc.conn].mean(axis=1)
        Ke = ops.diffusion(disc, S * (rho_e * open_e)[:, None])
        if inject:
            F, _ = self._gate_mass(p)
        flux = dt * (_scatter(disc, Ke) @ p)
        # the mass by the balance of the step everywhere but the held gate: equal to V rho(p) where the step converged, and the mass is
        # kept exactly even where the active set did not settle
        m_new = np.where(fixed, self.volume * rho, m_old - flux + dt * F)
        inflow = np.where(fixed, (m_new - m_old + flux) / dt, F)
        self.p = p
        self._step = (Ke, inflow, open_e)
        return m_new, inflow, k, clamped

    def _fill_compressible(self, t_end, progress, every, on_step, switchover=None):

        """The filling with the compressible melt: _pack_step with the gates injecting, each step until the next node is full."""

        n = len(self.xy)
        gate_nodes = self._gate_nodes()
        if getattr(self, 'mass', None) is None:
            self.mass = self.fill_factor * self.volume * self.full_density()
        rate = getattr(self, '_fill_rate', None)
        if rate is None:                                                           # the first step: the incompressible flow rates
            reaction, _ = self.solve_pressure()
            rate = reaction * self.full_density()
        start, step, dt = time.perf_counter(), 0, np.inf
        while t_end - self.time > 1e-12 * self.time:
            full = self.volume * self.full_density()
            short = full - self.mass
            if not np.any(np.isnan(self.fill_time)):
                break
            if switchover is not None and self.fill_factor @ self.volume >= switchover * self.volume.sum():
                break
            unfilled = np.isnan(self.fill_time)                                    # the front: nodes not filled yet (a filled node that
            receiving = (rate > 0) & (short > 1e-3 * full) & unfilled              # cools below its mass stays filled, at ambient pressure)
            refilling = (rate > 0) & ~unfilled                                     # filled nodes that the cooling left short: they take
            if not np.any(receiving) and np.any(refilling) and np.isfinite(dt):   # melt first, the front waits for them
                receiving, short, rate = refilling, np.full_like(short, dt), np.ones_like(rate)
            if not np.any(receiving):
                if self.thermal is not None:
                    self.short_shot = self.time
                    print(f"   tiro corto a t = {self.time:.4g} s: ningun nodo recibe fundido (la capa congelada cerro el paso)", flush=True)
                    break
                raise RuntimeError("Ningun nodo recibe fundido: la cavidad esta cerrada sin venteo o la entrada no tiene caudal.")
            # the rates of the previous step, with the step at most 4 times the last (they miss the nodes that only start to receive)
            dt = min(float(np.min(short[receiving] / rate[receiving])), 4.0 * dt, t_end - self.time)
            m_old, p_old = self.mass.copy(), self.p.copy()
            for attempt in range(10):
                m_new, inflow, iterations, clamped = self._pack_step(dt, None, inject=True, front=unfilled)
                # the step is redone shorter when, with its own rates, a node fills well before its end, or a node that received nothing
                # at its start fills (the front ran over several nodes): the rates of the previous step miss the nodes that only start
                gain = np.where(unfilled, np.maximum(m_new - m_old, 0.0) / dt, 0.0)
                done = unfilled & (~clamped | (m_new >= (1 - 1e-3) * full))         # within 0.1 %: the rest comes at ambient pressure
                waiting = receiving & (gain > 0) & (short > 1e-3 * full)
                dt_fill = float(np.min(short[waiting] / gain[waiting])) if np.any(waiting) else dt
                if attempt == 9 or (not np.any(done & ~receiving) and dt <= 1.1 * dt_fill):
                    break
                self.p = p_old.copy()
                dt = min(dt_fill, 0.5 * dt)
            # filled: out of the set at ambient pressure, or at its full mass, both at the temperature of the step (before it cools)
            self.fill_factor[unfilled] = np.minimum(m_new[unfilled] / full[unfilled], 1.0)
            self.fill_factor[done] = 1.0
            rate = np.where(clamped, (m_new - m_old) / dt, 0.0)
            if self.thermal is not None:
                self._advance_temperature(dt, m_old, m_new, p_old)
            self.mass = m_new
            self.time += dt
            step += 1
            new = done
            self.fill_time[new] = self.time
            self.history.append((self.time, float(self.fill_factor @ self.volume), float(self.p[gate_nodes].mean()), iterations,
                                 int(np.sum(self.fill_factor >= 1.0)), float(self.mass.sum())))
            if on_step is not None:
                on_step(self)
            if progress and step % every == 0:
                print(f"   llenado compresible: paso {step}, t = {self.time:.4g} s, {100 * np.mean(self.fill_factor >= 1):.1f} % de los nodos "
                      f"llenos, p de entrada {self.history[-1][2] / 1e6:.4g} MPa, masa {self.mass.sum() * 1e3:.4g} g, {iterations} it. "
                      f"({time.perf_counter() - start:.0f} s)", flush=True)
        self._fill_rate = rate
        return self.history

    def pack(self, p_gate, t_pack, t_cool=0.0, dt_max=0.05, dt_first=1e-3, growth=1.25, T_room=25.0, progress=True, every=20,
             on_step=None):

        """Packing after the filling (every node filled, setDensity given): the gate held at p_gate [Pa] (a number or f(t) of the time since
        the switchover) for t_pack, then closed for t_cool. The step grows from dt_first to dt_max. Keeps self.pack_history of (t, mass,
        gate pressure, gate mass flow [kg/s], iterations, nodes at ambient pressure) and, with setThermal and T_noflow, self.v_frozen (the
        specific volume of each node when it froze through the gap) and self.shrinkage() relative to T_room. Returns pack_history."""

        if self.density is None:
            raise ValueError("El empaque necesita setDensity (la ecuacion de estado del fundido).")
        if self.connectors is not None:
            raise NotImplementedError("El empaque de la doble superficie (conectores) no esta hecho: solo su llenado incompresible.")
        if np.any(np.isnan(self.fill_time)) and getattr(self, 'mass', None) is None:
            raise ValueError("El empaque empieza tras el llenado: llamar fill() primero (compresible si se cambia antes de llenar).")
        n = len(self.xy)
        if getattr(self, 'mass', None) is None:
            self.mass = self.volume * self.nodal_density()[0]
        if getattr(self, 'pack_history', None) is None:
            self.v_frozen = np.full(n, np.nan)
            self.pack_history = []
            self.T_room = float(T_room)
            self._switchover = self.time
        gate_nodes = self._gate_nodes()
        t0, end = self._switchover, self._switchover + t_pack + t_cool
        dt, step, start = dt_first, 0, time.perf_counter()
        while self.time < end - 1e-12:
            dt = min(dt, end - self.time)
            if self.time < t0 + t_pack - 1e-12:
                dt = min(dt, t0 + t_pack - self.time)
                s = self.time + dt - t0
                pg = float(p_gate(s)) if callable(p_gate) else float(p_gate)
            else:
                pg = None
            m_old, p_old = self.mass, self.p.copy()
            unfilled = np.isnan(self.fill_time)
            full = self.volume * self.full_density()
            m_new, inflow, iterations, clamped = self._pack_step(dt, pg, front=(m_old < full) | unfilled)
            # the rest of the filling, after a switchover: a node that cools keeps raising its full mass, so here 99 % of it is filled (the
            # rest is a void that the shrinkage counts)
            done = unfilled & (~clamped | (m_new >= 0.99 * full))
            self.fill_factor[unfilled] = np.minimum(m_new[unfilled] / full[unfilled], 1.0)
            self.fill_factor[done] = 1.0
            if self.thermal is not None:
                self._advance_temperature(dt, m_old, m_new, p_old)
            self.mass = m_new
            self.time += dt
            self.fill_time[done] = self.time
            step += 1
            if self._freezes():
                frozen = np.isnan(self.v_frozen) & (self.frozen_fraction() >= 1.0 - 1e-12)
                self.v_frozen[frozen] = self.volume[frozen] / self.mass[frozen]
            self.pack_history.append((self.time, float(self.mass.sum()), float(self.p[gate_nodes].mean()), float(inflow.sum()),
                                      iterations, int(clamped.sum())))
            if on_step is not None:
                on_step(self)
            if progress and step % every == 0:
                print(f"   empaque: paso {step}, t = {self.time:.4g} s, masa {self.mass.sum() * 1e3:.4f} g, p de entrada "
                      f"{self.pack_history[-1][2] / 1e6:.4g} MPa, caudal {inflow.sum() * 1e3:.4g} g/s, "
                      f"{100 * np.mean(self.frozen_fraction() >= 1 - 1e-12):.0f} % congelado, {iterations} it. "
                      f"({time.perf_counter() - start:.0f} s)", flush=True)
            dt = min(dt * growth, dt_max)
        return self.pack_history

    def shrinkage(self):

        """Volumetric shrinkage of each node, 1 - v(ambient, T_room) / v_frozen; a node not yet frozen uses its specific volume now, a node
        never filled (a short shot) gets NaN."""

        with np.errstate(divide='ignore', invalid='ignore'):
            v = np.where(np.isnan(self.v_frozen), self.volume / self.mass, self.v_frozen)
        X = self.crystallinity() @ (self.thermal['dz'] if self.thermal is not None else np.ones(1))   # the crystallinity reached
        v_room = 1.0 / np.asarray(self.density['law'](self.density['ambient'], np.full_like(X, self.T_room), X))
        return np.where(np.isnan(self.fill_time), np.nan, 1.0 - v_room / v)

    def melt_volume(self):

        """The melt in the cavity, and in the feed system if there is one."""

        cavity = float(self.fill_factor @ self.volume)
        return cavity if self.runners is None else cavity + float(self.runners.fill_factor @ self.runners.volume)

    def cavity_volume_filled(self):
        return float(self.fill_factor @ self.volume)

    def _neighbours(self):

        """Weights between neighbouring nodes for the overflow: the off-diagonal conductance of the gap, h^3, clipped at zero (n, n)."""

        if getattr(self, '_neighbour_weights', None) is None:
            K = _scatter(self.disc, ops.diffusion(self.disc, self.h ** 3)).tocsr()
            W = (-K).tocsr()
            W.setdiag(0.0)
            W.data = np.maximum(W.data, 0.0)
            if self.connectors is not None:                             # across the wall too, as strongly as the coupling
                W = (W + self.coupling * sp.diags(K.diagonal()) @ self._link_graph).tocsr()
            W.eliminate_zeros()
            self._neighbour_weights = W
        return self._neighbour_weights

    def _overflow(self, excess, reaction):

        """Passes the melt that the nodes filled before the end of the step received beyond their volume (excess [m3] per node) on to their
        neighbours not yet full, by the conductance of the gap; a node that overflows in turn passes its excess on. A node with no
        neighbour short of melt (the end of a branch) passes it to the nearest nodes short of melt, through the filled ones. Mass is kept
        exactly; only with nothing open left in reach (a full cavity) the excess goes to the front by its flow rate. Updates fill_factor;
        returns the transfers [m3] (n, n), entry [j, i] from node i into node j."""

        from scipy.sparse.csgraph import dijkstra
        W = self._neighbours()
        n = len(self.xy)
        rows, cols, vals = [], [], []
        excess = excess.copy()
        tiny = 1e-15 * self.volume.max()
        for _ in range(200):
            src = np.flatnonzero(excess > tiny)
            if not len(src):
                break
            open_ = self.fill_factor < 1.0
            to_open = (sp.diags(open_.astype(float)) @ W[:, src]).tocsc()
            total = np.asarray(to_open.sum(axis=0)).ravel()
            near = total > 0
            moved = np.zeros(n)
            if np.any(near):
                weights = (to_open[:, near] @ sp.diags(excess[src[near]] / total[near])).tocoo()
                rows.append(weights.row); cols.append(src[near][weights.col]); vals.append(weights.data)
                np.add.at(moved, weights.row, weights.data)
            stuck = src[~near]
            if len(stuck):
                if np.any(open_):
                    dist = dijkstra(W, directed=False, indices=stuck, unweighted=True)
                    dist[:, ~open_] = np.inf
                    for s, d in zip(stuck, dist):
                        best = np.min(d)
                        if np.isfinite(best):
                            tgt = np.flatnonzero(d == best)
                            share = excess[s] * self.volume[tgt] / self.volume[tgt].sum()
                            rows.append(tgt); cols.append(np.full(len(tgt), s)); vals.append(share)
                            np.add.at(moved, tgt, share)
                            excess[s] = 0.0
                if not np.any(near) and np.all(excess[stuck] > 0):
                    break
            excess[src[near]] = 0.0
            room = (1.0 - self.fill_factor) * self.volume
            take = np.minimum(moved, room)
            self.fill_factor += take / self.volume
            self.fill_factor[(moved > 0) & (room - take <= 1e-12 * self.volume)] = 1.0            # full, to round-off
            excess += moved - take
        left = excess[excess > tiny]
        if len(left):                                                   # nothing open in reach: to the front, by its flow rate
            front = (self.fill_factor < 1.0) & (reaction > 0)
            if not np.any(front):
                front = self.fill_factor < 1.0
            if np.any(front):
                share = np.where(front, reaction if np.any(reaction[front] > 0) else self.volume, 0.0)
                share = share / share.sum()
                tgt = np.flatnonzero(share)
                for s in np.flatnonzero(excess > tiny):
                    rows.append(tgt); cols.append(np.full(len(tgt), s)); vals.append(excess[s] * share[tgt])
                self.fill_factor = np.minimum(self.fill_factor + left.sum() * share / self.volume, 1.0)
        if not rows:
            return sp.csr_matrix((n, n))
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))

    def fill(self, t_end=np.inf, progress=True, every=50, on_step=None, compressible=None, switchover=None, batch=0.0):

        """Fills node by node until t_end or until every node is filled. Keeps self.history of (t, melt volume, mean gate pressure, Newton
        iterations, filled nodes) and returns it; self.fill_time holds when each node filled; with setThermal, self.T the temperature of each
        cell across the gap at each node. compressible (default: with setDensity): the compressible filling, history then
        (t, filled volume, gate pressure, iterations, filled nodes, mass) and self.mass the mass of each node. switchover: stop at
        this fraction of the cavity volume filled (the velocity-to-pressure switchover, before pack(), which fills the rest; at constant
        flow rate the last nodes of a nearly full compressible cavity only take melt as the whole cavity compresses). on_step(self) is called after each step (to take
        snapshots for an animation). batch (incompressible filling only): each step lasts until this fraction of the nodes of the front has
        filled (0: the next node only, one node per step; the cost of the filling then grows as the square of the nodes). The nodes that
        fill before the end of the step pass their excess on to their neighbours (mass and energy kept exactly); their fill time is when
        they filled inside the step, by their rate."""

        if compressible is None:
            compressible = self.density is not None
        if compressible:
            if self.runners is not None:
                raise NotImplementedError("La red de alimentacion solo tiene el llenado incompresible (sin compresible ni empaque por ahora).")
            if self.connectors is not None:
                raise NotImplementedError("La doble superficie (conectores) solo tiene el llenado incompresible: el compresible y el empaque "
                                          "no llevan aun los conectores.")
            if batch:
                raise ValueError("batch solo para el llenado incompresible: el compresible avanza nodo a nodo (con lotes la presion "
                                 "diverge con un fundido casi incompresible).")
            return self._fill_compressible(t_end, progress, every, on_step, switchover)
        gate_nodes = self._gate_nodes()
        start, step = time.perf_counter(), 0
        while t_end - self.time > 1e-12 * self.time and not np.all(self.fill_factor >= 1.0):   # t_end to round-off: no last step of 1e-13 s
            if switchover is not None and self.fill_factor @ self.volume >= switchover * self.volume.sum():
                break
            reaction, iterations = self.solve_pressure()
            p_in = float(self.runners.p[self.runners.inlet[0]]) if self.runners is not None else float(self.p[gate_nodes].mean())
            self.history.append((self.time, self.melt_volume(), p_in, iterations, int(np.sum(self.fill_factor >= 1.0))))
            receiving = reaction > 0
            if self.runners is not None:                                   # the network: no node fills past its end inside the step
                Rn = self.runners
                r_net = self._runner_reaction
                net = (r_net > 0) & (Rn.fill_factor < 1.0)
                dt_net = float(np.min((1.0 - Rn.fill_factor[net]) * Rn.volume[net] / r_net[net])) if np.any(net) else np.inf
            if not np.any(receiving) and (self.runners is None or not np.any(net)):
                if self.thermal is not None:
                    self.short_shot = self.time
                    print(f"   tiro corto a t = {self.time:.4g} s: ningun nodo recibe fundido (la capa congelada cerro el paso)", flush=True)
                    break
                raise RuntimeError("Ningun nodo recibe fundido: la cavidad esta cerrada sin venteo o la entrada no tiene caudal.")
            remaining = (1.0 - self.fill_factor[receiving]) * self.volume[receiving] / reaction[receiving]
            # the step: until the k-th node of the front fills, k a fraction `batch` of the nodes receiving (1: the next node only)
            dt = min(_kth(remaining, batch) if len(remaining) else np.inf, t_end - self.time)
            if self.runners is not None and np.any(net):
                dt = min(dt, dt_net)
                fr = Rn.fill_factor[net] + dt * r_net[net] / Rn.volume[net]
                done = fr > 1.0 - 1e-12
                Rn.fill_factor[net] = np.where(done, 1.0, fr)
                Rn.fill_time[np.flatnonzero(net)[done]] = self.time + dt
            f_old = self.fill_factor.copy()
            f = self.fill_factor[receiving] + dt * reaction[receiving] / self.volume[receiving]
            idx = np.flatnonzero(receiving)
            excess = np.zeros(len(self.xy))
            excess[idx] = np.where(f > 1.0 + 1e-12, f - 1.0, 0.0) * self.volume[idx]   # beyond round-off (ties fill together)
            early = idx[f >= 1.0 - 1e-12]
            self.fill_time[early] = self.time + np.minimum(remaining[f >= 1.0 - 1e-12], dt)   # when each filled inside the step
            self.fill_factor[receiving] = np.where(f > 1.0 - 1e-12, 1.0, f)           # the node that set dt is full, to round-off
            transfer = self._overflow(excess, reaction) / dt if dt > 0 and np.any(excess > 0) else None
            if self.connectors is not None:                                    # the melt that crosses the wall between the skins
                transfer = self._connector_flow if transfer is None else transfer + self._connector_flow
            if self.thermal is not None and dt > 0:
                self._advance_temperature(dt, f_old * self.volume, self.fill_factor * self.volume, transfer=transfer)
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
