# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++++

# Turbulence (RANS) with the one-equation Spalart-Allmaras model.
#
# The flow physics ff solves the averaged equations with the effective viscosity mu + mu_t; the turbulence physics
# sa transports the working variable nu~ (field 'nut'), from which mu_t = rho nu~ f_v1. Both are solved together:
#
#     model = Model(name='canal', dim=2, mesh=mesh, mat=material('aire', miu=mu, rho=rho), psc=[ff, sa])
#     model.physics.fluid.setInertia(True)
#     model.physics.turbulence.setWalls(wall_edges)       # nu~ = 0 there and the wall distance d
#     model.physics.turbulence.coupleFlow()               # mu + mu_t(nu~) in the flow
#
# Spalart-Allmaras, negative version (SA-neg, Allmaras, Johnson & Spalart 2012), without the trip term f_t2:
#
#     dnu~/dt + u . grad nu~ = P - D + [div((nu + nu~ f_n) grad nu~) + cb2 |grad nu~|^2] / sigma
#     nu~ >= 0:  P = cb1 S~ nu~,           D = cw1 f_w (nu~/d)^2
#     nu~ <  0:  P = cb1 (1 - ct3) W nu~,  D = -cw1 (nu~/d)^2,  f_n = (cn1 + chi^3)/(cn1 - chi^3)  (else f_n = 1)
#
# with W the vorticity magnitude, chi = nu~/nu, S~ = W + nu~ f_v2 / (kappa d)^2 (with the limiter of SA-neg) and the
# usual f_v1, f_v2, f_w. The convection uses the computed velocity with SUPG (FieldConvectionTerm); diffusion and
# sources are one pointwise law of nu~, grad nu~, grad u and d with automatic derivatives (GradientLawTerm).
#
# The wall distance d is the minimum distance to the straight segments of the walls (2D), or to the flat triangles of the wall
# faces (3D: quadrilateral and quadratic faces split into triangles through all their nodes). It is stored as the field
# 'wall_distance', fixed at every node. Axisymmetric studies need nothing else for the scalars, whose integrals carry r;
# k-omega SST adds the hoop strain to S^2.

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from Source.Physics.Physics import physics
from Source.Physics.Rheology import ViscosityLaw
from Source.Primals.Scalar import scalarField
from Source.Simulation.PointwiseLaw import PointwiseLaw, cached_law
import Source.Simulation.Operators as ops
from Source.enums import ShapeFunctionType

# Spalart-Allmaras constants
CB1, CB2, SIGMA, KAPPA = 0.1355, 0.622, 2.0 / 3.0, 0.41
CW1 = CB1 / KAPPA ** 2 + (1.0 + CB2) / SIGMA
CW2, CW3, CV1 = 0.3, 2.0, 7.1
CT3, CV2, CV3, CN1 = 1.2, 0.7, 0.9, 16.0


# Triangles through all the nodes of a wall face, by its number of nodes (the node order of Elements.py)
_FACE_TRIANGLES = {3: [(0, 1, 2)], 4: [(0, 1, 2), (0, 2, 3)],
                   6: [(0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)],
                   9: [(0, 4, 8), (0, 8, 7), (4, 1, 5), (4, 5, 8), (7, 8, 6), (7, 6, 3), (8, 5, 2), (8, 2, 6)]}


def _segment_distance2(P, a, b):

    """Squared distances (n, m) from the points P (n, 3) to the segments a-b (m, 3)."""

    t = b - a
    length2 = np.maximum(np.sum(t * t, axis=1), 1e-300)
    s = np.clip(np.einsum('nma,ma->nm', P[:, None, :] - a[None], t) / length2, 0.0, 1.0)
    gaps = P[:, None, :] - (a[None] + s[..., None] * t[None])
    return np.sum(gaps * gaps, axis=-1)


def _triangle_distance(P, A, B, C):

    """Exact distances from the points P (n, 3) to the nearest of the triangles A B C (m, 3): the distance to the plane where the
    projection falls inside, else the nearest of the three sides."""

    normal = np.cross(B - A, C - A)
    unit = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-300)
    height = np.einsum('nma,ma->nm', P[:, None, :] - A[None], unit)
    foot = P[:, None, :] - height[..., None] * unit[None]
    inside = np.ones(height.shape, dtype=bool)
    for X, Y in ((A, B), (B, C), (C, A)):
        inside &= np.einsum('nma,ma->nm', np.cross(Y - X, foot - X[None]), normal) >= 0.0
    d2 = np.where(inside, height ** 2, np.inf)
    for X, Y in ((A, B), (B, C), (C, A)):
        d2 = np.minimum(d2, _segment_distance2(P, X, Y))
    return np.sqrt(np.min(d2, axis=1))


def wall_distance(mesh, walls):

    """Nodal distance to the walls (list of boundary facets): the minimum over their straight segments in 2D, over the flat
    triangles of the faces in 3D. Exact for straight (flat) walls; on a curved wall the error is that of its chords, O(h^2)."""

    if getattr(mesh, 'PD', 2) == 3:
        xyz = np.array([node.coor[:3] for node in mesh.NL], dtype=float)
        tris = np.array([[face[i] for i in tri] for face in walls for tri in _FACE_TRIANGLES[len(face)]], dtype=int)
        A, B, C = xyz[tris[:, 0]], xyz[tris[:, 1]], xyz[tris[:, 2]]
        d = np.zeros(len(xyz))
        block = max(1, int(2e7 // max(1, 9 * len(tris))))              # nodes per block: n x triangles x 3 doubles stay small
        for start in range(0, len(xyz), block):
            d[start:start + block] = _triangle_distance(xyz[start:start + block], A, B, C)
        d[np.unique(np.concatenate([np.asarray(face) for face in walls]))] = 0.0
        return d

    xy = np.array([node.coor[:2] for node in mesh.NL], dtype=float)

    # An edge of three nodes gives two segments, which follow a curved wall more closely
    a = np.array([xy[edge[i]] for edge in walls for i in range(len(edge) - 1)])
    b = np.array([xy[edge[i + 1]] for edge in walls for i in range(len(edge) - 1)])
    t = b - a
    length2 = np.sum(t * t, axis=1)

    d = np.zeros(len(xy))
    for start in range(0, len(xy), 2000):                    # by blocks: nodes x segments can be large
        points = xy[start:start + 2000]
        s = np.clip(np.einsum('nsa,sa->ns', points[:, None, :] - a[None], t) / length2, 0.0, 1.0)
        gaps = points[:, None, :] - (a[None] + s[..., None] * t[None])
        d[start:start + 2000] = np.sqrt(np.min(np.sum(gaps * gaps, axis=-1), axis=1))

    d[np.unique(np.ravel(walls))] = 0.0

    return d


def _fv1(chi):
    chi3 = chi ** 3
    return chi3 / (chi3 + CV1 ** 3)


class SAViscosity(ViscosityLaw):

    """Effective viscosity of Spalart-Allmaras, mu + rho nu~ f_v1(nu~ / nu) (no eddy viscosity for nu~ < 0). It reads
    the field nu~ through the second argument of the law (ff.setViscosityLaw(law, temperature='nut'))."""

    uses_temperature = True

    def __init__(self, mu, rho):
        self.mu, self.rho = float(mu), float(rho)
        self.nu = self.mu / self.rho

    def __call__(self, gamma2, nut):
        positive = jnp.maximum(nut, 0.0)
        return self.mu + self.rho * positive * _fv1(positive / self.nu) + 0.0 * gamma2


def sa_law(nu, dim=2):

    """Pointwise law of the Spalart-Allmaras equation for GradientLawTerm.

    Inputs (per Gauss point): nu~, grad nu~ (n), u (n), grad u (n^2, row by row), d, grad d (n), n = dim (2 or 3).
    Output for nu~: source s (tested by N) and flux F (n, tested by grad N), with the residual
    integral N s + grad N . F:  F = (nu + nu~ f_n) grad nu~ / sigma,  s = -cb2 |grad nu~|^2 / sigma - P + D.
    The vorticity magnitude is sqrt(2 W:W), W the antisymmetric part of grad u (|dv/dx - du/dy| in the plane).
    """

    n = int(dim)

    def law(w):
        nut, g_nut = w[0], w[1:1 + n]
        G = w[1 + 2 * n:1 + 2 * n + n * n].reshape(n, n)          # G_ab = du_a/dx_b
        d = jnp.maximum(w[1 + 2 * n + n * n], 1e-12)

        W = 0.5 * (G - G.T)
        vort = jnp.sqrt(2.0 * jnp.sum(W * W) + 1e-30)
        chi = nut / nu
        fv1 = _fv1(chi)
        fv2 = 1.0 - chi / (1.0 + chi * fv1)
        kd2 = (KAPPA * d) ** 2

        s_bar = nut * fv2 / kd2
        s_tilde = jnp.where(s_bar >= -CV2 * vort, vort + s_bar,
                            vort + vort * (CV2 ** 2 * vort + CV3 * s_bar) / ((CV3 - 2.0 * CV2) * vort - s_bar))
        s_safe = jnp.maximum(s_tilde, 1e-16)
        r = jnp.minimum(nut / (s_safe * kd2), 10.0)
        g = r + CW2 * (r ** 6 - r)
        fw = g * ((1.0 + CW3 ** 6) / (g ** 6 + CW3 ** 6)) ** (1.0 / 6.0)

        positive = nut >= 0.0
        production = jnp.where(positive, CB1 * s_tilde * nut, CB1 * (1.0 - CT3) * vort * nut)
        destruction = jnp.where(positive, CW1 * fw * (nut / d) ** 2, -CW1 * (nut / d) ** 2)
        chi3 = chi ** 3
        fn = jnp.where(positive, 1.0, (CN1 + chi3) / (CN1 - chi3))

        flux = (nu + nut * fn) * g_nut / SIGMA
        source = -CB2 * (g_nut @ g_nut) / SIGMA - production + destruction

        return jnp.concatenate([jnp.array([source]), flux])

    return law


class sa(physics):

    part_name = 'turbulence'
    ConstRel = 'dnu~/dt + u.grad(nu~) = P - D + [div((nu + nu~) grad nu~) + cb2 |grad nu~|^2] / sigma'

    def __init__(self, model):
        super().__init__(model)

        self.physics_description = 'Turbulence (Spalart-Allmaras)'
        self.var = {'nut': scalarField('nut', 'SA working variable', 'm2/s', ShapeFunctionType.linear, model.mesh),
                    'wall_distance': scalarField('wall_distance', 'Wall distance', 'm', ShapeFunctionType.linear,
                                                 model.mesh)}

        self.mu = float(getattr(self.mat, 'miu', 1.0))
        self.rho = float(getattr(self.mat, 'rho', 1.0))
        self.nu = self.mu / self.rho
        self.var['nut'].initField(3.0 * self.nu)       # a usual free-stream value

        self.walls = []
        self.vel = 'u'                                 # convected by the computed flow, with SUPG
        self.Convection = True
        self.Stab = 'SUPG'
        self.C_const = 1.0
        self.M_const = 1.0


    # ----- setup -----

    def setWalls(self, edges):

        """Walls (no-slip boundaries): nu~ = 0 on them and the wall distance is measured from them."""

        self.walls = list(self.walls) + list(edges)
        self.addBC_TurbulenceEdges(edges, 0.0)

    def addBC_TurbulenceEdges(self, edges, nut):

        """nu~ on the edges (inlets: 3 to 5 times the laminar nu is the usual free-stream value)."""

        nodes = self.modelRef._mesh.NL
        for edge in edges:
            for node_id in edge:
                nodes[node_id].BC['nut'] = self.setDirichletBC(float(nut))

    def coupleFlow(self):

        """Gives the flow physics of the same model the effective viscosity mu + rho nu~ f_v1."""

        flow = next(part for part in self.modelRef.physics.parts.values() if getattr(part, 'computes_velocity', False))
        flow.setViscosityLaw(SAViscosity(self.mu, self.rho), temperature='nut')

    def eddyViscosity(self):

        """Nodal turbulent viscosity mu_t = rho nu~ f_v1."""

        nut = np.maximum(np.asarray(self.var['nut'].values, dtype=float), 0.0)
        return self.rho * nut * np.asarray(_fv1(nut / self.nu))

    def derived_fields(self, model):
        return {'eddy_viscosity': self.eddyViscosity()}

    # ----- weak form -----

    def add_terms(self, system, model, disc, layout):

        if not self.walls:
            raise ValueError("Spalart-Allmaras necesita las paredes: model.physics.turbulence.setWalls(aristas).")

        d = wall_distance(model.mesh, self.walls)
        self.var['wall_distance'].updateField(d)
        system.set_dirichlet(layout.dofs('wall_distance')[:, 0], d)

        rows = layout.element_dofs(disc.conn, 'nut')
        system.add_mass(layout.scatter_matrix(ops.mass(disc, self.M_const), rows, rows))
        self.add_convection(system, layout, disc, 'nut', k=self.nu, k_in_residual=False)

        law = cached_law(('spalart-allmaras', self.nu, disc.dim), lambda: PointwiseLaw(sa_law(self.nu, disc.dim)))
        system.add_gradient_law(law, ['nut', 'u', 'wall_distance'], ['nut'])


# ----------------------------------------------------------------------------------------------------------------
# k-omega SST (Menter, Kuntz & Langtry 2003). Solve it with globalization='pseudo-transient': Newton alone diverges.
#
#     dk/dt + u . grad k = P~_k - beta* k omega + div((nu + sigma_k nu_t) grad k)
#     dg/dt + u . grad g = alpha S^2 / omega - beta omega + div((nu + sigma_w nu_t) grad g) + (nu + sigma_w nu_t) |grad g|^2
#                          + 2 (1 - F1) sigma_w2 grad k . grad g / omega
#     nu_t = a1 k / max(a1 omega, S F2),   P~_k = min(nu_t S^2, 10 beta* k omega),   S = sqrt(2 S_ij S_ij)
#
# The unknown is g = ln(omega) (Ilinca & Pelletier 1998; Bassi et al. 2005): omega > 0 by construction and its range next
# to a wall drops from ~1e6 to ~14, the second equation being that of omega divided by omega. Constants blended by F1
# (inner k-omega set 1, outer k-epsilon set 2). At walls k = 0 and omega = 60 nu / (beta1 dy^2), dy the distance of the
# first node off the wall. k enters the nonlinear terms through its positive part, so an overshoot below zero stays finite.
# ----------------------------------------------------------------------------------------------------------------

BETA_STAR, A1 = 0.09, 0.31
SK1, SW1, BETA1, ALPHA1 = 0.85, 0.5, 0.075, 5.0 / 9.0
SK2, SW2, BETA2, ALPHA2 = 1.0, 0.856, 0.0828, 0.44
_EPS = 1e-30


def _sst_functions(k, g, grad_k, grad_g, d, nu):

    """Positive part of k, omega = exp(g), the blending functions F1, F2 and the cross-diffusion grad k . grad g."""

    kp = jnp.maximum(k, 0.0)
    wp = jnp.exp(g)
    d = jnp.maximum(d, 1e-12)
    sqrt_k = jnp.sqrt(kp + _EPS)
    cross = grad_k @ grad_g
    cd = jnp.maximum(2.0 * SW2 * cross, 1e-20)
    arg1 = jnp.minimum(jnp.maximum(sqrt_k / (BETA_STAR * wp * d), 500.0 * nu / (d ** 2 * wp)),
                       4.0 * SW2 * kp / (cd * d ** 2))
    arg2 = jnp.maximum(2.0 * sqrt_k / (BETA_STAR * wp * d), 500.0 * nu / (d ** 2 * wp))

    return kp, wp, jnp.tanh(arg1 ** 4), jnp.tanh(arg2 ** 2), cross


def _sst_nut(kp, wp, strain, f2):
    return A1 * kp / jnp.maximum(A1 * wp, strain * f2)


class SSTViscosity(ViscosityLaw):

    """Effective viscosity of k-omega SST, mu + rho a1 k / max(a1 omega, S F2), with omega = exp(g): a law of several
    fields, ff.setViscosityLaw(law, fields=('k', 'ln_omega', 'wall_distance')) (sst.coupleFlow does it)."""

    def __init__(self, mu, rho):
        self.mu, self.rho = float(mu), float(rho)
        self.nu = self.mu / self.rho

    def __call__(self, gamma2, k, g, d):
        kp, wp = jnp.maximum(k, 0.0), jnp.exp(g)
        d = jnp.maximum(d, 1e-12)
        arg2 = jnp.maximum(2.0 * jnp.sqrt(kp + _EPS) / (BETA_STAR * wp * d), 500.0 * self.nu / (d ** 2 * wp))
        strain = jnp.sqrt(gamma2 + _EPS)
        return self.mu + self.rho * _sst_nut(kp, wp, strain, jnp.tanh(arg2 ** 2))


def sst_law(nu, axisymmetric=False, dim=2):

    """Pointwise law of the k and ln(omega) equations for GradientLawTerm.

    Inputs: k, grad k (n), g = ln(omega), grad g (n), u (n), grad u (n^2, row by row), d, grad d (n), n = dim (2 or 3);
    axisymmetric (2D): also r, for the hoop strain u_r / r in S^2. The scalars need no other change: the measure of
    GradientLawTerm carries r. Outputs: for k and for g a source s (tested by N) and a flux F (n, tested by grad N), the residual
    being integral N s + grad N . F (convection is a separate term).
    """

    n = int(dim)
    i_g = 1 + n
    i_u = i_g + 1 + n
    i_G = i_u + n
    i_d = i_G + n * n

    def law(w):
        k, g_k, g, g_g = w[0], w[1:i_g], w[i_g], w[i_g + 1:i_u]
        G = w[i_G:i_d].reshape(n, n)
        d = w[i_d]

        kp, wp, f1, f2, cross = _sst_functions(k, g, g_k, g_g, d, nu)
        E = 0.5 * (G + G.T)
        s2 = 2.0 * jnp.sum(E * E)                                   # S^2
        if axisymmetric:
            s2 = s2 + 2.0 * (w[i_u] / w[i_d + 1 + n]) ** 2          # hoop strain: u_r / r
        nut = _sst_nut(kp, wp, jnp.sqrt(s2 + _EPS), f2)

        sigma_k = f1 * SK1 + (1.0 - f1) * SK2
        sigma_w = f1 * SW1 + (1.0 - f1) * SW2
        alpha = f1 * ALPHA1 + (1.0 - f1) * ALPHA2
        beta = f1 * BETA1 + (1.0 - f1) * BETA2

        production_k = jnp.minimum(nut * s2, 10.0 * BETA_STAR * kp * wp)
        s_k = -production_k + BETA_STAR * k * wp
        diffusivity = nu + sigma_w * nut
        s_g = -alpha * s2 / wp + beta * wp - diffusivity * (g_g @ g_g) - 2.0 * (1.0 - f1) * SW2 * cross / wp
        flux_k = (nu + sigma_k * nut) * g_k
        flux_g = diffusivity * g_g

        return jnp.concatenate([jnp.array([s_k]), flux_k, jnp.array([s_g]), flux_g])

    return law


class sst(physics):

    part_name = 'turbulence'
    ConstRel = 'k-omega SST (Menter 2003)'

    def __init__(self, model):
        super().__init__(model)

        self.physics_description = 'Turbulence (k-omega SST)'
        mesh = model.mesh
        self.var = {'k': scalarField('k', 'Turbulent kinetic energy', 'm2/s2', ShapeFunctionType.linear, mesh),
                    'ln_omega': scalarField('ln_omega', 'Logarithm of the specific dissipation rate', '-',
                                            ShapeFunctionType.linear, mesh),
                    'wall_distance': scalarField('wall_distance', 'Wall distance', 'm', ShapeFunctionType.linear, mesh)}

        self.mu = float(getattr(self.mat, 'miu', 1.0))
        self.rho = float(getattr(self.mat, 'rho', 1.0))
        self.nu = self.mu / self.rho

        self.walls = []
        self.k_inflow = None         # smallest k given on an inflow: sets the floor of k
        self.vel = 'u'
        self.Convection = True
        self.Stab = 'SUPG'
        self.C_const = 1.0
        self.M_const = 1.0


    def setWalls(self, edges):

        """Walls: k = 0 and omega = 60 nu / (beta1 dy^2) (set when the system is built); distance measured from them."""

        self.walls = list(self.walls) + list(edges)
        nodes = self.modelRef._mesh.NL
        for edge in edges:
            for node_id in edge:
                nodes[node_id].BC['k'] = self.setDirichletBC(0.0)

    def addBC_TurbulenceEdges(self, edges, k, omega):

        """k and omega on the edges (inlets): e.g. k = 1.5 (I U)^2 with the turbulence intensity I, and
        omega = k / nu_t for a chosen ratio nu_t / nu."""

        if k > 0.0:
            self.k_inflow = float(k) if self.k_inflow is None else min(self.k_inflow, float(k))
        nodes = self.modelRef._mesh.NL
        for edge in edges:
            for node_id in edge:
                nodes[node_id].BC['k'] = self.setDirichletBC(float(k))
                nodes[node_id].BC['ln_omega'] = self.setDirichletBC(float(np.log(omega)))

    def coupleFlow(self):

        """Gives the flow physics of the same model the effective viscosity mu + rho nu_t(k, omega)."""

        flow = next(part for part in self.modelRef.physics.parts.values() if getattr(part, 'computes_velocity', False))
        flow.setViscosityLaw(SSTViscosity(self.mu, self.rho), fields=('k', 'ln_omega', 'wall_distance'))

    def _wall_omega(self, mesh, d):

        """Wall value of omega at each wall node: 60 nu / (beta1 dy^2), dy the wall distance of its nearest neighbour."""

        wall_nodes = np.unique(np.ravel(self.walls))
        conn = np.array([e.getNodesId() for e in mesh.EL], dtype=int)
        dy = np.full(len(d), np.inf)
        for element in conn:
            others = d[element][d[element] > 1e-14]
            if others.size:
                for a in element:
                    dy[a] = min(dy[a], others.min())

        return wall_nodes, 60.0 * self.nu / (BETA1 * dy[wall_nodes] ** 2)

    def eddyViscosity(self):

        """Nodal turbulent viscosity mu_t = rho a1 k / max(a1 omega, S F2), with S from the current velocity."""

        model = self.modelRef
        disc = model.geometry
        G = disc.gradient(np.asarray(model.physics.var['u'].values, dtype=float))
        E = 0.5 * (G + np.swapaxes(G, -1, -2))
        g2 = 2.0 * np.einsum('egab,egab->eg', E, E)
        if disc.axisymmetric:
            g2 = g2 + 2.0 * (disc.interpolate(np.asarray(model.physics.var['u'].values, dtype=float))[..., 0] / disc.radius) ** 2
        values = [disc.interpolate(np.asarray(self.var[n].values, dtype=float)) for n in ('k', 'ln_omega', 'wall_distance')]
        law = SSTViscosity(self.mu, self.rho)
        mu_eff = np.asarray(jax.vmap(law)(jnp.asarray(g2.ravel()), *[jnp.asarray(v.ravel()) for v in values]))

        return disc.project(mu_eff.reshape(g2.shape) - self.mu)

    def derived_fields(self, model):
        return {'eddy_viscosity': self.eddyViscosity(), 'omega': np.exp(np.asarray(self.var['ln_omega'].values, dtype=float))}

    def add_terms(self, system, model, disc, layout):

        if not self.walls:
            raise ValueError("k-omega SST necesita las paredes: model.physics.turbulence.setWalls(aristas).")

        d = wall_distance(model.mesh, self.walls)
        self.var['wall_distance'].updateField(d)
        system.set_dirichlet(layout.dofs('wall_distance')[:, 0], d)

        wall_nodes, omega_wall = self._wall_omega(model.mesh, d)
        system.set_dirichlet(layout.node_dofs('ln_omega', wall_nodes)[:, 0], np.log(omega_wall))
        # Above zero: k = 0 is a kink of the model (no production, no eddy viscosity), and the nodes at the edge of a boundary
        # layer jump on and off it, so the residual of a flat plate does not close. A thousandth of the inflow value changes nothing.
        system.require_positive(layout.dofs('k')[:, 0], floor=1e-3 * self.k_inflow if self.k_inflow else 0.0)

        for name in ('k', 'ln_omega'):
            rows = layout.element_dofs(disc.conn, name)
            system.add_mass(layout.scatter_matrix(ops.mass(disc, self.M_const), rows, rows))
            self.add_convection(system, layout, disc, name, k=self.nu, k_in_residual=False)

        axisymmetric = disc.axisymmetric
        law = cached_law(('k-omega-sst', self.nu, axisymmetric, disc.dim),
                         lambda: PointwiseLaw(sst_law(self.nu, axisymmetric, disc.dim)))
        system.add_gradient_law(law, ['k', 'ln_omega', 'u', 'wall_distance'], ['k', 'ln_omega'],
                                gauss_inputs=disc.radius[..., None] if axisymmetric else None)
