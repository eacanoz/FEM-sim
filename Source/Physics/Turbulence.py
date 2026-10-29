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
# The wall distance d comes from Spalding's Poisson approximation: grad^2 phi = -1 with phi = 0 on the walls,
# d = sqrt(|grad phi|^2 + 2 phi) - |grad phi| (exact between parallel plates). It is stored as the field
# 'wall_distance', fixed at every node.

import numpy as np
import scipy.sparse.linalg as spla

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from Source.Physics.Physics import physics
from Source.Physics.Rheology import ViscosityLaw
from Source.Primals.Scalar import scalarField
from Source.Simulation.Discretization import Discretization
from Source.Simulation.DofLayout import DofLayout, FieldSpec
from Source.Simulation.PointwiseLaw import PointwiseLaw, cached_law
import Source.Simulation.Operators as ops
from Source.enums import ShapeFunctionType

# Spalart-Allmaras constants
CB1, CB2, SIGMA, KAPPA = 0.1355, 0.622, 2.0 / 3.0, 0.41
CW1 = CB1 / KAPPA ** 2 + (1.0 + CB2) / SIGMA
CW2, CW3, CV1 = 0.3, 2.0, 7.1
CT3, CV2, CV3, CN1 = 1.2, 0.7, 0.9, 16.0


def wall_distance(mesh, walls):

    """Nodal distance to the walls (list of boundary edges) by Spalding's Poisson approximation, on the plane mesh."""

    disc = Discretization(mesh)
    layout = DofLayout([FieldSpec('phi')], disc.n_nodes)
    rows = layout.element_dofs(disc.conn, 'phi')
    K = layout.scatter_matrix(ops.diffusion(disc, 1.0), rows, rows).tolil()
    b = layout.scatter_vector(ops.source(disc, 1.0), rows)

    wall_nodes = np.unique(np.ravel(walls))
    for k in wall_nodes:
        K.rows[k], K.data[k] = [k], [1.0]
        b[k] = 0.0

    phi = spla.spsolve(K.tocsr(), b)
    # nodal gradient by patch recovery (second order also on the boundary, unlike a lumped projection)
    grad = np.linalg.norm(disc.recovery() @ disc.gradient(phi).reshape(-1, disc.dim), axis=1)
    d = np.sqrt(grad ** 2 + 2.0 * np.maximum(phi, 0.0)) - grad
    d[wall_nodes] = 0.0

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


def sa_law(nu):

    """Pointwise law of the Spalart-Allmaras equation for GradientLawTerm.

    Inputs (per Gauss point): nu~, grad nu~ (2), u (2), grad u (4, row by row), d, grad d (2).
    Output for nu~: source s (tested by N) and flux F (2, tested by grad N), with the residual
    integral N s + grad N . F:  F = (nu + nu~ f_n) grad nu~ / sigma,  s = -cb2 |grad nu~|^2 / sigma - P + D.
    """

    def law(w):
        nut, g_nut = w[0], w[1:3]
        G = w[5:9].reshape(2, 2)                       # G_ab = du_a/dx_b
        d = jnp.maximum(w[9], 1e-12)

        vort = jnp.sqrt((G[1, 0] - G[0, 1]) ** 2 + 1e-30)
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

        if getattr(model, 'axisymmetric', False):
            raise NotImplementedError("Spalart-Allmaras solo esta implementado en 2D plano.")

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
        self.add_convection(system, layout, disc, 'nut', k=self.nu)

        law = cached_law(('spalart-allmaras', self.nu), lambda: PointwiseLaw(sa_law(self.nu)))
        system.add_gradient_law(law, ['nut', 'u', 'wall_distance'], ['nut'])
