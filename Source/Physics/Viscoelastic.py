# Viscoelastic flow: polymer modes on top of the flow physics (Oldroyd-B, Giesekus), on Taylor-Hood elements.
#
#     sigma = -p I + 2 mu_s eps(u) + sum_m tau_m,     tau_m = (mu_m / lam_m) (c_m - I)
#     dc/dt + u . grad c = L c + c L^T - (c - I) / lam - (alpha / lam) (c - I)^2,     L_ab = du_a / dx_b
#
# with mu_s the viscosity of the flow physics (the solvent). alpha = 0 is Oldroyd-B; 0 < alpha <= 1/2 is Giesekus.
#
# The unknown of each mode is the symmetric square root of the conformation, c = b b (Balci, Thomases, Renardy & Doering
# 2011), which keeps c positive definite without eigenvalues, so it stays differentiable at rest (c = I):
#
#     db/dt + u . grad b = L b + b a + (1/2) f(c) b^-1,   a = (b L^T - L b) / tr(b),   f(c) = -(c - I)/lam - (alpha/lam) (c - I)^2
#
# a is the antisymmetric matrix that keeps b symmetric (a closed form in the plane). Axisymmetric: the (r, z) block is the
# same, and the hoop component is db_t/dt + u . grad b_t = (u_r / r) b_t + (1/2) f(c_t) / b_t.
#
# b is equal to a linear function in each element (System.restrict_to_linear_subspace), with SUPG on its equation:
#
#     R_b = integral (N + tau_s u . grad N) (u . grad b - rhs(b, L)) + tau_s (u . grad N) db/dt [+ M db/dt, by the solver]
#     R_u += integral tau : grad v  [+ integral (tau_tt / r) v_r, axisymmetric]
#
# Without a solvent the momentum equation loses its elliptic term. DEVSS-G (setDEVSS) restores it consistently: a continuous
# projection G of the velocity gradient, linear in each element, replaces grad u in the equation of b, and
#
#     integral psi : (G - grad u) = 0,      R_u += integral mu_a (grad u + grad u^T - G - G^T) : grad v
#
# The added term vanishes as G -> grad u, so it does not change the solution in the limit of a fine mesh.
#
# At high Wi the equation of b has no diffusion and sharp layers form next to a reentrant corner. setDiffusion adds a small
# artificial one (Sureshkumar & Beris 1995), R_b += integral kappa grad N . grad b, with kappa = c h^2 (1 / lam + gamma_dot) frozen in each
# iteration: it scales with the mesh, so it vanishes at second order. It changes the solution, so it is better kept to where it is
# needed (a function of the coordinates, 1 near the corner and 0 away from it): check it against c = 0 where possible.
#
# An indefinite b also squares to a conformation, and it is a spurious solution: the solvers reject any state whose b is not
# positive definite (System.require_admissible). Checking the nodes is enough: the cone is convex and b is linear in an element.
#
# The equation of b is hyperbolic: it takes values only at the inflow (addBC_InflowEdges). Its tangent is automatic, with
# tau_s frozen. An outlet of the flow physics fixes the total normal traction; addBC_OutletEdges of this physics takes the
# polymer stress out of it, so that the given pressure is the pressure of a developed flow.

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from Source.Physics.Physics import physics
from Source.Physics.Rheology import ViscosityLaw
from Source.Primals.Vector import vectorField
from Source.Simulation.PointwiseLaw import PointwiseLaw, cached_law
from Source.Simulation.System import GradientLawTerm
import Source.Simulation.Operators as ops
from Source.enums import ShapeFunctionType


def _plane(b):
    return jnp.array([[b[0], b[1]], [b[1], b[2]]])


def conformation_rate(b, L, hoop, lam, alpha):

    """The right-hand side of the equation of b: (xx, xy, yy) plus (tt) if b has four components. L is 2 x 2 and hoop = u_r / r."""

    B = _plane(b)
    I = jnp.eye(2)
    det = b[0] * b[2] - b[1] * b[1]
    inverse = jnp.array([[b[2], -b[1]], [-b[1], b[0]]]) / det

    a = (B @ L.T - L @ B) / (b[0] + b[2])
    E = B @ B - I
    rate = L @ B + B @ a - 0.5 / lam * (E + alpha * E @ E) @ inverse
    plane = jnp.array([rate[0, 0], 0.5 * (rate[0, 1] + rate[1, 0]), rate[1, 1]])

    if b.shape[0] == 3:
        return plane

    e = b[3] * b[3] - 1.0
    return jnp.concatenate([plane, jnp.array([hoop * b[3] - 0.5 / lam * (e + alpha * e * e) / b[3]])])


def conformation_law(mu, lam, alpha, nc, devss=False):

    """Pointwise law of a mode for GradientLawTerm.

    Inputs: b (nc), grad b (nc x 2), u (2), grad u (4), with DEVSS-G also G (4) and grad G (8), then the data at the Gauss
    points: tau_s, db/dt (nc), r, kappa. Outputs: for b a source (nc) and a flux (nc x 2), for u a source (2) and a flux (4).
    """

    data = 3 * nc + 6 + (12 if devss else 0)

    def law(w):
        b, grad_b = w[:nc], w[nc:3 * nc].reshape(nc, 2)
        u, L = w[3 * nc:3 * nc + 2], w[3 * nc + 2:3 * nc + 6].reshape(2, 2)
        if devss:
            L = w[3 * nc + 6:3 * nc + 10].reshape(2, 2)
        tau_s, b_t, r, kappa = w[data], w[data + 1:data + 1 + nc], w[data + 1 + nc], w[data + 2 + nc]

        residual = grad_b @ u - conformation_rate(b, L, u[0] / r, lam, alpha)
        supg = tau_s * jnp.outer(residual + b_t, u) + kappa * grad_b

        B = _plane(b)
        stress = mu / lam * (B @ B - jnp.eye(2))
        hoop_force = mu / lam * (b[3] * b[3] - 1.0) / r if nc == 4 else 0.0

        return jnp.concatenate([residual, supg.ravel(), jnp.array([hoop_force, 0.0]), stress.ravel()])

    return law


class SpectrumViscosity(ViscosityLaw):

    """The steady shear viscosity of a spectrum of modes plus a solvent, as an inelastic law: the same flow curve without elasticity.
    Closed form of Giesekus (Bird, Armstrong & Hassager), with its limit eta = mu for Oldroyd-B, written without the differences
    sqrt(1 + 2 s) - 1 and 1 - chi: at low rates they vanish by round-off, and the derivative of the law would be infinite."""

    def __init__(self, modes, solvent=0.0):
        self.modes, self.solvent = list(modes), float(solvent)

    def __call__(self, gamma2, T):
        eta = self.solvent
        for mu, lam, alpha in self.modes:
            if alpha == 0.0:
                eta = eta + mu
                continue
            s = 8.0 * alpha * (1.0 - alpha) * lam * lam * gamma2
            root = jnp.sqrt(1.0 + 2.0 * s)
            chi = jnp.sqrt(2.0 / (1.0 + root))
            f = 2.0 * s / (1.0 + root) ** 2 / (1.0 + chi) / (1.0 + (1.0 - 2.0 * alpha) * chi)
            eta = eta + mu * (1.0 - f) ** 2 / (1.0 + (1.0 - 2.0 * alpha) * f)
        return eta


class ConformationTerm(GradientLawTerm):

    """The equation of b of one mode and its stress in the momentum equation, all through its pointwise law."""

    def __init__(self, system, law, field, nc, lam, devss=False, diffusion=0.0, where=None):
        super().__init__(system, law, [field, 'u', 'G'] if devss else [field, 'u'], [field, 'u'])
        self.field, self.nc, self.lam, self.diffusion = field, nc, lam, diffusion
        self._where = 1.0 if where is None else ops.at_gauss(system.disc, where)
        self.tau_state = None        # evaluate tau_s at this state instead of x (to test the tangent)
        self._h = 0.5 * ops.element_size(system.disc)[:, None]               # the spacing of the nodes of the element

    def contribution(self, x, tangent=True):

        system, disc, layout = self.system, self.system.disc, self.system.layout
        nel, ngp, nne, _ = disc.B.shape
        nc = self.nc

        time = system.time
        if time is not None and not time.get('old_level', False) and time['theta'] < 1.0:
            raise ValueError("El flujo viscoelastico no admite Crank-Nicolson: use timeScheme='BDF2' o Euler implicito.")

        fields = layout.split(x)
        u = disc.interpolate(np.asarray(fields['u'], dtype=float))
        speed_field = fields['u'] if self.tau_state is None else layout.split(self.tau_state)['u']
        speed_u = disc.interpolate(np.asarray(speed_field, dtype=float))
        tau_s = 1.0 / np.sqrt((2.0 * np.linalg.norm(speed_u, axis=-1) / self._h) ** 2 + (2.0 / self.lam) ** 2)

        b_t = np.zeros((nel, ngp, nc))
        if time is not None:
            b_t = disc.interpolate(np.asarray(layout.split(time['alpha'] * x + time['beta'])[self.field], dtype=float))

        r = disc.radius if disc.axisymmetric else np.ones(disc.detJw.shape)

        kappa = np.zeros(disc.detJw.shape)
        if self.diffusion:
            G = disc.gradient(np.asarray(speed_field, dtype=float))
            E = 0.5 * (G + np.swapaxes(G, -1, -2))
            kappa = self.diffusion * self._where * self._h ** 2 * (1.0 / self.lam + np.sqrt(2.0 * np.einsum('egab,egab->eg', E, E)))

        self.gauss_inputs = np.concatenate([tau_s[..., None], b_t, r[..., None], kappa[..., None]], axis=-1)

        R, J = super().contribution(x, tangent)

        if tangent and time is not None:
            # db/dt in the SUPG flux: tau_s (u . grad N_i) alpha N_j, on each component
            K = np.einsum('eg,egib,egb,gj->eij', disc.detJw * tau_s * time['alpha'], disc.B, u, disc.N)
            K = np.einsum('eij,ac->eiajc', K, np.eye(nc)).reshape(nel, nne * nc, nne * nc)
            rows = layout.element_dofs(disc.conn, self.field)
            J = J + layout.scatter_matrix(K, rows, rows)

        return R, J


class PolymerTractionTerm:

    """R_u -= boundary integral of (tau . n) . v: takes the polymer stress out of the natural traction of an outlet."""

    def __init__(self, system, facets, field, mu, lam):
        self.system, self.facets, self.field = system, facets, field

        def traction(w):
            b, normal = w[:-2], w[-2:]
            B = _plane(b)
            return mu / lam * (B @ B - jnp.eye(2)) @ normal

        self.law = cached_law(('polymer traction', mu, lam), lambda: PointwiseLaw(traction))

    def contribution(self, x, tangent=True):

        layout, fs = self.system.layout, self.facets
        nc = layout.spec(self.field).ncomp
        nq, nnf = fs.N.shape

        values = np.einsum('qk,fkc->fqc', fs.N, layout.split(x)[self.field][fs.nodes])
        data = np.concatenate([values, fs.normals], axis=-1).reshape(-1, nc + 2)
        ru = layout.element_dofs(fs.nodes, 'u')

        if not tangent:
            t = self.law.values(data).reshape(len(fs.nodes), nq, 2)
            return layout.scatter_vector(-np.einsum('fq,qi,fqa->fia', fs.detJw, fs.N, t).reshape(len(fs.nodes), -1), ru), None

        t, dt = self.law(data)
        t = t.reshape(len(fs.nodes), nq, 2)
        dt = dt.reshape(len(fs.nodes), nq, 2, nc + 2)[..., :nc]
        R = layout.scatter_vector(-np.einsum('fq,qi,fqa->fia', fs.detJw, fs.N, t).reshape(len(fs.nodes), -1), ru)
        K = -np.einsum('fq,qi,fqac,qj->fiajc', fs.detJw, fs.N, dt, fs.N).reshape(len(fs.nodes), nnf * 2, nnf * nc)

        return R, layout.scatter_matrix(K, ru, layout.element_dofs(fs.nodes, self.field))


class ve(physics):

    part_name = 'viscoelastic'
    ConstRel = 'Dc/Dt = L c + c L^T - (c - I)/lam - (alpha/lam) (c - I)^2'

    def __init__(self, model):
        super().__init__(model)

        self.physics_description = 'Viscoelastic polymer modes'
        self.axisymmetric = bool(getattr(model, 'axisymmetric', False))
        self.nc = 4 if self.axisymmetric else 3
        self.modes = []
        self.outlets = []
        self.inflows = []
        self.devss = 0.0
        self.diffusion = 0.0
        self.diffusion_where = None
        self._rest = np.array([1.0, 0.0, 1.0, 1.0])[:self.nc]

    def setModes(self, modes):

        """Polymer modes [(mu, lam, alpha), ...]: viscosity, relaxation time and Giesekus parameter (0: Oldroyd-B). Their
        unknowns, 'b0', 'b1', ..., start at rest. The solvent is the viscosity of the flow physics, which may be zero."""

        self.modes = [(float(mu), float(lam), float(alpha)) for mu, lam, alpha in modes]
        self._set_fields()

    def setDEVSS(self, mu_a):

        """Stabilization DEVSS-G with the viscosity mu_a, of the order of the polymer viscosity; 0 removes it. Necessary with
        little or no solvent."""

        self.devss = float(mu_a)
        self._set_fields()

    def setDiffusion(self, c, where=None):

        """Artificial diffusion in the equation of b, kappa = c h^2 (1 / lam + gamma_dot), of order c ~ 0.01 to 1. See the header.
        where: optional function of the coordinates that multiplies it, to keep it next to a corner."""

        self.diffusion = float(c)
        self.diffusion_where = where

    def _set_fields(self):

        mesh = self.modelRef.mesh
        self.var = {}

        for m in range(len(self.modes)):
            self.var[f'b{m}'] = vectorField(f'b{m}', 'Square root of the conformation', '-', ShapeFunctionType.linear, mesh,
                                            ncomp=self.nc)
            self.var[f'b{m}'].initField(self._rest)

        if self.devss:
            self.var['G'] = vectorField('G', 'Projected velocity gradient', '1/s', ShapeFunctionType.linear, mesh, ncomp=4)

        self.modelRef._layout = None

    def setWeissenberg(self, factor):

        """Multiplies the relaxation times: a continuation in Wi, keeping the state of the previous solve. The values at the
        inflows follow."""

        self.modes = [(mu, lam * factor, alpha) for mu, lam, alpha in self.modes]

        for edges, gradient in self.inflows:
            self._set_inflow(edges, gradient)

    def developed(self, L, hoop=0.0, mode=0):

        """The steady b of a homogeneous flow with velocity gradient L (2 x 2) and hoop strain rate, from the rest state."""

        mu, lam, alpha = self.modes[mode]
        L = jnp.asarray(L, dtype=float)
        rate = jax.jit(lambda b: conformation_rate(b, L, hoop, lam, alpha))
        solution = solve_ivp(lambda t, b: np.asarray(rate(jnp.asarray(b))), (0.0, 100.0 * lam), self._rest, method='LSODA',
                             rtol=1e-10, atol=1e-12)

        return solution.y[:, -1]

    def addBC_InflowEdges(self, edges, gradient):

        """Values of b at an inflow: the steady state of the velocity gradient there, gradient(x, y) -> L (2 x 2). On the
        vertices of the edges, for each mode. A function that returns zeros gives the rest state."""

        self.inflows.append((list(edges), gradient))
        self._set_inflow(edges, gradient)

    def _set_inflow(self, edges, gradient):

        nodes = self.modelRef._mesh.NL
        vertices = sorted({int(n) for edge in edges for n in (edge[0], edge[-1])})

        for m in range(len(self.modes)):
            for k in vertices:
                x, y = nodes[k].coor[:2]
                L = np.asarray(gradient(x, y), dtype=float)
                hoop = 0.0
                nodes[k].BC[f'b{m}'] = self.setDirichletBC(list(self.developed(L, hoop, m)))

    def addBC_OutletEdges(self, edges):

        """Outlet of a developed flow: use with the outlet of the flow physics, whose pressure then excludes the polymer stress."""

        self.outlets.append(list(edges))

    def stress(self, mode=0, values=None):

        """Nodal polymer stress of a mode: (xx, xy, yy[, tt])."""

        mu, lam, _ = self.modes[mode]
        b = self.var[f'b{mode}'].values if values is None else np.asarray(values, dtype=float)

        c = np.stack([b[:, 0] ** 2 + b[:, 1] ** 2, b[:, 1] * (b[:, 0] + b[:, 2]), b[:, 1] ** 2 + b[:, 2] ** 2]
                     + ([b[:, 3] ** 2] if self.nc == 4 else []), axis=1)
        rest = np.array([1.0, 0.0, 1.0, 1.0])[:self.nc]

        return mu / lam * (c - rest)

    def inelasticEquivalent(self, solvent=0.0):

        """An inelastic law with the steady shear viscosity of these modes: flow.setViscosityLaw(polymer.inelasticEquivalent(mu_s)),
        in a flow without this physics. What differs from the viscoelastic solution is due to elasticity."""

        return SpectrumViscosity(self.modes, solvent)

    def energy_balance(self, mode=0):

        """Balance of the free energy of a mode in a steady state: power of the stress = dissipation + what convection carries out.

        With psi = (mu / (2 lam)) (tr c - ln det c - n), Dpsi/Dt = tau : grad u - D, D = -(mu / (2 lam)) tr((I - c^-1) f(c)). Returns
        |power - D - flux| / |power|. An indicator of error, not a proof: it falls with the mesh and jumps at a wrong solution of a high
        Wi (Couette: from 1e-13 to 5e-5), but a good solution with 1 % of error on a coarse mesh gives the same value, so no threshold
        tells them apart. Watch it along the continuation and the refinement of the mesh."""

        mu, lam, alpha = self.modes[mode]
        disc = self.modelRef.geometry
        fields = self.modelRef.physics.var
        U = np.asarray(fields['u'].values, dtype=float)
        b = disc.interpolate(np.asarray(self.var[f'b{mode}'].values, dtype=float))
        L = disc.gradient(U)
        u = disc.interpolate(U)

        def psi_of(xx, xy, yy, tt=None):
            det = xx * yy - xy * xy
            c_xx, c_xy, c_yy = xx * xx + xy * xy, xy * (xx + yy), xy * xy + yy * yy
            psi = c_xx + c_yy - 2.0 * np.log(det) - 2.0
            if tt is not None:
                psi = psi + tt * tt - 2.0 * np.log(tt) - 1.0
            return mu / (2.0 * lam) * psi

        B = np.stack([np.stack([b[..., 0], b[..., 1]], axis=-1), np.stack([b[..., 1], b[..., 2]], axis=-1)], axis=-2)
        C = B @ B
        I = np.eye(2)
        E = C - I
        stress = mu / lam * E
        inverse = np.linalg.inv(C)
        f = -(E + alpha * E @ E) / lam
        power = np.einsum('egab,egab->eg', stress, L)
        dissipation = -mu / (2.0 * lam) * np.einsum('egab,egba->eg', I - inverse, f)

        nodal = np.asarray(self.var[f'b{mode}'].values, dtype=float)
        if self.nc == 4:
            hoop = u[..., 0] / disc.radius
            ct = b[..., 3] ** 2
            et = ct - 1.0
            power = power + mu / lam * et * hoop
            dissipation = dissipation + mu / (2.0 * lam) * (1.0 - 1.0 / ct) * (et + alpha * et * et) / lam
            psi = psi_of(nodal[:, 0], nodal[:, 1], nodal[:, 2], nodal[:, 3])
        else:
            psi = psi_of(nodal[:, 0], nodal[:, 1], nodal[:, 2])

        flux = np.einsum('ega,ega->eg', u, disc.gradient(psi))
        w = disc.detJw
        total = np.sum(w * power)

        return abs(total - np.sum(w * dissipation) - np.sum(w * flux)) / max(abs(total), 1e-30)

    def derived_fields(self, model):

        return {'polymer_stress': sum(self.stress(m) for m in range(len(self.modes)))} if self.modes else {}

    def _add_devss(self, system, disc, layout):

        """The linear terms of DEVSS-G: the projection and the term in the momentum equation."""

        nel, _, nne, _ = disc.B.shape
        w = disc.detJw
        ru, rG = layout.element_dofs(disc.conn, 'u'), layout.element_dofs(disc.conn, 'G')

        system.restrict_to_linear_subspace('G')
        system.require_constraint(layout.dofs('G').ravel())

        # G_ab = du_a / dx_b, stored row by row
        mass = np.einsum('eg,gi,gj,ac->eiajc', w, disc.N, disc.N, np.eye(4)).reshape(nel, nne * 4, nne * 4)
        D = np.zeros((nel, nne, 4, nne, 2))                                  # integral N_i dN_j/dx_b on the row (a, b), column (j, a)
        block = np.einsum('eg,gi,egjb->eijb', w, disc.N, disc.B)
        for a in range(2):
            for b in range(2):
                D[:, :, 2 * a + b, :, a] = block[..., b]

        A = layout.scatter_matrix(mass, rG, rG) - layout.scatter_matrix(D.reshape(nel, nne * 4, nne * 2), rG, ru)

        # mu_a (grad u + grad u^T - G - G^T) : grad v, tested by dN_i/dx_b on the component a
        K = np.einsum('eg,egib,egjb,ac->eiajc', w, disc.B, disc.B, np.eye(2)) + np.einsum('eg,egic,egja->eiajc', w, disc.B, disc.B)
        A = A + self.devss * layout.scatter_matrix(K.reshape(nel, nne * 2, nne * 2), ru, ru)

        C = np.zeros((nel, nne, 2, nne, 4))
        cross = np.einsum('eg,egib,gj->eibj', w, disc.B, disc.N)
        for a in range(2):
            for b in range(2):
                C[:, :, a, :, 2 * a + b] += cross[:, :, b, :]                # G_ab tested by dv_a/dx_b
                C[:, :, b, :, 2 * a + b] += cross[:, :, a, :]                # G_ab tested by dv_b/dx_a (the transpose)

        system.add_matrix(A - self.devss * layout.scatter_matrix(C.reshape(nel, nne * 2, nne * 4), ru, rG))

    def _positive_definite(self, layout, field):

        d = layout.dofs(field)

        def check(x):
            xx, xy, yy = x[d[:, 0]], x[d[:, 1]], x[d[:, 2]]
            hoop = x[d[:, 3]] if self.nc == 4 else 1.0
            return bool(np.all(xx + yy > 0.0) and np.all(xx * yy - xy * xy > 0.0) and np.all(hoop > 0.0))

        return check

    def add_terms(self, system, model, disc, layout):

        if not self.modes:
            raise ValueError("Defina los modos: model.physics.viscoelastic.setModes([(mu, lam, alpha), ...]).")
        if not disc.element.linear_subspace:
            raise ValueError("El flujo viscoelastico necesita Taylor-Hood: una malla cuadratica (order=2).")

        if self.devss:
            self._add_devss(system, disc, layout)

        for m, (mu, lam, alpha) in enumerate(self.modes):
            field = f'b{m}'
            system.restrict_to_linear_subspace(field)
            system.require_admissible(self._positive_definite(layout, field))
            rows = layout.element_dofs(disc.conn, field)
            system.add_mass(layout.scatter_matrix(ops.vector_mass(disc, 1.0, self.nc), rows, rows))

            devss = bool(self.devss)
            law = cached_law(('conformation', mu, lam, alpha, self.nc, devss),
                             lambda: PointwiseLaw(conformation_law(mu, lam, alpha, self.nc, devss)))
            system.add_term(ConformationTerm(system, law, field, self.nc, lam, devss, self.diffusion, self.diffusion_where))

            for edges in self.outlets:
                system.add_term(PolymerTractionTerm(system, disc.facets(edges), field, mu, lam))
