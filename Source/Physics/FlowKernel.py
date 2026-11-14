"""Compiled residual of FlowTerm (docs/PLAN_RENDIMIENTO_GPU_ML.md, phase B).

The residual of the flow without its tangent is what a modified Newton evaluates most (transient studies: about ten
residuals per tangent). FlowTerm.contribution builds it with numpy, array by array; here the same expressions are written
with jax.numpy and compiled once per configuration of the term, which fuses them into a few loops over the Gauss points.

Only the local part is compiled. What couples elements stays in scipy, as in FlowTerm: the divergence of the recovered
stress (V @ x for a constant viscosity, Q @ sigma for a viscosity law, whose sigma comes from the compiled `stress`) enters
the kernel as data at the Gauss points, like the residual of the old time level. The scatter to the global vector is a
bincount.

The numpy path of FlowTerm.contribution(x, tangent=False) is the reference: Test/test_flow_kernel.py compares both.
"""

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def _mu(flags, law, g2, Fv, mu_const):

    """Viscosity at the Gauss points (nel, ngp): the law of (gamma2, fields), point by point as PointwiseLaw does, or the
    constant one."""

    if law is None:
        return mu_const

    shape = g2.shape
    n_fields = flags['n_fields']
    if n_fields == 0:
        inputs = jnp.stack([g2.ravel(), jnp.zeros(g2.size)], axis=-1)
        values = jax.vmap(lambda w: law(w[0], w[1]))(inputs)
    else:
        inputs = jnp.concatenate([g2.reshape(-1, 1), Fv.reshape(-1, n_fields)], axis=-1)
        values = jax.vmap(lambda w: law(w[0], *[w[1 + i] for i in range(n_fields)]))(inputs)

    return values.reshape(shape)


def _kinematics(flags, geo, U):

    """u (nel, ngp, dim), G_ab = du_a/dx_b, eps, the hoop strain and gamma2 = 2 eps:eps."""

    u = jnp.einsum('gj,ejc->egc', geo['N'], U)
    G = jnp.einsum('egjb,eja->egab', geo['B'], U)
    eps = 0.5 * (G + jnp.swapaxes(G, -1, -2))
    e_hh = u[..., 0] / geo['radius'] if flags['axi'] else jnp.zeros(u.shape[:2])
    g2 = 2.0 * (jnp.einsum('egab,egab->eg', eps, eps) + e_hh ** 2)

    return u, G, eps, e_hh, g2


def build(flags, law):

    """(stress, residual) compiled for a term with these flags (a dict of Python values: they choose the code)."""

    axi, dim = flags['axi'], flags['dim']

    def stress(geo, U, Fv, mu_const):

        """sigma = 2 mu eps at the Gauss points (nel, ngp, dim, dim) and, on an axisymmetric mesh, the terms of
        div(sigma) without derivatives, (sigma_rr - sigma_tt) / r and sigma_rz / r (zeros otherwise)."""

        u, G, eps, e_hh, g2 = _kinematics(flags, geo, U)
        mu = _mu(flags, law, g2, jnp.einsum('gj,ejk->egk', geo['N'], Fv), mu_const)
        sigma = 2.0 * mu[..., None, None] * eps

        if axi:
            r = geo['radius']
            extra = jnp.stack([(sigma[..., 0, 0] - 2.0 * mu * e_hh) / r, sigma[..., 1, 0] / r], axis=-1)
        else:
            extra = jnp.zeros(u.shape)

        return sigma, extra

    def residual(geo, U, P, Fv, phi, Ut, divs, old, f, mu_const, rho, c_s, c_p, buoyancy):

        """Element residuals: momentum (nel, nne, dim), PSPG (nel, nne) and heating (nel, nne).

        U, P: element velocities and pressures; Fv (nel, nne, n_fields): fields of the viscosity law; phi (nel, nne, nb):
        fields of the buoyancy; Ut: element values of the time derivative of u; divs, old, f (nel, ngp, dim): divergence
        of the recovered stress, space residual of the old level and body force; c_s, c_p: weights of the stabilization
        residual (time, grad p, space, old) in the momentum and PSPG rows; buoyancy (nb, dim + 1): c and the reference."""

        N, B, w = geo['N'], geo['B'], geo['w']

        u, G, eps, e_hh, g2 = _kinematics(flags, geo, U)
        gp = jnp.einsum('egjb,ej->egb', B, P)
        mu = _mu(flags, law, g2, jnp.einsum('gj,ejk->egk', N, Fv), mu_const)

        speed = jnp.linalg.norm(u, axis=-1) if flags['inertia'] else jnp.zeros(w.shape)
        h_tau = flags['tau_scale'] * geo['h'][:, None]
        tau = 1.0 / jnp.sqrt((12.0 * mu / h_tau ** 2) ** 2 + (2.0 * rho * speed / h_tau) ** 2)

        Ru = jnp.zeros(U.shape)

        if law is not None:
            sigma = 2.0 * mu[..., None, None] * eps
            Ru = Ru + jnp.einsum('eg,egib,egab->eia', w, B, sigma)
            if axi:
                Ru = Ru.at[:, :, 0].add(jnp.einsum('eg,egi,eg->ei', w, geo['hoop'], 2.0 * mu * e_hh))

        RM_space = -divs - f

        for k in range(flags['n_buoyancy']):
            fb = (jnp.einsum('gj,ej->eg', N, phi[..., k]) - buoyancy[k, dim])[..., None] * buoyancy[k, :dim]
            RM_space = RM_space - fb
            Ru = Ru - jnp.einsum('eg,gi,ega->eia', w, N, fb)

        if flags['inertia']:
            conv = rho[..., None] * jnp.einsum('egb,egab->ega', u, G)
            RM_space = RM_space + conv
            uB = jnp.einsum('egb,egib->egi', u, B)

        u_t = jnp.einsum('gj,ejc->egc', N, Ut) if flags['time'] else jnp.zeros(u.shape)

        def stabilized(c):
            return c[0] * rho[..., None] * u_t + c[1] * gp + c[2] * RM_space + c[3] * old

        if flags['inertia']:
            h = geo['h'][:, None]
            tau_c = 0.5 * rho * speed * h * jnp.minimum(rho * speed * h / (2.0 * mu) / 3.0, 1.0)
            divu = jnp.einsum('egaa->eg', G) + e_hh
            divB = B.at[..., 0].add(geo['hoop']) if axi else B

            Ru = Ru + jnp.einsum('eg,gi,ega->eia', w, N, conv) + jnp.einsum('eg,egia->eia', w * tau_c * divu, divB)
            if flags['supg']:
                Ru = Ru + jnp.einsum('eg,egi,ega->eia', w * tau * rho * flags['supg_weight'], uB, stabilized(c_s))

        if flags['pspg']:
            Rp = jnp.einsum('eg,egia,ega->ei', w * tau, B, stabilized(c_p))
        else:
            Rp = jnp.zeros(P.shape)

        if flags['heating']:
            RH = -jnp.einsum('eg,gi,eg->ei', w, N, mu * g2)
        else:
            RH = jnp.zeros(P.shape)

        return Ru, Rp, RH

    return jax.jit(stress), jax.jit(residual)


_COMPILED = {}


class CompiledFlowResidual:

    """Residual of a FlowTerm without its tangent, R (ndof,), the same as term.contribution(x, tangent=False)[0]."""

    def __init__(self, term):

        self.term = term
        system = term.system
        disc, layout = system.disc, system.layout
        nel, ngp, nne, dim = disc.B.shape
        self.shape = (nel, ngp, nne, dim)

        import Source.Simulation.Operators as ops

        axi = bool(disc.axisymmetric)
        self.geo = {'N': jnp.asarray(disc.N), 'B': jnp.asarray(disc.B), 'w': jnp.asarray(disc.detJw),
                    'h': jnp.asarray(ops.element_size(disc))}
        if axi:
            self.geo['radius'] = jnp.asarray(disc.radius)
            self.geo['hoop'] = jnp.asarray(disc.hoop)

        self.ru = layout.element_dofs(disc.conn, 'u')
        self.rp = layout.element_dofs(disc.conn, 'p')
        self.rF = [layout.element_dofs(disc.conn, name) for name in term.fields]
        self.r_phi = [layout.element_dofs(disc.conn, name) for name, _, _ in term.buoyancy]
        self.rH = None if term.heating is None else layout.element_dofs(disc.conn, term.heating)
        self.ndof = layout.ndof

        self.mu_const = jnp.asarray(np.array(ops.at_gauss(disc, term.mu_const), dtype=float)) if term.law is None \
            else jnp.zeros(disc.detJw.shape)
        self.rho = jnp.asarray(np.array(ops.at_gauss(disc, term.rho), dtype=float))
        self.f = jnp.zeros((nel, ngp, dim)) if term.f is None else jnp.asarray(np.broadcast_to(term.f, (nel, ngp, dim)))
        self.buoyancy = jnp.asarray(np.array([np.r_[c, ref] for _, c, ref in term.buoyancy], dtype=float).reshape(-1, dim + 1))
        self.zeros = jnp.zeros((nel, ngp, dim))
        self.no_rate = jnp.zeros((nel, nne, dim))
        self.no_fields = jnp.zeros((nel, nne, 0))

        self.law = term.viscosity_law

    def _functions(self, time, old_level):

        term = self.term
        flags = {'axi': 'radius' in self.geo, 'dim': self.shape[3], 'n_fields': len(term.fields),
                 'inertia': bool(term.inertia), 'supg': bool(term.supg), 'pspg': bool(term.pspg),
                 'heating': term.heating is not None, 'n_buoyancy': len(term.buoyancy), 'tau_scale': float(term.tau_scale),
                 'time': time, 'supg_weight': 0.0 if old_level else 1.0}
        key = (id(self.law), tuple(sorted(flags.items())))

        if key not in _COMPILED:
            _COMPILED[key] = build(flags, self.law) + (self.law,)       # the law is kept: its id is part of the key

        return _COMPILED[key][:2]

    def __call__(self, x):

        term, system = self.term, self.term.system
        nel, ngp, nne, dim = self.shape
        x = np.asarray(x, dtype=float)

        time = system.time
        old_level = time is not None and time.get('old_level', False)
        if time is None or old_level:
            c_p = c_s = (0.0, 1.0, 1.0, 0.0)
        else:
            th = time['theta']
            c_p = (1.0, 1.0, th, 1.0 - th)
            c_s = tuple(c / th for c in c_p)

        stress, residual = self._functions(bool(c_p[0]), old_level)

        U = x[self.ru].reshape(nel, nne, dim)
        P = x[self.rp]
        Fv = np.stack([x[r] for r in self.rF], axis=-1) if self.rF else self.no_fields
        phi = np.stack([x[r] for r in self.r_phi], axis=-1) if self.r_phi else self.no_fields

        divs = self.zeros
        if term.law is not None:
            if term.Q is not None:
                sigma, extra = stress(self.geo, U, Fv, self.mu_const)
                divs = (term.Q @ np.asarray(sigma).ravel()).reshape(nel, ngp, dim)
                if 'radius' in self.geo:
                    divs = divs + np.asarray(extra)
        elif term.V is not None:
            divs = (term.V @ x).reshape(nel, ngp, dim)

        Ut, old = self.no_rate, self.zeros
        if c_p[0]:
            Ut = (time['alpha'] * x + time['beta'])[self.ru].reshape(nel, nne, dim)
            if c_p[3]:
                old = term._old_space_residual(time)

        Ru, Rp, RH = residual(self.geo, U, P, Fv, phi, Ut, divs, old, self.f, self.mu_const, self.rho,
                              jnp.asarray(c_s), jnp.asarray(c_p), self.buoyancy)

        R = np.bincount(self.ru.ravel(), weights=np.asarray(Ru).ravel(), minlength=self.ndof)
        if term.pspg:
            R += np.bincount(self.rp.ravel(), weights=np.asarray(Rp).ravel(), minlength=self.ndof)
        if self.rH is not None:
            R += np.bincount(self.rH.ravel(), weights=np.asarray(RH).ravel(), minlength=self.ndof)

        return R
