"""Compiled residual and tangent of FlowTerm (docs/PLAN_RENDIMIENTO_GPU_ML.md, phase B).

FlowTerm.contribution builds the residual and the tangent of the flow with numpy, array by array. Here the residual is
written once with jax.numpy and compiled per configuration of the term, which fuses it into a few loops over the Gauss
points; the tangent is its automatic derivative, element by element, with respect to the element values of the velocity,
the pressure and the fields of the viscosity law and of the buoyancy.

Only the local part is compiled. What couples elements stays in scipy, as in FlowTerm: the divergence of the recovered
stress (V @ x for a constant viscosity, Q @ sigma for a viscosity law, whose sigma and derivative come from the compiled
`stress`) enters the kernel as data at the Gauss points, like the residual of the old time level, and its part of the
tangent is -(weights of SUPG and PSPG) @ d(div sigma)/dx with the weights the kernel returns.

The tangent is the one of FlowTerm: exact except for tau and tau_c, which do not depend on the state that is derived
(stop_gradient, or the state of term.tau_state); 'picard' also keeps the convecting velocity out of the derivative.

The numpy path of FlowTerm.contribution is the reference: Test/test_flow_kernel.py compares both.
"""

import numpy as np
import scipy.sparse as sp

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

CHUNK_ENTRIES = 30_000_000      # entries of the element tangents evaluated at once (0.24 GB)


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

    """Compiled functions of a term with these flags (a dict of Python values: they choose the code):
    stress and residual on all the elements, and their derivatives element by element."""

    axi, dim = flags['axi'], flags['dim']
    frozen = jax.lax.stop_gradient

    def stress(geo, U, Fv, mu_const):

        """sigma = 2 mu eps at the Gauss points (nel, ngp, dim, dim)."""

        u, G, eps, e_hh, g2 = _kinematics(flags, geo, U)
        mu = _mu(flags, law, g2, jnp.einsum('gj,ejk->egk', geo['N'], Fv), mu_const)

        return 2.0 * mu[..., None, None] * eps

    def residual(geo, U, P, Fv, phi, U_tau, Fv_tau, alpha, beta, divs, old, f, mu_const, rho, c_s, c_p, buoyancy):

        """Element residuals, momentum (nel, nne, dim), PSPG (nel, nne) and heating (nel, nne), and the weights of
        div(sigma~) in them: SUPG (nel, ngp, nne), the same for every component, and PSPG (nel, ngp, nne, dim).

        U, P: element velocities and pressures; Fv (nel, nne, n_fields): fields of the viscosity law; phi (nel, nne, nb):
        fields of the buoyancy; U_tau, Fv_tau: the state of tau when it is not this one (flags['tau_state']); alpha, beta:
        u' = alpha u + beta (element values); divs, old, f (nel, ngp, dim): divergence of the recovered stress through
        the nodes, space residual of the old level and body force; c_s, c_p: weights of the stabilization residual (time,
        grad p, space, old) in the momentum and PSPG rows; buoyancy (nb, dim + 1): c and the reference."""

        N, B, w = geo['N'], geo['B'], geo['w']

        u, G, eps, e_hh, g2 = _kinematics(flags, geo, U)
        gp = jnp.einsum('egjb,ej->egb', B, P)
        mu = _mu(flags, law, g2, jnp.einsum('gj,ejk->egk', N, Fv), mu_const)

        # tau and tau_c: frozen in the tangent
        if flags['tau_state']:
            u_tau, _, _, _, g2_tau = _kinematics(flags, geo, U_tau)
            mu_tau = _mu(flags, law, g2_tau, jnp.einsum('gj,ejk->egk', N, Fv_tau), mu_const)
        else:
            u_tau, mu_tau = frozen(u), frozen(mu)
        speed = jnp.sqrt(jnp.sum(u_tau ** 2, axis=-1)) if flags['inertia'] else jnp.zeros(w.shape)
        h_tau = flags['tau_scale'] * geo['h'][:, None]
        tau = 1.0 / jnp.sqrt((12.0 * mu_tau / h_tau ** 2) ** 2 + (2.0 * rho * speed / h_tau) ** 2)

        Ru = jnp.zeros(U.shape)

        if law is not None:
            sigma = 2.0 * mu[..., None, None] * eps
            Ru = Ru + jnp.einsum('eg,egib,egab->eia', w, B, sigma)
            if axi:
                s_hh = 2.0 * mu * e_hh
                Ru = Ru.at[:, :, 0].add(jnp.einsum('eg,egi,eg->ei', w, geo['hoop'], s_hh))
                if flags['hoop_divergence']:          # the terms of div(sigma) without derivatives: local
                    r = geo['radius']
                    divs = divs + jnp.stack([(sigma[..., 0, 0] - s_hh) / r, sigma[..., 1, 0] / r], axis=-1)

        RM_space = -divs - f

        for k in range(flags['n_buoyancy']):
            fb = (jnp.einsum('gj,ej->eg', N, phi[..., k]) - buoyancy[k, dim])[..., None] * buoyancy[k, :dim]
            RM_space = RM_space - fb
            Ru = Ru - jnp.einsum('eg,gi,ega->eia', w, N, fb)

        if flags['inertia']:
            carrier = u if flags['newton'] else frozen(u)          # Picard: the convecting velocity is not derived
            conv = rho[..., None] * jnp.einsum('egb,egab->ega', carrier, G)
            RM_space = RM_space + conv
            uB = jnp.einsum('egb,egib->egi', carrier, B)

        if flags['time']:
            u_t = jnp.einsum('gj,ejc->egc', N, alpha * U + beta)
        else:
            u_t = jnp.zeros(u.shape)

        def stabilized(c):
            return c[0] * rho[..., None] * u_t + c[1] * gp + c[2] * RM_space + c[3] * old

        W_supg = jnp.zeros(w.shape + (U.shape[1],))
        if flags['inertia']:
            h = geo['h'][:, None]
            tau_c = 0.5 * rho * speed * h * jnp.minimum(rho * speed * h / (2.0 * mu_tau) / 3.0, 1.0)
            divu = jnp.einsum('egaa->eg', G) + e_hh
            divB = B.at[..., 0].add(geo['hoop']) if axi else B

            Ru = Ru + jnp.einsum('eg,gi,ega->eia', w, N, conv) + jnp.einsum('eg,egia->eia', w * tau_c * divu, divB)
            if flags['supg']:
                weight = (w * tau * rho * flags['supg_weight'])[..., None] * uB
                Ru = Ru + jnp.einsum('egi,ega->eia', weight, stabilized(c_s))
                W_supg = c_s[2] * weight

        if flags['pspg']:
            weight = (w * tau)[..., None, None] * B
            Rp = jnp.einsum('egia,ega->ei', weight, stabilized(c_p))
            W_pspg = c_p[2] * weight
        else:
            Rp = jnp.zeros(P.shape)
            W_pspg = jnp.zeros(B.shape)

        if flags['heating']:
            RH = -jnp.einsum('eg,gi,eg->ei', w, N, mu * g2)
        else:
            RH = jnp.zeros(P.shape)

        return (Ru, Rp, RH), (W_supg, W_pspg)

    # ----- one element: the same functions on a mesh of one element, to derive them -----

    def as_mesh(geo):
        return {name: value if name == 'N' else value[None] for name, value in geo.items()}

    geo_axes = {name: None if name == 'N' else 0 for name in ('N', 'B', 'w', 'h') + (('radius', 'hoop') if axi else ())}

    def element_residual(U, P, Fv, phi, geo, U_tau, Fv_tau, alpha, beta, divs, old, f, mu_const, rho, c_s, c_p, buoyancy):
        values, weights = residual(as_mesh(geo), U[None], P[None], Fv[None], phi[None], U_tau[None], Fv_tau[None], alpha,
                                   beta[None], divs[None], old[None], f[None], mu_const[None], rho[None], c_s, c_p, buoyancy)
        values = tuple(v[0] for v in values)
        return values, (values, tuple(v[0] for v in weights))

    def element_stress(U, Fv, geo, mu_const):
        return stress(as_mesh(geo), U[None], Fv[None], mu_const[None])[0]

    tangent = jax.vmap(jax.jacfwd(element_residual, argnums=(0, 1, 2, 3), has_aux=True),
                       in_axes=(0, 0, 0, 0, geo_axes, 0, 0, None, 0, 0, 0, 0, 0, 0, None, None, None))
    stress_tangent = jax.vmap(jax.jacfwd(element_stress, argnums=(0, 1)), in_axes=(0, 0, geo_axes, 0))

    return {'stress': jax.jit(stress), 'residual': jax.jit(residual), 'tangent': jax.jit(tangent),
            'stress_tangent': jax.jit(stress_tangent)}


_COMPILED = {}


class CompiledFlowResidual:

    """Residual and tangent of a FlowTerm: __call__(x) is term.contribution(x, tangent=False)[0] and tangent(x) is
    term.contribution(x), both as the numpy path of the term gives them."""

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
        self.zeros = np.zeros((nel, ngp, dim))
        self.no_rate = np.zeros((nel, nne, dim))
        self.no_fields = np.zeros((nel, nne, 0))

        self.law = term.viscosity_law

    # ----- the state of a call -----

    def _functions(self, time, old_level):

        term = self.term
        axi = 'radius' in self.geo
        flags = {'axi': axi, 'dim': self.shape[3], 'n_fields': len(term.fields),
                 'inertia': bool(term.inertia), 'supg': bool(term.supg), 'pspg': bool(term.pspg),
                 'newton': bool(term.newton), 'heating': term.heating is not None, 'n_buoyancy': len(term.buoyancy),
                 'tau_scale': float(term.tau_scale), 'tau_state': term.tau_state is not None,
                 'hoop_divergence': axi and term.law is not None and term.Q is not None,
                 'time': time, 'supg_weight': 0.0 if old_level else 1.0}
        key = (id(self.law), tuple(sorted(flags.items())))

        if key not in _COMPILED:
            _COMPILED[key] = (build(flags, self.law), self.law)         # the law is kept: its id is part of the key

        return _COMPILED[key][0]

    def _gather(self, x):
        nel, ngp, nne, dim = self.shape
        return (x[self.ru].reshape(nel, nne, dim), x[self.rp],
                np.stack([x[r] for r in self.rF], axis=-1) if self.rF else self.no_fields,
                np.stack([x[r] for r in self.r_phi], axis=-1) if self.r_phi else self.no_fields)

    def _arguments(self, x):

        """(compiled functions, arguments of the kernel after the geometry, element state) at x and the step of the system."""

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

        functions = self._functions(bool(c_p[0]), old_level)

        U, P, Fv, phi = self._gather(x)
        U_tau, Fv_tau = (U, Fv) if term.tau_state is None else self._gather(np.asarray(term.tau_state, dtype=float))[::2]

        divs = self.zeros
        if term.law is not None:
            if term.Q is not None:
                sigma = functions['stress'](self.geo, U, Fv, self.mu_const)
                divs = (term.Q @ np.asarray(sigma).ravel()).reshape(nel, ngp, dim)
        elif term.V is not None:
            divs = (term.V @ x).reshape(nel, ngp, dim)

        alpha, beta, old = 0.0, self.no_rate, self.zeros
        if c_p[0]:
            alpha, beta = float(time['alpha']), np.asarray(time['beta'])[self.ru].reshape(nel, nne, dim)
            if c_p[3]:
                old = term._old_space_residual(time)

        arguments = (U, P, Fv, phi, U_tau, Fv_tau, alpha, beta, divs, old, self.f, self.mu_const, self.rho,
                     jnp.asarray(c_s), jnp.asarray(c_p), self.buoyancy)

        return functions, arguments

    def _scatter(self, Ru, Rp, RH):

        R = np.bincount(self.ru.ravel(), weights=np.asarray(Ru).ravel(), minlength=self.ndof)
        if self.term.pspg:
            R += np.bincount(self.rp.ravel(), weights=np.asarray(Rp).ravel(), minlength=self.ndof)
        if self.rH is not None:
            R += np.bincount(self.rH.ravel(), weights=np.asarray(RH).ravel(), minlength=self.ndof)

        return R

    # ----- residual -----

    def __call__(self, x):

        functions, arguments = self._arguments(x)
        (Ru, Rp, RH), _ = functions['residual'](self.geo, *arguments)

        return self._scatter(Ru, Rp, RH)

    # ----- residual and tangent -----

    def _chunks(self, entries_per_element):

        nel = self.shape[0]
        n = max(1, -(-nel * entries_per_element // CHUNK_ENTRIES))
        size = -(-nel // n)
        return [slice(start, min(nel, start + size)) for start in range(0, nel, size)]

    @staticmethod
    def _slice(value, part, n):
        """The elements `part` of an argument given per element (leading axis n); the others pass whole."""
        return value[part] if getattr(value, 'ndim', 0) and value.shape[0] == n else value

    def tangent(self, x):

        term, layout = self.term, self.term.system.layout
        nel, ngp, nne, dim = self.shape
        functions, arguments = self._arguments(x)
        U, P, Fv, phi = arguments[:4]

        # Arguments by element: all but alpha (6) and the constants c_s, c_p, buoyancy (13, 14, 15)
        whole = (6, 13, 14, 15)
        per_element = {name: value for name, value in self.geo.items() if name != 'N'}

        n_rows = nne * (dim + 2)
        n_cols = nne * (dim + 1 + Fv.shape[-1] + phi.shape[-1])
        parts = []
        for part in self._chunks(n_rows * n_cols + ngp * nne * (dim + 1)):
            geo = dict({name: value[part] for name, value in per_element.items()}, N=self.geo['N'])
            args = [a if k in whole else a[part] for k, a in enumerate(arguments)]
            jacobians, (values, weights) = functions['tangent'](args[0], args[1], args[2], args[3], geo, *args[4:])
            parts.append(([[np.asarray(j) for j in row] for row in jacobians], [np.asarray(v) for v in values],
                          [np.asarray(v) for v in weights]))

        def joined(pick):
            return pick(parts[0]) if len(parts) == 1 else np.concatenate([pick(p) for p in parts])

        R = self._scatter(*(joined(lambda p, o=o: p[1][o]) for o in range(3)))

        rows = [(self.ru, nne * dim), (self.rp if term.pspg else None, nne), (self.rH, nne)]
        cols = [(self.ru, None), (self.rp, None)] + [(r, i) for i, r in enumerate(self.rF)] \
            + [(r, k) for k, r in enumerate(self.r_phi)]
        inputs = [0, 1] + [2] * len(self.rF) + [3] * len(self.r_phi)          # which derivative holds each column field

        blocks = {}                                                            # one matrix per pair of dof arrays
        for o, (r, nr) in enumerate(rows):
            if r is None:
                continue
            for (c, component), i in zip(cols, inputs):
                Ke = joined(lambda p, o=o, i=i: p[0][o][i])
                if component is not None:
                    Ke = Ke[..., component]
                if not np.any(Ke):
                    continue
                Ke = Ke.reshape(nel, nr, -1)
                key = (id(r), id(c))
                blocks[key] = (blocks[key][0] + Ke, r, c) if key in blocks else (Ke, r, c)

        J = sp.csr_matrix((layout.ndof, layout.ndof))
        for Ke, r, c in blocks.values():
            J = J + layout.scatter_matrix(Ke, r, c)

        # ----- nonlocal part: -(SUPG and PSPG weights) @ d(div sigma~)/dx -----
        d_divs = None
        if term.law is not None:
            if term.Q is not None:
                dsig, dsigF = [], []
                for part in self._chunks(ngp * dim * dim * nne * (dim + Fv.shape[-1])):
                    geo = dict({name: value[part] for name, value in per_element.items()}, N=self.geo['N'])
                    dU, dF = functions['stress_tangent'](U[part], Fv[part], geo, self.mu_const[part])
                    dsig.append(np.asarray(dU)), dsigF.append(np.asarray(dF))
                dsig, dsigF = np.concatenate(dsig), np.concatenate(dsigF)
                d_divs = term.Q @ term._stress_tangent(dsig, [dsigF[..., i] for i in range(len(self.rF))], self.ru, self.rF)
        elif term.V is not None:
            d_divs = term.V

        if d_divs is not None and (term.pspg or (term.inertia and term.supg)):
            columns = np.arange(nel * ngp).reshape(nel, ngp)[:, :, None] * dim + np.arange(dim)       # (nel, ngp, dim)
            W = sp.csr_matrix((layout.ndof, nel * ngp * dim))
            if term.inertia and term.supg:
                Ws = joined(lambda p: p[2][0])                                                  # (nel, ngp, nne)
                r = self.ru.reshape(nel, nne, dim)
                shape = (nel, ngp, nne, dim)
                W = W + sp.coo_matrix((np.broadcast_to(Ws[..., None], shape).ravel(),
                                       (np.broadcast_to(r[:, None], shape).ravel(),
                                        np.broadcast_to(columns[:, :, None, :], shape).ravel())), shape=W.shape).tocsr()
            if term.pspg:
                Wp = joined(lambda p: p[2][1])                                                  # (nel, ngp, nne, dim)
                W = W + sp.coo_matrix((Wp.ravel(), (np.broadcast_to(self.rp[:, None, :, None], Wp.shape).ravel(),
                                                    np.broadcast_to(columns[:, :, None, :], Wp.shape).ravel())),
                                      shape=W.shape).tocsr()
            J = J - W @ d_divs

        return R, J
