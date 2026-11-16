"""Compiled residual and tangent of FieldConvectionTerm (docs/PLAN_RENDIMIENTO_GPU_ML.md, phase B).

The same expressions as FieldConvectionTerm.contribution, written once with jax.numpy; the tangent is the automatic
derivative of the element residual with respect to the element values of the field and of the velocity. tau comes from
numpy (Operators.supg_tau at the state of the term) and enters as data, so it stays out of the derivative, like the
residual of the old time level. The numpy path of the term is the reference (FieldConvectionTerm.compiled = False):
Test/test_flow_kernel.py compares both.
"""

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def _residual(N, B, w, Phi, U, c, tau, k_lap_N, source, mt, alpha, beta):

    """Element residuals (nel, nne) of integral N_i c u . grad(phi) + integral tau (c u . grad N_i) res, with
    res = c u . grad(phi) - k lap(phi) + mt (alpha phi + beta) - source (source: f and the old level, already weighted)."""

    u = jnp.einsum('gj,ejc->egc', N, U)
    gphi = jnp.einsum('egja,ej->ega', B, Phi)
    a = c[..., None] * u
    conv = jnp.einsum('ega,ega->eg', a, gphi)
    aB = jnp.einsum('egb,egjb->egj', a, B)                                    # c u . grad N_j

    res = conv - jnp.einsum('egj,ej->eg', k_lap_N, Phi) + mt * jnp.einsum('gj,ej->eg', N, alpha * Phi + beta) - source

    return jnp.einsum('eg,gi,eg->ei', w, N, conv) + jnp.einsum('eg,egi,eg->ei', w * tau, aB, res)


def _element(Phi, U, N, B, w, c, tau, k_lap_N, source, mt, alpha, beta):
    value = _residual(N, B[None], w[None], Phi[None], U[None], c[None], tau[None], k_lap_N[None], source[None], mt[None],
                      alpha, beta[None])[0]
    return value, value


residual = jax.jit(_residual)
tangent = jax.jit(jax.vmap(jax.jacfwd(_element, argnums=(0, 1), has_aux=True),
                           in_axes=(0, 0, None, 0, 0, 0, 0, 0, 0, 0, None, 0)))


class CompiledFieldConvection:

    """contribution(x, tangent) of a FieldConvectionTerm, as its numpy path gives it."""

    def __init__(self, term):

        import Source.Simulation.Operators as ops

        self.term = term
        disc, layout = term.system.disc, term.system.layout
        self.shape = disc.B.shape
        self.N, self.B, self.w = jnp.asarray(disc.N), jnp.asarray(disc.B), jnp.asarray(disc.detJw)
        self.rows = layout.element_dofs(disc.conn, term.field)
        self.cols_u = layout.element_dofs(disc.conn, term.velocity)
        self.c = np.array(ops.at_gauss(disc, term.c), dtype=float)
        self.f = np.array(ops.at_gauss(disc, term.f), dtype=float) if term.f is not None else np.zeros(disc.detJw.shape)
        self.m = np.array(ops.at_gauss(disc, term.m), dtype=float)
        laplacian = disc.laplacian if term.k is not None and term.k_in_residual else None
        self.k_lap_N = np.zeros(self.shape[:3]) if laplacian is None else ops.at_gauss(disc, term.k)[..., None] * laplacian
        self.zeros = np.zeros(disc.detJw.shape)
        self.ops = ops

    def __call__(self, x, with_tangent):

        term, system = self.term, self.term.system
        disc, layout = system.disc, system.layout
        nel, ngp, nne, dim = self.shape
        x = np.asarray(x, dtype=float)

        Phi, U = x[self.rows], x[self.cols_u].reshape(nel, nne, dim)

        time = system.time
        tau, k_lap_N, source, mt, alpha, beta = self.zeros, np.zeros_like(self.k_lap_N), self.zeros, self.zeros, 0.0, np.zeros_like(Phi)

        if term.supg and not (time is not None and time.get('old_level', False)):
            state = x if term.tau_state is None else np.asarray(term.tau_state, dtype=float)
            u_tau = np.matmul(disc.N, state[self.cols_u].reshape(nel, nne, dim))
            tau = self.ops.supg_tau(disc, term.c, u_tau, term.k)
            k_lap_N, source = self.k_lap_N, self.f

            if time is not None:
                th = time['theta']
                if th < 1.0:
                    source = source - (1.0 - th) / th * term._old_space_residual(time)
                if np.any(self.m != 0):
                    mt, alpha, beta = self.m / th, float(time['alpha']), np.asarray(time['beta'])[self.rows]

        arguments = (self.c, tau, k_lap_N, source, mt, alpha, beta)

        if not with_tangent:
            Re = np.asarray(residual(self.N, self.B, self.w, Phi, U, *arguments))
            return np.bincount(self.rows.ravel(), weights=Re.ravel(), minlength=layout.ndof), None

        (Kpp, Kpu), Re = tangent(Phi, U, self.N, self.B, self.w, *arguments)
        R = np.bincount(self.rows.ravel(), weights=np.asarray(Re).ravel(), minlength=layout.ndof)
        J = (layout.scatter_matrix(np.asarray(Kpp), self.rows, self.rows)
             + layout.scatter_matrix(np.asarray(Kpu).reshape(nel, nne, nne * dim), self.rows, self.cols_u))

        return R, J
