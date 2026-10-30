"""Nonlinear laws evaluated point by point, with derivatives from automatic differentiation.

A law is a pure JAX function f(u) of the local state u (a vector with the values of the input
fields at one point) returning one value or a vector of values (reaction rates, h(T) * (T - T_inf),
temperature-dependent conductivity, a stress from a strain, ...).

PointwiseLaw compiles `vmap(f)` and `vmap(jacfwd(f))` once and applies them to all the quadrature
points of the mesh in a single call, so there is no compilation per element. Keep the same law
object between assemblies (or use cached_law) so it is compiled only once.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np


class PointwiseLaw:

    def __init__(self, fn):

        self._value = jax.jit(jax.vmap(fn))
        self._jacobian = jax.jit(jax.vmap(jax.jacfwd(fn)))

    def __call__(self, U):

        """U (npts, n_in) -> values (npts, n_out) and derivatives (npts, n_out, n_in)."""

        U = jnp.asarray(U, dtype=float)
        npts, n_in = U.shape

        values = np.asarray(self._value(U)).reshape(npts, -1)
        derivatives = np.asarray(self._jacobian(U)).reshape(npts, values.shape[1], n_in)

        return values, derivatives

    def values(self, U):

        """U (npts, n_in) -> values (npts, n_out), without the derivatives."""

        U = jnp.asarray(U, dtype=float)

        return np.asarray(self._value(U)).reshape(U.shape[0], -1)


_LAWS = {}


def cached_law(key, factory):

    """Returns the law stored under `key` (hashable, e.g. a tuple with the user's functions), building it once."""

    law = _LAWS.get(key)

    if law is None:
        law = _LAWS[key] = factory()

    return law
