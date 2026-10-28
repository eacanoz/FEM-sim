"""Weak-form operators for scalar fields, evaluated on all elements (or facets) at once.

Every operator takes a Discretization (or FacetSet) and coefficients given at the quadrature
points, and returns the element arrays that DofLayout then scatters into the global system:

    diffusion(disc, D)          (nel, nne, nne)    integral of D grad(N_i) . grad(N_j)
    diffusion_tensor(disc, D)   (nel, nne, nne)    same with a tensor D (dim, dim) or (nel, ngp, dim, dim)
    convection(disc, c, v)      (nel, nne, nne)    integral of N_i c v . grad(N_j)
    mass(disc, c)               (nel, nne, nne)    integral of c N_i N_j
    source(disc, f)             (nel, nne)         integral of f N_i
    facet_mass(facets, c)       (nf, nnf, nnf)     boundary integral of c N_i N_j
    facet_source(facets, f)     (nf, nnf)          boundary integral of f N_i

A coefficient is a number, an array already at the quadrature points, or a function of the
coordinates f(x), f(x, y), ... that is evaluated at the quadrature points.

Extension points (not implemented yet): vector fields (B^T C B for solids, divergence-pressure
coupling for fluids) follow the same pattern with `ncomp > 1` blocks.
"""

import numpy as np


def _coordinates(points):
    return [points[..., a] for a in range(points.shape[-1])]


def at_gauss(disc, c):

    """Coefficient at the Gauss points: (nel, ngp)."""

    if callable(c):
        c = c(*_coordinates(disc.x_gp))

    return np.broadcast_to(np.asarray(c, dtype=float), disc.detJw.shape)


def at_facets(facets, c):

    """Coefficient at the facet quadrature points: (nf, nq)."""

    if callable(c):
        c = c(*_coordinates(facets.x))

    return np.broadcast_to(np.asarray(c, dtype=float), facets.detJw.shape)


def diffusion(disc, D):

    return np.einsum('eg,egia,egja->eij', disc.detJw * at_gauss(disc, D), disc.B, disc.B)


def diffusion_tensor(disc, D):

    D = np.asarray(D, dtype=float)

    if D.ndim == 2:
        D = np.broadcast_to(D, disc.detJw.shape + D.shape)

    return np.einsum('eg,egia,egab,egjb->eij', disc.detJw, disc.B, D, disc.B)


def convection(disc, c, v):

    """v: scalar (1D), constant vector (dim,), array (nel, ngp, dim) or function returning the components."""

    if callable(v):
        v = np.stack(np.broadcast_arrays(*v(*_coordinates(disc.x_gp))), axis=-1)

    v = np.asarray(v, dtype=float)

    if v.ndim == 0:
        v = v.reshape(1)

    if v.ndim == 1:
        if v.shape[0] != disc.dim:
            raise ValueError(f"La velocidad tiene {v.shape[0]} componentes y la malla es {disc.dim}D.")
        v = np.broadcast_to(v, disc.detJw.shape + (disc.dim,))

    return np.einsum('eg,gi,egja,ega->eij', disc.detJw * at_gauss(disc, c), disc.N, disc.B, v)


def mass(disc, c):

    return np.einsum('eg,gi,gj->eij', disc.detJw * at_gauss(disc, c), disc.N, disc.N)


def source(disc, f):

    return np.einsum('eg,gi->ei', disc.detJw * at_gauss(disc, f), disc.N)


def facet_mass(facets, c):

    return np.einsum('fq,qi,qj->fij', facets.detJw * at_facets(facets, c), facets.N, facets.N)


def facet_source(facets, f):

    return np.einsum('fq,qi->fi', facets.detJw * at_facets(facets, f), facets.N)
