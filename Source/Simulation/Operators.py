"""Weak-form operators for scalar fields, evaluated on all elements (or facets) at once.

Every operator takes a Discretization (or FacetSet) and coefficients given at the quadrature
points, and returns the element arrays that DofLayout then scatters into the global system:

    diffusion(disc, D)          (nel, nne, nne)    integral of D grad(N_i) . grad(N_j)
    diffusion_tensor(disc, D)   (nel, nne, nne)    same with a tensor D (dim, dim) or (nel, ngp, dim, dim)
    convection(disc, c, v)      (nel, nne, nne)    integral of N_i c v . grad(N_j)
    supg_tau, supg_terms        SUPG stabilization of the convection (streamline-upwind Petrov-Galerkin)
    mass(disc, c)               (nel, nne, nne)    integral of c N_i N_j
    source(disc, f)             (nel, nne)         integral of f N_i
    facet_mass(facets, c)       (nf, nnf, nnf)     boundary integral of c N_i N_j
    facet_source(facets, f)     (nf, nnf)          boundary integral of f N_i
    gradient_source(disc, g)    (nel, nne)         integral of g . grad(N_i), g a vector

Vector fields (velocity, displacement) with nc components use node-major element dofs (i, a) -> i * nc + a,
the order of DofLayout.element_dofs:

    vector_mass(disc, c, nc)            (nel, nne*nc, nne*nc)   integral of c u . v
    vector_diffusion(disc, mu)          (nel, nne*dim, nne*dim) integral of mu grad(u) : grad(v)
    viscous(disc, mu)                   (nel, nne*dim, nne*dim) integral of 2 mu eps(u) : eps(v), eps = sym(grad)
    divergence(disc, c)                 (nel, nne, nne*dim)     integral of c N_i div(u)   (test: scalar, trial: vector)
    vector_source(disc, f)              (nel, nne*dim)          integral of f . v
    facet_vector_source(facets, t)      (nf, nnf*dim)           boundary integral of t . v (tractions)

The pressure term of the momentum equation, -integral of p div(v), is minus the transpose of `divergence`:
np.swapaxes(divergence(disc), 1, 2) has the shape (nel, nne*dim, nne) of a block (test: vector, trial: scalar).

A coefficient is a number, an array already at the quadrature points, or a function of the
coordinates f(x), f(x, y), ... that is evaluated at the quadrature points. A vector coefficient is a
constant vector (dim,), an array (n, nq, dim) or a function returning its components.
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


def _velocity(disc, v):

    """v: scalar (1D), constant vector (dim,), array (nel, ngp, dim) or function returning the components."""

    if callable(v):
        v = np.stack(np.broadcast_arrays(*v(*_coordinates(disc.x_gp))), axis=-1)

    v = np.asarray(v, dtype=float)

    if v.ndim == 0:
        v = v.reshape(1)

    if v.ndim == 1:
        if v.shape[0] != disc.dim:
            raise ValueError(f"El vector (velocidad, fuerza, gravedad) tiene {v.shape[0]} componentes y la malla es {disc.dim}D.")
        v = np.broadcast_to(v, disc.detJw.shape + (disc.dim,))

    return v


def convection(disc, c, v):

    """v: scalar (1D), constant vector (dim,), array (nel, ngp, dim) or function returning the components."""

    return np.einsum('eg,gi,egja,ega->eij', disc.detJw * at_gauss(disc, c), disc.N, disc.B, _velocity(disc, v))


def supg_tau(disc, c, v, k=None):

    """SUPG stabilization parameter tau (nel, ngp) for the advection a = c v, with the diffusion k (None: pure advection).

    tau = h xi(Pe) / (2 |a|), with the streamline length h = 2 |a| / sum_i |a . grad N_i| (Hughes), the element
    Peclet number Pe = |a| h / (2 k) and the optimal xi = coth(Pe) - 1/Pe. In 1D with constant coefficients this
    makes the steady solution nodally exact.
    """

    a = at_gauss(disc, c)[..., None] * _velocity(disc, v)
    amag = np.linalg.norm(a, axis=-1)

    s = np.abs(np.einsum('egia,ega->egi', disc.B, a)).sum(axis=-1)
    safe_a, safe_s = np.where(amag > 0, amag, 1.0), np.where(s > 0, s, 1.0)
    h = 2.0 * safe_a / safe_s

    if k is None:
        xi = np.ones_like(h)
    else:
        Pe = np.maximum(safe_a * h / (2.0 * at_gauss(disc, k)), 1e-12)
        xi = np.where(Pe < 1e-3, Pe / 3.0, 1.0 / np.tanh(np.minimum(Pe, 350.0)) - 1.0 / Pe)

    return np.where(amag > 0, h * xi / (2.0 * safe_a), 0.0)


def supg_terms(disc, c, v, tau, m=None, f=None):

    """SUPG matrices with the weight tau (a . grad N_i), a = c v: (stiffness, mass, source).

    stiffness_ij = integral tau (a . grad N_i)(a . grad N_j);  mass_ij = integral tau (a . grad N_i) m N_j;
    source_i = integral tau (a . grad N_i) f. m and f are None when the equation has no mass or source.
    """

    a = at_gauss(disc, c)[..., None] * _velocity(disc, v)
    wa = np.einsum('egia,ega->egi', disc.B, a)
    wt = disc.detJw * tau

    K = np.einsum('eg,egi,egj->eij', wt, wa, wa)
    M = None if m is None else np.einsum('eg,egi,gj->eij', wt * at_gauss(disc, m), wa, disc.N)
    F = None if f is None else np.einsum('eg,egi->ei', wt * at_gauss(disc, f), wa)

    return K, M, F


def mass(disc, c):

    return np.einsum('eg,gi,gj->eij', disc.detJw * at_gauss(disc, c), disc.N, disc.N)


def source(disc, f):

    return np.einsum('eg,gi->ei', disc.detJw * at_gauss(disc, f), disc.N)


def facet_mass(facets, c):

    return np.einsum('fq,qi,qj->fij', facets.detJw * at_facets(facets, c), facets.N, facets.N)


def facet_source(facets, f):

    return np.einsum('fq,qi->fi', facets.detJw * at_facets(facets, f), facets.N)


def gradient_source(disc, g):

    return np.einsum('eg,egia,ega->ei', disc.detJw, disc.B, _velocity(disc, g))


# ----- vector fields -----

def expand_components(K, nc):

    """Scalar element blocks (n, a, b) -> blocks (n, a*nc, b*nc) acting on every component alike (K ⊗ I)."""

    n, a, b = K.shape

    return np.einsum('eij,cd->eicjd', K, np.eye(nc)).reshape(n, a * nc, b * nc)


def vector_mass(disc, c, nc=None):

    return expand_components(mass(disc, c), disc.dim if nc is None else nc)


def vector_diffusion(disc, mu):

    return expand_components(diffusion(disc, mu), disc.dim)


def viscous(disc, mu):

    """integral of 2 mu eps(u) : eps(v) = integral of mu (grad(u) + grad(u)^T) : grad(v).

    K[(i, a), (j, c)] = integral of mu (delta_ac grad(N_i) . grad(N_j) + dN_i/dx_c dN_j/dx_a).
    """

    nel, _, nne, dim = disc.B.shape
    w = disc.detJw * at_gauss(disc, mu)

    cross = np.einsum('eg,egic,egja->eiajc', w, disc.B, disc.B).reshape(nel, nne * dim, nne * dim)

    return expand_components(np.einsum('eg,egia,egja->eij', w, disc.B, disc.B), dim) + cross


def divergence(disc, c=1.0):

    nel, _, nne, dim = disc.B.shape

    return np.einsum('eg,gi,egjb->eijb', disc.detJw * at_gauss(disc, c), disc.N, disc.B).reshape(nel, nne, nne * dim)


def vector_source(disc, f):

    nel, _, nne, dim = disc.B.shape

    return np.einsum('eg,gi,ega->eia', disc.detJw, disc.N, _velocity(disc, f)).reshape(nel, nne * dim)


def facet_vector_source(facets, t):

    """t: constant vector (dim,), array (nf, nq, dim) or function of the coordinates returning the components."""

    dim = facets.x.shape[-1]

    if callable(t):
        t = np.stack(np.broadcast_arrays(*t(*_coordinates(facets.x))), axis=-1)

    t = np.broadcast_to(np.asarray(t, dtype=float), facets.detJw.shape + (dim,))

    return np.einsum('fq,qi,fqa->fia', facets.detJw, facets.N, t).reshape(len(facets.detJw), -1)
