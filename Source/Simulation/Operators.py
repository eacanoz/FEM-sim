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

Pressure stabilization (PSPG) of equal-order velocity-pressure elements:

    element_size(disc)                  (nel,)                  h of each element
    pspg_tau(disc, mu, rho, speed, dt)  (nel, ngp)              tau of the term integral of tau grad(q) . (momentum residual)
    pspg_mass(disc, c)                  (nel, nne, nne*dim)     integral of c grad(N_i) . u   (test: scalar, trial: vector)
    recovered_stress(disc, mu)          sparse                  continuous nodal stress recovered from 2 mu eps(u_h)
    recovered_stress_divergence         sparse                  div of the recovered stress at the Gauss points
    gauss_stress_divergence(disc)       sparse                  same from a stress given at the Gauss points (nonlinear mu)
    hoop_stress_terms(disc, mu)         sparse                  (sigma_rr - sigma_tt)/r, sigma_rz/r of div(sigma) (axisymmetric)
    pspg_weight(disc, tau)              sparse                  integral of tau grad(N_i) . g, g a vector at the Gauss points

Axisymmetric discretizations (x = r, y = z) carry the factor r in detJw, so every operator integrates over the
volume of revolution (without 2 pi); viscous, vector_diffusion and divergence add the hoop strain u_r / r.
    lsic_tau(disc, mu, rho, speed)      (nel, ngp)              grad-div stabilization coefficient (Navier-Stokes)

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


def _add_hoop(K, disc, w):

    """Adds integral of w (N_i / r)(N_j / r) to the radial-radial block of vector element matrices (axisymmetric)."""

    nel, _, nne, dim = disc.B.shape
    K = K.reshape(nel, nne, dim, nne, dim).copy()
    K[:, :, 0, :, 0] += np.einsum('eg,egi,egj->eij', w, disc.hoop, disc.hoop)

    return K.reshape(nel, nne * dim, nne * dim)


def vector_diffusion(disc, mu):

    """integral of mu grad(u) : grad(v); axisymmetric: plus integral of mu u_r v_r / r^2 (vector Laplacian)."""

    K = expand_components(diffusion(disc, mu), disc.dim)

    return _add_hoop(K, disc, disc.detJw * at_gauss(disc, mu)) if disc.axisymmetric else K


def viscous(disc, mu):

    """integral of 2 mu eps(u) : eps(v) = integral of mu (grad(u) + grad(u)^T) : grad(v).

    K[(i, a), (j, c)] = integral of mu (delta_ac grad(N_i) . grad(N_j) + dN_i/dx_c dN_j/dx_a).
    Axisymmetric: eps also has the hoop strain u_r / r, which adds integral of 2 mu (N_i / r)(N_j / r) to the r-r block.
    """

    nel, _, nne, dim = disc.B.shape
    w = disc.detJw * at_gauss(disc, mu)

    cross = np.einsum('eg,egic,egja->eiajc', w, disc.B, disc.B).reshape(nel, nne * dim, nne * dim)
    K = expand_components(np.einsum('eg,egia,egja->eij', w, disc.B, disc.B), dim) + cross

    return _add_hoop(K, disc, 2.0 * w) if disc.axisymmetric else K


def divergence(disc, c=1.0):

    """integral of c N_i div(u); axisymmetric: div u = du_r/dr + u_r / r + du_z/dz."""

    nel, _, nne, dim = disc.B.shape
    w = disc.detJw * at_gauss(disc, c)
    D = np.einsum('eg,gi,egjb->eijb', w, disc.N, disc.B)

    if disc.axisymmetric:
        D = D.copy()
        D[..., 0] += np.einsum('eg,gi,egj->eij', w, disc.N, disc.hoop)

    return D.reshape(nel, nne, nne * dim)


def element_size(disc):

    """h of each element: the length of the line, the side of the square of the same area (quadrilaterals) or the
    leg of the right isosceles triangle of the same area (triangles), (nel,). Uses the plane area (also axisymmetric)."""

    measure = disc.area.sum(axis=1)

    if disc.element.name.startswith('tri'):
        measure = 2.0 * measure

    return measure ** (1.0 / disc.dim)


def pspg_tau(disc, mu, rho=0.0, speed=0.0, dt=None):

    """PSPG parameter (nel, ngp): tau = 1 / sqrt((2 rho / dt)^2 + (2 rho |u| / h)^2 + (12 mu / h^2)^2).

    It multiplies the momentum residual (force per unit volume) in integral of tau grad(q) . R_M, so it has units
    of time / density; this is the tau of Tezduyar and Shakib divided by rho, which stays defined for Stokes flow
    (rho = 0). Creeping flow: tau = h^2 / (12 mu). speed: |u| at the Gauss points (Navier-Stokes); dt: time step.
    """

    h = element_size(disc)[:, None]
    rho = at_gauss(disc, rho)

    inv2 = (12.0 * at_gauss(disc, mu) / h ** 2) ** 2 + (2.0 * rho * np.asarray(speed) / h) ** 2
    if dt is not None:
        inv2 = inv2 + (2.0 * rho / dt) ** 2

    return 1.0 / np.sqrt(inv2)


def recovered_stress(disc, mu):

    """Sparse P (n_nodes*dim*dim, n_nodes*dim): continuous nodal stress sigma~ recovered from 2 mu eps(u_h).

    The Gauss-point stress is passed through Discretization.recovery (patch least squares of the element means,
    second order also on the boundary). Rows are node-major (k, a, b) -> k*dim*dim + a*dim + b; columns are the
    nodal velocity (j, c) -> j*dim + c.
    """

    import scipy.sparse as sp

    dim = disc.dim

    return sp.kron(disc.recovery(), sp.identity(dim * dim), format='csr') @ _gauss_stress(disc, mu)


def _gauss_stress(disc, mu):

    """Sparse S (nel*ngp*dim*dim, n_nodes*dim): 2 mu eps(u_h) at the Gauss points from the nodal velocity,
    S[(e, g), a, b; j, c] = mu (dN_j/dx_b delta_ac + dN_j/dx_a delta_bc)."""

    import scipy.sparse as sp

    nel, ngp, nne, dim = disc.B.shape
    dd = dim * dim

    eye = np.eye(dim)
    m = at_gauss(disc, mu)
    Se = (np.einsum('eg,egjb,ac->egabjc', m, disc.B, eye) + np.einsum('eg,egja,bc->egabjc', m, disc.B, eye))

    point = np.arange(nel * ngp).reshape(nel, ngp)
    r = (point[:, :, None, None] * dd + np.arange(dim)[:, None] * dim + np.arange(dim)[None, :])
    c = disc.conn[:, :, None] * dim + np.arange(dim)

    return sp.coo_matrix((Se.ravel(), (np.broadcast_to(r[..., None, None], Se.shape).ravel(),
                                       np.broadcast_to(c[:, None, None, None], Se.shape).ravel())),
                         shape=(nel * ngp * dd, disc.n_nodes * dim)).tocsr()


def _stress_rows(disc):
    dim = disc.dim
    return disc.conn[:, :, None, None] * dim * dim + np.arange(dim)[:, None] * dim + np.arange(dim)[None, :]


def pspg_weight(disc, tau):

    """Sparse Wp (n_nodes, nel*ngp*dim): integral of tau grad(N_i) . g for a vector g given at the Gauss points,
    Wp @ g.ravel(). With g = div(sigma~) = V @ u, -Wp @ V is the viscous part of the PSPG term of the continuity
    equation. Inside a linear element div(2 mu eps(u_h)) vanishes, which makes plain PSPG inconsistent (a first-order
    error in the pressure); the recovered continuous stress sigma~ restores it."""

    import scipy.sparse as sp

    nel, ngp, nne, dim = disc.B.shape
    values = np.einsum('eg,egia->egia', disc.detJw * tau, disc.B)
    cols = np.arange(nel * ngp).reshape(nel, ngp)[:, :, None] * dim + np.arange(dim)       # (nel, ngp, dim)

    return sp.coo_matrix((values.ravel(), (np.broadcast_to(disc.conn[:, None, :, None], values.shape).ravel(),
                                           np.broadcast_to(cols[:, :, None, :], values.shape).ravel())),
                         shape=(disc.n_nodes, nel * ngp * dim)).tocsr()


def hoop_stress_terms(disc, mu):

    """Sparse E (nel*ngp*dim, n_nodes*dim): the non-derivative terms of div(sigma) in cylindrical coordinates at the
    Gauss points, (sigma_rr - sigma_tt) / r for the radial and sigma_rz / r for the axial component, with
    sigma = 2 mu eps(u) and the hoop stress sigma_tt = 2 mu u_r / r. Zero matrix for a plane discretization."""

    import scipy.sparse as sp

    nel, ngp, nne, dim = disc.B.shape
    shape = (nel * ngp * dim, disc.n_nodes * dim)
    if not disc.axisymmetric:
        return sp.csr_matrix(shape)

    r = disc.radius
    m = at_gauss(disc, mu) / r
    E = np.zeros((nel, ngp, dim, nne, dim))
    E[:, :, 0, :, 0] = 2.0 * m[..., None] * (disc.B[..., 0] - disc.hoop)      # (sigma_rr - sigma_tt) / r
    E[:, :, 1, :, 0] = m[..., None] * disc.B[..., 1]                          # sigma_rz / r, du_r/dz part
    E[:, :, 1, :, 1] = m[..., None] * disc.B[..., 0]                          # sigma_rz / r, du_z/dr part

    rows = np.arange(nel * ngp).reshape(nel, ngp)[:, :, None] * dim + np.arange(dim)       # (nel, ngp, dim)
    cols = disc.conn[:, :, None] * dim + np.arange(dim)                                    # (nel, nne, dim)

    return sp.coo_matrix((E.ravel(), (np.broadcast_to(rows[..., None, None], E.shape).ravel(),
                                      np.broadcast_to(cols[:, None, None], E.shape).ravel())), shape=shape).tocsr()


def gauss_stress_divergence(disc):

    """Sparse Q (nel*ngp*dim, nel*ngp*dim*dim): div(sigma~) at the Gauss points from any stress given at the Gauss
    points, with sigma~ its continuous nodal recovery (Discretization.recovery). Geometry only, so it is built once;
    a nonlinear stress (mu(gamma_dot)) is passed through it at every iteration. Rows (e, g, a) -> (e*ngp + g)*dim + a;
    columns (e, g, a, b) -> (e*ngp + g)*dim*dim + a*dim + b."""

    import scipy.sparse as sp

    nel, ngp, nne, dim = disc.B.shape
    rows_k = _stress_rows(disc)                                                   # (nel, nne, dim, dim)

    # D[(e, g, a), (k, a, b)] = dN_k/dx_b
    values = np.broadcast_to(disc.B[:, :, :, None, :], (nel, ngp, nne, dim, dim))  # (e, g, k, a, b)
    r = (np.arange(nel * ngp).reshape(nel, ngp)[:, :, None, None, None] * dim
         + np.arange(dim)[None, None, None, :, None])
    c = rows_k[:, None]
    D = sp.coo_matrix((values.ravel(), (np.broadcast_to(r, values.shape).ravel(),
                                        np.broadcast_to(c, values.shape).ravel())),
                      shape=(nel * ngp * dim, disc.n_nodes * dim * dim)).tocsr()

    return D @ sp.kron(disc.recovery(), sp.identity(dim * dim), format='csr')


def recovered_stress_divergence(disc, mu):

    """Sparse V (nel*ngp*dim, n_nodes*dim): div(sigma~) at the Gauss points from the nodal velocity for a viscosity
    mu that does not depend on the flow, (V @ u).reshape(nel, ngp, dim). Rows (e, g, a) -> (e*ngp + g)*dim + a.
    Axisymmetric: plus the hoop terms (hoop_stress_terms), evaluated directly at the Gauss points."""

    V = gauss_stress_divergence(disc) @ _gauss_stress(disc, mu)

    return V + hoop_stress_terms(disc, mu) if disc.axisymmetric else V


def lsic_tau(disc, mu, rho, speed):

    """Grad-div (LSIC) coefficient [Pa s] (nel, ngp): rho |u| h / 2 * min(Re_h / 3, 1), Re_h = rho |u| h / (2 mu)."""

    h = element_size(disc)[:, None]
    rho, mu = at_gauss(disc, rho), at_gauss(disc, mu)
    Re_h = rho * speed * h / (2.0 * mu)

    return 0.5 * rho * speed * h * np.minimum(Re_h / 3.0, 1.0)


def pspg_mass(disc, c):

    nel, _, nne, dim = disc.B.shape

    return np.einsum('eg,egia,gj->eija', disc.detJw * at_gauss(disc, c), disc.B, disc.N).reshape(nel, nne, nne * dim)


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
