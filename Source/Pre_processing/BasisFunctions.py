## Shape functions of the bilinear quadrilateral (Q4) on the reference square.
## The 1D Lagrange basis and the element registry live in Elements.py.

import numpy as np

# Bilinear quadrilateral (Q4) on the reference square; nodes ordered BL, BR, TR, TL
Q4_NODES = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


def q4_shape_functions(xi, eta):

    """Q4 shape functions and their reference derivatives at points (xi, eta).

    Returns N (npts, 4), dN (npts, 4, 2) with dN[..., 0] = dN/dxi and dN[..., 1] = dN/deta.
    """

    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    eta = np.atleast_1d(np.asarray(eta, dtype=float))

    xk = Q4_NODES[:, 0][None, :]
    ek = Q4_NODES[:, 1][None, :]
    x = xi[:, None]
    e = eta[:, None]

    N = (1 + xk * x) * (1 + ek * e) / 4
    dN = np.stack([xk * (1 + ek * e) / 4, ek * (1 + xk * x) / 4], axis=-1)

    return N, dN
