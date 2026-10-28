"""Checks for the physics-independent foundation: elements, discretization, facets, DOF layout,
operators and pointwise laws.

Run from the project root:  python Test/test_foundation.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax.numpy as jnp

from Source.Pre_processing.Mesh import Mesh
from Source.Pre_processing.Elements import ELEMENTS
from Source.Simulation.Discretization import Discretization
from Source.Simulation.DofLayout import DofLayout, FieldSpec
from Source.Simulation.PointwiseLaw import PointwiseLaw
import Source.Simulation.Operators as ops
from Source.enums import ElementType


def mesh_1d(n=4, L=2.0):
    mesh = Mesh()
    mesh.Generate_Mesh(1, L, n, ElementType.line)
    return mesh


def mesh_2d(nx=3, ny=2, Lx=3.0, Ly=1.0):
    mesh = Mesh()
    mesh.generate2DMesh((Lx, Ly), (nx, ny), ElementType.quadrilateral)
    return mesh


def test_reference_elements_partition_of_unity():
    """N sums to 1 and dN sums to 0 at any point, for every element in the registry."""

    for name, element in ELEMENTS.items():
        points, weights = element.quadrature(3)
        N, dN = element.shape(points)

        assert np.allclose(N.sum(axis=1), 1.0), name
        assert np.allclose(dN.sum(axis=1), 0.0), name

        if element.dim > 0:
            measure = 0.5 if name.startswith('tri') else 2.0 ** element.dim
            assert np.isclose(weights.sum(), measure), name  # measure of the reference domain


def test_volume_and_measure_1d_2d():
    """Sum of detJw is the length (1D) or the area (2D) of the domain."""

    d1 = Discretization(mesh_1d(5, 2.0))
    assert np.isclose(d1.detJw.sum(), 2.0)

    d2 = Discretization(mesh_2d(3, 2, 3.0, 1.0))
    assert np.isclose(d2.detJw.sum(), 3.0)


def test_gradient_of_linear_field():
    """B reproduces the exact gradient of u = 2x + 3y (2D) and u = 5x (1D) at every Gauss point."""

    mesh = mesh_2d(4, 3, 2.0, 1.5)
    disc = Discretization(mesh)
    u = 2 * disc.node_xy[:, 0] + 3 * disc.node_xy[:, 1]
    grad = np.einsum('egia,ei->ega', disc.B, u[disc.conn])
    assert np.allclose(grad[..., 0], 2.0) and np.allclose(grad[..., 1], 3.0)

    disc1 = Discretization(mesh_1d(4, 2.0))
    u1 = 5 * disc1.node_xy[:, 0]
    grad1 = np.einsum('egia,ei->ega', disc1.B, u1[disc1.conn])
    assert np.allclose(grad1, 5.0)


def test_facets_1d_are_points_with_outward_normals():
    disc = Discretization(mesh_1d(4, 2.0))

    tuples = disc.boundary_facet_nodes()
    assert sorted(tuples) == [(0,), (4,)]

    fs = disc.facets(tuples)
    assert np.allclose(fs.detJw, 1.0)
    normal = {int(n[0]): float(v[0, 0]) for n, v in zip(fs.nodes, fs.normals)}
    assert normal == {0: -1.0, 4: 1.0}


def test_facets_2d_length_and_normals():
    mesh = mesh_2d(3, 2, 3.0, 1.0)
    disc = Discretization(mesh)

    assert len(disc.boundary_facet_nodes()) == 10

    right = disc.facets(disc.boundary_facet_nodes(lambda x, y: x > 3.0 - 1e-12))
    assert np.isclose(right.detJw.sum(), 1.0)                      # length of the right side
    assert np.allclose(right.normals, [1.0, 0.0])

    bottom = disc.facets(disc.boundary_facet_nodes(lambda x, y: y < 1e-12))
    assert np.isclose(bottom.detJw.sum(), 3.0)
    assert np.allclose(bottom.normals, [0.0, -1.0])

    # Divergence theorem: the integral of n over the whole boundary is zero
    allf = disc.facets(disc.boundary_facet_nodes())
    assert np.allclose(np.einsum('fq,fqa->a', allf.detJw, allf.normals), 0.0)


def test_facet_that_is_not_on_the_boundary_is_rejected():
    disc = Discretization(mesh_2d(3, 2))
    try:
        disc.facets([(5, 6)])  # an interior edge
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_dof_layout_roundtrip_and_blocks():
    layout = DofLayout([FieldSpec('A'), FieldSpec('u', 2), FieldSpec('B')], n_nodes=3)

    assert layout.n_total == 4 and layout.ndof == 12
    assert layout.dofs('A')[:, 0].tolist() == [0, 4, 8]
    assert layout.dofs('u').tolist() == [[1, 2], [5, 6], [9, 10]]
    assert layout.dofs('B')[:, 0].tolist() == [3, 7, 11]

    values = {'A': np.array([1., 2., 3.]), 'u': np.arange(6.).reshape(3, 2), 'B': np.array([7., 8., 9.])}
    x = layout.join(values)
    back = layout.split(x)
    for name in values:
        assert np.allclose(back[name], values[name]), name


def test_operators_match_known_1d_matrices():
    """For an element of length h: K = k/h [[1,-1],[-1,1]], M = c h/6 [[2,1],[1,2]], load = f h/2."""

    h, k, c, f, v = 0.5, 3.0, 2.0, 4.0, 1.5
    disc = Discretization(mesh_1d(1, h))

    assert np.allclose(ops.diffusion(disc, k)[0], k / h * np.array([[1, -1], [-1, 1]]))
    assert np.allclose(ops.mass(disc, c)[0], c * h / 6 * np.array([[2, 1], [1, 2]]))
    assert np.allclose(ops.source(disc, f)[0], f * h / 2)
    assert np.allclose(ops.convection(disc, c, v)[0], c * v / 2 * np.array([[-1, 1], [-1, 1]]))


def test_operator_coefficients_as_functions_and_tensors():
    mesh = mesh_2d(2, 2, 1.0, 1.0)
    disc = Discretization(mesh)

    # A constant function gives the same matrix as the number
    assert np.allclose(ops.diffusion(disc, lambda x, y: 2.0 + 0 * x), ops.diffusion(disc, 2.0))
    # An isotropic tensor gives the same matrix as the scalar
    assert np.allclose(ops.diffusion_tensor(disc, 2.0 * np.eye(2)), ops.diffusion(disc, 2.0))
    # Constant velocity vector
    assert np.allclose(ops.convection(disc, 1.0, [1.0, 0.0]),
                       ops.convection(disc, 1.0, lambda x, y: (1.0 + 0 * x, 0.0 * x)))


def test_pointwise_law_derivatives_match_finite_differences():
    """Reaction-like law with two inputs and two outputs."""

    def law(u):
        a, b = u[0], u[1]
        r = 2.0 * jnp.exp(-0.3 / 1.0) * a * jnp.maximum(b, 1e-12) ** 1.5
        return jnp.array([-r, r])

    pl = PointwiseLaw(law)
    U = np.array([[1.0, 2.0], [0.5, 4.0], [3.0, 0.7]])
    values, derivs = pl(U)

    assert values.shape == (3, 2) and derivs.shape == (3, 2, 2)

    eps = 1e-6
    for j in range(2):
        dU = U.copy()
        dU[:, j] += eps
        fd = (pl(dU)[0] - values) / eps
        assert np.allclose(derivs[:, :, j], fd, rtol=1e-4, atol=1e-6), j


def test_pointwise_law_is_compiled_once():
    """A second call with the same shapes must not trace the function again."""

    calls = []

    def law(u):
        calls.append(1)
        return u[0] ** 2

    pl = PointwiseLaw(law)
    U = np.linspace(0, 1, 10).reshape(5, 2)
    pl(U)
    traced = len(calls)
    pl(U + 1.0)
    pl(2 * U)

    assert len(calls) == traced


if __name__ == "__main__":
    tests = [test_reference_elements_partition_of_unity, test_volume_and_measure_1d_2d,
             test_gradient_of_linear_field, test_facets_1d_are_points_with_outward_normals,
             test_facets_2d_length_and_normals, test_facet_that_is_not_on_the_boundary_is_rejected,
             test_dof_layout_roundtrip_and_blocks, test_operators_match_known_1d_matrices,
             test_operator_coefficients_as_functions_and_tensors,
             test_pointwise_law_derivatives_match_finite_differences, test_pointwise_law_is_compiled_once]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
