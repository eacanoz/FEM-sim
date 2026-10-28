"""Checks for the transport of chemical species (diffusion, convection, reactions), 1D and 2D.

Run from the project root:  python Test/test_species.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.MassTransfer import mt
from Source.Simulation.Discretization import Discretization
from Source.enums import ElementType, ShapeFunctionType, StudyType, ProblemType, SolverType, DirectSolvers


def species_model(species, NoE=20, L=1.0, D=1.0, convection=False, vel=1.0):

    mesh = Mesh()
    mesh.Generate_Mesh(1, L, NoE, ElementType.line)
    mesh.setElementShapeFunction(ShapeFunctionType.linear)

    model = Model(name='species', dim=1, mesh=mesh, mat=material('X', k=1, rho=1, Cp=1), psc=mt)

    for s in species:
        model.physics.setChemSpecies(s, s)
        model.physics.setDiffusivity(s, D[s] if isinstance(D, dict) else D)

    model.physics.Convection = convection
    model.physics.vel = vel

    return model, mesh


def configure(model, study=StudyType.steady_state, problem=ProblemType.nonlinear, **time):
    model.solverConfiguration(Study=study, Type=problem, Method=SolverType.direct,
                              Solver=DirectSolvers.PARDISO, **time)


def reaction_first_order(model, k):
    model.physics.addReaction(stoich={'A': -1}, k0=k, E_R=0.0, T=300.0, order={'A': 1})


# ---------------------------------------------------------------------------------------------
# 1D analytic solutions
# ---------------------------------------------------------------------------------------------

def error_diffusion_reaction(n):

    """D c'' = k c, c(0) = 1, c(L) = 0:  c = sinh(m (L - x)) / sinh(m L), m = sqrt(k / D)."""

    D, k, L = 0.5, 2.0, 1.0
    model, mesh = species_model(['A'], NoE=n, L=L, D=D)
    reaction_first_order(model, k)
    model.physics.addBC_Concentration(0, 'A', 1.0)
    model.physics.addBC_Concentration(n, 'A', 0.0)
    configure(model, problem=ProblemType.linear)
    model.solve()

    x = np.array(mesh.getXCoor())
    m = np.sqrt(k / D)
    exact = np.sinh(m * (L - x)) / np.sinh(m * L)

    return np.max(np.abs(np.asarray(model.sol['A']) - exact))


def test_diffusion_first_order_reaction_converges_with_order_two():
    errors = [error_diffusion_reaction(n) for n in (8, 16, 32, 64)]
    rates = [np.log2(errors[i] / errors[i + 1]) for i in range(3)]

    print("   nodal errors:", ["%.2e" % e for e in errors], "rates:", ["%.2f" % r for r in rates])
    assert all(r > 1.9 for r in rates), rates


def test_convection_diffusion_reaction_exponentials():
    """D c'' - v c' - k c = 0, c(0) = 1, c(L) = 0."""

    D, v, k, L, n = 0.5, 1.0, 1.0, 1.0, 40
    model, mesh = species_model(['A'], NoE=n, L=L, D=D, convection=True, vel=v)
    reaction_first_order(model, k)
    model.physics.addBC_Concentration(0, 'A', 1.0)
    model.physics.addBC_Concentration(n, 'A', 0.0)
    configure(model, problem=ProblemType.linear)
    model.solve()

    r_plus = (v + np.sqrt(v ** 2 + 4 * D * k)) / (2 * D)
    r_minus = (v - np.sqrt(v ** 2 + 4 * D * k)) / (2 * D)
    coeffs = np.linalg.solve([[1.0, 1.0], [np.exp(r_plus * L), np.exp(r_minus * L)]], [1.0, 0.0])

    x = np.array(mesh.getXCoor())
    exact = coeffs[0] * np.exp(r_plus * x) + coeffs[1] * np.exp(r_minus * x)

    assert np.max(np.abs(np.asarray(model.sol['A']) - exact)) < 5e-4


def test_second_order_reaction_conserves_stoichiometric_sums():
    """A + B -> C with equal diffusivities. A + C and B + C satisfy D S'' = 0, so they are linear.

    This holds for the discrete system too, which verifies the stoichiometry, the coupling between
    species and the Newton solution of a nonlinear (second-order) reaction.
    """

    n = 20
    model, mesh = species_model(['A', 'B', 'C'], NoE=n, D=0.7)
    model.physics.addReaction(stoich={'A': -1, 'B': -1, 'C': 1}, k0=5.0, E_R=0.0, T=300.0,
                              order={'A': 1, 'B': 1})

    left = {'A': 3.0, 'B': 2.0, 'C': 0.5}
    right = {'A': 1.0, 'B': 0.5, 'C': 1.5}
    for s in 'ABC':
        model.physics.addBC_Concentration(0, s, left[s])
        model.physics.addBC_Concentration(n, s, right[s])
        model.physics.initField(s, 1.0)

    configure(model)
    model.solve()

    x = np.array(mesh.getXCoor()) / 1.0
    A, B, C = (np.asarray(model.sol[s]) for s in 'ABC')

    for first, name in ((A, 'A'), (B, 'B')):
        total = first + C
        line = (left[name] + left['C']) * (1 - x) + (right[name] + right['C']) * x
        assert np.allclose(total, line, atol=1e-7), (name, np.max(np.abs(total - line)))

    # The reaction really consumed A and B in the interior
    assert A[n // 2] < 3.0 * 0.5 + 1.0 * 0.5
    assert np.all(A > 0) and np.all(B > 0)


def test_jacobian_with_cross_derivatives_matches_finite_differences():
    """Three coupled species: the automatic Jacobian of the system must match finite differences."""

    n = 6
    model, _ = species_model(['A', 'B', 'C'], NoE=n, D={'A': 1.0, 'B': 0.5, 'C': 2.0}, convection=True, vel=0.8)
    model.physics.addReaction(stoich={'A': -1, 'B': -1, 'C': 1}, k0=4.0, E_R=100.0, T=300.0,
                              order={'A': 1, 'B': 1.5})
    model.physics.addReaction(stoich={'C': -1, 'A': 1}, k0=0.5, E_R=0.0, T=300.0, order={'C': 2})

    for s in 'ABC':
        model.physics.addBC_Concentration(0, s, 1.0)
    configure(model)
    system = model._system()

    rng = np.random.default_rng(1)
    x = 0.5 + rng.random(model.layout.ndof)

    J, R = system.residual_and_tangent(x)
    J = J.toarray()

    eps = 1e-6
    J_fd = np.zeros_like(J)
    for j in range(len(x)):
        xp = x.copy()
        xp[j] += eps
        J_fd[:, j] = (system.residual_and_tangent(xp)[1] - R) / eps

    assert np.allclose(J, J_fd, rtol=1e-4, atol=1e-5), np.max(np.abs(J - J_fd))


def test_transient_species_decay_with_different_diffusivities():
    """Each species decays as exp(-D pi^2 t) sin(pi x) when there is no reaction."""

    n, total_time = 40, 0.1
    D = {'A': 1.0, 'B': 0.4}
    model, mesh = species_model(['A', 'B'], NoE=n, D=D)

    x = np.array(mesh.getXCoor())
    for s in D:
        model.physics.var[s].updateField(list(np.sin(np.pi * x)))
        model.physics.addBC_Concentration(0, s, 0.0)
        model.physics.addBC_Concentration(n, s, 0.0)

    configure(model, study=StudyType.transient, problem=ProblemType.linear,
              timeDisc=0.5, timeStep=0.005, totalTime=total_time)
    model.solve()

    for s, d in D.items():
        exact = np.exp(-d * np.pi ** 2 * total_time) * np.sin(np.pi * x)
        assert model.sol[s].shape == (n + 1, 21)
        assert np.max(np.abs(model.sol[s][:, -1] - exact)) < 2e-3, s


# ---------------------------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------------------------

def species_model_2d(n, species=('A',), D=1.0):

    mesh = Mesh()
    mesh.generate2DMesh((1.0, 1.0), (n, n), ElementType.quadrilateral)

    model = Model(name='species2d', dim=2, mesh=mesh, mat=material('X', k=1, rho=1, Cp=1), psc=mt)
    for s in species:
        model.physics.setChemSpecies(s, s)
        model.physics.setDiffusivity(s, D)
    model.physics.Convection = False

    return model, mesh


def l2_error_nonlinear_reaction(n):

    """-D lap(c) + k c^2 = S with c = 1 + sin(pi x) sin(pi y) (manufactured), c = 1 on the boundary."""

    D, k = 0.8, 1.5
    model, mesh = species_model_2d(n, D=D)
    model.physics.addReaction(stoich={'A': -1}, k0=k, E_R=0.0, T=300.0, order={'A': 2})

    exact = lambda x, y: 1.0 + np.sin(np.pi * x) * np.sin(np.pi * y)
    model.physics.setSource('A', lambda x, y: D * 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
                            + k * exact(x, y) ** 2)
    model.physics.addBC_ConcentrationEdges(mesh.boundary_edges(), 'A', 1.0)
    model.physics.initField('A', 1.0)
    configure(model)
    model.solve()

    disc = Discretization(mesh)
    c_nodes = np.asarray(model.sol['A'])
    c_gauss = np.einsum('gi,ei->eg', disc.N, c_nodes[disc.conn])
    err2 = np.sum(disc.detJw * (c_gauss - exact(disc.x_gp[..., 0], disc.x_gp[..., 1])) ** 2)

    return np.sqrt(err2)


def test_reaction_diffusion_2d_manufactured_order_two():
    errors = [l2_error_nonlinear_reaction(n) for n in (4, 8, 16)]
    rates = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]

    print("   L2 errors:", ["%.3e" % e for e in errors], "rates:", ["%.2f" % r for r in rates])
    assert all(r > 1.9 for r in rates), rates


def test_edge_conditions_2d_flux_and_film_transfer():
    """c = 2x + 1 (D = 1): the flux leaving through x = 1 is -D dc/dx = -2 and equals k_m (c - c_ext) = -2 at c = 3."""

    for kind in ('flux', 'film'):
        model, mesh = species_model_2d(4)
        model.physics.addBC_ConcentrationEdges(mesh.boundary_edges(lambda x, y: x < 1 - 1e-12), 'A',
                                               lambda x, y: 2 * x + 1)
        right = mesh.boundary_edges(lambda x, y: x > 1 - 1e-12)

        if kind == 'flux':
            model.physics.addBC_FluxEdges(right, 'A', -2.0)
        else:
            model.physics.addBC_MassTransferEdges(right, 'A', 4.0, 3.5)  # 4 (3 - 3.5) = -2

        configure(model, problem=ProblemType.linear)
        model.solve()

        xy = np.array([node.coor for node in mesh.NL])
        assert np.max(np.abs(np.asarray(model.sol['A']) - (2 * xy[:, 0] + 1))) < 1e-10, kind


def test_clear_errors_for_missing_species_or_diffusivity():
    model, _ = species_model(['A'])

    try:
        model.physics.setDiffusivity('Z', 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an undefined species")

    model2, _ = species_model(['A'])
    model2.physics.setChemSpecies('B', 'B')  # no diffusivity for B
    model2.physics.addBC_Concentration(0, 'A', 1.0)
    configure(model2, problem=ProblemType.linear)

    try:
        model2.solve()
    except ValueError:
        return

    raise AssertionError("expected ValueError for a missing diffusivity")


if __name__ == "__main__":
    tests = [test_diffusion_first_order_reaction_converges_with_order_two,
             test_convection_diffusion_reaction_exponentials,
             test_second_order_reaction_conserves_stoichiometric_sums,
             test_jacobian_with_cross_derivatives_matches_finite_differences,
             test_transient_species_decay_with_different_diffusivities,
             test_reaction_diffusion_2d_manufactured_order_two,
             test_edge_conditions_2d_flux_and_film_transfer,
             test_clear_errors_for_missing_species_or_diffusivity]
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
