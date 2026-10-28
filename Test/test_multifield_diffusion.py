"""Diffusion coefficients that depend on other fields: D(c), D(T) and k(c), alone and coupled.

Run from the project root:  python Test/test_multifield_diffusion.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
from Source.Physics.MassTransfer import mt
from Source.enums import ElementType, ShapeFunctionType, StudyType, ProblemType, SolverType, DirectSolvers


def configure_steady(model):
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear,
                              Method=SolverType.direct, Solver=DirectSolvers.PARDISO)


def line_mesh(n):
    mesh = Mesh()
    mesh.Generate_Mesh(1, 1.0, n, ElementType.line)
    mesh.setElementShapeFunction(ShapeFunctionType.linear)
    return mesh


def dirichlet_ends(mesh, left, right):

    """left/right: {field: value} at x = 0 and x = 1."""

    for node in mesh.NL:
        x = node.coor[0]
        values = left if x < 1e-12 else right if x > 1 - 1e-12 else None
        if values is not None:
            node.BC = {name: {'type': 'Dirichlet', 'value': v} for name, v in values.items()}


def x_of(mesh):
    return np.array([node.coor[0] for node in mesh.NL])


# ---------------------------------------------------------------------------------------------
# Manufactured / exact solutions
# ---------------------------------------------------------------------------------------------

BETA = 0.8


def test_concentration_dependent_diffusivity_is_exact_by_kirchhoff():

    """D(c) = D0 (1 + BETA c): c + BETA c^2 / 2 is linear in x. P1 elements with 2-point Gauss are nodally exact."""

    c1, c2 = 0.1, 1.0
    psi = lambda c: c + BETA * c ** 2 / 2

    for n in (4, 8, 16):
        mesh = line_mesh(n)
        model = Model(name='D(c)', dim=1, mesh=mesh, mat=material('X', k=1, rho=1, Cp=1), psc=mt)
        model.physics.Convection = False
        model.physics.setChemSpecies('A', 'A')
        model.physics.setDiffusivity('A', lambda A: 3.0 * (1.0 + BETA * A), depends_on=('A',))
        dirichlet_ends(mesh, {'A': c1}, {'A': c2})
        model.physics.initField('A', 0.5)
        configure_steady(model)
        model.solve()

        p = psi(c1) + (psi(c2) - psi(c1)) * x_of(mesh)
        exact = (-1.0 + np.sqrt(1.0 + 2.0 * BETA * p)) / BETA
        assert np.max(np.abs(np.asarray(model.sol['A']) - exact)) < 1e-9, n


def cross_error(n, kind):

    """Error of the exact 1D solution when the coefficient of one field depends on a second, linear field.

    kind 'k(A)': heat with k = k0 (1 + g A), A fixed linear by its ends (constant D).
    kind 'D(T)': species with D = D0 (1 + g T), T fixed linear by its ends (constant k).
    """

    g = 0.05 if kind == 'D(T)' else 0.9
    mesh = line_mesh(n)
    model = Model(name=kind, dim=1, mesh=mesh, mat=material('X', k=2.0, rho=1, Cp=1), psc=[ht, mt])
    heat, mass = model.physics.heat, model.physics.mass
    heat.Convection = False
    mass.Convection = False
    mass.setChemSpecies('A', 'A')

    if kind == 'k(A)':
        heat.setConductivityLaw(lambda T, A: 2.0 * (1.0 + g * A), depends_on=('T', 'A'))
        mass.setDiffusivity('A', 1.0)
        a1, a2, f1, f2 = 0.2, 1.5, 20.0, 80.0           # A ends, T ends
        dirichlet_ends(mesh, {'T': f1, 'A': a1}, {'T': f2, 'A': a2})
        driver, target = 'A', 'T'
        d1, d2 = a1, a2
    else:
        mass.setDiffusivity('A', lambda T: 1.5 * (1.0 + g * T), depends_on=('T',))
        t1, t2, f1, f2 = 20.0, 80.0, 0.2, 1.5           # T ends, A ends
        dirichlet_ends(mesh, {'T': t1, 'A': f1}, {'T': t2, 'A': f2})
        driver, target = 'T', 'A'
        d1, d2 = t1, t2

    model.physics.initField('T', 50.0)
    model.physics.initField('A', 0.8)
    configure_steady(model)
    model.solve()

    # The driver is linear; the flux of the target is constant, so ln(1 + g u) is linear in the target
    u = d1 + (d2 - d1) * x_of(mesh)
    exact = f1 + (f2 - f1) * np.log((1 + g * u) / (1 + g * d1)) / np.log((1 + g * d2) / (1 + g * d1))

    assert np.allclose(np.asarray(model.sol[driver]), u, atol=1e-10)
    return np.max(np.abs(np.asarray(model.sol[target]) - exact))


def test_conductivity_depending_on_concentration_converges_to_the_exact_profile():
    errors = [cross_error(n, 'k(A)') for n in (8, 16, 32)]

    assert errors[2] < 5e-3, errors
    assert errors[0] / errors[1] > 3.3 and errors[1] / errors[2] > 3.3, errors   # second order


def test_diffusivity_depending_on_temperature_converges_to_the_exact_profile():
    errors = [cross_error(n, 'D(T)') for n in (8, 16, 32)]

    assert errors[2] < 5e-3, errors
    assert errors[0] / errors[1] > 3.3 and errors[1] / errors[2] > 3.3, errors


# ---------------------------------------------------------------------------------------------
# Jacobian and behavior of the coupled system
# ---------------------------------------------------------------------------------------------

def finite_difference_jacobian(system, x):
    J, R = system.residual_and_tangent(x)
    eps = 1e-6
    J_fd = np.zeros(J.shape)
    for j in range(len(x)):
        xp = x.copy()
        xp[j] += eps
        J_fd[:, j] = (system.residual_and_tangent(xp)[1] - R) / eps
    return J.toarray(), J_fd


def coupled_model(dim, with_laws=True):
    mesh = Mesh()
    if dim == 1:
        mesh.Generate_Mesh(1, 1.0, 5, ElementType.line)
        mesh.setElementShapeFunction(ShapeFunctionType.linear)
    else:
        mesh.generate2DMesh((1.0, 0.5), (4, 2), ElementType.quadrilateral)

    model = Model(name='cross', dim=dim, mesh=mesh, mat=material('X', k=2.0, rho=2.0, Cp=1.0), psc=[ht, mt])
    heat, mass = model.physics.heat, model.physics.mass
    heat.Convection = False
    mass.Convection = False

    for s in ('A', 'C'):
        mass.setChemSpecies(s, s)
    mass.addReaction({'A': -1, 'C': 1}, k0=300.0, E_R=1500.0, T='T', order={'A': 1}, dH=-30.0)

    if with_laws:
        heat.setConductivityLaw(lambda T, A, C: 1.0 + 0.01 * T + 0.5 * A * C, depends_on=('T', 'A', 'C'))
        mass.setDiffusivity('A', lambda T: 0.5 + 0.01 * T, depends_on=('T',))
        mass.setDiffusivity('C', lambda A, C: 1.0 + A * C, depends_on=('A', 'C'))
    else:
        mass.setDiffusivity('A', 1.0)
        mass.setDiffusivity('C', 1.0)

    return model, mesh


def test_jacobian_of_multifield_laws_matches_finite_differences():
    for dim in (1, 2):
        model, mesh = coupled_model(dim)
        configure_steady(model)
        system = model._system()

        rng = np.random.default_rng(3)
        x = np.empty(model.layout.ndof)
        for name, low, span in (('T', 20.0, 40.0), ('A', 0.2, 0.8), ('C', 0.1, 0.6)):
            x[model.layout.dofs(name)[:, 0]] = low + span * rng.random(mesh.getNoN())

        J, J_fd = finite_difference_jacobian(system, x)
        assert np.allclose(J, J_fd, rtol=1e-4, atol=1e-4), (dim, np.max(np.abs(J - J_fd)))

        # k(A, C) puts A and C columns in the T rows; D_A(T) puts T columns in the A rows
        T, A, C = (model.layout.dofs(n)[:, 0] for n in ('T', 'A', 'C'))
        assert np.abs(J[np.ix_(T, C)]).max() > 1e-3 and np.abs(J[np.ix_(A, T)]).max() > 1e-3


def test_constant_laws_reproduce_the_constant_coefficients():

    """depends_on with a constant function gives the same fields as the plain coefficient."""

    def solve(constant_law):
        mesh = line_mesh(10)
        model = Model(name='c', dim=1, mesh=mesh, mat=material('X', k=2.0, rho=1, Cp=1), psc=[ht, mt])
        model.physics.heat.Convection = False
        model.physics.mass.Convection = False
        model.physics.mass.setChemSpecies('A', 'A')
        if constant_law:
            model.physics.heat.setConductivityLaw(lambda T, A: 2.0 + 0.0 * T, depends_on=('T', 'A'))
            model.physics.mass.setDiffusivity('A', lambda T: 1.5 + 0.0 * T, depends_on=('T',))
        else:
            model.physics.mass.setDiffusivity('A', 1.5)
        model.physics.mass.addReaction({'A': -1}, k0=50.0, E_R=0.0, T=300.0, order={'A': 1})
        dirichlet_ends(mesh, {'T': 20.0, 'A': 1.0}, {'T': 60.0, 'A': 0.0})
        model.physics.initField('T', 30.0)
        model.physics.initField('A', 0.5)
        configure_steady(model)
        model.solve()
        return np.asarray(model.sol['T']), np.asarray(model.sol['A'])

    (T0, A0), (T1, A1) = solve(False), solve(True)
    assert np.allclose(T0, T1, atol=1e-9) and np.allclose(A0, A1, atol=1e-9)


def test_coupled_reactor_with_laws_converges_and_differs_from_constants():
    results = []
    for with_laws in (False, True):
        model, mesh = coupled_model(1, with_laws)
        dirichlet_ends(mesh, {'T': 25.0, 'A': 1.0, 'C': 0.0}, {'T': 40.0, 'A': 0.2, 'C': 0.5})
        for name, v in (('T', 25.0), ('A', 0.5), ('C', 0.2)):
            model.physics.initField(name, v)
        configure_steady(model)
        model.solve()
        results.append({n: np.asarray(model.sol[n]) for n in ('T', 'A', 'C')})

    assert all(np.all(np.isfinite(v)) for r in results for v in r.values())
    assert np.max(np.abs(results[0]['A'] - results[1]['A'])) > 1e-3


def test_transient_conserves_mass_with_concentration_dependent_diffusivity():

    """Insulated ends and D(c): the total amount of species is constant in time (discrete conservation)."""

    mesh = line_mesh(30)
    model = Model(name='D(c)', dim=1, mesh=mesh, mat=material('X', k=1, rho=1, Cp=1), psc=mt)
    model.physics.Convection = False
    model.physics.setChemSpecies('A', 'A')
    model.physics.setDiffusivity('A', lambda A: 0.05 * (1.0 + 4.0 * A), depends_on=('A',))
    model.physics.initField('A', lambda x: 1.0 if x < 0.5 else 0.0)
    model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, Method=SolverType.direct,
                              Solver=DirectSolvers.PARDISO, timeDisc=1.0, timeStep=0.01, totalTime=0.5)
    model.solve()

    A = np.asarray(model.sol['A'])
    x = x_of(mesh)
    totals = np.trapezoid(A, x, axis=0)
    assert np.max(np.abs(totals - totals[0])) < 1e-10, np.max(np.abs(totals - totals[0]))
    assert A[:, -1].max() < A[:, 0].max() - 0.05          # the step spreads out


def test_errors():

    # A law over a field that the model does not have
    mesh = line_mesh(4)
    model = Model(name='x', dim=1, mesh=mesh, mat=material('X', k=1, rho=1, Cp=1), psc=mt)
    model.physics.setChemSpecies('A', 'A')
    model.physics.setDiffusivity('A', lambda T: 1.0 + 0.0 * T, depends_on=('T',))
    configure_steady(model)

    try:
        model.solve()
    except ValueError as error:
        assert "'T'" in str(error)
    else:
        raise AssertionError("expected ValueError for a diffusivity over a missing field")

    # Going back to a plain number drops the law
    model.physics.setDiffusivity('A', 2.0)
    assert 'A' not in model.physics.diffusivity_inputs

    # Heat conductivity over a missing species
    mesh2 = line_mesh(4)
    model2 = Model(name='y', dim=1, mesh=mesh2, mat=material('X', k=1, rho=1, Cp=1), psc=ht)
    model2.physics.setConductivityLaw(lambda T, A: 1.0 + 0.0 * A, depends_on=('T', 'A'))
    configure_steady(model2)

    try:
        model2.solve()
    except ValueError as error:
        assert "'A'" in str(error)
    else:
        raise AssertionError("expected ValueError for a conductivity over a missing field")


if __name__ == "__main__":
    tests = [test_concentration_dependent_diffusivity_is_exact_by_kirchhoff,
             test_conductivity_depending_on_concentration_converges_to_the_exact_profile,
             test_diffusivity_depending_on_temperature_converges_to_the_exact_profile,
             test_jacobian_of_multifield_laws_matches_finite_differences,
             test_constant_laws_reproduce_the_constant_coefficients,
             test_coupled_reactor_with_laws_converges_and_differs_from_constants,
             test_transient_conserves_mass_with_concentration_dependent_diffusivity,
             test_errors]
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
