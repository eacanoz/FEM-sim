"""Regression checks of the vectorized heat assembly against reference values.

The reference values were computed with the vectorized path after verifying that it reproduced the
old element-based assembly (matrices, vectors, mass, residual, tangent, transient and Newton solution),
and before that path was removed. The Jacobian is also checked against finite differences, which does
not depend on any reference.

Run from the project root:  python Test/test_vectorized.py
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
from Source.enums import ElementType, ShapeFunctionType, StudyType, ProblemType, SolverType, DirectSolvers

REFERENCE = {'linear_0_dirichlet_newton': {'A': [[1.0, 0.0, 0.0, 0.0, 0.0], [-1.4, 2.8, -1.4, 0.0, 0.0], [0.0, -1.4, 2.8, -1.4, 0.0], [0.0, 0.0, -1.4, 2.8, -1.4], [0.0, 0.0, 0.0, -1.4, 6.4]], 'b': [60.0, 0.0, 0.0, 0.0, 100.0]}, 'linear_0_dirichlet_neumann': {'A': [[1.0, 0.0, 0.0, 0.0, 0.0], [-1.4, 2.8, -1.4, 0.0, 0.0], [0.0, -1.4, 2.8, -1.4, 0.0], [0.0, 0.0, -1.4, 2.8, -1.4], [0.0, 0.0, 0.0, -1.4, 1.4]], 'b': [60.0, 0.0, 0.0, 0.0, -3.0]}, 'linear_1_dirichlet_newton': {'A': [[1.0, 0.0, 0.0, 0.0, 0.0], [-4.4, 2.8, 1.6, 0.0, 0.0], [0.0, -4.4, 2.8, 1.6, 0.0], [0.0, 0.0, -4.4, 2.8, 1.6], [0.0, 0.0, 0.0, -4.4, 9.4]], 'b': [60.0, 0.0, 0.0, 0.0, 100.0]}, 'linear_1_dirichlet_neumann': {'A': [[1.0, 0.0, 0.0, 0.0, 0.0], [-4.4, 2.8, 1.6, 0.0, 0.0], [0.0, -4.4, 2.8, 1.6, 0.0], [0.0, 0.0, -4.4, 2.8, 1.6], [0.0, 0.0, 0.0, -4.4, 4.4]], 'b': [60.0, 0.0, 0.0, 0.0, -3.0]}, 'mass': [[1.0, 0.5, 0.0, 0.0, 0.0], [0.5, 2.0, 0.5, 0.0, 0.0], [0.0, 0.5, 2.0, 0.5, 0.0], [0.0, 0.0, 0.5, 2.0, 0.5], [0.0, 0.0, 0.0, 0.5, 1.0]], 'radiation_state': [60.0, 53.75, 47.5, 41.25, 35.0], 'radiation_J': [[1.0, 0.0, 0.0, 0.0, 0.0], [-1.4, 2.8, -1.4, 0.0, 0.0], [0.0, -1.4, 2.8, -1.4, 0.0], [0.0, 0.0, -1.4, 2.8, -1.4], [0.0, 0.0, 0.0, -1.4, 6.045775757366]], 'radiation_R': [0.0, 0.0, 0.0, -0.0, 31.265612033057], 'transient_final': [60.0, 85.269466865097, 107.525725626455, 124.763491501244, 136.449267816168, 143.288297118472, 146.577024750477, 147.56666308003, 147.087957305837], 'radiation_solution': [60.0, 56.855623764843, 53.711247529687, 50.56687129453, 47.422495059374, 44.278118824217, 41.13374258906, 37.989366353904, 34.844990118747, 31.70061388359, 28.556237648434]}


def make_model(NoE=10, convection=False, bc='dirichlet_newton', T0=40.0):

    mesh = Mesh()
    mesh.Generate_Mesh(1, 2.0, NoE, ElementType.line)
    mesh.setElementShapeFunction(ShapeFunctionType.linear)

    model = Model(name='vec', dim=1, mesh=mesh, mat=material('X', k=0.7, rho=2.0, Cp=3.0), psc=ht)
    model.physics.Convection = convection
    model.physics.initField('T', T0)

    model.physics.addBC_Temperature(id=0, T=60.0)
    if bc == 'dirichlet_newton':
        model.physics.addBC_Convection(id=NoE, h_c=5.0, T_ext=20.0)
    elif bc == 'dirichlet_neumann':
        model.physics.addBC_HeatFlux(id=NoE, q_flux=3.0)
    elif bc == 'dirichlet_radiation':
        model.physics.addBC_Radiation(id=NoE, epsilon=0.7, T_ext=26.0)

    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear,
                              Method=SolverType.direct, Solver=DirectSolvers.PARDISO)
    return model


def test_linear_matrices_match_reference():
    for convection in (False, True):
        for bc in ('dirichlet_newton', 'dirichlet_neumann'):
            A, b = make_model(NoE=4, convection=convection, bc=bc).assembleGlobalSystem()
            ref = REFERENCE[f"linear_{int(convection)}_{bc}"]

            assert np.allclose(A.toarray(), ref["A"], atol=1e-10, rtol=1e-10), (convection, bc)
            assert np.allclose(b, ref["b"], atol=1e-10, rtol=1e-10), (convection, bc)


def test_mass_matrix_matches_reference():
    M = make_model(NoE=4).assembleMassMatrix().toarray()

    assert np.allclose(M, REFERENCE["mass"], atol=1e-10), "mass matrix"
    # rho * Cp * length: the total mass of the bar (length 2, rho Cp = 6)
    assert np.isclose(M.sum(), 2.0 * 6.0)


def test_nonlinear_residual_and_tangent_match_reference():
    model = make_model(NoE=4, bc='dirichlet_radiation')
    model.physics.var['T'].updateField(list(REFERENCE["radiation_state"]))

    J, R = model.assembleGlobalSystemNonLinear()

    assert np.allclose(R, REFERENCE["radiation_R"], atol=1e-8, rtol=1e-10)
    assert np.allclose(J.toarray(), REFERENCE["radiation_J"], atol=1e-8, rtol=1e-10)


def test_tangent_matches_finite_differences():
    """Radiation + convection: dR/dT from the system against finite differences (no reference needed)."""

    model = make_model(NoE=6, convection=True, bc='dirichlet_radiation')
    system = model._system()

    x = 30.0 + 40.0 * np.random.default_rng(0).random(model.layout.ndof)
    J, R = system.residual_and_tangent(x)

    eps = 1e-6
    J_fd = np.zeros(J.shape)
    for j in range(len(x)):
        xp = x.copy()
        xp[j] += eps
        J_fd[:, j] = (system.residual_and_tangent(xp)[1] - R) / eps

    assert np.allclose(J.toarray(), J_fd, rtol=1e-5, atol=1e-6), np.max(np.abs(J.toarray() - J_fd))


def test_transient_matches_reference():
    model = make_model(NoE=8, bc='dirichlet_neumann', T0=150.0)
    model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.linear,
                              Method=SolverType.direct, Solver=DirectSolvers.PARDISO, totalTime=2)
    model.solve()

    assert np.allclose(np.array(model.sol['T'])[:, -1], REFERENCE["transient_final"], atol=1e-6)


def test_nonlinear_solution_matches_reference_and_is_fast():
    """Full radiation problem (Newton)."""

    model = make_model(NoE=10, bc='dirichlet_radiation', T0=60.0)
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear,
                              Method=SolverType.direct, Solver=DirectSolvers.PARDISO)

    t0 = time.time()
    model.solve()
    elapsed = time.time() - t0

    print(f"   nonlinear solve: {elapsed:.2f} s")
    assert np.allclose(model.sol['T'], REFERENCE["radiation_solution"], atol=1e-8)
    assert elapsed < 10.0  # the old element-based path needed about 15 s


if __name__ == "__main__":
    tests = [test_linear_matrices_match_reference, test_mass_matrix_matches_reference,
             test_nonlinear_residual_and_tangent_match_reference, test_tangent_matches_finite_differences,
             test_transient_matches_reference, test_nonlinear_solution_matches_reference_and_is_fast]
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
