"""Checks that the vectorized 1D assembly reproduces the element-based one.

For each case the same model is assembled twice: with the vectorized path (default) and
with the element-based path (switched off on the physics instance). The results must match.
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


def use_element_path(model):
    model.physics.supports_vectorized_assembly = False


def test_linear_matrices_match():
    for convection in (False, True):
        for bc in ('dirichlet_newton', 'dirichlet_neumann'):
            fast = make_model(convection=convection, bc=bc)
            slow = make_model(convection=convection, bc=bc)
            use_element_path(slow)

            A_fast, b_fast = fast.assembleGlobalSystem()
            A_slow, b_slow = slow.assembleGlobalSystem()

            assert np.allclose(A_fast.toarray(), A_slow.toarray(), atol=1e-10, rtol=1e-10), (convection, bc)
            assert np.allclose(b_fast, b_slow, atol=1e-10, rtol=1e-10), (convection, bc)


def test_mass_matrix_matches():
    fast = make_model()
    slow = make_model()
    use_element_path(slow)

    M_fast = fast.assembleMassMatrix().toarray()
    M_slow = slow.assembleMassMatrix().toarray()

    assert np.allclose(M_fast, M_slow, atol=1e-10), "mass matrix"


def test_nonlinear_residual_and_tangent_match():
    rng = np.random.default_rng(0)

    for bc in ('dirichlet_radiation', 'dirichlet_newton'):
        fast = make_model(bc=bc)
        slow = make_model(bc=bc)
        use_element_path(slow)

        T = 40.0 + 30.0 * rng.random(fast.mesh.getNoN())  # some state away from the solution
        for model in (fast, slow):
            model.physics.var['T'].updateField(list(T))

        J_fast, R_fast = fast.assembleGlobalSystemNonLinear()
        J_slow, R_slow = slow.assembleGlobalSystemNonLinear()

        assert np.allclose(R_fast, R_slow, atol=1e-8, rtol=1e-10), (bc, R_fast - R_slow)
        assert np.allclose(J_fast.toarray(), J_slow.toarray(), atol=1e-8, rtol=1e-10), bc


def test_nonlinear_solution_matches_and_timing():
    """Full radiation problem (Newton). Solutions must agree; timings are printed."""

    results = {}
    for label, slow_path in (('element-based', True), ('vectorized', False)):
        model = make_model(NoE=10, bc='dirichlet_radiation', T0=60.0)
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear,
                                  Method=SolverType.direct, Solver=DirectSolvers.PARDISO)
        if slow_path:
            use_element_path(model)

        t0 = time.time()
        model.solve()
        results[label] = (np.array(model.sol['T']), time.time() - t0)

    T_slow, t_slow = results['element-based']
    T_fast, t_fast = results['vectorized']

    print(f"   nonlinear solve: element-based {t_slow:.2f} s, vectorized {t_fast:.2f} s")
    assert np.allclose(T_fast, T_slow, atol=1e-8), np.max(np.abs(T_fast - T_slow))


def test_transient_matches():
    """Short transient run (uses the mass matrix and the BDF integrator)."""

    results = {}
    for label, slow_path in (('element-based', True), ('vectorized', False)):
        model = make_model(NoE=8, bc='dirichlet_neumann', T0=150.0)
        model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.linear,
                                  Method=SolverType.direct, Solver=DirectSolvers.PARDISO, totalTime=2)
        if slow_path:
            use_element_path(model)
        model.solve()
        results[label] = np.array(model.sol['T'])[:, -1]

    assert np.allclose(results['vectorized'], results['element-based'], atol=1e-6), \
        np.max(np.abs(results['vectorized'] - results['element-based']))


if __name__ == "__main__":
    tests = [test_linear_matrices_match, test_mass_matrix_matches, test_transient_matches,
             test_nonlinear_residual_and_tangent_match, test_nonlinear_solution_matches_and_timing]
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
