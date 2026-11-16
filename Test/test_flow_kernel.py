"""Phase B of docs/PLAN_RENDIMIENTO_GPU_ML.md: the compiled residual of the flow against its numpy reference, and the
refinement on demand of the direct solver.

Run from the project root:  python Test/test_flow_kernel.py
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff, FlowTerm
from Source.Physics.HeatTransfer import ht
from Source.Physics.Rheology import CarreauYasuda, CrossWLF
from Source.Simulation.DirectSolver import FactorizedSolver
from Source.enums import ElementType, StudyType, ProblemType, DirectSolvers

QUAD, TRI, HEX, TET = ElementType.quadrilateral, ElementType.triangle, ElementType.hexahedron, ElementType.tetrahedron


def mesh_2d(element, order=1, n=3):
    mesh = Mesh()
    mesh.generate2DMesh((1.0, 1.0), (n, n), element, **({'order': 2} if order == 2 else {}))
    for node in mesh.NL:                                          # away from the axis, for the axisymmetric cases
        node.coor = (node.coor[0] + 0.5, node.coor[1])
    return mesh


def mesh_3d(element, order=1, n=2):
    mesh = Mesh()
    mesh.generate3DMesh((1.0, 1.0, 1.0), (n, n, n), element, **({'order': 2} if order == 2 else {}))
    return mesh


def flow_term(model):
    return next(term for term in model._system()._terms if isinstance(term, FlowTerm))


def compiled_matches(model, seed=5, tol=1e-12):

    """The compiled residual and tangent of every term that has them (FlowTerm, FieldConvectionTerm) against the numpy ones:
    steady, in two time steps and at the old level, with tau at the state and at another one."""

    system = model._system()
    terms = [term for term in system._terms if hasattr(type(term), 'compiled')]
    assert any(isinstance(term, FlowTerm) for term in terms)
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=0.3, size=system.layout.ndof)
    old = x + rng.normal(scale=0.01, size=system.layout.ndof)

    contexts = (None, {'alpha': 10.0, 'beta': -10.0 * old, 'theta': 0.5, 'x_old': old, 'dt': 0.1},
                {'alpha': 15.0, 'beta': -15.0 * old, 'theta': 1.0, 'x_old': old, 'dt': 0.1}, {'old_level': True})
    try:
        for context in contexts:
            system.time = context
            for term in terms:
                where = (type(term).__name__, context and sorted(context))
                term.compiled = False
                reference = term.contribution(x, tangent=False)[0]
                term.compiled = True
                fast = term.contribution(x, tangent=False)[0]
                error = np.max(np.abs(fast - reference))
                assert np.max(np.abs(reference)) > 0.0, where
                assert error <= tol * max(1.0, np.max(np.abs(reference))), (where, error)

                # The tangent, with tau at another state as the tests against finite differences use it, and at this one
                for tau_state in (None, old):
                    term.tau_state = tau_state
                    term.compiled = False
                    R_ref, J_ref = term.contribution(x)
                    term.compiled = True
                    R_fast, J_fast = term.contribution(x)
                    size = max(1.0, abs(J_ref).max())
                    assert np.max(np.abs(R_fast - R_ref)) <= tol * max(1.0, np.max(np.abs(R_ref))), where
                    assert abs(J_fast - J_ref).max() <= 1e-11 * size, (where, tau_state is None,
                                                                        abs(J_fast - J_ref).max() / size)
                term.tau_state = None
    finally:
        system.time = None
        for term in terms:
            term.compiled = True
            term.tau_state = None


def build(mesh, dim=2, law=None, inertia=True, heat=False, heating=False, buoyancy=False, axisymmetric=False, supg=False,
          force=None, mu=0.1):

    model = Model(name='flow', dim=dim, mesh=mesh, mat=material('X', miu=mu, rho=1.0, k=1.0, Cp=1.0),
                  psc=[ff, ht] if heat else ff, axisymmetric=axisymmetric)
    fluid = model.physics.fluid if heat else model.physics
    fluid.setInertia(inertia)
    if law is not None:
        fluid.setViscosityLaw(law)
    if heating:
        fluid.setViscousHeating(True)
    if buoyancy:
        fluid.setBoussinesq((0.0, -9.8) if dim == 2 else (0.0, 0.0, -9.8), 0.01, 0.5)
    if supg:
        fluid.setStabilization('SUPG')
    if force is not None:
        fluid.setBodyForce(force)
    if heat:                                                     # heat convected by the computed velocity, with SUPG
        model.physics.heat.setVelocity('u')
        model.physics.heat.setStabilization('SUPG')
        model.physics.heat.F_const = 0.7
    return model


carreau = CarreauYasuda(mu0=1.0, mu_inf=0.1, lam=1.0, a=2.0, n=0.5)
cross_wlf = CrossWLF(n=0.3, tau_star=5.0, D1=3.0, D2=-0.5, A1=2.0, A2=5.0)


# ----- tests -----

def test_compiled_residual_on_equal_order_elements():

    """Q4 and T3 with PSPG and the recovered stress: constant viscosity (V), a law without fields and with the temperature (Q),
    creeping and with inertia, body force, buoyancy, heating, axisymmetric."""

    for element in (QUAD, TRI):
        compiled_matches(build(mesh_2d(element), force=(0.3, -0.2)))                       # Navier-Stokes, constant mu
        compiled_matches(build(mesh_2d(element), law=carreau, inertia=False))              # creeping, a law alone
        compiled_matches(build(mesh_2d(element), law=carreau, heat=True, heating=True, buoyancy=True, force=(0.1, 0.2)))
        compiled_matches(build(mesh_2d(element), law=cross_wlf, heat=True, heating=True))  # mu(gamma, T)
        compiled_matches(build(mesh_2d(element), law=carreau, heat=True, heating=True, axisymmetric=True))
        compiled_matches(build(mesh_2d(element), axisymmetric=True))


def test_compiled_residual_on_taylor_hood_elements():

    """Q9 and T6: Galerkin with LSIC, and with SUPG/PSPG (stress divergence of the element)."""

    for element in (QUAD, TRI):
        compiled_matches(build(mesh_2d(element, 2, n=2)))
        compiled_matches(build(mesh_2d(element, 2, n=2), law=carreau, heat=True, heating=True, buoyancy=True))
        compiled_matches(build(mesh_2d(element, 2, n=2), supg=True))
        compiled_matches(build(mesh_2d(element, 2, n=2), law=carreau, supg=True, axisymmetric=True))


def test_compiled_residual_in_3d():

    for element, order in ((HEX, 1), (TET, 1), (HEX, 2), (TET, 2)):
        compiled_matches(build(mesh_3d(element, order, n=2 if order == 1 else 1), dim=3, force=(0.1, 0.2, -0.3)))
        compiled_matches(build(mesh_3d(element, order, n=2 if order == 1 else 1), dim=3, law=carreau, heat=True, heating=True,
                               buoyancy=True))


def test_compiled_convection_of_turbulence_fields():

    """Spalart-Allmaras and k-omega SST: their transport uses FieldConvectionTerm without the diffusion in the SUPG residual."""

    from Source.Physics.Turbulence import sa, sst
    from Source.Physics.Physics import FieldConvectionTerm

    for kind in (sa, sst):
        mesh = mesh_2d(QUAD)
        model = Model(name='t', dim=2, mesh=mesh, mat=material('X', miu=0.05, rho=1.0), psc=[ff, kind])
        model.physics.fluid.setInertia(True)
        model.physics.turbulence.setWalls(mesh.boundary_edges(lambda x, y: y < 1e-9))
        model.physics.turbulence.coupleFlow()
        assert any(isinstance(term, FieldConvectionTerm) for term in model._system()._terms)
        compiled_matches(model)


def test_compiled_residual_with_a_viscosity_by_region():

    mesh = mesh_2d(QUAD, n=4)
    model = build(mesh)
    centres = np.array([np.mean([mesh.NL[k].coor[1] for k in conn]) for conn in model.geometry.conn])
    model.physics.setViscosity(np.where(centres < 0.5, 0.1, 2.0)[:, None])
    compiled_matches(model)


def test_compiled_residual_is_faster():

    """Not a strict benchmark (the machine may be busy): on 6400 Q4 elements the compiled residual must be the faster one."""

    mesh = Mesh()
    mesh.generate2DMesh((1.0, 1.0), (80, 80), QUAD)
    model = build(mesh, law=carreau)
    term = flow_term(model)
    x = np.random.default_rng(0).normal(scale=0.3, size=model.layout.ndof)

    def best(compiled):
        term.compiled = compiled
        term.contribution(x, tangent=False)                      # compiles
        times = []
        for _ in range(5):
            start = time.perf_counter()
            term.contribution(x, tangent=False)
            times.append(time.perf_counter() - start)
        return min(times)

    slow, fast = best(False), best(True)
    print(f"   residual of 6400 Q4 with a viscosity law: numpy {1e3 * slow:.1f} ms, compiled {1e3 * fast:.1f} ms")
    assert fast < 0.8 * slow, (slow, fast)


def test_transient_with_the_compiled_residual_matches_numpy():

    def run(compiled):
        mesh = Mesh()
        mesh.generate2DMesh((1.0, 1.0), (12, 12), QUAD)
        model = build(mesh, law=carreau, mu=0.05)
        model.physics.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), (1.0, 0.0))
        model.physics.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or x < 1e-9 or x > 1 - 1e-9))
        model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, timeScheme='BDF2', timeStep=0.05,
                                  totalTime=0.5)
        FlowTerm.compiled = compiled
        try:
            model.solve()
        finally:
            FlowTerm.compiled = True
        return np.asarray(model.sol['u'])[..., -1], np.asarray(model.sol['p'])[..., -1], model.tangent_assemblies

    u_fast, p_fast, tangents_fast = run(True)
    u_slow, p_slow, tangents_slow = run(False)
    assert tangents_fast == tangents_slow
    assert np.max(np.abs(u_fast - u_slow)) < 1e-9, np.max(np.abs(u_fast - u_slow))
    assert np.max(np.abs(p_fast - p_slow)) < 1e-8, np.max(np.abs(p_fast - p_slow))


def test_direct_solver_refines_on_demand():

    """Without the two refinement steps PARDISO runs by default the residual stays at round-off on a well-conditioned
    matrix, and a badly scaled one is still refined to the same accuracy."""

    rng = np.random.default_rng(1)
    n = 400
    A = (sp.random(n, n, density=0.03, random_state=3) + 8.0 * sp.identity(n)).tocsr()
    scale = sp.diags(10.0 ** rng.uniform(-2, 2, n))
    hard = (scale @ A @ scale).tocsr()
    b = rng.random(n)

    for matrix in (A, hard):
        answers = []
        for on_demand in (True, False):
            solver = FactorizedSolver.__new__(FactorizedSolver)
            solver.refine_on_demand = on_demand
            solver.__init__(DirectSolvers.PARDISO)
            x = solver.solve(matrix, b)
            assert np.linalg.norm(matrix @ x - b) <= 1e-12 * np.linalg.norm(b)
            answers.append(x)
        assert np.max(np.abs(answers[0] - answers[1])) <= 1e-12 * np.max(np.abs(answers[1]))


def test_threaded_sparse_product_is_the_same_matrix():

    import Source.Simulation.SparseTools as tools

    A = sp.random(3000, 2500, density=0.004, random_state=1, format='csr')
    B = sp.random(2500, 1800, density=0.004, random_state=2, format='csr')
    A[10:40] = 0.0                                               # empty rows, and blocks with very different sizes
    A.eliminate_zeros()
    reference = (A @ B).tocsr()
    limit = tools.MIN_ENTRIES
    tools.MIN_ENTRIES = 0
    try:
        product = tools.matmat(A, B)
    finally:
        tools.MIN_ENTRIES = limit
    assert product.shape == reference.shape and product.nnz == reference.nnz
    assert abs(product - reference).max() == 0.0


def test_newton_stops_at_the_round_off_floor():

    """A Newtonian melt in a Taylor-Hood slit die with convected heat: one Newton step solves it, but the linear solves of
    the saddle point leave the residual at 1e-9 of the initial one, above the relative tolerance. Without the stagnation
    test, Newton went on with the smallest step of the line search until a full one happened to pass: 2 to 175 iterations
    from one run to the next (PARDISO in parallel is not reproducible to round-off). That stall also hid a refinement of the
    direct solver that did not work on these matrices (commit 641ad6b)."""

    H, L, T_in = 1.0e-3, 20.0e-3, 230.0
    mu = float(CrossWLF(n=0.28, tau_star=2.5e4, D1=5.0e13, D2=263.15, A1=30.0, A2=51.6).zero_shear(T_in))     # a PP melt

    def solve(kind):
        mesh = Mesh()
        mesh.generate2DMesh((L, H), (80, 16), QUAD, order=2)
        model = Model(name='slit', dim=2, mesh=mesh, mat=material('PP', miu=mu, rho=750.0, Cp=2700.0, k=0.17), psc=[ff, ht])
        fluid, heat = model.physics.fluid, model.physics.heat
        inlet = mesh.boundary_edges(lambda x, y: x < 1e-12)
        wall = mesh.boundary_edges(lambda x, y: y > H - 1e-12)
        fluid.addBC_VelocityEdges(inlet, lambda x, y: (0.15 * (1 - (y / H) ** 2), 0.0))
        fluid.addBC_SymmetryEdges(mesh.boundary_edges(lambda x, y: y < 1e-12))
        fluid.addBC_NoSlipEdges(wall)
        fluid.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x > L - 1e-12), 0.0)
        heat.addBC_TemperatureEdges(inlet, T_in)
        heat.addBC_TemperatureEdges(wall, T_in)
        model.physics.setVelocity('u')
        model.physics.setStabilization('SUPG')
        model.physics.initField('T', T_in)
        model.solverConfiguration(Study=StudyType.steady_state, Type=kind)
        model.solve()
        return model

    reference = np.asarray(solve(ProblemType.linear).sol['u'])        # one Newton step from rest: the problem is linear in u, p

    for _ in range(3):
        model = solve(ProblemType.nonlinear)
        assert model.nonlinear_converged and model.newton_iterations <= 4, model.newton_iterations
        error = np.max(np.abs(np.asarray(model.sol['u']) - reference)) / np.max(np.abs(reference))
        assert error < 1e-7, error


if __name__ == "__main__":
    tests = [test_compiled_residual_on_equal_order_elements, test_compiled_residual_on_taylor_hood_elements,
             test_compiled_residual_in_3d, test_compiled_convection_of_turbulence_fields,
             test_compiled_residual_with_a_viscosity_by_region,
             test_compiled_residual_is_faster, test_transient_with_the_compiled_residual_matches_numpy,
             test_direct_solver_refines_on_demand, test_threaded_sparse_product_is_the_same_matrix,
             test_newton_stops_at_the_round_off_floor]
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
