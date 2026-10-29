"""Stage 11 of the fluids plan: steady incompressible Navier-Stokes with SUPG/PSPG/LSIC on equal-order elements.

Run from the project root:  python Test/test_navier_stokes.py
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
from Source.Physics.FluidFlow import ff
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers

# Ghia, Ghia & Shin (1982), Re = 100: u along the vertical centerline x = 0.5
GHIA_RE100 = np.array([[1.0000, 1.00000], [0.9766, 0.84123], [0.9688, 0.78871], [0.9609, 0.73722], [0.9531, 0.68717],
                       [0.8516, 0.23151], [0.7344, 0.00332], [0.6172, -0.13641], [0.5000, -0.20581],
                       [0.4531, -0.21090], [0.2813, -0.15662], [0.1719, -0.10150], [0.1016, -0.06434],
                       [0.0703, -0.04775], [0.0625, -0.04192], [0.0547, -0.03717], [0.0000, 0.00000]])


def configure(model, kind=ProblemType.nonlinear):
    model.solverConfiguration(Study=StudyType.steady_state, Type=kind, Method=SolverType.direct,
                              Solver=DirectSolvers.PARDISO)


def rect_mesh(size, div, element=ElementType.quadrilateral, origin=(0.0, 0.0)):
    mesh = Mesh()
    mesh.generate2DMesh(size, div, element)
    for node in mesh.NL:
        node.coor = (node.coor[0] + origin[0], node.coor[1] + origin[1])
    return mesh


def xy_of(mesh):
    return np.array([n.coor[:2] for n in mesh.NL], dtype=float)


def kovasznay(Re):

    """Exact Navier-Stokes solution (Kovasznay 1948) with rho = 1, mu = 1 / Re, and its traction on x = const."""

    lam = Re / 2 - np.sqrt(Re ** 2 / 4 + 4 * np.pi ** 2)
    mu = 1.0 / Re
    k = 2 * np.pi

    U = lambda x, y: np.stack(np.broadcast_arrays(1 - np.exp(lam * x) * np.cos(k * y),
                                                  lam / k * np.exp(lam * x) * np.sin(k * y)), axis=-1)
    P = lambda x, y: 0.5 * (1 - np.exp(2 * lam * x))

    def traction(x, y):
        e = np.exp(lam * x)
        ux, uy, vx = -lam * e * np.cos(k * y), k * e * np.sin(k * y), lam ** 2 / k * e * np.sin(k * y)
        return (-P(x, y) + 2 * mu * ux, mu * (uy + vx))

    return U, P, traction, mu


def solve_kovasznay(n, element, Re=40.0, linearization='newton'):

    U, P, traction, mu = kovasznay(Re)
    mesh = rect_mesh((1.5, 2.0), (n, int(round(n * 4 / 3))), element, origin=(-0.5, -0.5))
    model = Model(name='kovasznay', dim=2, mesh=mesh, mat=material('X', miu=mu, rho=1.0), psc=ff)
    model.physics.setInertia(True, linearization)
    model.physics.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: x < 1 - 1e-9), lambda x, y: tuple(U(x, y)))
    model.physics.addBC_TractionEdges(mesh.boundary_edges(lambda x, y: x > 1 - 1e-9), traction)
    configure(model)
    model.solve()

    xy = xy_of(mesh)
    return (model, np.max(np.abs(model.sol['u'] - U(xy[:, 0], xy[:, 1]))),
            np.max(np.abs(model.sol['p'] - P(xy[:, 0], xy[:, 1]))))


def cavity(n, Re, element=ElementType.quadrilateral):

    mesh = rect_mesh((1.0, 1.0), (n, n), element)
    model = Model(name='cavity', dim=2, mesh=mesh, mat=material('X', miu=1.0 / Re, rho=1.0), psc=ff)
    model.physics.setInertia(True)
    model.physics.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), (1.0, 0.0))
    model.physics.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or x < 1e-9 or x > 1 - 1e-9))
    configure(model)
    model.solve()
    return model, mesh


def centerline_u(model, mesh, y):
    xy = xy_of(mesh)
    line = np.abs(xy[:, 0] - 0.5) < 1e-9
    order = np.argsort(xy[line, 1])
    return np.interp(y, xy[line, 1][order], model.sol['u'][line, 0][order])


# ----- tests -----

def test_tangent_against_finite_differences():

    """Newton tangent of convection + SUPG + PSPG + LSIC + recovered stress, with tau frozen at the state."""

    for element in (ElementType.quadrilateral, ElementType.triangle):
        mesh = rect_mesh((1.0, 1.0), (3, 3), element)
        model = Model(name='tangent', dim=2, mesh=mesh, mat=material('X', miu=0.05, rho=1.3), psc=ff)
        model.physics.setInertia(True)
        model.physics.setBodyForce(lambda x, y: (np.sin(x), x * y))

        layout = model.layout
        x = np.random.default_rng(1).normal(size=layout.ndof)
        system = model._system()
        system._terms[0].tau_state = x.copy()

        J, _ = system._assemble(x)
        h = 1e-6
        FD = np.zeros((layout.ndof, layout.ndof))
        for k in range(layout.ndof):
            e = np.zeros(layout.ndof)
            e[k] = h
            FD[:, k] = (system._assemble(x + e)[1] - system._assemble(x - e)[1]) / (2 * h)

        assert np.max(np.abs(J.toarray() - FD)) < 1e-8 * np.max(np.abs(FD)), np.max(np.abs(J.toarray() - FD))


def test_kovasznay_flow_converges():

    for element in (ElementType.quadrilateral, ElementType.triangle):
        eu, ep, its = [], [], []
        for n in (8, 16, 32):
            model, err_u, err_p = solve_kovasznay(n, element)
            eu.append(err_u)
            ep.append(err_p)
            its.append(model.newton_iterations)

        ru = np.log2(eu[-2] / eu[-1])
        rp = np.log2(ep[-2] / ep[-1])
        print(f"   {element.name}: u {['%.1e' % e for e in eu]} rate {ru:.2f}; p {['%.1e' % e for e in ep]} rate {rp:.2f}; "
              f"Newton iterations {its}")
        assert ru > 1.6 and rp > 1.5 and eu[-1] < 5e-3 and max(its) <= 8


def test_lid_driven_cavity_re100_against_ghia():

    for element in (ElementType.quadrilateral, ElementType.triangle):
        model, mesh = cavity(32, 100.0, element)
        u = centerline_u(model, mesh, GHIA_RE100[:, 0])
        diff = np.max(np.abs(u - GHIA_RE100[:, 1]))
        print(f"   {element.name} 32x32: max |u - Ghia| = {diff:.4f}, u_min = {u.min():.4f}, "
              f"{model.newton_iterations} Newton iterations")
        assert diff < 0.01


def test_fully_developed_channel_is_exact_with_inertia():

    """Poiseuille flow has no convective acceleration: with inertia the Q4 solution is still nodally exact
    (SUPG, PSPG and grad-div terms vanish for the exact solution)."""

    L, H, mu, rho, Umax = 3.0, 1.0, 0.01, 1.0, 1.0       # Re = rho Umax H / mu = 100
    mesh = rect_mesh((L, H), (12, 6))
    model = Model(name='channel', dim=2, mesh=mesh, mat=material('X', miu=mu, rho=rho), psc=ff)
    model.physics.setInertia(True)
    model.physics.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: x < 1e-9), lambda x, y: (4 * Umax * y * (H - y), 0.0))
    model.physics.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x > L - 1e-9))
    model.physics.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or y > H - 1e-9))
    configure(model)
    model.solve()

    xy = xy_of(mesh)
    assert np.max(np.abs(model.sol['u'][:, 0] - 4 * Umax * xy[:, 1] * (H - xy[:, 1]))) < 1e-10
    assert np.max(np.abs(model.sol['u'][:, 1])) < 1e-10
    assert np.max(np.abs(model.sol['p'] - 8 * mu * Umax * (L - xy[:, 0]))) < 1e-10


def test_low_reynolds_limit_is_stokes():

    results = []
    for inertia in (False, True):
        mesh = rect_mesh((1.0, 1.0), (12, 12), ElementType.triangle)
        model = Model(name='limit', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1e-6), psc=ff)
        if inertia:
            model.physics.setInertia(True)
        model.physics.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), (1.0, 0.0))
        model.physics.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or x < 1e-9 or x > 1 - 1e-9))
        configure(model, ProblemType.nonlinear if inertia else ProblemType.linear)
        model.solve()
        results.append((model.sol['u'], model.sol['p']))

    (us, ps), (un, pn) = results
    assert np.max(np.abs(us - un)) < 1e-6 and np.max(np.abs(ps - pn)) < 1e-5


def test_picard_reaches_the_same_solution():

    newton, eu_n, _ = solve_kovasznay(12, ElementType.quadrilateral)
    picard, eu_p, _ = solve_kovasznay(12, ElementType.quadrilateral, linearization='picard')

    print(f"   Newton {newton.newton_iterations} iterations, Picard {picard.newton_iterations}")
    assert np.max(np.abs(newton.sol['u'] - picard.sol['u'])) < 1e-8
    assert newton.newton_iterations < picard.newton_iterations


def test_errors():

    mesh = rect_mesh((1.0, 1.0), (2, 2))
    model = Model(name='err', dim=2, mesh=mesh, mat=material('X', miu=1.0), psc=ff)

    for call, text in ((lambda: model.physics.setInertia(True), 'densidad'),
                       (lambda: model.physics.setInertia(True, 'secant'), 'Linealizacion')):
        try:
            call()
        except ValueError as error:
            assert text in str(error), str(error)
        else:
            raise AssertionError(f"expected an error containing {text!r}")


if __name__ == "__main__":
    tests = [test_tangent_against_finite_differences, test_kovasznay_flow_converges,
             test_lid_driven_cavity_re100_against_ghia, test_fully_developed_channel_is_exact_with_inertia,
             test_low_reynolds_limit_is_stokes, test_picard_reaches_the_same_solution, test_errors]
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
