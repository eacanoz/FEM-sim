"""Checks for the scalar physics in 3D (phase 1 of docs/PLAN_3D.md): heat, species and Darcy on tetrahedra and hexahedra,
through the same Model and physics as in 2D.

Run from the project root:  python Test/test_3d_scalars.py
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
from Source.Physics.MassTransfer import mt
from Source.Physics.Darcy import darcy
from Source.Simulation.Discretization import Discretization
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers

KIND = {'tet4': (ElementType.tetrahedron, 1), 'tet10': (ElementType.tetrahedron, 2),
        'hex8': (ElementType.hexahedron, 1), 'hex27': (ElementType.hexahedron, 2)}


def box(name, n, L=(1.0, 1.0, 1.0)):
    mesh = Mesh()
    mesh.generate3DMesh(L, n if np.ndim(n) else (n, n, n), *KIND[name])
    return mesh


def xyz_of(mesh):
    return np.array([node.coor for node in mesh.NL], dtype=float)


def configure(model, study=StudyType.steady_state, kind=ProblemType.linear, **time_options):
    model.solverConfiguration(Study=study, Type=kind, Method=SolverType.direct, Solver=DirectSolvers.PARDISO, **time_options)


def heat_model(mesh, k=1.0, rho_cp=1.0):
    model = Model(name='3d', dim=3, mesh=mesh, mat=material('X', k=k, rho=rho_cp, Cp=1.0), psc=ht)
    model.physics.Convection = False
    return model


def l2_error(mesh, values, exact):
    disc = Discretization(mesh)
    u_gp = disc.interpolate(np.asarray(values, dtype=float))
    return np.sqrt(np.sum(disc.detJw * (u_gp - exact(*np.moveaxis(disc.x_gp, -1, 0))) ** 2))


def rates_of(errors):
    return [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]


# ----- heat -----

def test_heat_patch_with_flux_and_convection_faces():
    """T = 2x + 1 with Dirichlet on x = 0, the flux -k dT/dx = -2 leaving through x = 2 (Neumann) or convection with
    h = 4 and T_ext = 5.5 there; insulated y and z faces. Exact on every element type, distorted or not."""

    f = lambda x, y, z: 2 * x + 1
    for name in KIND:
        for condition in ('flux', 'convection'):
            mesh = box(name, (4, 2, 2), (2.0, 1.0, 1.0))
            model = heat_model(mesh)
            model.physics.addBC_TemperatureEdges(mesh.boundary_faces(lambda x, y, z: x < 1e-12), f)
            right = mesh.boundary_faces(lambda x, y, z: x > 2.0 - 1e-12)
            if condition == 'flux':
                model.physics.addBC_HeatFluxEdges(right, -2.0)
            else:
                model.physics.addBC_ConvectionEdges(right, 4.0, 5.5)
            configure(model)
            model.solve()
            err = np.max(np.abs(np.asarray(model.sol['T']) - f(*xyz_of(mesh).T)))
            assert err < 1e-10, (name, condition, err)


def test_heat_poisson_manufactured():
    """-k lap T = 3 k pi^2 sin sin sin in the unit cube through ht: L2 order 2 (hex8, tet4) and 3 (hex27, tet10)."""

    exact = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
    k = 2.5

    for name, sizes, expected in (('hex8', (4, 8, 16), 2.0), ('tet4', (4, 8, 16), 2.0),
                                  ('hex27', (2, 4, 8), 3.0), ('tet10', (2, 4, 8), 3.0)):
        start = time.perf_counter()
        errors = []
        for n in sizes:
            mesh = box(name, n)
            model = heat_model(mesh, k=k)
            model.physics.F_const = lambda x, y, z: 3 * k * np.pi ** 2 * exact(x, y, z)
            model.physics.addBC_TemperatureEdges(mesh.boundary_faces(), 0.0)
            configure(model)
            model.solve()
            errors.append(l2_error(mesh, model.sol['T'], exact))
        rates = rates_of(errors)
        print(f"   {name}: L2 {['%.2e' % e for e in errors]}, rates {['%.2f' % r for r in rates]} "
              f"({time.perf_counter() - start:.1f} s)")
        assert rates[-1] > expected - 0.15, (name, rates)


def cube_series(x, y, z, t, alpha=1.0, terms=40):

    """Unit cube at T = 1 with its faces at T = 0 from t = 0: T = sum over odd m, n, p of
    64 / (pi^3 m n p) sin(m pi x) sin(n pi y) sin(p pi z) exp(-alpha pi^2 (m^2 + n^2 + p^2) t)."""

    odd = np.arange(1, 2 * terms, 2)
    s = [np.sin(np.pi * np.multiply.outer(c, odd)) / odd for c in (x, y, z)]
    decay = np.exp(-alpha * np.pi ** 2 * t * (odd[:, None, None] ** 2 + odd[None, :, None] ** 2 + odd[None, None, :] ** 2))
    return 64 / np.pi ** 3 * np.einsum('...m,...n,...p,mnp->...', s[0], s[1], s[2], decay)


def test_transient_cube_against_the_series():
    """Cooling of a unit cube (T = 1 inside, faces at 0, alpha = 1) with BDF2: the centre and the whole field at t = 0.05 match
    the Fourier series; halving dt divides the time error by about 4."""

    alpha, t_end = 1.0, 0.05
    centre = float(cube_series(0.5, 0.5, 0.5, t_end, alpha))

    results = {}
    for name, n, dt in (('hex27', 6, 0.0025), ('hex27', 6, 0.00125), ('tet10', 6, 0.0025)):
        mesh = box(name, n)
        model = heat_model(mesh, k=alpha)
        model.physics.initField('T', 1.0)
        model.physics.addBC_TemperatureEdges(mesh.boundary_faces(), 0.0)
        configure(model, StudyType.transient, timeScheme='BDF2', timeStep=dt, totalTime=t_end)
        model.solve()

        xyz = xyz_of(mesh)
        T_end = np.asarray(model.sol['T'])[:, -1]
        k_centre = int(np.argmin(np.linalg.norm(xyz - 0.5, axis=1)))
        exact = cube_series(xyz[:, 0], xyz[:, 1], xyz[:, 2], t_end, alpha)
        results[(name, dt)] = (abs(T_end[k_centre] - centre), np.max(np.abs(T_end - exact)))
        print(f"   {name} n = {n}, dt = {dt}: centre {T_end[k_centre]:.6f} (series {centre:.6f}), "
              f"max error {results[(name, dt)][1]:.2e}")

    assert results[('hex27', 0.0025)][0] < 2e-3 and results[('tet10', 0.0025)][0] < 3e-3
    assert results[('hex27', 0.0025)][1] < 1e-2 and results[('tet10', 0.0025)][1] < 1.5e-2
    ratio = results[('hex27', 0.0025)][0] / results[('hex27', 0.00125)][0]
    assert ratio > 2.5, ratio


def boundary_layer_exact(x, Pe):
    return (np.exp(Pe * x) - 1.0) / (np.exp(Pe) - 1.0)


def test_supg_boundary_layer_3d_equals_1d():
    """v = (1, 0, 0), T(0) = 0, T(1) = 1, Pe = 50, insulated sides: with SUPG the hexahedra are nodally exact (the 1D optimal
    upwinding); Galerkin oscillates. On tetrahedra SUPG cuts the error of Galerkin by more than 3 but does not remove it."""

    Pe = 50.0
    errors = {}
    for name in ('hex8', 'tet4'):
        for stab in ('SUPG', None):
            mesh = box(name, (10, 2, 2))
            model = Model(name='cd', dim=3, mesh=mesh, mat=material('X', k=1.0 / Pe, rho=1.0, Cp=1.0), psc=ht)
            model.physics.setVelocity([1.0, 0.0, 0.0])
            if stab:
                model.physics.setStabilization(stab)
            model.physics.addBC_TemperatureEdges(mesh.boundary_faces(lambda x, y, z: x < 1e-12), 0.0)
            model.physics.addBC_TemperatureEdges(mesh.boundary_faces(lambda x, y, z: x > 1 - 1e-12), 1.0)
            model.physics.initField('T', 0.0)
            configure(model)
            model.solve()
            x = xyz_of(mesh)[:, 0]
            errors[(name, stab)] = np.max(np.abs(np.asarray(model.sol['T']) - boundary_layer_exact(x, Pe)))
            print(f"   {name} {stab or 'Galerkin'}: max nodal error {errors[(name, stab)]:.2e}")

    assert errors[('hex8', 'SUPG')] < 1e-10
    assert errors[('hex8', None)] > 0.1 and errors[('tet4', None)] > 0.1
    # Tetrahedra, like triangles in 2D, keep an undershoot before the layer (0.33 here against 0.19 for T3 on the same grid, and
    # 0.16 / 0.10 with 40 cells): SUPG is not monotone on simplices. Only the improvement over Galerkin is checked.
    assert errors[('tet4', 'SUPG')] < errors[('tet4', None)] / 3


def test_supg_manufactured_convection_diffusion():
    """c v . grad T - k lap T = f with v = (1, 0.5, 0.25), k = 0.02 and T = sin sin sin: SUPG is consistent, second order
    (nodal) on hex8 and tet4, and works on the quadratic elements (where it includes the diffusion in its residual)."""

    k, v = 0.02, np.array([1.0, 0.5, 0.25])
    s = lambda x: np.sin(np.pi * x)
    c = lambda x: np.pi * np.cos(np.pi * x)
    u = lambda x, y, z: s(x) * s(y) * s(z)
    f = lambda x, y, z: (v[0] * c(x) * s(y) * s(z) + v[1] * s(x) * c(y) * s(z) + v[2] * s(x) * s(y) * c(z)
                         + 3 * k * np.pi ** 2 * u(x, y, z))

    def error(name, n):
        mesh = box(name, n)
        model = Model(name='mms', dim=3, mesh=mesh, mat=material('X', k=k, rho=1.0, Cp=1.0), psc=ht)
        model.physics.setVelocity(list(v))
        model.physics.setStabilization('SUPG')
        model.physics.F_const = f
        model.physics.addBC_TemperatureEdges(mesh.boundary_faces(), 0.0)
        model.physics.initField('T', 0.0)
        configure(model)
        model.solve()
        return np.max(np.abs(np.asarray(model.sol['T']) - u(*xyz_of(mesh).T)))

    for name, sizes, expected in (('hex8', (8, 16), 1.8), ('tet4', (8, 16), 1.5), ('hex27', (4, 8), 2.5),
                                  ('tet10', (4, 8), 2.5)):
        errors = [error(name, n) for n in sizes]
        rate = rates_of(errors)[0]
        print(f"   {name}: nodal max {['%.2e' % e for e in errors]}, rate {rate:.2f}")
        assert rate > expected, (name, errors)


# ----- species -----

def test_species_diffusion_reaction_3d_equals_1d():
    """D lap c = k c with c = 1 on x = 0, c = 0 on x = 1 and no flux on the sides: the 1D profile sinh(m (1 - x)) / sinh(m)
    with L2 order 2 (hex8) and 3 (tet10); a source given as f(x, y, z) and the face flux work too."""

    D, k = 0.5, 2.0
    m = np.sqrt(k / D)
    exact = lambda x, y, z: np.sinh(m * (1 - x)) / np.sinh(m)

    for name, sizes, expected in (('hex8', (4, 8, 16), 2.0), ('tet10', (2, 4, 8), 3.0)):
        errors = []
        for n in sizes:
            mesh = box(name, (n, 2, 2))
            model = Model(name='species', dim=3, mesh=mesh, mat=material('X', k=1, rho=1, Cp=1), psc=mt)
            model.physics.setChemSpecies('A', 'A')
            model.physics.setDiffusivity('A', D)
            model.physics.addReaction(stoich={'A': -1}, k0=k, E_R=0.0, T=300.0, order={'A': 1})
            model.physics.addBC_ConcentrationEdges(mesh.boundary_faces(lambda x, y, z: x < 1e-12), 'A', 1.0)
            model.physics.addBC_ConcentrationEdges(mesh.boundary_faces(lambda x, y, z: x > 1 - 1e-12), 'A', exact)
            configure(model)
            model.solve()
            errors.append(l2_error(mesh, model.sol['A'], exact))
        rates = rates_of(errors)
        print(f"   {name}: L2 {['%.2e' % e for e in errors]}, rates {['%.2f' % r for r in rates]}")
        assert rates[-1] > expected - 0.15, (name, rates)

    # Flux J leaving through x = 1 (-D dc/dx = J), c = 0 at x = 0, no reaction: c = -J x / D
    J = 0.3
    mesh = box('tet4', (3, 2, 2))
    model = Model(name='flux', dim=3, mesh=mesh, mat=material('X', k=1, rho=1, Cp=1), psc=mt)
    model.physics.setChemSpecies('A', 'A')
    model.physics.setDiffusivity('A', D)
    model.physics.addBC_ConcentrationEdges(mesh.boundary_faces(lambda x, y, z: x < 1e-12), 'A', 0.0)
    model.physics.addBC_FluxEdges(mesh.boundary_faces(lambda x, y, z: x > 1 - 1e-12), 'A', J)
    configure(model)
    model.solve()
    assert np.max(np.abs(np.asarray(model.sol['A']) + J * xyz_of(mesh)[:, 0] / D)) < 1e-10


# ----- Darcy -----

def test_darcy_3d_tensor_permeability_and_gravity():
    """Linear pressure with an anisotropic 3 x 3 permeability and gravity along -z: pressure and velocity exact; a closed box
    with the top at p0 is hydrostatic with no flow."""

    kappa = np.array([[2.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.5]])
    mu, rho, g = 0.8, 1.2, np.array([0.0, 0.0, -9.81])
    grad = np.array([-1.5, 2.0, 0.5])
    p_exact = lambda x, y, z: 4.0 + grad[0] * x + grad[1] * y + grad[2] * z

    for name in ('tet4', 'hex8', 'tet10', 'hex27'):
        mesh = box(name, (3, 2, 2), (2.0, 1.0, 1.0))
        model = Model(name='darcy3d', dim=3, mesh=mesh, mat=material('soil', miu=mu, rho=rho), psc=darcy)
        model.physics.setPermeability(kappa)
        model.physics.setGravity(g)
        model.physics.addBC_PressureEdges(mesh.boundary_faces(), p_exact)
        configure(model)
        model.solve()
        assert np.max(np.abs(model.sol['p'] - p_exact(*xyz_of(mesh).T))) < 1e-10, name
        u_exact = -kappa @ (grad - rho * g) / mu
        assert np.allclose(model.sol['u'], u_exact, atol=1e-10), name
        assert np.asarray(model.sol['u']).shape == (mesh.getNoN(), 3)

    rho, H = 1000.0, 2.0
    mesh = box('hex8', (2, 2, 4), (1.0, 1.0, H))
    model = Model(name='hydro', dim=3, mesh=mesh, mat=material('water', kappa=1e-3, miu=1e-3, rho=rho), psc=darcy)
    model.physics.setGravity((0.0, 0.0, -9.81))
    model.physics.addBC_PressureEdges(mesh.boundary_faces(lambda x, y, z: z > H - 1e-9), 1e5)
    configure(model)
    model.solve()
    assert np.max(np.abs(model.sol['p'] - (1e5 + rho * 9.81 * (H - xyz_of(mesh)[:, 2])))) < 1e-7
    assert np.max(np.abs(model.sol['u'])) < 1e-10 * rho * 9.81


def test_darcy_injection_convects_heat_3d():
    """Injection U through x = 0 of a porous bar (p = 0 at x = L): uniform velocity (U, 0, 0); its Darcy velocity carries the
    heat equation to the same solution as the prescribed constant velocity."""

    L, kappa, mu, U = 3.0, 0.5, 2.0, 0.1
    mesh = box('tet4', (6, 2, 2), (L, 1.0, 1.0))
    flow = Model(name='inj', dim=3, mesh=mesh, mat=material('soil', kappa=kappa, miu=mu), psc=darcy)
    flow.physics.addBC_NormalVelocityEdges(mesh.boundary_faces(lambda x, y, z: x < 1e-9), -U)
    flow.physics.addBC_PressureEdges(mesh.boundary_faces(lambda x, y, z: x > L - 1e-9), 0.0)
    configure(flow)
    flow.solve()
    assert np.max(np.abs(flow.sol['p'] - mu * U * (L - xyz_of(mesh)[:, 0]) / kappa)) < 1e-10
    assert np.allclose(flow.sol['u'], [U, 0.0, 0.0], atol=1e-12)

    def transport(velocity):
        model = Model(name='heat', dim=3, mesh=mesh, mat=material('X', k=0.05, rho=1.0, Cp=1.0), psc=ht)
        model.physics.setVelocity(velocity)
        model.physics.setStabilization('SUPG')
        model.physics.addBC_TemperatureEdges(mesh.boundary_faces(lambda x, y, z: x < 1e-9), 1.0)
        model.physics.addBC_TemperatureEdges(mesh.boundary_faces(lambda x, y, z: x > L - 1e-9), 0.0)
        model.physics.initField('T', 0.0)
        configure(model)
        model.solve()
        return np.asarray(model.sol['T'])

    assert np.allclose(transport(flow.physics.velocity()), transport([U, 0.0, 0.0]), atol=1e-11)


def test_auto_solver_in_3d():
    """Method auto: a symmetric 3D scalar system above ITERATIVE_SCALAR_3D free unknowns goes to CG + AMG (iterations counted,
    same solution as PARDISO); a small one stays direct; non-symmetric scalars keep the 2D threshold, flow systems go to the block
    preconditioner from ITERATIVE_FLOW_3D."""

    from Source.Simulation import SolverSelection as selection
    from Source.enums import IterativeSolvers, Preconditioners

    big = selection.ITERATIVE_SCALAR_3D
    assert selection.choose_options(big, True, False, dim=3)['prec'] == Preconditioners.AMG
    assert selection.choose_options(big - 1, True, False, dim=3)['Method'] == SolverType.direct
    assert selection.choose_options(big, False, False, dim=3) == selection.choose_options(big, False, False)
    assert selection.choose_options(big, True, True, dim=3) == selection.choose_options(big, True, True)
    flow = selection.ITERATIVE_FLOW_3D
    assert selection.choose_options(flow, False, True, dim=3)['prec'] == Preconditioners.SaddlePoint
    assert selection.choose_options(flow - 1, False, True, dim=3)['Method'] == SolverType.direct
    assert selection.choose_options(flow, False, True)['Method'] == SolverType.direct                 # 2D: 500 k
    assert selection.choose_options(big, True, False)['Method'] == SolverType.direct          # 2D unchanged

    solutions = {}
    for method in (SolverType.auto, SolverType.direct):
        mesh = box('hex8', 24)                                       # 23^3 = 12 167 free unknowns
        model = heat_model(mesh)
        model.physics.F_const = 1.0
        model.physics.addBC_TemperatureEdges(mesh.boundary_faces(), 0.0)
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear, Method=method,
                                  Solver=DirectSolvers.PARDISO, tol=1e-12)
        model.solve()
        solutions[method] = np.asarray(model.sol['T'])
        if method == SolverType.auto:
            assert sum(model.solver_iterations) > 0
    assert np.max(np.abs(solutions[SolverType.auto] - solutions[SolverType.direct])) < 1e-9 * solutions[SolverType.direct].max()


def main():
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    failures = 0

    for name, fn in tests:
        start = time.perf_counter()
        try:
            fn()
            print(f"PASS  {name} ({time.perf_counter() - start:.1f} s)")
        except Exception as exc:
            failures += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}: {exc!r}")

    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
