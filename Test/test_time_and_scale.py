"""Stage 17, priority 5: time-dependent boundary conditions, adaptive BDF2, second-order Crank-Nicolson pressure, exact wall
distance, axisymmetric turbulence and the block preconditioner of the flow.

Run from the project root:  python Test/test_time_and_scale.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import gmsh
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.HeatTransfer import ht
from Source.Physics.Rheology import CarreauYasuda
from Source.Physics.Turbulence import sa, sst, wall_distance
from Source.enums import ElementType, StudyType, ProblemType, SolverType, Preconditioners, IterativeSolvers


def rect(size, div, element=ElementType.quadrilateral, order=1):
    mesh = Mesh()
    mesh.generate2DMesh(size, div, element, order=order)
    return mesh


def xy_of(mesh):
    return np.array([n.coor[:2] for n in mesh.NL], dtype=float)


def transient(model, kind=ProblemType.linear, **options):
    model.solverConfiguration(Study=StudyType.transient, Type=kind, **options)
    model.solve()


# ----- boundary conditions in time -----

def test_oscillating_wall():

    """u(1, t) = cos(w t) between plates (Stokes' second problem): exact solution, second order in time."""

    nu, om = 0.5, 2 * np.pi
    lam = np.sqrt(1j * om / nu)
    exact = lambda y, t: np.real(np.exp(1j * om * t) * np.sinh(lam * y) / np.sinh(lam))

    for scheme in (dict(timeScheme='BDF2'), dict(timeScheme='theta', timeDisc=0.5)):
        errors = []
        for steps in (20, 40, 80):
            mesh = rect((0.1, 1.0), (1, 32), order=2)
            model = Model(name='wall', dim=2, mesh=mesh, mat=material('X', miu=nu, rho=1.0), psc=ff)
            flow = model.physics
            top = mesh.boundary_edges(lambda x, y: y > 1 - 1e-9)
            flow.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9))
            flow.addBC_VelocityEdges(top, (1.0, 0.0))
            flow.setBCTime('u', top, lambda t: np.cos(om * t))
            # Later conditions on the corners keep the time function
            flow.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x < 1e-9))
            flow.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x > 0.1 - 1e-9))
            xy = xy_of(mesh)
            flow.var['u'].updateField(np.c_[exact(xy[:, 1], 0.0), np.zeros(len(xy))])
            transient(model, timeStep=1.0 / steps, totalTime=1.0, **scheme)
            errors.append(np.max(np.abs(model.sol['u'][:, 0, -1] - exact(xy[:, 1], 1.0))))

        rate = np.log2(errors[-2] / errors[-1])
        print(f"   {scheme['timeScheme']}: errors {['%.1e' % e for e in errors]}, rate {rate:.2f}")
        assert rate > 1.9 and errors[-1] < 1e-3


def test_pressure_pulse():

    """Pressure at the inlet times cos(w t): Womersley flow between plates; a traction with a time function."""

    rho, mu, om, G = 1.0, 0.2, 2 * np.pi, 3.0
    a = np.sqrt(1j * om * rho / mu)
    exact = lambda y, t: np.real(G / (1j * om * rho) * np.exp(1j * om * t) * (1 - np.cosh(a * (y - 0.5)) / np.cosh(a * 0.5)))

    errors = []
    for steps in (40, 80):
        mesh = rect((0.5, 1.0), (2, 24), order=2)
        model = Model(name='pulse', dim=2, mesh=mesh, mat=material('X', miu=mu, rho=rho), psc=ff)
        flow = model.physics
        inlet = mesh.boundary_edges(lambda x, y: x < 1e-9)
        flow.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or y > 1 - 1e-9))
        flow.addBC_OutletEdges(inlet, G * 0.5)
        flow.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x > 0.5 - 1e-9), 0.0)
        flow.setBCTime('u', inlet, lambda t: np.cos(om * t))
        xy = xy_of(mesh)
        flow.var['u'].updateField(np.c_[exact(xy[:, 1], 0.0), np.zeros(len(xy))])
        transient(model, timeStep=1.0 / steps, totalTime=1.0, timeScheme='BDF2')
        errors.append(np.max(np.abs(model.sol['u'][:, 0, -1] - exact(xy[:, 1], 1.0))))

    print(f"   errors {['%.1e' % e for e in errors]}")
    assert errors[-1] < 1e-3 and errors[0] / errors[1] > 3.0


def test_temperature_ramp_is_exact():

    """T = 2 a t + x^2: linear in time and quadratic in space, exact for BDF2 on quadratic elements."""

    alpha = 0.3
    mesh = rect((1.0, 0.2), (8, 2), order=2)
    model = Model(name='ramp', dim=2, mesh=mesh, mat=material('X', k=alpha, rho=1.0, Cp=1.0), psc=ht)
    heat = model.physics
    left, right = mesh.boundary_edges(lambda x, y: x < 1e-9), mesh.boundary_edges(lambda x, y: x > 1 - 1e-9)
    heat.addBC_TemperatureEdges(left, 1.0)
    heat.addBC_TemperatureEdges(right, 1.0)
    heat.setBCTime('T', left, lambda t: 2 * alpha * t)
    heat.setBCTime('T', right, lambda t: 2 * alpha * t + 1)
    xy = xy_of(mesh)
    heat.var['T'].updateField(xy[:, 0] ** 2)
    transient(model, timeStep=0.1, totalTime=1.0, timeScheme='BDF2')
    assert np.max(np.abs(model.sol['T'][:, -1] - (2 * alpha + xy[:, 0] ** 2))) < 1e-11


def test_steady_studies_ignore_the_time_function_and_errors():

    mesh = rect((1.0, 0.2), (4, 2))
    model = Model(name='steady', dim=2, mesh=mesh, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
    heat = model.physics
    left = mesh.boundary_edges(lambda x, y: x < 1e-9)
    heat.addBC_TemperatureEdges(left, 5.0)
    heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x > 1 - 1e-9), 1.0)
    heat.setBCTime('T', left, lambda t: 0.0)
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear)
    model.solve()
    assert np.max(np.abs(model.sol['T'] - (5.0 - 4.0 * xy_of(mesh)[:, 0]))) < 1e-11

    try:
        heat.setBCTime('T', mesh.boundary_edges(lambda x, y: y < 1e-9 and 0.3 < x < 0.7), lambda t: t)
    except ValueError:
        pass
    else:
        raise AssertionError("a time function needs a condition on the edges")


# ----- adaptive steps -----

def start_up(element=ElementType.quadrilateral):
    mesh = rect((0.1, 1.0), (1, 32), element, order=2)
    model = Model(name='start', dim=2, mesh=mesh, mat=material('X', miu=0.5, rho=1.0), psc=ff)
    flow = model.physics
    flow.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9))
    flow.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), (1.0, 0.0))
    flow.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x < 1e-9))
    flow.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x > 0.1 - 1e-9))
    return model, xy_of(mesh)


def fourier(y, t, nu=0.5, terms=400):
    k = np.arange(1, terms)
    return y + 2 / np.pi * np.sum(((-1.0) ** k / k)[None] * np.sin(k[None] * np.pi * y[:, None])
                                  * np.exp(-nu * (k * np.pi) ** 2 * t)[None], axis=1)


def test_adaptive_steps_follow_the_tolerance():

    """A wall that starts suddenly: steps from 1e-5 to 0.1 and more; the number of steps grows as tol^(-1/3), and the end
    time is reached exactly. The error follows the tolerance, and it takes fewer steps than a fixed step."""

    T = 2.0
    counts, errors = [], []
    for tol in (1e-2, 1e-3, 1e-4):
        model, xy = start_up()
        transient(model, ProblemType.nonlinear, timeStep=1e-4, totalTime=T, timeScheme='BDF2', adaptiveTol=tol)
        times = np.asarray(model.timeVector)
        steps = np.diff(times)
        k = int(np.argmin(np.abs(times - 0.05)))
        counts.append(len(steps))
        errors.append(np.max(np.abs(model.sol['u'][:, 0, k] - fourier(xy[:, 1], times[k]))))
        assert abs(times[-1] - T) < 1e-12 and steps.max() / steps.min() > 1e2
        assert np.all(steps[1:] / steps[:-1] <= 2.0 + 1e-9) or model.rejected_steps > 0
    print(f"   steps {counts}, errors at t = 0.05 {['%.1e' % e for e in errors]}")
    assert 1.5 < counts[1] / counts[0] < 3.0 and 1.5 < counts[2] / counts[1] < 3.0
    assert errors[0] > errors[1] > errors[2]

    model, xy = start_up()
    transient(model, ProblemType.nonlinear, timeStep=T / 100, totalTime=T, timeScheme='BDF2')
    times = np.asarray(model.timeVector)
    k = int(np.argmin(np.abs(times - 0.05)))
    fixed = np.max(np.abs(model.sol['u'][:, 0, k] - fourier(xy[:, 1], times[k])))
    assert counts[0] < 100 and errors[0] < fixed                                 # fewer steps, smaller error


def test_adaptive_steps_with_a_time_function_and_errors():

    """A pulse between two quiet periods: dt must shrink for the pulse and grow again. The control cannot foresee the pulse:
    without a limit it may jump over it, which is what the limit on the step is for."""

    mesh = rect((0.2, 1.0), (2, 2), order=2)
    model = Model(name='pulse', dim=2, mesh=mesh, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
    heat = model.physics
    left = mesh.boundary_edges(lambda x, y: y < 1e-9)
    heat.addBC_TemperatureEdges(left, 1.0)
    heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), 0.0)
    heat.setBCTime('T', left, lambda t: np.exp(-((t - 5.0) / 0.3) ** 2))
    transient(model, ProblemType.nonlinear, timeStep=1e-2, totalTime=10.0, timeScheme='BDF2', adaptiveTol=1e-3,
              adaptiveMaxStep=0.2)
    times = np.asarray(model.timeVector)
    middle = (times[:-1] > 4.5) & (times[:-1] < 5.5)
    steps = np.diff(times)
    print(f"   {len(steps)} steps: largest {steps.max():.2f}, around the pulse {steps[middle].min():.3f}")
    assert steps.max() <= 0.2 + 1e-12 and steps[middle].min() < 0.25 * steps.max()
    assert model.sol['T'][:, -1].max() < 0.1 and np.max(model.sol['T'][:, np.argmin(np.abs(times - 5.0))]) > 0.9

    model, _ = start_up()
    try:
        transient(model, ProblemType.nonlinear, timeStep=1e-2, totalTime=1.0, adaptiveTol=1e-3)
    except ValueError:
        pass
    else:
        raise AssertionError("adaptive steps need BDF2")


def test_newton_settles_with_a_zero_pressure():

    """A pure shear flow has p = 0: round-off alone in that field must not make Newton go on (nor the step shrink)."""

    model, _ = start_up()
    transient(model, ProblemType.nonlinear, timeStep=0.01, totalTime=0.2, timeScheme='BDF2')
    assert max(model.step_iterations) < 15


# ----- Crank-Nicolson -----

def test_crank_nicolson_pressure_is_second_order():

    """Taylor-Green: the implicit pressure belongs to t_n + dt / 2; reported at t_{n+1} it is second order."""

    nu = 0.1
    decay = lambda t: np.exp(-2 * nu * t)
    U = lambda x, y, t: np.stack([np.sin(x) * np.cos(y) * decay(t), -np.cos(x) * np.sin(y) * decay(t)], axis=-1)
    P = lambda x, y, t: 0.25 * (np.cos(2 * x) + np.cos(2 * y)) * decay(t) ** 2

    def run(steps, order, n):
        mesh = rect((np.pi, np.pi), (n, n), order=order)
        model = Model(name='tg', dim=2, mesh=mesh, mat=material('X', miu=nu, rho=1.0), psc=ff)
        flow = model.physics
        flow.setInertia(True)
        for selector in (lambda x, y: y < 1e-9, lambda x, y: y > np.pi - 1e-9, lambda x, y: x < 1e-9, lambda x, y: x > np.pi - 1e-9):
            flow.addBC_SymmetryEdges(mesh.boundary_edges(selector))
        flow.setPressureReference(point=(0.0, 0.0), value=float(P(0.0, 0.0, 1.0)))
        xy = xy_of(mesh)
        flow.var['u'].updateField(U(xy[:, 0], xy[:, 1], 0.0))
        flow.var['p'].updateField(P(xy[:, 0], xy[:, 1], 0.0))
        transient(model, ProblemType.nonlinear, timeStep=1.0 / steps, totalTime=1.0, timeScheme='theta', timeDisc=0.5)
        return model.sol['p'][:, -1]

    for order, n in ((1, 12), (2, 6)):
        reference = run(320, order, n)
        errors = [np.max(np.abs(run(steps, order, n) - reference)) for steps in (20, 40, 80)]
        rate = np.log2(errors[-2] / errors[-1])
        print(f"   order {order}: pressure {['%.1e' % e for e in errors]}, rate {rate:.2f}")
        assert rate > 1.8


# ----- wall distance -----

def test_wall_distance_is_exact():

    for element in (ElementType.quadrilateral, ElementType.triangle):
        mesh = rect((2.0, 1.0), (8, 10), element)
        xy = xy_of(mesh)
        d = wall_distance(mesh, mesh.boundary_edges(lambda x, y: y < 1e-9 or y > 1 - 1e-9))
        assert np.max(np.abs(d - np.minimum(xy[:, 1], 1 - xy[:, 1]))) < 1e-13

    # A wall that ends: the distance goes around the end, which a Poisson approximation cannot do
    mesh = rect((2.0, 1.0), (8, 4))
    xy = xy_of(mesh)
    d = wall_distance(mesh, mesh.boundary_edges(lambda x, y: y < 1e-9 and x > 1.0))
    exact = np.hypot(np.maximum(1.0 - xy[:, 0], 0.0), xy[:, 1])
    assert np.max(np.abs(d - exact)) < 1e-13

    # Around a cylinder, convex: the error is that of the chords, second order; quadratic meshes follow the wall better
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        occ = gmsh.model.occ
        fluid, _ = occ.cut([(2, occ.addRectangle(-1, -1, 0, 2, 2))], [(2, occ.addDisk(0, 0, 0, 0.2, 0.2))])
        occ.synchronize()
        circle = [tag for _, tag in gmsh.model.getBoundary(fluid, oriented=False) if occ.getMass(1, tag) < 1.5]
        gmsh.model.addPhysicalGroup(1, circle, name='cylinder')
        gmsh.option.setNumber('Mesh.MeshSizeMax', 0.04)
        gmsh.model.mesh.generate(2)
        linear = Mesh.from_gmsh_model()
        gmsh.model.mesh.setOrder(2)
        quadratic = Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()

    errors = []
    for mesh in (linear, quadratic):
        xy = xy_of(mesh)
        d = wall_distance(mesh, mesh.boundary_edges('cylinder'))
        errors.append(np.max(np.abs(d - (np.hypot(xy[:, 0], xy[:, 1]) - 0.2))))
    print(f"   cylinder: {errors[0]:.1e} (T3), {errors[1]:.1e} (T6)")
    assert errors[0] < 2e-3 and errors[1] < 0.5 * errors[0]


# ----- axisymmetric turbulence -----

def pipe(kind, re_tau, ny):

    """Pipe of radius 1 driven by a body force 2: tau_w = 1, u_tau = 1, nu = 1 / Re_tau, so u is u+."""

    mesh = rect((1.0, 0.1), (ny, 2))
    for node in mesh.NL:
        e, z = node.coor
        node.coor = (-np.tanh(3.0 * (e - 1.0)) / np.tanh(3.0), z)        # clustered at r = 1
    for element in mesh.EL:
        element.nodes = [element.nodes[1], element.nodes[0], element.nodes[3], element.nodes[2]]
    xy = xy_of(mesh)

    model = Model(name='pipe', dim=2, mesh=mesh, mat=material('X', miu=1.0 / re_tau, rho=1.0), psc=[ff, kind], axisymmetric=True)
    fluid, turbulence = model.physics.fluid, model.physics.turbulence
    fluid.setInertia(True)
    fluid.setBodyForce((0.0, 2.0))
    wall = mesh.boundary_edges(lambda x, y: x > 1 - 1e-9)
    fluid.addBC_NoSlipEdges(wall)
    fluid.addBC_SymmetryEdges(mesh.boundary_edges(lambda x, y: x < 1e-9))
    fluid.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: y < 1e-9))
    fluid.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: y > 0.1 - 1e-9))
    turbulence.setWalls(wall)
    turbulence.coupleFlow()

    d = 1.0 - xy[:, 0]
    y_plus = d * re_tau
    u = np.log(1 + 0.41 * y_plus) / 0.41 + 7.8 * (1 - np.exp(-y_plus / 11) - y_plus / 11 * np.exp(-y_plus / 3))
    fluid.var['u'].updateField(np.c_[np.zeros(len(d)), u])
    if kind is sst:
        ds = np.maximum(d, 1e-6)
        omega = np.sqrt((6 / re_tau / (0.075 * ds ** 2)) ** 2 + (1 / (0.3 * 0.41 * ds)) ** 2)
        turbulence.var['ln_omega'].updateField(np.log(omega))
        turbulence.var['k'].updateField((0.41 * d * (1 - 0.5 * d) + 1e-4) * omega * (d > 0))
    else:
        turbulence.var['nut'].updateField(0.41 * d * (1 - d) + 1e-3)
    return model, mesh


def test_turbulent_pipe_against_prandtl():

    """Friction factor against Prandtl's law 1 / sqrt(f) = 2 log10(Re sqrt(f)) - 0.8. SST is further away at low Re."""

    for kind, name, tol in ((sa, 'SA', 0.03), (sst, 'SST', 0.08)):
        for re_tau, ny in ((360.0, 64), (1000.0, 80)):
            model, mesh = pipe(kind, re_tau, ny)
            model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, globalization='pseudo-transient')
            model.solve()
            bulk = model.physics.fluid.flowRate(mesh.boundary_edges(lambda x, y: y > 0.1 - 1e-9)) / np.pi
            f, reynolds = 8.0 / bulk ** 2, bulk * 2.0 * re_tau
            prandtl = f
            for _ in range(60):
                prandtl = (2.0 * np.log10(reynolds * np.sqrt(prandtl)) - 0.8) ** -2
            print(f"   {name}, Re_tau = {re_tau:.0f}: {model.newton_iterations} iterations, f {f:.5f}, Prandtl {prandtl:.5f} "
                  f"({f / prandtl - 1:+.1%})")
            assert model.nonlinear_converged and abs(f / prandtl - 1) < tol


def test_tangent_of_axisymmetric_sst():

    mesh = rect((1.0, 1.0), (2, 2))
    for node in mesh.NL:
        node.coor = (node.coor[0] + 0.5, node.coor[1])
    model = Model(name='sst', dim=2, mesh=mesh, mat=material('X', miu=0.05, rho=1.0), psc=[ff, sst], axisymmetric=True)
    model.physics.fluid.setInertia(True)
    model.physics.turbulence.setWalls(mesh.boundary_edges(lambda x, y: x > 1.5 - 1e-9))
    model.physics.turbulence.coupleFlow()
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
    system = model._system()
    layout = system.layout
    rng = np.random.default_rng(5)
    x = rng.normal(scale=0.5, size=layout.ndof)
    x[layout.dofs('k')[:, 0]] = 0.2 * np.abs(rng.normal(size=layout.n_nodes))
    x[layout.dofs('ln_omega')[:, 0]] = 2.0 + rng.normal(size=layout.n_nodes)
    x[layout.dofs('wall_distance')[:, 0]] = model.physics.turbulence.var['wall_distance'].values
    free = ~system.is_dirichlet
    for term in system._terms:
        if hasattr(term, 'tau_state'):
            term.tau_state = x.copy()
    J, _ = system._assemble(x)
    FD = np.zeros((layout.ndof, layout.ndof))
    for k in np.nonzero(free)[0]:
        e = np.zeros(layout.ndof)
        e[k] = 1e-7
        FD[:, k] = (system._assemble(x + e)[1] - system._assemble(x - e)[1]) / 2e-7
    err = np.max(np.abs(J.toarray()[:, free] - FD[:, free])) / np.max(np.abs(FD))
    assert err < 1e-6, err


# ----- block preconditioner -----

def stokes_cavity(n, element, law=None, **options):
    mesh = rect((1.0, 1.0), (n, n), element, order=2)
    model = Model(name='cavity', dim=2, mesh=mesh, mat=material('X', miu=2.0), psc=ff)
    if law is not None:
        model.physics.setViscosityLaw(law)
    model.physics.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), (1.0, 0.0))
    model.physics.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or x < 1e-9 or x > 1 - 1e-9))
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear if law else ProblemType.linear, **options)
    model.solve()
    return model


BLOCK = dict(Method=SolverType.iterative, Solver=IterativeSolvers.GMRES, prec=Preconditioners.SaddlePoint, tol=1e-8)


def test_block_preconditioner():

    """The iterations of GMRES barely grow with the mesh, and the solution is that of the direct solver."""

    for element in (ElementType.quadrilateral, ElementType.triangle):
        counts = []
        for n in (16, 32, 64):
            model = stokes_cavity(n, element, **BLOCK)
            counts.append(model.solver_iterations[0])
        direct = stokes_cavity(64, element)
        du, dp = np.max(np.abs(model.sol['u'] - direct.sol['u'])), np.max(np.abs(model.sol['p'] - direct.sol['p']))
        print(f"   {element.name}: GMRES {counts}; difference with PARDISO u {du:.1e}, p {dp:.1e}")
        assert counts[-1] < 1.5 * counts[0] and du < 1e-6 and dp < 1e-3


def test_block_preconditioner_with_a_viscosity_law():

    model = stokes_cavity(24, ElementType.quadrilateral, law=CarreauYasuda(mu0=100.0, lam=10.0, n=0.3, mu_inf=1.0), **BLOCK)
    direct = stokes_cavity(24, ElementType.quadrilateral, law=CarreauYasuda(mu0=100.0, lam=10.0, n=0.3, mu_inf=1.0))
    print(f"   GMRES per Newton iteration, at most {max(model.solver_iterations)}")
    assert model.nonlinear_converged and max(model.solver_iterations) < 150
    assert np.max(np.abs(model.sol['u'] - direct.sol['u'])) < 1e-5


def test_block_preconditioner_needs_a_flow():

    mesh = rect((1.0, 1.0), (2, 2), order=2)
    model = Model(name='heat', dim=2, mesh=mesh, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(None), 1.0)
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear, **BLOCK)
    try:
        model.solve()
    except ValueError:
        pass
    else:
        raise AssertionError("no flow, no block preconditioner")


if __name__ == "__main__":
    tests = [test_oscillating_wall, test_pressure_pulse, test_temperature_ramp_is_exact,
             test_steady_studies_ignore_the_time_function_and_errors, test_adaptive_steps_follow_the_tolerance,
             test_adaptive_steps_with_a_time_function_and_errors, test_newton_settles_with_a_zero_pressure,
             test_crank_nicolson_pressure_is_second_order, test_wall_distance_is_exact, test_turbulent_pipe_against_prandtl,
             test_tangent_of_axisymmetric_sst, test_block_preconditioner, test_block_preconditioner_with_a_viscosity_law,
             test_block_preconditioner_needs_a_flow]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
