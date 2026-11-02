"""Stage 17: the small gaps left by Taylor-Hood. SUPG on quadratic elements, turbulence and species on Q9 and T6, the block
preconditioner with more fields, drawing on quadratic meshes, and the assemblies of the adaptive time step.

Run from the project root:  python Test/test_quadratic_gaps.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.HeatTransfer import ht
from Source.Physics.MassTransfer import mt
from Source.Physics.Turbulence import sa, sst
from Source.Simulation import Operators as ops
from Source.enums import ElementType, StudyType, ProblemType, SolverType, IterativeSolvers, Preconditioners

QUAD, TRI = ElementType.quadrilateral, ElementType.triangle
BLOCK = dict(Method=SolverType.iterative, Solver=IterativeSolvers.GMRES, prec=Preconditioners.SaddlePoint, tol=1e-8)


def rect(size, div, element=QUAD, order=2):
    mesh = Mesh()
    mesh.generate2DMesh(size, div, element, order=order)
    return mesh


def xy_of(mesh):
    return np.array([node.coor for node in mesh.NL])


def distort(mesh):
    for node in mesh.NL:
        x, y = node.coor
        node.coor = (x + 0.05 * np.sin(np.pi * x) * np.sin(np.pi * y), y - 0.04 * np.sin(np.pi * x) * np.sin(2 * np.pi * y))


# ----- second derivatives -----

def test_second_derivatives_on_curved_elements():

    """The coordinates are in the space even on a curved element, so their second derivatives vanish; a quadratic is
    reproduced exactly on a straight one. Linear elements have none."""

    for element in (QUAD, TRI):
        mesh = rect((1.0, 1.0), (4, 4), element)
        distort(mesh)
        model = Model(name='h', dim=2, mesh=mesh, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
        disc, xy = model.geometry, xy_of(mesh)
        H = disc.hessian
        assert np.max(np.abs(np.einsum('egiab,ei->egab', H, xy[:, 0][disc.conn]))) < 1e-10
        assert np.max(np.abs(np.einsum('egi,ei->eg', disc.laplacian, xy[:, 1][disc.conn]))) < 1e-10

        mesh = rect((1.0, 1.0), (4, 4), element)
        model = Model(name='h', dim=2, mesh=mesh, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
        disc, xy = model.geometry, xy_of(mesh)
        q = (xy[:, 0] ** 2 + 3 * xy[:, 0] * xy[:, 1] - 2 * xy[:, 1] ** 2)[disc.conn]
        assert np.max(np.abs(np.einsum('egi,ei->eg', disc.laplacian, q) - (2.0 - 4.0))) < 1e-10

        S = ops.stress_divergence(disc, 2.0)
        u = np.c_[xy[:, 0] ** 2, -2 * xy[:, 0] * xy[:, 1]]               # div(u) = 0: 2 div(sym grad u) * 2 = 2 lap(u) = (4, 0)
        Su = np.asarray(S @ u.ravel()).reshape(*disc.detJw.shape, 2)
        assert np.max(np.abs(Su - np.array([4.0, 0.0]))) < 1e-10

    model = Model(name='h', dim=2, mesh=rect((1.0, 1.0), (2, 2), QUAD, order=1), mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
    assert model.geometry.hessian is None and model.geometry.laplacian is None


# ----- SUPG of a scalar on quadratic elements -----

def layer(Pe, supg, element=QUAD, n=20):

    """u T' = (1 / Pe) T'', T(0) = 0, T(1) = 1."""

    mesh = rect((1.0, 0.2), (n, 2), element)
    model = Model(name='layer', dim=2, mesh=mesh, mat=material('X', k=1.0 / Pe, rho=1.0, Cp=1.0), psc=ht)
    heat = model.physics
    heat.setVelocity((1.0, 0.0))
    if supg:
        heat.setStabilization('SUPG')
    heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x < 1e-9), 0.0)
    heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x > 1 - 1e-9), 1.0)
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear)
    model.solve()
    x = xy_of(mesh)[:, 0]
    return model.sol['T'], (np.exp(x * Pe) - 1.0) / (np.exp(Pe) - 1.0)


def test_supg_of_a_scalar_on_quadratic_elements():

    """A layer that the mesh does not resolve loses most of its oscillation, and one that it resolves keeps its accuracy."""

    for element in (QUAD, TRI):
        T_supg, exact = layer(500.0, True, element)
        T_gal, _ = layer(500.0, False, element)
        print(f"   {element.name} Pe 500: T min {T_supg.min():+.3f} with SUPG, {T_gal.min():+.3f} without")
        assert abs(T_supg.min()) < 0.4 * abs(T_gal.min())

        T_supg, exact = layer(10.0, True, element)
        T_gal, _ = layer(10.0, False, element)
        e_supg, e_gal = np.max(np.abs(T_supg - exact)), np.max(np.abs(T_gal - exact))
        print(f"   {element.name} Pe 10: error {e_supg:.1e} with SUPG, {e_gal:.1e} without")
        assert e_supg < 2.0 * e_gal and e_supg < 3e-3


def test_crank_nicolson_keeps_the_steady_state_with_supg_on_quadratic_elements():

    """A coupled convecting field in its steady state: the old-level SUPG residual has the diffusion too, or the steady state
    would drift with theta < 1."""

    mesh = rect((1.0, 0.2), (12, 2))
    Pe = 40.0
    model = Model(name='cn', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.0, k=1.0 / Pe, Cp=1.0), psc=[ff, ht])
    fluid, heat = model.physics.fluid, model.physics.heat
    model.physics.setVelocity('u')
    heat.setStabilization('SUPG')
    x = xy_of(mesh)[:, 0]
    fluid.addBC_VelocityEdges(mesh.boundary_edges(None), (1.0, 0.0))
    fluid.setPressureReference(0, 0.0)
    heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x < 1e-9), 0.0)
    heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x > 1 - 1e-9), 1.0)
    fluid.var['u'].updateField(np.c_[np.ones(len(x)), np.zeros(len(x))])
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
    model.solve()
    steady = np.asarray(model.sol['T']).copy()

    model = Model(name='cn', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.0, k=1.0 / Pe, Cp=1.0), psc=[ff, ht])
    fluid, heat = model.physics.fluid, model.physics.heat
    model.physics.setVelocity('u')
    heat.setStabilization('SUPG')
    fluid.addBC_VelocityEdges(mesh.boundary_edges(None), (1.0, 0.0))
    fluid.setPressureReference(0, 0.0)
    heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x < 1e-9), 0.0)
    heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x > 1 - 1e-9), 1.0)
    fluid.var['u'].updateField(np.c_[np.ones(len(x)), np.zeros(len(x))])
    model.physics.initField('T', steady)
    model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, timeScheme='theta', timeDisc=0.5,
                              timeStep=0.05, totalTime=0.5)
    model.solve()
    drift = np.max(np.abs(model.sol['T'][:, -1] - steady))
    print(f"   drift of the steady state in 10 steps: {drift:.1e}")
    assert drift < 1e-8


# ----- Taylor-Hood with SUPG -----

def cavity(n, Re, supg, element=QUAD):
    mesh = rect((1.0, 1.0), (n, n), element)
    model = Model(name='cavity', dim=2, mesh=mesh, mat=material('X', miu=1.0 / Re, rho=1.0), psc=ff)
    flow = model.physics
    flow.setInertia(True)
    if supg:
        flow.setStabilization('SUPG')
    flow.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), (1.0, 0.0))
    flow.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or x < 1e-9 or x > 1 - 1e-9))
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, globalization='pseudo-transient')
    model.solve()
    return model


def test_taylor_hood_with_supg_at_high_reynolds():

    """A cavity at Re = 5000 on a coarse mesh: from rest, the pseudo-transient march does not finish without SUPG, and it does
    with it. On a fine mesh both reach the same solution. Measured: SUPG buys robustness, not accuracy."""

    plain, stabilized = cavity(16, 5000.0, False), cavity(16, 5000.0, True)
    print(f"   16^2: {plain.newton_iterations} iterations, converged {plain.nonlinear_converged} without SUPG; "
          f"{stabilized.newton_iterations}, converged {stabilized.nonlinear_converged} with it")
    assert stabilized.nonlinear_converged and stabilized.newton_iterations < 200
    assert stabilized.newton_iterations < plain.newton_iterations

    plain, stabilized = cavity(32, 1000.0, False), cavity(32, 1000.0, True)
    xy = xy_of(plain.mesh)
    centre = np.abs(xy[:, 0] - 0.5) < 1e-9
    difference = np.max(np.abs(plain.sol['u'][centre, 0] - stabilized.sol['u'][centre, 0]))
    print(f"   Re 1000, 32^2: difference on the centreline {difference:.1e}")
    assert plain.nonlinear_converged and stabilized.nonlinear_converged and difference < 0.02


def test_stabilization_is_refused_where_it_does_not_belong():

    mesh = rect((1.0, 1.0), (2, 2))
    model = Model(name='th', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.0), psc=ff)
    model.physics.setInertia(True)
    try:
        model.physics.setStabilization('PSPG')
        model._system()
    except ValueError:
        pass
    else:
        raise AssertionError("Taylor-Hood is inf-sup stable: PSPG must be refused")

    mesh = rect((1.0, 1.0), (2, 2), order=1)
    model = Model(name='q1', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.0), psc=ff)
    try:
        model.physics.setStabilization('SUPG')
        model._system()
    except ValueError:
        pass
    else:
        raise AssertionError("equal order flow needs PSPG")


# ----- turbulence on Q9 and T6 -----

def channel(kind, ny, element, order, **options):

    """Half channel Re_tau = 395, x-independent, unit body force: u is u+. Independent 1D SST: U_c+ = 19.49."""

    re_tau, beta = 395.0, 3.0
    mesh = Mesh()
    mesh.generate2DMesh((0.1, 1.0), (2, ny), element, order=order)
    for node in mesh.NL:
        x, e = node.coor
        node.coor = (x, 1.0 + np.tanh(beta * (e - 1.0)) / np.tanh(beta))
    xy = xy_of(mesh)
    model = Model(name='channel', dim=2, mesh=mesh, mat=material('X', miu=1.0 / re_tau, rho=1.0), psc=[ff, kind])
    fluid, turbulence = model.physics.fluid, model.physics.turbulence
    fluid.setInertia(True)
    fluid.setBodyForce((1.0, 0.0))
    wall = mesh.boundary_edges(lambda x, y: y < 1e-9)
    fluid.addBC_NoSlipEdges(wall)
    fluid.addBC_SymmetryEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9))
    fluid.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x < 1e-9))
    fluid.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x > 0.1 - 1e-9))
    turbulence.setWalls(wall)
    turbulence.coupleFlow()
    y = xy[:, 1]
    yp = y * re_tau
    fluid.var['u'].updateField(np.c_[np.log(1 + 0.41 * yp) / 0.41 + 7.8 * (1 - np.exp(-yp / 11) - yp / 11 * np.exp(-yp / 3)), 0 * y])
    if kind is sst:
        ys = np.maximum(y, 1e-6)
        omega = np.sqrt((6 / re_tau / (0.075 * ys ** 2)) ** 2 + (1 / (0.3 * 0.41 * ys)) ** 2)
        turbulence.var['ln_omega'].updateField(np.log(omega))
        turbulence.var['k'].updateField((0.41 * y * (1 - 0.5 * y) + 1e-4) * omega * (y > 0))
    else:
        turbulence.var['nut'].updateField(0.41 * y * (1 - y) + 1e-3)
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, globalization='pseudo-transient', **options)
    model.solve()
    return model


def test_turbulence_on_quadratic_elements():

    """SA and SST: the channel on Q9 and T6 reaches what Q4 and the independent 1D solver give, x-independent. The linear
    triangles too, whose k is clipped at two nodes next to the wall and used to hide the convergence from the stopping test."""

    for kind, name, reference in ((sa, 'SA', 20.0), (sst, 'SST', 19.49)):
        for element, order, ny in ((QUAD, 2, 64), (QUAD, 2, 32), (TRI, 2, 64), (TRI, 1, 128)):
            model = channel(kind, ny, element, order)
            u = model.sol['u']
            print(f"   {name} {element.name} order {order} ny {ny}: U_c {u[:, 0].max():.3f} (reference {reference}), "
                  f"{model.newton_iterations} iterations, v {np.abs(u[:, 1]).max():.1e}")
            assert model.nonlinear_converged and model.newton_iterations < 100
            assert abs(u[:, 0].max() / reference - 1.0) < 0.012
            assert np.abs(u[:, 1]).max() < 5e-3                          # a spurious flow gave 1e-2 and lost 10 % of U_c


def test_sst_flat_plate_closes_its_residual():

    """Re_L = 1e6, Q4 40 x 48, from a uniform state: k stays above a thousandth of its inflow value and the residual closes, with a C_f
    within 12 % of Schultz-Grunow. It takes about five minutes. On Q9 it does not close: see docs/ESTADO.md."""

    L, H, x0, Re = 2.0, 1.0, -0.33, 1e6
    nu = 1.0 / Re
    mesh = rect((1.0, 1.0), (40, 48), order=1)
    for node in mesh.NL:
        a, b = node.coor
        s_ = np.sinh(3.0 * (a - 0.25)) / np.sinh(3.0 * 0.75)
        x = L * s_ if s_ >= 0 else -x0 * s_ / (np.sinh(3.0 * 0.25) / np.sinh(3.0 * 0.75))
        node.coor = (x, H * (np.exp(10.0 * b) - 1) / (np.exp(10.0) - 1))
    xy = xy_of(mesh)
    model = Model(name='plate', dim=2, mesh=mesh, mat=material('X', miu=nu, rho=1.0), psc=[ff, sst])
    fluid, turbulence = model.physics.fluid, model.physics.turbulence
    fluid.setInertia(True)
    wall = mesh.boundary_edges(lambda x, y: y < 1e-12 and x > -1e-12)
    inlet = mesh.boundary_edges(lambda x, y: x < xy[:, 0].min() + 1e-9)
    fluid.addBC_VelocityEdges(inlet, (1.0, 0.0))
    fluid.addBC_NoSlipEdges(wall)
    fluid.addBC_SymmetryEdges(mesh.boundary_edges(lambda x, y: y < 1e-12 and x < 1e-12))
    fluid.addBC_SymmetryEdges(mesh.boundary_edges(lambda x, y: y > H - 1e-9))
    fluid.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x > L - 1e-9))
    turbulence.setWalls(wall)
    k_in, n = 1e-6, len(xy)
    turbulence.addBC_TurbulenceEdges(inlet, k_in, k_in / nu)
    fluid.var['u'].updateField(np.c_[np.ones(n), np.zeros(n)])
    turbulence.var['k'].updateField(np.full(n, k_in))
    turbulence.var['ln_omega'].updateField(np.full(n, np.log(k_in / nu)))
    turbulence.coupleFlow()
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, globalization='pseudo-transient', cfl0=1.0)
    model.solve()
    print(f"   {model.newton_iterations} iterations, converged {model.nonlinear_converged}")
    assert model.nonlinear_converged
    free = (xy[:, 1] > 1e-12) & (xy[:, 0] > xy[:, 0].min() + 1e-9)
    assert model.sol['k'][free].min() >= 1e-3 * k_in * (1.0 - 1e-12)
    u = model.sol['u'][:, 0]
    for xq in (0.5, 1.0, 1.5):
        xc = xy[np.argmin(np.abs(xy[:, 0] - xq)), 0]
        column = np.abs(xy[:, 0] - xc) < 1e-9
        order = np.argsort(xy[column, 1])
        cf = 2 * nu * u[column][order][1] / xy[column, 1][order][1]
        reference = 0.37 / np.log10(Re * xc) ** 2.584
        print(f"   x {xc:.2f}: C_f {cf:.5f}, Schultz-Grunow {reference:.5f} ({100 * (cf / reference - 1):+.1f} %)")
        assert abs(cf / reference - 1.0) < 0.12


# ----- species on Q9 and T6 -----

def species(n, element, order, supg=False, D=0.05, v=1.0, rate=1.0):

    """v c' = D c'' - k c, c(0) = 1, c(1) = 0, which has an exact solution; the second species B is what A leaves: A -> B."""

    mesh = rect((1.0, 0.25), (n, 2), element, order=order)
    model = Model(name='species', dim=2, mesh=mesh, mat=material('X', rho=1.0), psc=mt)
    mass = model.physics
    for name in ('A', 'B'):
        mass.setChemSpecies(name, name)
        mass.setDiffusivity(name, D)
        mass.addBC_ConcentrationEdges(mesh.boundary_edges(lambda x, y: x > 1 - 1e-9), name, 0.0)
    mass.addBC_ConcentrationEdges(mesh.boundary_edges(lambda x, y: x < 1e-9), 'A', 1.0)
    mass.addBC_ConcentrationEdges(mesh.boundary_edges(lambda x, y: x < 1e-9), 'B', 0.0)
    mass.addReaction({'A': -1, 'B': 1}, rate, 0.0, 300.0, {'A': 1})
    mass.setVelocity((v, 0.0))
    if supg:
        mass.setStabilization('SUPG')
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
    model.solve()
    x = xy_of(mesh)[:, 0]
    r1 = (v + np.sqrt(v ** 2 + 4 * D * rate)) / (2 * D)
    r2 = (v - np.sqrt(v ** 2 + 4 * D * rate)) / (2 * D)
    exact = (np.exp(r1 + r2 * x) - np.exp(r2 + r1 * x)) / (np.exp(r1) - np.exp(r2))
    return np.max(np.abs(model.sol['A'] - exact)), model


def test_species_on_quadratic_elements():

    """Against the exact solution, with a reaction and a convected product: second order on Q4, third or more on Q9, and about
    2.5 at the nodes of T6 on this mesh of aligned diagonals."""

    for element, order, expected in ((QUAD, 1, 1.8), (QUAD, 2, 3.0), (TRI, 2, 2.4)):
        errors = [species(n, element, order)[0] for n in (16, 32, 64)]
        rate = np.log2(errors[-2] / errors[-1])
        print(f"   {element.name} order {order}: {['%.1e' % e for e in errors]}, rate {rate:.2f}")
        assert rate > expected

    error, model = species(16, QUAD, 2, supg=True)
    plain, _ = species(16, QUAD, 2, supg=False)
    print(f"   SUPG on Q9: {error:.1e}, without {plain:.1e} (the reaction is not in its residual)")
    assert error < 5e-2
    assert model.sol['B'].min() > -1e-3 and model.sol['B'].max() < 1.0


# ----- block preconditioner with more fields -----

def buoyant_cavity(n, Ra=1e4, **options):
    mesh = rect((1.0, 1.0), (n, n))
    model = Model(name='ra', dim=2, mesh=mesh, mat=material('X', miu=0.71, rho=1.0, Cp=1.0, k=1.0), psc=[ff, ht])
    fluid, heat = model.physics.fluid, model.physics.heat
    fluid.setInertia(True)
    model.physics.setVelocity('u')
    fluid.addBC_NoSlipEdges(mesh.boundary_edges(None))
    heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x < 1e-9), 1.0)
    heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x > 1 - 1e-9), 0.0)
    fluid.setBoussinesq((0.0, -Ra * 0.71), beta=1.0, reference=0.5)
    model.physics.initField('T', 0.5)
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, **options)
    model.solve()
    return model


def test_block_preconditioner_with_more_fields():

    """Natural convection: flow and heat, with the buoyancy in the velocity block. The solution is that of the direct solver."""

    counts = []
    for n in (12, 24, 48):
        model = buoyant_cavity(n, **BLOCK)
        counts.append(max(model.solver_iterations))
        assert model.nonlinear_converged
    direct = buoyant_cavity(48)
    du, dT = np.max(np.abs(model.sol['u'] - direct.sol['u'])), np.max(np.abs(model.sol['T'] - direct.sol['T']))
    print(f"   Ra 1e4: GMRES at most {counts} per Newton iteration; difference with PARDISO u {du:.1e}, T {dT:.1e}")
    assert du < 1e-4 and dT < 1e-5 and counts[-1] < 400


def test_block_preconditioner_with_turbulence():

    direct = channel(sst, 32, QUAD, 2)
    model = channel(sst, 32, QUAD, 2, **BLOCK)
    difference = np.max(np.abs(model.sol['u'] - direct.sol['u']))
    print(f"   SST on Q9: GMRES at most {max(model.solver_iterations)}, difference with PARDISO {difference:.1e}")
    assert model.nonlinear_converged and difference < 1e-3


def test_block_preconditioner_needs_a_flow():

    mesh = rect((1.0, 1.0), (2, 2))
    model = Model(name='heat', dim=2, mesh=mesh, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(None), 1.0)
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear, **BLOCK)
    try:
        model.solve()
    except ValueError:
        pass
    else:
        raise AssertionError("no flow, no block preconditioner")


# ----- drawing -----

def test_triangulation_of_every_element():

    """Every node is used, the triangles cover the domain once, with positive area, on any element and on a curved one. The
    linear interpolant of a quadratic field is exact at the nodes, and postProcess2D draws it."""

    for element, order, per_element in ((TRI, 1, 1), (QUAD, 1, 2), (TRI, 2, 4), (QUAD, 2, 8)):
        mesh = rect((2.0, 1.0), (4, 3), element, order=order)
        triangles = mesh.plot_triangles()
        assert len(triangles) == per_element * mesh.getNoE()
        assert np.array_equal(np.unique(triangles), np.arange(mesh.getNoN()))
        xy = xy_of(mesh)
        a, b, c = xy[triangles[:, 0]], xy[triangles[:, 1]], xy[triangles[:, 2]]
        area = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        assert np.all(area > 0) and abs(area.sum() - 2.0) < 1e-12
        assert len(mesh.triangulation(scale=1e3).triangles) == len(triangles)
        assert abs(mesh.triangulation(scale=1e3).x.max() - 2e3) < 1e-9

        model = Model(name='p', dim=2, mesh=mesh, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
        model.physics.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x < 1e-9), 1.0)
        model.physics.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x > 2 - 1e-9), 0.0)
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear)
        model.solve()
        fig, ax = plt.subplots()
        model.postProcess2D('T', ax=ax)
        plt.close(fig)


# ----- assemblies of the adaptive step -----

def test_adaptive_step_assembles_a_linear_system_once():

    """A pulse in a linear problem: however dt moves, one assembly; only alpha M changes. The solution keeps its accuracy."""

    def pulse(reuse):
        mesh = rect((1.0, 1.0), (12, 12), order=1)
        model = Model(name='pulse', dim=2, mesh=mesh, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
        heat = model.physics
        left = mesh.boundary_edges(lambda x, y: y < 1e-9)
        heat.addBC_TemperatureEdges(left, 1.0)
        heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), 0.0)
        heat.setBCTime('T', left, lambda t: np.exp(-((t - 5.0) / 0.3) ** 2))
        model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, timeStep=1e-2, totalTime=7.0,
                                  timeScheme='BDF2', adaptiveTol=1e-3, adaptiveMaxStep=0.2, reuseTangent=reuse)
        model.solve()
        return model

    model, full = pulse(True), pulse(False)
    k5 = np.argmin(np.abs(np.asarray(model.timeVector) - 5.0))
    print(f"   {model.tangent_assemblies} tangent with reuse, {full.tangent_assemblies} without; "
          f"{model._factorizer.n_factorizations} factorizations, {len(model.timeVector) - 1} steps")
    assert model.tangent_assemblies == 1
    assert model._factorizer.n_factorizations < len(model.timeVector)
    assert abs(model.sol['T'][:, k5].max() - 1.0) < 0.02


def test_adaptive_step_keeps_dt_for_a_small_growth():

    """A growth below 20 % is left out: the steps come in runs of the same dt, so the factorization lasts."""

    mesh = rect((1.0, 1.0), (16, 16), order=1)
    model = Model(name='cavity', dim=2, mesh=mesh, mat=material('X', miu=1.0 / 400, rho=1.0), psc=ff)
    model.physics.setInertia(True)
    model.physics.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), (1.0, 0.0))
    model.physics.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or x < 1e-9 or x > 1 - 1e-9))
    model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, timeStep=1e-3, totalTime=2.0,
                              timeScheme='BDF2', adaptiveTol=1e-3)
    model.solve()
    steps = np.diff(model.timeVector)
    print(f"   {len(steps)} steps, {model.tangent_assemblies} tangents, {np.sum(model.step_iterations)} iterations")
    growth = steps[1:-1] / steps[:-2]
    assert np.all((growth >= 1.2 - 1e-9) | (growth <= 1.0 + 1e-9))
    assert model.tangent_assemblies < len(steps)


if __name__ == "__main__":
    tests = [test_second_derivatives_on_curved_elements, test_supg_of_a_scalar_on_quadratic_elements,
             test_crank_nicolson_keeps_the_steady_state_with_supg_on_quadratic_elements,
             test_taylor_hood_with_supg_at_high_reynolds, test_stabilization_is_refused_where_it_does_not_belong,
             test_turbulence_on_quadratic_elements, test_sst_flat_plate_closes_its_residual, test_species_on_quadratic_elements,
             test_block_preconditioner_with_more_fields, test_block_preconditioner_with_turbulence,
             test_block_preconditioner_needs_a_flow, test_triangulation_of_every_element,
             test_adaptive_step_assembles_a_linear_system_once, test_adaptive_step_keeps_dt_for_a_small_growth]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__} {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
