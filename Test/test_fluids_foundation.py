"""Stage 9 of the fluids plan: vector operators, gradient laws, vector fields in the model and Darcy flow.

Run from the project root:  python Test/test_fluids_foundation.py
"""

import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")

import numpy as np
import scipy.sparse.linalg as spla
import sympy as sym

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.Physics import physics
from Source.Physics.Darcy import darcy
from Source.Physics.HeatTransfer import ht
from Source.Physics.MassTransfer import mt
from Source.Primals.Scalar import scalarField
from Source.Primals.Vector import vectorField
from Source.Simulation.Discretization import Discretization
from Source.Simulation.DofLayout import DofLayout, FieldSpec
from Source.Simulation.PointwiseLaw import PointwiseLaw
from Source.Simulation.System import System
import Source.Simulation.Operators as ops
from Source.enums import ElementType, ShapeFunctionType, StudyType, ProblemType, SolverType, DirectSolvers


def configure(model, study=StudyType.steady_state, kind=ProblemType.linear, **time):
    model.solverConfiguration(Study=study, Type=kind, Method=SolverType.direct, Solver=DirectSolvers.PARDISO, **time)


def square_mesh(n, element=ElementType.quadrilateral, size=(1.0, 1.0), ny=None):
    mesh = Mesh()
    mesh.generate2DMesh(size, (n, n if ny is None else ny), element)
    return mesh


def line_mesh(n, L=1.0):
    mesh = Mesh()
    mesh.Generate_Mesh(1, L, n, ElementType.line)
    mesh.setElementShapeFunction(ShapeFunctionType.linear)
    return mesh


def distorted(mesh, amount=0.15, seed=0):

    """Moves the interior nodes randomly (keeps the elements valid) so the tests do not rely on regular grids."""

    rng = np.random.default_rng(seed)
    xy = np.array([n.coor for n in mesh.NL], dtype=float)
    h = 1.0 / (round(np.sqrt(len(xy))) - 1)

    for node in mesh.NL:
        x, y = node.coor
        if 1e-12 < x < 1 - 1e-12 and 1e-12 < y < 1 - 1e-12:
            node.coor = tuple(np.array([x, y]) + amount * h * rng.uniform(-1, 1, 2))

    return mesh


def xy_of(mesh):
    return np.array([n.coor[:2] for n in mesh.NL], dtype=float)


def on_boundary(x, y, L=(1.0, 1.0)):
    return min(x, y, L[0] - x, L[1] - y) < 1e-12


# ----- a vector physics used only by these tests -----

class vectorTest(physics):

    """-div(2 mu eps(u)) = f (operator='viscous') or m du/dt - mu lap(u) = f (operator='laplace')."""

    part_name = 'vector'

    def __init__(self, model):
        super().__init__(model)
        self.physics_description = 'Vector test physics'
        self.var = {'u': vectorField('u', 'Test vector', '-', ShapeFunctionType.linear, model.mesh, ncomp=2)}
        self.mu = 1.0
        self.f = None
        self.M_const = 0.0
        self.operator = 'viscous'

    def add_terms(self, system, model, disc, layout):
        rows = layout.element_dofs(disc.conn, 'u')
        K = ops.viscous(disc, self.mu) if self.operator == 'viscous' else ops.vector_diffusion(disc, self.mu)
        system.add_matrix(layout.scatter_matrix(K, rows, rows))
        if self.f is not None:
            system.add_vector(layout.scatter_vector(ops.vector_source(disc, self.f), rows))
        system.add_mass(layout.scatter_matrix(ops.vector_mass(disc, self.M_const), rows, rows))


class lawTest(physics):

    """One scalar field 'w' whose whole weak form is a gradient law (inputs and outputs: 'w')."""

    part_name = 'law'

    def __init__(self, model):
        super().__init__(model)
        self.physics_description = 'Gradient law test physics'
        self.var = {'w': scalarField('w', 'Test scalar', '-', ShapeFunctionType.linear, model.mesh)}
        self.law = None

    def add_terms(self, system, model, disc, layout):
        system.add_gradient_law(self.law, ['w'], ['w'])


# ----- operators -----

def test_vector_operators():

    disc = Discretization(distorted(square_mesh(3)))
    nel, nne = disc.conn.shape
    xy = disc.node_xy

    M, K = ops.mass(disc, 2.0), ops.diffusion(disc, 3.0)
    Mv, Kv = ops.vector_mass(disc, 2.0), ops.vector_diffusion(disc, 3.0)
    assert Mv.shape == (nel, 2 * nne, 2 * nne)
    assert np.allclose(Mv[:, 0::2, 0::2], M) and np.allclose(Mv[:, 1::2, 1::2], M) and np.allclose(Mv[:, 0::2, 1::2], 0)
    assert np.allclose(Kv[:, 1::2, 1::2], K)

    V = ops.viscous(disc, 1.5)
    assert np.allclose(V, np.swapaxes(V, 1, 2))

    # Rigid motions have no strain: translation and rotation are in the kernel of the viscous operator
    for u in (np.c_[np.ones(len(xy)), np.zeros(len(xy))], np.c_[-xy[:, 1], xy[:, 0]]):
        ue = u[disc.conn].reshape(nel, -1)
        assert np.max(np.abs(np.einsum('eij,ej->ei', V, ue))) < 1e-12

    # Energy of a pure shear u = (y, 0): integral of 2 mu eps:eps = 2 mu * 2 * (1/2)^2 = mu (unit square)
    ue = np.c_[xy[:, 1], np.zeros(len(xy))][disc.conn].reshape(nel, -1)
    assert np.isclose(np.einsum('ei,eij,ej->', ue, V, ue), 1.5)

    # Divergence: div(x, y) = 2 -> integral of q * 2; transpose is the pressure-gradient block
    D = ops.divergence(disc)
    ue = xy[disc.conn].reshape(nel, -1)
    assert np.allclose(np.einsum('eij,ej->ei', D, ue), 2.0 * ops.source(disc, 1.0))

    p = 1.0 + xy[:, 0] ** 2
    v = np.c_[xy[:, 1], xy[:, 0] * xy[:, 1]]
    lhs = np.einsum('ei,eij,ej->', p[disc.conn], D, v[disc.conn].reshape(nel, -1))
    G = np.swapaxes(D, 1, 2)
    assert np.isclose(lhs, np.einsum('ei,eij,ej->', v[disc.conn].reshape(nel, -1), G, p[disc.conn]))

    # Vector source and gradient source
    F = ops.vector_source(disc, (2.0, -1.0))
    assert np.allclose(F[:, 0::2], 2.0 * ops.source(disc, 1.0)) and np.allclose(F[:, 1::2], -ops.source(disc, 1.0))
    assert np.isclose(ops.gradient_source(disc, (1.0, 0.0)).sum(), 0.0)   # sum of grad(N_i) = 0

    # Gradient of a linear field is exact; the lumped projection is exact for constants
    g = disc.gradient(np.c_[3 * xy[:, 0] - xy[:, 1], xy[:, 1]])
    assert np.allclose(g[..., 0, :], [3.0, -1.0]) and np.allclose(g[..., 1, :], [0.0, 1.0])
    assert np.allclose(disc.project(np.broadcast_to([1.0, 2.0], disc.detJw.shape + (2,))), [1.0, 2.0])


def manufactured_viscous(mu=1.3):

    x, y = sym.symbols('x y')
    u = [sym.sin(sym.pi * x) * sym.sin(sym.pi * y), x ** 2 * y * (1 - y) + sym.cos(sym.pi * x) * y]
    div = sym.diff(u[0], x) + sym.diff(u[1], y)
    f = [-mu * (sym.diff(u[a], x, 2) + sym.diff(u[a], y, 2) + sym.diff(div, [x, y][a])) for a in range(2)]

    exact = [sym.lambdify((x, y), c, 'numpy') for c in u]
    force = [sym.lambdify((x, y), c, 'numpy') for c in f]

    return (lambda X, Y: tuple(np.broadcast_to(g(X, Y), np.shape(X)) for g in exact),
            lambda X, Y: tuple(np.broadcast_to(g(X, Y), np.shape(X)) for g in force))


def test_viscous_manufactured_solution_converges_at_second_order():

    mu = 1.3
    exact, force = manufactured_viscous(mu)

    for element in (ElementType.quadrilateral, ElementType.triangle):
        errors = []
        for n in (4, 8, 16, 32):
            mesh = square_mesh(n, element)
            model = Model(name='mms', dim=2, mesh=mesh, mat=material('X'), psc=vectorTest)
            model.physics.mu = mu
            model.physics.f = force

            for node in mesh.NL:
                if on_boundary(*node.coor):
                    node.BC['u'] = {'type': 'Dirichlet', 'value': [float(c) for c in exact(*node.coor)]}

            configure(model)
            model.solve()

            xy = xy_of(mesh)
            ref = np.stack(exact(xy[:, 0], xy[:, 1]), axis=-1)
            assert model.sol['u'].shape == (len(xy), 2)
            errors.append(np.max(np.abs(model.sol['u'] - ref)))

        rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
        print(f"   {element.name}: nodal errors {['%.2e' % e for e in errors]} rates {['%.2f' % r for r in rates]}")
        assert rates[-1] > 1.8, rates


def test_traction_and_component_dirichlet_patch():

    """u = (a x, -a y) is divergence free with constant stress: symmetry walls (one component fixed) on
    x = 0 and y = 0 and tractions sigma . n on x = 1 and y = 1 must reproduce it exactly."""

    a, mu = 0.7, 2.0

    for element in (ElementType.quadrilateral, ElementType.triangle):
        mesh = distorted(square_mesh(5, element), seed=3)
        model = Model(name='patch', dim=2, mesh=mesh, mat=material('X'), psc=vectorTest)
        model.physics.mu = mu

        for node in mesh.NL:
            x, y = node.coor
            ux = 0.0 if x < 1e-12 else None
            uy = 0.0 if y < 1e-12 else None
            if ux is not None or uy is not None:
                node.BC['u'] = {'type': 'Dirichlet', 'value': [ux, uy]}

        model.physics.facet_bcs += [
            {'type': 'Traction', 'field': 'u', 'facets': mesh.boundary_edges(lambda x, y: x > 1 - 1e-9),
             'traction': (2 * mu * a, 0.0)},
            {'type': 'Traction', 'field': 'u', 'facets': mesh.boundary_edges(lambda x, y: y > 1 - 1e-9),
             'traction': lambda x, y: (0.0 * x, -2 * mu * a + 0.0 * x)},
        ]

        configure(model)
        model.solve()

        xy = xy_of(mesh)
        assert np.max(np.abs(model.sol['u'] - np.c_[a * xy[:, 0], -a * xy[:, 1]])) < 1e-10

        # Only the fixed component is Dirichlet
        fixed = model.dirichletDofs().reshape(-1, 2)
        assert fixed[:, 0].sum() == 6 and fixed[:, 1].sum() == 6

    try:
        mesh = square_mesh(2)
        model = Model(name='bad', dim=2, mesh=mesh, mat=material('X'), psc=vectorTest)
        mesh.NL[0].BC['u'] = {'type': 'Dirichlet', 'value': [0.0]}
        configure(model)
        model.solve()
    except ValueError as error:
        assert 'componentes' in str(error)
    else:
        raise AssertionError("expected ValueError for a Dirichlet value with the wrong number of components")


def test_transient_vector_field_and_vtk_output():

    mesh = square_mesh(16)
    model = Model(name='decay', dim=2, mesh=mesh, mat=material('X'), psc=vectorTest)
    model.physics.operator = 'laplace'
    model.physics.M_const = 1.0

    s = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    model.physics.initField('u', lambda x, y: (s(x, y), 2 * s(x, y)))
    for node in mesh.NL:
        if on_boundary(*node.coor):
            node.BC['u'] = {'type': 'Dirichlet', 'value': [0.0, 0.0]}

    configure(model, StudyType.transient, timeDisc=0.5, timeStep=0.01, totalTime=0.1)
    model.solve()

    U = model.sol['u']
    xy = xy_of(mesh)
    assert U.shape == (len(xy), 2, 11)
    decay = np.exp(-2 * np.pi ** 2 * 0.1)
    ref = np.c_[s(xy[:, 0], xy[:, 1]), 2 * s(xy[:, 0], xy[:, 1])] * decay
    assert np.max(np.abs(U[..., -1] - ref)) < 0.02 * decay * 2, np.max(np.abs(U[..., -1] - ref))

    with tempfile.TemporaryDirectory() as folder:
        files = model.exportVTK(os.path.join(folder, 'decay'), every=5)
        assert len(files) == 4                     # steps 0, 5, 10 and the .pvd
        text = open(files[-2], encoding='utf-8').read()
        assert 'Name="u" NumberOfComponents="3"' in text


# ----- gradient laws -----

def system_with(fields, mesh):
    disc = Discretization(mesh)
    layout = DofLayout(fields, disc.n_nodes)
    return System(disc, layout), disc, layout


def test_gradient_law_reproduces_linear_operators():

    mesh = distorted(square_mesh(4), seed=1)
    rng = np.random.default_rng(2)
    v = np.array([0.8, -0.3])

    # Scalar: diffusion k grad c and convection v . grad c
    system, disc, layout = system_with([FieldSpec('c')], mesh)
    law = PointwiseLaw(lambda w: jnp.array([v[0] * w[1] + v[1] * w[2], 2.5 * w[1], 2.5 * w[2]]))
    system.add_gradient_law(law, ['c'], ['c'])

    x = rng.normal(size=layout.ndof)
    rows = layout.element_dofs(disc.conn, 'c')
    A = layout.scatter_matrix(ops.diffusion(disc, 2.5) + ops.convection(disc, 1.0, v), rows, rows)
    J, R = system._assemble(x)
    assert np.allclose(R, A @ x) and np.allclose(J.toarray(), A.toarray())

    # Vector: viscous stress mu (grad u + grad u^T), next to a scalar field that does not take part
    system, disc, layout = system_with([FieldSpec('p'), FieldSpec('u', 2)], mesh)
    mu = 1.7

    def stress(w):
        G = w[2:].reshape(2, 2)
        return jnp.concatenate([jnp.zeros(2), (mu * (G + G.T)).ravel()])

    system.add_gradient_law(PointwiseLaw(stress), ['u'], ['u'])

    x = rng.normal(size=layout.ndof)
    rows = layout.element_dofs(disc.conn, 'u')
    A = layout.scatter_matrix(ops.viscous(disc, mu), rows, rows)
    J, R = system._assemble(x)
    assert np.allclose(R, A @ x) and np.allclose(J.toarray(), A.toarray())

    # Wrong number of outputs
    bad, *_ = system_with([FieldSpec('c')], mesh)
    bad.add_gradient_law(PointwiseLaw(lambda w: jnp.array([w[0], w[1]])), ['c'], ['c'])
    try:
        bad._assemble(np.zeros(bad.layout.ndof))
    except ValueError as error:
        assert 'se esperaban 3' in str(error)
    else:
        raise AssertionError("expected ValueError for a law with the wrong number of outputs")


def test_gradient_law_jacobian_with_coupled_vector_and_scalar():

    """Nonlinear law reading the values and gradients of a vector u and a scalar c and writing to both."""

    for element in (ElementType.quadrilateral, ElementType.triangle):
        mesh = distorted(square_mesh(2, element), seed=4)
        system, disc, layout = system_with([FieldSpec('u', 2), FieldSpec('c')], mesh)

        def law(w):
            u, Gu, c, gc = w[0:2], w[2:6].reshape(2, 2), w[6], w[7:9]
            gamma2 = jnp.sum((Gu + Gu.T) ** 2)
            mu = 1.0 + 0.5 * gamma2 + 0.3 * c ** 2                      # viscosity of shear rate and c
            out_u = jnp.concatenate([c * u + jnp.sin(gc), (mu * (Gu + Gu.T)).ravel()])
            out_c = jnp.concatenate([jnp.array([u @ gc + c ** 3]), (1.0 + c ** 2) * gc + 0.1 * Gu @ u])
            return jnp.concatenate([out_c, out_u])

        system.add_gradient_law(PointwiseLaw(law), ['u', 'c'], ['c', 'u'])

        x = np.random.default_rng(5).normal(scale=0.5, size=layout.ndof)
        J, _ = system._assemble(x)

        h = 1e-6
        FD = np.zeros((layout.ndof, layout.ndof))
        for k in range(layout.ndof):
            e = np.zeros(layout.ndof)
            e[k] = h
            FD[:, k] = (system._assemble(x + e)[1] - system._assemble(x - e)[1]) / (2 * h)

        assert np.max(np.abs(J.toarray() - FD)) < 1e-7 * max(1.0, np.max(np.abs(FD))), np.max(np.abs(J.toarray() - FD))


def power_law_channel(n_index, n_el, K=1.0, G=1.0, eps=1e-8):

    """Power-law fluid between plates: -(K |w'|^(n-1) w')' = G, w(0) = w(1) = 0 (a p-Laplacian), solved with Newton."""

    mesh = line_mesh(n_el)
    model = Model(name='power', dim=1, mesh=mesh, mat=material('X'), psc=lawTest)

    def law(w):
        mu = K * (w[1] ** 2 + eps ** 2) ** ((n_index - 1.0) / 2.0)
        return jnp.array([-G, mu * w[1]])

    model.physics.law = PointwiseLaw(law)
    model.physics.initField('w', lambda y: G / (2 * K) * y * (1 - y) + 1e-3)   # Newtonian guess
    mesh.NL[0].BC['w'] = {'type': 'Dirichlet', 'value': 0.0}
    mesh.NL[-1].BC['w'] = {'type': 'Dirichlet', 'value': 0.0}

    configure(model, kind=ProblemType.nonlinear)
    model.solve()

    y = np.asarray(mesh.getXCoor(), dtype=float)
    m = (n_index + 1.0) / n_index
    exact = n_index / (n_index + 1.0) * (G / K) ** (1.0 / n_index) * (0.5 ** m - np.abs(y - 0.5) ** m)

    return np.max(np.abs(np.asarray(model.sol['w']) - exact)) / np.max(exact)


def test_power_law_channel_flow():

    for n_index in (0.5, 1.5):
        errors = [power_law_channel(n_index, n) for n in (8, 16, 32, 64)]
        rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
        print(f"   n = {n_index}: relative errors {['%.2e' % e for e in errors]} rates {['%.2f' % r for r in rates]}")
        assert errors[-1] < 2e-3 and np.all(rates > 1.0), (errors, rates)


# ----- Darcy -----

def test_darcy_1d_with_source():

    L, kappa, mu, Q = 2.0, 2.0, 0.5, 0.4
    K = kappa / mu
    mesh = line_mesh(10, L)
    model = Model(name='darcy1d', dim=1, mesh=mesh, mat=material('soil', kappa=kappa, miu=mu), psc=darcy)
    model.physics.setSource(Q)
    model.physics.addBC_Pressure(0, 3.0)
    model.physics.addBC_Pressure(len(mesh.NL) - 1, 1.0)
    configure(model)
    model.solve()

    x = np.asarray(mesh.getXCoor(), dtype=float)
    exact = Q / (2 * K) * x * (L - x) + 3.0 - x
    assert np.max(np.abs(np.asarray(model.sol['p']) - exact)) < 1e-12

    u = model.sol['u'][:, 0]
    u_exact = -K * (Q / (2 * K) * (L - 2 * x) - 1.0)
    assert np.allclose(u[1:-1], u_exact[1:-1], atol=1e-12)        # interior nodes: central difference, exact


def test_darcy_2d_linear_pressure_with_tensor_permeability_and_gravity():

    kappa = np.array([[2.0, 0.5], [0.5, 1.0]])
    mu, rho, g = 0.8, 1.2, np.array([0.0, -9.81])
    p_exact = lambda x, y: 4.0 - 1.5 * x + 2.0 * y

    for element in (ElementType.quadrilateral, ElementType.triangle):
        mesh = square_mesh(4, element, size=(2.0, 1.0))
        model = Model(name='darcy2d', dim=2, mesh=mesh, mat=material('soil', miu=mu, rho=rho), psc=darcy)
        model.physics.setPermeability(kappa)
        model.physics.setGravity(g)
        model.physics.addBC_PressureEdges(mesh.boundary_edges(), p_exact)
        configure(model)
        model.solve()

        xy = xy_of(mesh)
        assert np.max(np.abs(model.sol['p'] - p_exact(xy[:, 0], xy[:, 1]))) < 1e-11

        u_exact = -kappa @ (np.array([-1.5, 2.0]) - rho * g) / mu
        assert np.allclose(model.sol['u'], u_exact, atol=1e-11)
        assert np.allclose(model.physics.velocity(), u_exact, atol=1e-11)


def test_darcy_injection_and_hydrostatics():

    # Injection U through x = 0, impermeable top and bottom, p = 0 at x = L: p = mu U (L - x) / kappa
    L, kappa, mu, U = 3.0, 0.5, 2.0, 0.1
    mesh = square_mesh(6, ElementType.triangle, size=(L, 1.0), ny=3)
    model = Model(name='inj', dim=2, mesh=mesh, mat=material('soil', kappa=kappa, miu=mu), psc=darcy)
    model.physics.addBC_NormalVelocityEdges(mesh.boundary_edges(lambda x, y: x < 1e-9), -U)
    model.physics.addBC_PressureEdges(mesh.boundary_edges(lambda x, y: x > L - 1e-9), 0.0)
    configure(model)
    model.solve()

    xy = xy_of(mesh)
    assert np.max(np.abs(model.sol['p'] - mu * U * (L - xy[:, 0]) / kappa)) < 1e-11
    assert np.allclose(model.sol['u'], [U, 0.0], atol=1e-12)

    # Closed box with the pressure fixed on top: hydrostatic p = p0 + rho g (H - y) and no flow
    rho, H = 1000.0, 2.0
    mesh = square_mesh(4, size=(1.0, H))
    model = Model(name='hydro', dim=2, mesh=mesh, mat=material('water', kappa=1e-3, miu=1e-3, rho=rho), psc=darcy)
    model.physics.setGravity((0.0, -9.81))
    model.physics.addBC_PressureEdges(mesh.boundary_edges(lambda x, y: y > H - 1e-9), 1e5)
    configure(model)
    model.solve()

    xy = xy_of(mesh)
    assert np.max(np.abs(model.sol['p'] - (1e5 + rho * 9.81 * (H - xy[:, 1])))) < 1e-7
    assert np.max(np.abs(model.sol['u'])) < 1e-10 * rho * 9.81      # kappa/mu = 1: round-off of grad p - rho g


def test_darcy_velocity_convects_heat_and_species():

    """The Gauss-point Darcy velocity drives convection exactly as the equivalent constant velocity."""

    kappa, mu, dp, Pe = 0.2, 1.0, 1.0, 5.0
    U = kappa / mu * dp
    mesh = square_mesh(20, ElementType.triangle, ny=4)

    flow = Model(name='flow', dim=2, mesh=mesh, mat=material('rock', kappa=kappa, miu=mu), psc=darcy)
    flow.physics.addBC_PressureEdges(mesh.boundary_edges(lambda x, y: x < 1e-9), dp)
    flow.physics.addBC_PressureEdges(mesh.boundary_edges(lambda x, y: x > 1 - 1e-9), 0.0)
    configure(flow)
    flow.solve()
    velocity = flow.physics.velocity()
    assert velocity.shape[-1] == 2 and np.allclose(velocity, [U, 0.0])

    def transport(v):
        model = Model(name='tr', dim=2, mesh=mesh, mat=material('X', k=U / Pe, rho=1.0, Cp=1.0), psc=[ht, mt])
        model.physics.mass.setChemSpecies('A', 'A')
        model.physics.mass.setDiffusivity('A', U / Pe)
        model.physics.setVelocity(v)
        model.physics.mass.setStabilization('SUPG')
        model.physics.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x < 1e-9), 0.0)
        model.physics.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x > 1 - 1e-9), 1.0)
        model.physics.addBC_ConcentrationEdges(mesh.boundary_edges(lambda x, y: x < 1e-9), 'A', 0.0)
        model.physics.addBC_ConcentrationEdges(mesh.boundary_edges(lambda x, y: x > 1 - 1e-9), 'A', 1.0)
        configure(model)
        model.solve()
        return model

    darcy_driven, constant = transport(velocity), transport((U, 0.0))

    x = xy_of(mesh)[:, 0]
    exact = np.expm1(Pe * x) / np.expm1(Pe)
    for name in ('T', 'A'):
        assert np.allclose(darcy_driven.sol[name], constant.sol[name], atol=1e-12)
        assert np.max(np.abs(darcy_driven.sol[name] - exact)) < 1e-2, name


if __name__ == "__main__":
    tests = [test_vector_operators, test_viscous_manufactured_solution_converges_at_second_order,
             test_traction_and_component_dirichlet_patch, test_transient_vector_field_and_vtk_output,
             test_gradient_law_reproduces_linear_operators, test_gradient_law_jacobian_with_coupled_vector_and_scalar,
             test_power_law_channel_flow, test_darcy_1d_with_source,
             test_darcy_2d_linear_pressure_with_tensor_permeability_and_gravity, test_darcy_injection_and_hydrostatics,
             test_darcy_velocity_convects_heat_and_species]
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
