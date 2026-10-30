"""Stage 17, priority 3: residual without the tangent and modified Newton in the nonlinear transient solver.

Run from the project root:  python Test/test_modified_newton.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.HeatTransfer import ht
from Source.Physics.Rheology import CarreauYasuda
from Source.Physics.Turbulence import sa, sst
from Source.Simulation.PointwiseLaw import PointwiseLaw
from Source.enums import ElementType, StudyType, ProblemType


def rect_mesh(n=4, element=ElementType.quadrilateral, size=(1.0, 1.0)):
    mesh = Mesh()
    mesh.generate2DMesh(size, (n, n), element)
    return mesh


def configure(model, **options):
    model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, **options)


def cavity(n=16, Re=400.0):

    mesh = rect_mesh(n)
    model = Model(name='cavity', dim=2, mesh=mesh, mat=material('X', miu=1.0 / Re, rho=1.0), psc=ff)
    model.physics.setInertia(True)
    model.physics.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), (1.0, 0.0))
    model.physics.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or x < 1e-9 or x > 1 - 1e-9))
    return model


def coupled_flow(element, axisymmetric=False):

    """Flow with inertia, a viscosity law of the temperature, viscous heating, buoyancy and convected heat."""

    mesh = rect_mesh(3, element)
    for node in mesh.NL:
        node.coor = (node.coor[0] + 0.5, node.coor[1])
    model = Model(name='flow', dim=2, mesh=mesh, mat=material('X', miu=0.1, rho=1.0, k=1.0, Cp=1.0), psc=[ff, ht],
                  axisymmetric=axisymmetric)
    fluid, heat = model.physics.fluid, model.physics.heat
    fluid.setInertia(True)
    fluid.setViscosityLaw(CarreauYasuda(mu0=1.0, mu_inf=0.1, lam=1.0, a=2.0, n=0.5))
    fluid.setViscousHeating(True)
    if not axisymmetric:
        fluid.setBoussinesq((0.0, -9.8), 0.01, 0.5)
    heat.setVelocity('u')
    heat.setStabilization('SUPG')
    return model


def residual_matches(model, seed=3):

    """R of residual(x) against that of residual_and_tangent(x): steady, in a time step and at the old level."""

    system = model._system()
    layout = system.layout
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=0.3, size=layout.ndof)
    if 'wall_distance' in layout.names:
        x[layout.dofs('wall_distance')[:, 0]] = model.physics.turbulence.var['wall_distance'].values
    old = x + rng.normal(scale=0.01, size=layout.ndof)

    contexts = (None, {'alpha': 10.0, 'beta': -10.0 * old, 'theta': 0.5, 'x_old': old, 'dt': 0.1},
                {'alpha': 15.0, 'beta': -15.0 * old, 'theta': 1.0, 'x_old': old, 'dt': 0.1}, {'old_level': True})
    try:
        for context in contexts:
            system.time = context
            _, full = system.residual_and_tangent(x)
            alone = system.residual(x)
            assert np.max(np.abs(full - alone)) <= 1e-12 * max(1.0, np.max(np.abs(full))), (context, np.max(np.abs(full - alone)))
    finally:
        system.time = None


# ----- tests -----

def test_residual_without_tangent_in_every_flow_configuration():

    for element in (ElementType.quadrilateral, ElementType.triangle):
        residual_matches(cavity(3))
        residual_matches(coupled_flow(element))
        residual_matches(coupled_flow(element, axisymmetric=True))


def test_residual_without_tangent_in_turbulence():

    for kind in (sa, sst):
        mesh = rect_mesh(3)
        model = Model(name='t', dim=2, mesh=mesh, mat=material('X', miu=0.05, rho=1.0), psc=[ff, kind])
        model.physics.fluid.setInertia(True)
        model.physics.turbulence.setWalls(mesh.boundary_edges(lambda x, y: y < 1e-9))
        model.physics.turbulence.coupleFlow()
        residual_matches(model)


def test_residual_without_tangent_in_the_generic_law_terms():

    """Volume, diffusion, gradient and facet laws of the System."""

    mesh = rect_mesh(3)
    model = Model(name='laws', dim=2, mesh=mesh, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: y < 1e-9), 1.0)
    system = model._system()
    system.add_volume_law(PointwiseLaw(lambda u: jnp.sin(u[0])), ['T'], ['T'])
    system.add_diffusion_law(PointwiseLaw(lambda u: 1.0 + u[0] ** 2), 'T')
    system.add_gradient_law(PointwiseLaw(lambda w: jnp.array([w[0] * w[1], w[1] ** 3, w[2]])), ['T'], ['T'])
    system.add_facet_law(PointwiseLaw(lambda u: u[0] ** 4), model.geometry.facets(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9)),
                         ['T'], ['T'])
    residual_matches(model)


def test_modified_newton_gives_the_same_transient():

    for options in (dict(timeScheme='BDF2'), dict(timeScheme='theta', timeDisc=0.5), dict(timeScheme='theta', timeDisc=1.0)):
        results = []
        for reuse in (False, True):
            model = cavity()
            configure(model, timeStep=0.05, totalTime=1.0, reuseTangent=reuse, **options)
            model.solve()
            results.append(model)

        full, modified = results
        err = np.max(np.abs(full.sol['u'] - modified.sol['u'])), np.max(np.abs(full.sol['p'] - modified.sol['p']))
        print(f"   {options}: tangents {full.tangent_assemblies} -> {modified.tangent_assemblies}, "
              f"iterations per step {np.mean(full.step_iterations):.1f} -> {np.mean(modified.step_iterations):.1f}, "
              f"difference u {err[0]:.1e}, p {err[1]:.1e}")
        assert err[0] < 1e-6 and err[1] < 1e-5
        assert modified.tangent_assemblies < 0.3 * full.tangent_assemblies


def test_modified_newton_rebuilds_the_tangent_when_it_stops_working():

    """A strongly nonlinear start (a shear-thinning melt set in motion) must still converge in every step."""

    mesh = rect_mesh(8)
    model = Model(name='start', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.0), psc=ff)
    model.physics.setViscosityLaw(CarreauYasuda(mu0=100.0, mu_inf=0.01, lam=10.0, a=2.0, n=0.3))
    model.physics.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), (1.0, 0.0))
    model.physics.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or x < 1e-9 or x > 1 - 1e-9))
    configure(model, timeStep=0.1, totalTime=1.0, timeScheme='BDF2')
    model.solve()

    print(f"   {model.tangent_assemblies} tangents in {len(model.step_iterations)} steps, "
          f"at most {max(model.step_iterations)} iterations")
    assert model.tangent_assemblies > 1
    assert max(model.step_iterations) < 30                                       # 30 is the limit: not converged


def test_a_non_finite_residual_stops_the_simulation():

    class Bomb:
        """Goes off after a number of evaluations."""
        calls = 0

        def contribution(self, x, tangent=True):
            Bomb.calls += 1
            value = np.nan if Bomb.calls > 40 else 0.0
            return np.full(len(x), value), (0.0 * sp_eye(len(x))).tocsr() if tangent else None

    from scipy.sparse import identity as sp_eye

    model = cavity(4)
    configure(model, timeStep=0.1, totalTime=2.0, timeScheme='BDF2')
    build = model._system
    def patched():
        system = build()
        if not any(isinstance(term, Bomb) for term in system._terms):
            system.add_term(Bomb())
        return system
    model._system = patched
    model.solve()

    steps = len(model.step_iterations)
    print(f"   stopped after {steps} of 20 steps, t = {model.timeVector[-1]:.2f}")
    assert 0 < steps < 20
    assert len(model.timeVector) == steps + 1 and model.sol['u'].shape[-1] == steps + 1
    assert np.all(np.isfinite(model.sol['u'])) and np.all(np.isfinite(model.sol['p']))


if __name__ == "__main__":
    tests = [test_residual_without_tangent_in_every_flow_configuration, test_residual_without_tangent_in_turbulence,
             test_residual_without_tangent_in_the_generic_law_terms, test_modified_newton_gives_the_same_transient,
             test_modified_newton_rebuilds_the_tangent_when_it_stops_working, test_a_non_finite_residual_stops_the_simulation]
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
