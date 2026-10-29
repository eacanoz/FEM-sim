"""Stage 15 of the fluids plan: Boussinesq buoyancy (natural convection), species carried by the computed flow and
boundary fluxes from the reactions.

Run from the project root:  python Test/test_buoyancy.py
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
from Source.Physics.HeatTransfer import ht
from Source.Physics.MassTransfer import mt
from Source.Physics.Rheology import CarreauYasuda
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers

PR = 0.71
# de Vahl Davis (1983): Nusselt of the hot wall, max u on x = 0.5, max v on y = 0.5 (scaled with alpha / L)
DE_VAHL_DAVIS = {1e3: (1.118, 3.649, 3.697), 1e4: (2.243, 16.178, 19.617), 1e5: (4.519, 34.73, 68.59)}


def configure(model, kind=ProblemType.nonlinear):
    model.solverConfiguration(Study=StudyType.steady_state, Type=kind, Method=SolverType.direct,
                              Solver=DirectSolvers.PARDISO)


def cavity_mesh(n, element=ElementType.quadrilateral, stretch=True):

    """Unit square, nodes clustered at the walls (Chebyshev distribution) to resolve the boundary layers."""

    mesh = Mesh()
    mesh.generate2DMesh((1.0, 1.0), (n, n), element)
    if stretch:
        for node in mesh.NL:
            node.coor = tuple(0.5 * (1 - np.cos(np.pi * c)) for c in node.coor)
    return mesh


def xy_of(mesh):
    return np.array([n.coor[:2] for n in mesh.NL], dtype=float)


def heated_cavity(mesh, species=False):

    """Nondimensional: L = rho = Cp = k = 1 (alpha = 1), mu = Pr, hot wall x = 0 (T = 1), cold wall x = 1 (T = 0),
    adiabatic top and bottom. Buoyancy set by the caller (Ra Pr as gravity)."""

    parts = [ff, ht, mt] if species else [ff, ht]
    model = Model(name='cavity', dim=2, mesh=mesh, mat=material('air', miu=PR, rho=1.0, Cp=1.0, k=1.0), psc=parts)
    fluid, heat = model.physics.fluid, model.physics.heat
    fluid.setInertia(True)
    model.physics.setVelocity('u')
    model.physics.setStabilization('SUPG')
    hot = mesh.boundary_edges(lambda x, y: x < 1e-9)
    cold = mesh.boundary_edges(lambda x, y: x > 1 - 1e-9)
    fluid.addBC_NoSlipEdges(mesh.boundary_edges(None))
    heat.addBC_TemperatureEdges(hot, 1.0)
    heat.addBC_TemperatureEdges(cold, 0.0)
    model.physics.initField('T', 0.5)
    if species:
        mass = model.physics.mass
        mass.setChemSpecies('A', 'A')
        mass.setDiffusivity('A', 1.0)
        mass.addBC_ConcentrationEdges(hot, 'A', 1.0)
        mass.addBC_ConcentrationEdges(cold, 'A', 0.0)
        model.physics.initField('A', 0.5)
    configure(model)

    return model, hot, cold


# ----- tests -----

def test_de_vahl_davis_cavity():

    """Differentially heated square cavity, Ra = 1e3, 1e4, 1e5 (continuation): Nusselt number of the hot wall from the
    reactions and maximum velocities on the mid-planes against de Vahl Davis (1983); heat in = heat out exactly."""

    mesh = cavity_mesh(32)
    model, hot, cold = heated_cavity(mesh)
    xy = xy_of(mesh)
    vertical, horizontal = np.abs(xy[:, 0] - 0.5) < 1e-9, np.abs(xy[:, 1] - 0.5) < 1e-9

    for Ra, (nu_ref, u_ref, v_ref) in DE_VAHL_DAVIS.items():
        model.physics.fluid.setBoussinesq((0.0, -Ra * PR), beta=1.0, reference=0.5)
        model.solve()

        nu_hot, nu_cold = model.physics.heat.boundaryFlux(hot), model.physics.heat.boundaryFlux(cold)
        u_max, v_max = model.sol['u'][vertical, 0].max(), model.sol['u'][horizontal, 1].max()
        print(f"   Ra = {Ra:.0e}: Nu {nu_hot:.4f} ({nu_hot / nu_ref - 1:+.2%}), u_max {u_max:.3f} ({u_max / u_ref - 1:+.2%}), "
              f"v_max {v_max:.3f} ({v_max / v_ref - 1:+.2%}), Newton {model.newton_iterations}")
        assert abs(nu_hot / nu_ref - 1) < 0.01 and abs(u_max / u_ref - 1) < 0.025 and abs(v_max / v_ref - 1) < 0.025
        assert abs(nu_hot + nu_cold) < 1e-10 * nu_hot                       # energy balance of the reactions


def test_stable_stratification_stays_at_rest():

    """Heated from above with adiabatic sides: T = y conducts and the buoyancy is balanced by the pressure, the fluid
    stays at rest. Q4: exact, also the quadratic pressure. T3: the quadratic pressure cannot be represented and a small
    part of the buoyancy leaks into spurious currents (the discretization is not pressure-robust); they vanish with
    refinement."""

    def stratified(n, element):
        mesh = cavity_mesh(n, element, stretch=False)
        model = Model(name='stratified', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.0, Cp=1.0, k=1.0),
                      psc=[ff, ht])
        model.physics.fluid.setInertia(True)
        model.physics.fluid.setBoussinesq((0.0, -1e4), beta=1.0, reference=0.5)
        model.physics.setVelocity('u')
        model.physics.fluid.addBC_NoSlipEdges(mesh.boundary_edges(None))
        model.physics.heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), 1.0)
        model.physics.heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: y < 1e-9), 0.0)
        configure(model)
        model.solve()
        return model, xy_of(mesh)

    model, xy = stratified(10, ElementType.quadrilateral)
    p_exact = 1e4 * (xy[:, 1] ** 2 / 2 - 0.5 * xy[:, 1])                      # dp/dy = rho beta G (T - 0.5)
    dp = model.sol['p'] - p_exact
    assert np.max(np.abs(model.sol['T'] - xy[:, 1])) < 1e-12
    assert np.max(np.abs(model.sol['u'])) < 1e-10 and np.max(np.abs(dp - dp.mean())) < 1e-8

    spurious = [np.max(np.abs(stratified(n, ElementType.triangle)[0].sol['u'])) for n in (8, 16, 32)]
    rates = np.log2(np.array(spurious[:-1]) / np.array(spurious[1:]))
    print(f"   T3 spurious currents {['%.1e' % s for s in spurious]} rates {['%.2f' % r for r in rates]} "
          f"(buoyancy scale G h^2 / mu = {1e4 / 32 ** 2:.1f} on the finest mesh)")
    assert rates[-1] > 1.5


def test_species_follow_the_computed_flow():

    """A species with the diffusivity and the boundary values of the temperature (Lewis number 1) is carried by the
    same velocity: c = T. The same buoyancy given through the species (solutal) gives the same flow."""

    Ra = 1e4
    mesh = cavity_mesh(16, ElementType.triangle)
    thermal, _, _ = heated_cavity(mesh, species=True)
    thermal.physics.fluid.setBoussinesq((0.0, -Ra * PR), beta=1.0, reference=0.5, field='T')
    thermal.solve()
    assert np.max(np.abs(thermal.sol['A'] - thermal.sol['T'])) < 1e-8

    solutal, _, _ = heated_cavity(mesh, species=True)
    solutal.physics.fluid.setBoussinesq((0.0, -Ra * PR), beta=1.0, reference=0.5, field='A')
    solutal.solve()
    assert np.max(np.abs(solutal.sol['u'] - thermal.sol['u'])) < 1e-7 * np.max(np.abs(thermal.sol['u']))
    assert 'Boussinesq buoyancy (A)' in solutal.physics_status


def test_tangent_with_buoyancy():

    """Flow (Carreau, inertia) with thermal and solutal buoyancy, heat and a species convected with SUPG, steady
    and with a Crank-Nicolson step context, against finite differences (tau frozen)."""

    for element in (ElementType.quadrilateral, ElementType.triangle):
        mesh = cavity_mesh(2, element, stretch=False)
        model = Model(name='tangent', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.2, Cp=1.5, k=0.3),
                      psc=[ff, ht, mt])
        fluid = model.physics.fluid
        fluid.setInertia(True)
        fluid.setViscosityLaw(CarreauYasuda(mu0=2.0, lam=0.7, n=0.4))
        fluid.setBoussinesq((0.3, -9.0), beta=0.02, reference=10.0, field='T')
        fluid.setBoussinesq((0.0, -9.0), beta=0.5, reference=0.2, field='A')
        model.physics.mass.setChemSpecies('A', 'A')
        model.physics.mass.setDiffusivity('A', 0.1)
        model.physics.setVelocity('u')
        model.physics.setStabilization('SUPG')
        configure(model)

        layout = model.layout
        rng = np.random.default_rng(8)
        x = rng.normal(scale=0.5, size=layout.ndof)
        system = model._system()
        for term in system._terms:
            if hasattr(term, 'tau_state'):
                term.tau_state = x.copy()

        for context in (None, {'alpha': 20.0, 'beta': rng.normal(size=layout.ndof), 'theta': 0.5,
                               'x_old': rng.normal(scale=0.5, size=layout.ndof), 'dt': 0.05}):
            system.time = context
            J, _ = system._assemble(x)
            FD = np.zeros((layout.ndof, layout.ndof))
            for k in range(layout.ndof):
                e = np.zeros(layout.ndof)
                e[k] = 1e-6
                FD[:, k] = (system._assemble(x + e)[1] - system._assemble(x - e)[1]) / 2e-6
            err = np.max(np.abs(J.toarray() - FD)) / np.max(np.abs(FD))
            assert err < 1e-7, (element, context is not None, err)
        system.time = None


def test_boundary_flux_of_conduction():

    """Plane wall: T = 1 at x = 0 and T = 0 at x = L, k = 2: the heat entering through the hot face is k / L per unit
    height, and leaves through the cold one."""

    k, L, H = 2.0, 0.5, 0.3
    mesh = Mesh()
    mesh.generate2DMesh((L, H), (5, 3), ElementType.triangle)
    model = Model(name='wall', dim=2, mesh=mesh, mat=material('X', k=k, rho=1.0, Cp=1.0), psc=ht)
    hot, cold = mesh.boundary_edges(lambda x, y: x < 1e-9), mesh.boundary_edges(lambda x, y: x > L - 1e-9)
    model.physics.addBC_TemperatureEdges(hot, 1.0)
    model.physics.addBC_TemperatureEdges(cold, 0.0)
    configure(model, ProblemType.linear)
    model.solve()
    assert np.isclose(model.physics.boundaryFlux(hot), k / L * H, rtol=1e-12)
    assert np.isclose(model.physics.boundaryFlux(cold), -k / L * H, rtol=1e-12)


if __name__ == "__main__":
    tests = [test_de_vahl_davis_cavity, test_stable_stratification_stays_at_rest, test_species_follow_the_computed_flow,
             test_tangent_with_buoyancy, test_boundary_flux_of_conduction]
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
