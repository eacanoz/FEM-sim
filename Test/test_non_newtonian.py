"""Stage 12 of the fluids plan: generalized Newtonian fluids, viscous heating and flow-heat coupling.

Run from the project root:  python Test/test_non_newtonian.py
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
from Source.Physics.Rheology import (Newtonian, PowerLaw, CarreauYasuda, Cross, CrossWLF, Bingham, HerschelBulkley,
                                     Arrhenius, Exponential)
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers


def configure(model, kind=ProblemType.nonlinear):
    model.solverConfiguration(Study=StudyType.steady_state, Type=kind, Method=SolverType.direct,
                              Solver=DirectSolvers.PARDISO)


def rect_mesh(size, div, element=ElementType.quadrilateral):
    mesh = Mesh()
    mesh.generate2DMesh(size, div, element)
    return mesh


def xy_of(mesh):
    return np.array([n.coor[:2] for n in mesh.NL], dtype=float)


def edges(mesh, selector):
    return mesh.boundary_edges(selector)


# ----- viscosity laws -----

def test_viscosity_laws():

    g = np.array([1e-3, 1e-1, 1.0, 10.0, 1e3])

    assert np.allclose(Newtonian(2.5).viscosity(g), 2.5)
    assert np.allclose(PowerLaw(K=3.0, n=0.4, gamma_min=0.0).viscosity(g), 3.0 * g ** -0.6)

    carreau = CarreauYasuda(mu0=100.0, lam=2.0, n=0.3)
    assert np.isclose(carreau.viscosity(1e-8), 100.0)
    assert np.isclose(carreau.viscosity(1e4) / carreau.viscosity(1e5), 10 ** 0.7, rtol=1e-3)   # power-law branch

    cross = Cross(mu0=1e3, tau_star=1e4, n=0.25)
    assert np.isclose(cross.viscosity(1e-8), 1e3)
    assert np.isclose(cross.viscosity(10.0), 1e3 / 2.0)         # mu0 gamma / tau_star = 1 -> mu0 / 2

    wlf = CrossWLF(n=0.3, tau_star=2e4, D1=1e12, D2=373.15, A1=25.0, A2=51.6)
    assert np.isclose(float(wlf.zero_shear(100.0)), 1e12)       # T = D2 (100 °C) -> D1
    assert wlf.viscosity(1.0, 250.0) < wlf.viscosity(1.0, 200.0)   # hotter melt flows easier

    bingham = Bingham(mu_p=2.0, tau_y=50.0, m=1e3)
    stress = bingham.viscosity(g[2:]) * g[2:]
    assert np.allclose(stress, 50.0 + 2.0 * g[2:], rtol=1e-6)   # tau = tau_y + mu_p gamma beyond the yield
    assert np.isfinite(bingham.viscosity(0.0)) and np.isclose(bingham.viscosity(0.0), 2.0 + 50.0 * 1e3)

    hb = HerschelBulkley(K=2.0, n=0.5, tau_y=10.0, m=1e3, gamma_min=0.0)
    assert np.allclose(hb.viscosity(g[2:]) * g[2:], 10.0 + 2.0 * g[2:] ** 0.5, rtol=1e-6)

    arr = Arrhenius(Newtonian(1.0), E_R=5000.0, T_ref=200.0)
    assert np.isclose(arr.viscosity(1.0, 200.0), 1.0)
    assert np.isclose(arr.viscosity(1.0, 250.0), np.exp(5000.0 * (1 / 523.15 - 1 / 473.15)))
    assert np.isclose(Exponential(Newtonian(4.0), b=0.02, T_ref=20.0).viscosity(1.0, 70.0), 4.0 * np.exp(-1.0))


# ----- channel flows with an analytical solution -----

def channel(law, dp, L=0.5, ny=16, newtonian_start=1.0):

    """Pressure-driven flow between plates y = 0 and y = 1: pressure dp at x = 0 and 0 at x = L, both fully developed.
    Solved first as Newtonian (mu = newtonian_start) and then with the law, from that solution."""

    mesh = rect_mesh((L, 1.0), (2, ny))
    model = Model(name='channel', dim=2, mesh=mesh, mat=material('X', miu=newtonian_start), psc=ff)
    model.physics.addBC_OutletEdges(edges(mesh, lambda x, y: x < 1e-9), dp)
    model.physics.addBC_OutletEdges(edges(mesh, lambda x, y: x > L - 1e-9), 0.0)
    model.physics.addBC_NoSlipEdges(edges(mesh, lambda x, y: y < 1e-9 or y > 1 - 1e-9))
    configure(model, ProblemType.linear)
    model.solve()

    model.physics.setViscosityLaw(law)
    configure(model)
    model.solve()

    return model, xy_of(mesh)


def test_power_law_channel():

    K, n, G = 2.0, 0.5, 4.0
    L = 0.5
    errors = []
    for ny in (8, 16, 32):
        model, xy = channel(PowerLaw(K, n, gamma_min=1e-4), G * L, L=L, ny=ny)
        s = np.abs(xy[:, 1] - 0.5)
        m = (n + 1) / n
        exact = n / (n + 1) * (G / K) ** (1 / n) * (0.5 ** m - s ** m)
        errors.append(np.max(np.abs(model.sol['u'][:, 0] - exact)) / exact.max())
        assert np.max(np.abs(model.sol['u'][:, 1])) < 1e-10
        assert np.max(np.abs(model.sol['p'] - G * (L - xy[:, 0]))) < 1e-6 * G * L
        assert model.newton_iterations < 30

    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    print(f"   power law n = {n}: relative errors {['%.1e' % e for e in errors]} rates {['%.2f' % r for r in rates]}")
    assert errors[-1] < 1e-3 and rates[-1] > 1.8


def test_bingham_channel_has_a_plug():

    mu_p, tau_y, G = 1.0, 0.5, 4.0          # yield surface at |y - 1/2| = tau_y / G = 0.125
    model, xy = channel(Bingham(mu_p, tau_y, m=2000.0), G * 0.5, ny=40)

    s = np.abs(xy[:, 1] - 0.5)
    y0 = tau_y / G
    yielded = (G / 2 * (0.25 - s ** 2) - tau_y * (0.5 - s)) / mu_p
    plug = (G / 2 * (0.25 - y0 ** 2) - tau_y * (0.5 - y0)) / mu_p
    exact = np.where(s < y0, plug, yielded)

    err = np.max(np.abs(model.sol['u'][:, 0] - exact)) / plug
    rate = model.sol['shear_rate']
    print(f"   Bingham: relative error {err:.1e}; shear rate in the plug {rate[s < 0.8 * y0].max():.1e}, "
          f"at the wall {rate[s > 0.49].mean():.2f}")
    assert err < 0.01
    assert rate[s < 0.8 * y0].max() < 2e-2 * rate[s > 0.49].mean()   # nearly rigid core


# ----- coupling with heat -----

def thermal_model(mesh, mu, k, rho=1.0, Cp=1.0):
    return Model(name='thermal', dim=2, mesh=mesh, mat=material('melt', miu=mu, rho=rho, Cp=Cp, k=k), psc=[ff, ht])


def test_viscous_heating_in_couette_flow():

    """Brinkman problem: Couette flow with both walls at T0. The dissipation mu (U/H)^2 gives
    T = T0 + mu U^2 / (2 k) (y/H)(1 - y/H), exact at the nodes."""

    mu, k, U, H, T0 = 3.0, 0.5, 2.0, 1.0, 20.0
    for element in (ElementType.quadrilateral, ElementType.triangle):
        mesh = rect_mesh((1.0, H), (3, 8), element)
        model = thermal_model(mesh, mu, k)
        fluid, heat = model.physics.fluid, model.physics.heat
        fluid.setViscousHeating(True)
        model.physics.setVelocity('u')                                  # heat convected by the computed flow
        fluid.addBC_VelocityEdges(edges(mesh, lambda x, y: x < 1e-9 or x > 1 - 1e-9), lambda x, y: (U * y / H, 0.0))
        fluid.addBC_NoSlipEdges(edges(mesh, lambda x, y: y < 1e-9))
        fluid.addBC_VelocityEdges(edges(mesh, lambda x, y: y > H - 1e-9), (U, 0.0))
        heat.addBC_TemperatureEdges(edges(mesh, lambda x, y: y < 1e-9 or y > H - 1e-9), T0)
        model.physics.initField('T', T0)
        configure(model)
        model.solve()

        xy = xy_of(mesh)
        y = xy[:, 1] / H
        assert np.max(np.abs(model.sol['u'] - np.c_[U * y, 0 * y])) < 1e-10
        exact = T0 + mu * U ** 2 / (2 * k) * y * (1 - y)
        assert np.max(np.abs(model.sol['T'] - exact)) < 1e-9, np.max(np.abs(model.sol['T'] - exact))
        assert np.allclose(model.sol['shear_rate'], U / H)


def mu_of_T_couette(law, ny, T1=0.0, T2=40.0, U=1.0):

    """Couette flow (no pressure gradient) across a linear temperature profile; returns (model, y)."""

    mesh = rect_mesh((0.5, 1.0), (2, ny))
    model = thermal_model(mesh, 1.0, 1.0)
    fluid, heat = model.physics.fluid, model.physics.heat
    fluid.setViscosityLaw(law)
    fluid.addBC_NoSlipEdges(edges(mesh, lambda x, y: y < 1e-9))
    fluid.addBC_VelocityEdges(edges(mesh, lambda x, y: y > 1 - 1e-9), (U, 0.0))
    fluid.addBC_OutletEdges(edges(mesh, lambda x, y: x < 1e-9))           # no pressure gradient: pure shear
    fluid.addBC_OutletEdges(edges(mesh, lambda x, y: x > 0.5 - 1e-9))
    heat.addBC_TemperatureEdges(edges(mesh, lambda x, y: y < 1e-9), T1)
    heat.addBC_TemperatureEdges(edges(mesh, lambda x, y: y > 1 - 1e-9), T2)
    model.physics.initField('T', T1)
    configure(model)
    model.solve()

    y = xy_of(mesh)[:, 1]
    assert np.max(np.abs(model.sol['T'] - (T1 + (T2 - T1) * y))) < 1e-9
    return model, y


def test_temperature_dependent_viscosity_in_couette_flow():

    """Linear temperature across the gap: the shear stress is uniform, so u(y) = U * int_0^y 1/mu / int_0^1 1/mu
    (one-way coupling T -> u, solved monolithically with the dmu/dT blocks).

    mu = mu0 exp(-b (T - T_ref)) is exponential in y: the 2-point Gauss mean of mu on every element is the exact mean
    times the same factor, which cancels, so linear elements are nodally exact. An Arrhenius law (exponential in 1/T)
    is not, and converges at second order."""

    b, T1, T2 = 0.05, 0.0, 40.0
    model, y = mu_of_T_couette(Exponential(Newtonian(2.0), b=b, T_ref=0.0), 8)
    a = b * (T2 - T1)
    assert np.max(np.abs(model.sol['u'][:, 0] - np.expm1(a * y) / np.expm1(a))) < 1e-12

    law = Arrhenius(Newtonian(2.0), E_R=6000.0, T_ref=0.0)
    s = np.linspace(0.0, 1.0, 200001)
    inv_mu = 1.0 / law.viscosity(1.0, T1 + (T2 - T1) * s)
    F = np.concatenate([[0.0], np.cumsum(0.5 * (inv_mu[1:] + inv_mu[:-1]) * np.diff(s))])
    errors = []
    for ny in (8, 16, 32):
        model, y = mu_of_T_couette(law, ny)
        exact = np.interp(y, s, F) / F[-1]
        errors.append(np.max(np.abs(model.sol['u'][:, 0] - exact)))

    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    print(f"   Arrhenius mu(T) Couette: errors {['%.1e' % e for e in errors]} rates {['%.2f' % r for r in rates]}")
    assert errors[-1] < 1e-4 and rates[-1] > 1.8


def test_heat_convected_by_the_computed_velocity_matches_a_prescribed_one():

    """Plug flow solved by ff (symmetry walls) carries the heat exactly as the same constant velocity given to ht."""

    U, k, L = 1.0, 0.2, 1.0
    mesh = rect_mesh((L, 0.25), (20, 2), ElementType.triangle)
    inlet, outlet = edges(mesh, lambda x, y: x < 1e-9), edges(mesh, lambda x, y: x > L - 1e-9)

    for stab in (None, 'SUPG'):
        coupled = thermal_model(mesh, 1.0, k)
        fluid, heat = coupled.physics.fluid, coupled.physics.heat
        fluid.addBC_VelocityEdges(inlet, (U, 0.0))
        fluid.addBC_SymmetryEdges(edges(mesh, lambda x, y: y < 1e-9))
        fluid.addBC_SymmetryEdges(edges(mesh, lambda x, y: y > 0.25 - 1e-9))
        fluid.addBC_OutletEdges(outlet)
        coupled.physics.setVelocity('u')
        coupled.physics.setStabilization(stab)
        heat.addBC_TemperatureEdges(inlet, 0.0)
        heat.addBC_TemperatureEdges(outlet, 1.0)
        coupled.physics.initField('T', 0.0)
        configure(coupled)
        coupled.solve()

        alone = Model(name='alone', dim=2, mesh=mesh, mat=material('melt', rho=1.0, Cp=1.0, k=k), psc=ht)
        alone.physics.setVelocity((U, 0.0))
        alone.physics.setStabilization(stab)
        alone.physics.addBC_TemperatureEdges(inlet, 0.0)
        alone.physics.addBC_TemperatureEdges(outlet, 1.0)
        alone.physics.initField('T', 0.0)
        configure(alone, ProblemType.linear)
        alone.solve()

        assert np.max(np.abs(coupled.sol['u'] - [U, 0.0])) < 1e-10
        assert np.max(np.abs(coupled.sol['T'] - alone.sol['T'])) < 1e-9, (stab, np.max(np.abs(coupled.sol['T'] - alone.sol['T'])))


def test_coupled_tangent_against_finite_differences():

    """Velocity, pressure and temperature: Carreau law with an Arrhenius shift, inertia (SUPG, LSIC), PSPG with the
    recovered stress, viscous heating, and heat convected by u with SUPG. tau frozen at the state."""

    for element in (ElementType.quadrilateral, ElementType.triangle):
        mesh = rect_mesh((1.0, 1.0), (2, 2), element)
        model = thermal_model(mesh, 1.0, 0.3, rho=1.2, Cp=1.5)
        fluid = model.physics.fluid
        fluid.setViscosityLaw(Arrhenius(CarreauYasuda(mu0=2.0, lam=0.7, n=0.4), E_R=800.0, T_ref=150.0))
        fluid.setInertia(True)
        fluid.setViscousHeating(True)
        fluid.setBodyForce((0.3, -0.2))
        model.physics.setVelocity('u')
        model.physics.setStabilization('SUPG')
        configure(model)

        layout = model.layout
        rng = np.random.default_rng(3)
        x = rng.normal(scale=0.5, size=layout.ndof)
        x[layout.dofs('T')[:, 0]] = 150.0 + 20.0 * rng.normal(size=layout.n_nodes)

        system = model._system()
        for term in system._terms:
            term.tau_state = x.copy()

        J, _ = system._assemble(x)
        FD = np.zeros((layout.ndof, layout.ndof))
        for kdof in range(layout.ndof):
            h = 1e-6 * max(1.0, abs(x[kdof]))
            e = np.zeros(layout.ndof)
            e[kdof] = h
            FD[:, kdof] = (system._assemble(x + e)[1] - system._assemble(x - e)[1]) / (2 * h)

        err = np.max(np.abs(J.toarray() - FD)) / np.max(np.abs(FD))
        assert err < 1e-7, (element, err)


def test_descriptions_and_errors():

    mesh = rect_mesh((1.0, 1.0), (2, 2))
    model = Model(name='desc', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.0), psc=ff)
    model.physics.setViscosityLaw(Cross(mu0=10.0, tau_star=1.0, n=0.3))
    assert model.physics_status == 'Creeping flow (Stokes), Cross viscosity'
    model.physics.setViscosity(1.0)
    assert model.physics_status == 'Creeping flow (Stokes)'

    model.physics.setViscosityLaw(CrossWLF(n=0.3, tau_star=1e4, D1=1e12, D2=373.15, A1=25.0, A2=51.6))
    configure(model)
    try:
        model.solve()
    except ValueError as error:
        assert "'T'" in str(error)
    else:
        raise AssertionError("expected an error: the law needs a temperature field")


if __name__ == "__main__":
    tests = [test_viscosity_laws, test_power_law_channel, test_bingham_channel_has_a_plug,
             test_viscous_heating_in_couette_flow, test_temperature_dependent_viscosity_in_couette_flow,
             test_heat_convected_by_the_computed_velocity_matches_a_prescribed_one,
             test_coupled_tangent_against_finite_differences, test_descriptions_and_errors]
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
