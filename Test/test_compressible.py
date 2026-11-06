"""Compressible flow of polymer melts: the two-domain Tait equation of state, its fit, and the compressible continuity of ff.setDensityLaw.
Each test prints its progress.

Run from the project root:  python -u Test/test_compressible.py
"""

import logging
import os
import sys
import time

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
from Source.Physics.Rheology import TaitPVT, Cross, CrossWLF
from Source.Physics.Fitting import fit_tait
from Source.enums import ElementType, StudyType, ProblemType
from Source.logger_config import sim_logger

sim_logger.setLevel(logging.WARNING)
START = time.perf_counter()
ELEMENTS = (ElementType.quadrilateral, ElementType.triangle)

# A glass-filled polypropylene: the fit of fit_tait to measured pvT data (0.1 to 200 MPa, 20 to 280 C), SI units
PP_GF = dict(b1m=1.006e-3, b2m=4.936e-7, b3m=7.989e7, b4m=3.53e-3, b1s=9.073e-4, b2s=1.49e-7, b3s=2.601e8, b4s=2.353e-3,
             b5=453.9, b6=1.709e-7, b7=1.009e-4, b8=4.788e-2, b9=1.553e-8)


def note(text):
    print(f"   [{time.perf_counter() - START:6.1f} s] {text}", flush=True)


def rect(size, div, element=ElementType.quadrilateral):
    mesh = Mesh()
    mesh.generate2DMesh(size, div, element, order=2)
    return mesh


def xy_of(mesh):
    return np.array([n.coor[:2] for n in mesh.NL], dtype=float)


def exponential(kappa, rho0):
    """rho = rho0 exp(kappa (p - ambient)): constant compressibility kappa [1/Pa], with the gauge pressure of the model."""
    return lambda P, T: rho0 * jnp.exp(kappa * (P - 1.0e5))


# ----- the equation of state and its fit -----

def test_tait_reduces_to_its_branches():

    """Far from the transition the smoothed law is the melt or the solid branch to round-off; the transition moves with b6."""

    law = TaitPVT(**PP_GF, width=1.0)
    sharp = lambda p, T, branch: np.asarray(TaitPVT(**{**PP_GF, 'b5': PP_GF['b5'] + (-500 if branch == 'melt' else 500)},
                                                    width=1.0).specific_volume(p, T))
    for p in (1e5, 1e8, 2e8):
        Tt = float(law.transition(p))
        for T, branch in ((Tt + 60, 'melt'), (Tt - 60, 'solid')):
            v = float(law.specific_volume(p, T))
            b = dict(PP_GF)
            C = 0.0894
            dT = T + 273.15 - b['b5']
            if branch == 'melt':
                exact = (b['b1m'] + b['b2m'] * dT) * (1 - C * np.log1p(p / (b['b3m'] * np.exp(-b['b4m'] * dT))))
            else:
                exact = ((b['b1s'] + b['b2s'] * dT) * (1 - C * np.log1p(p / (b['b3s'] * np.exp(-b['b4s'] * dT))))
                         + b['b7'] * np.exp(b['b8'] * dT - b['b9'] * p))
            assert abs(v / exact - 1) < 1e-12, (p, T, branch)
        note(f"p {p / 1e6:5.1f} MPa: transition {Tt:.2f} C")
    assert abs(float(law.transition(2e8)) - float(law.transition(1e5)) - PP_GF['b6'] * (2e8 - 1e5)) < 1e-9


def test_tait_fit_recovers_its_parameters():

    """pvT data of a known law (seven isobars from 0.1 to 200 MPa, 20 to 280 C, 0.02 % scatter as a dilatometer): the fit gives back the
    specific volume to the scatter everywhere, and the transition line within 1 K."""

    law = TaitPVT(**PP_GF, width=1.0)
    p = np.repeat([0.1e6, 20e6, 40e6, 80e6, 120e6, 160e6, 200e6], 80)
    T = np.tile(np.linspace(20.0, 280.0, 80), 7)
    v = np.asarray(law.specific_volume(p, T)) * np.exp(2e-4 * np.random.default_rng(5).standard_normal(len(p)))
    fit = fit_tait(p, T, v, width=1.0)
    exact = np.asarray(law.specific_volume(p, T))
    err = np.max(np.abs(np.asarray(fit.model.specific_volume(p, T)) / exact - 1))
    dTt = max(abs(float(fit.model.transition(q)) - float(law.transition(q))) for q in (1e5, 1e8, 2e8))
    note(f"rms {fit.rms:.3%} (scatter 0.02 %), largest error against the exact law {err:.3%}, transition within {dTt:.2f} K")
    assert fit.rms < 3e-4 and err < 2e-3 and dTt < 1.0


# ----- compressible flow -----

def tangent_error(model):
    system = model._system()
    layout = system.layout
    rng = np.random.default_rng(3)
    x = rng.normal(scale=0.3, size=layout.ndof)
    if 'T' in layout.names:
        x[layout.dofs('T').ravel()] += 200.0
    J, _ = system.residual_and_tangent(x)
    FD = np.zeros((layout.ndof, layout.ndof))
    for k in range(layout.ndof):
        e = np.zeros(layout.ndof)
        e[k] = 1e-6
        FD[:, k] = (system.residual(x + e) - system.residual(x - e)) / 2e-6
    return np.max(np.abs(J.toarray() - FD)) / np.max(np.abs(FD))


def time_part_error(model):

    """The time derivative alone: R(x) with a step minus R(x) without it, its tangent against finite differences. Taken apart because
    it is small next to the viscous entries, which would hide it in the relative error of the whole tangent."""

    system = model._system()
    layout = system.layout
    rng = np.random.default_rng(3)
    x = rng.normal(scale=0.3, size=layout.ndof)
    if 'T' in layout.names:
        x[layout.dofs('T').ravel()] += 200.0
    x_old = np.random.default_rng(8).normal(scale=0.3, size=layout.ndof)
    step = {'alpha': 7.0, 'beta': -7.0 * x_old, 'theta': 1.0, 'x_old': x_old, 'dt': 1 / 7.0}

    def residual(y, timed):
        system.time = step if timed else None
        return system.residual(y)

    system.time = step
    J1, _ = system.residual_and_tangent(x)
    system.time = None
    J0, _ = system.residual_and_tangent(x)
    J = (J1 - J0).toarray()
    FD = np.zeros_like(J)
    for k in range(layout.ndof):
        e = np.zeros(layout.ndof)
        e[k] = 1e-6
        FD[:, k] = ((residual(x + e, True) - residual(x + e, False)) - (residual(x - e, True) - residual(x - e, False))) / 2e-6
    system.time = None
    assert np.max(np.abs(FD)) > 0
    return np.max(np.abs(J - FD)) / np.max(np.abs(FD))


def test_tangent():

    """Continuity, -2/3 mu div u and the compression heating against finite differences, with a thinning viscosity law (without its d mu /
    d gamma_dot in -2/3 mu div u, Newton converged linearly in a packing: 9 to 30 iterations a step instead of 2 to 4): steady, and the
    part of a time step (BDF) alone."""

    for element in ELEMENTS:
        for thermal in (False, True):
            mesh = rect((1.0, 1.0), (2, 2), element)
            psc = [ff, ht] if thermal else ff
            model = Model(name='tangent', dim=2, mesh=mesh, mat=material('X', miu=0.7, rho=900.0, Cp=2000.0, k=0.2), psc=psc)
            flow = model.physics.fluid if thermal else model.physics
            # a thinning melt (the Cross-WLF of a glass-filled PP): the tangent needs d mu / d gamma_dot of -2/3 mu div u
            flow.setViscosityLaw(CrossWLF(n=0.2619, tau_star=1.778e4, D1=4.378e17, D2=266.7, A1=39.22, A2=51.6, gamma_min=1e-2) if thermal
                                 else Cross(4565.0, 1.778e4, 0.2619, gamma_min=1e-2))
            flow.setDensityLaw(TaitPVT(**PP_GF, width=3.0) if thermal else exponential(0.3, 900.0), ambient=1e5,   # kappa large: the time part weighs
                               compression_heating=thermal, T=None if thermal else 230.0)
            model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
            steady = tangent_error(model)
            step = time_part_error(model)
            note(f"{element.name}, {'Tait with heating' if thermal else 'exponential, isothermal'}: steady {steady:.1e}, "
                 f"the part of the time step alone {step:.1e}")
            assert steady < 1e-7 and step < 1e-7


def test_constant_density_is_a_grad_div_term():

    """With a constant density only -2/3 mu div u div v remains: the incompressible flow with grad-div of gamma = -2/3 mu, to round-off."""

    for element in ELEMENTS:
        results = []
        for compressible in (True, False):
            mesh = rect((2.0, 1.0), (8, 4), element)
            model = Model(name='cavity', dim=2, mesh=mesh, mat=material('X', miu=0.5, rho=1.0), psc=ff)
            flow = model.physics
            flow.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or x < 1e-9 or x > 2 - 1e-9))
            flow.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), lambda x, y: (np.sin(np.pi * x / 2), 0.0))
            flow.setPressureReference((0.0, 0.0), 0.0)
            if compressible:
                flow.setDensityLaw(lambda P, T: 1.0 + 0.0 * P, T=0.0)
            else:
                flow.setGradDiv(-2.0 / 3.0 * 0.5)
            model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear if compressible else ProblemType.linear)
            model.solve()
            results.append((np.asarray(model.sol['u']), np.asarray(model.sol['p'])))
        du, dp = np.max(np.abs(results[0][0] - results[1][0])), np.max(np.abs(results[0][1] - results[1][1]))
        note(f"{element.name}: |u - u_graddiv| {du:.1e}, |p - p_graddiv| {dp:.1e}")
        assert du < 1e-10 and dp < 1e-9


def channel(element, nx, ny, L=20.0, H=1.0, kappa=1e-3, rho0=1.0, mu=1.0):
    mesh = rect((L, H), (nx, ny), element)
    model = Model(name='channel', dim=2, mesh=mesh, mat=material('X', miu=mu, rho=rho0), psc=ff)
    flow = model.physics
    flow.setDensityLaw(exponential(kappa, rho0), T=0.0)
    flow.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or y > H - 1e-9))
    return model, mesh, flow


def test_steady_channel_against_lubrication():

    """Steady flow of a fluid of constant compressibility, rho = rho0 exp(kappa p), in a channel of L/H = 20 with the density changing by
    27 %: the mass flow is constant along it and rho dp/dx = -12 mu m / H^3 gives p(x) = ln(1 + 12 mu m kappa (L - x) / (rho0 H^3)) / kappa
    (lubrication, exact as H / L -> 0), m = rho(p_in) Q. The centreline pressure against it, and the mass flow in against out."""

    from scipy.optimize import brentq
    L, H, kappa, rho0, mu, Q = 20.0, 1.0, 1e-3, 1.0, 1.0, 1.0
    p_in = brentq(lambda q: q - np.log1p(12 * mu * rho0 * np.exp(kappa * q) * Q * kappa * L / (rho0 * H ** 3)) / kappa, 0.0, 1e4)
    m = rho0 * np.exp(kappa * p_in) * Q
    exact = lambda x: np.log1p(12 * mu * m * kappa * (L - x) / (rho0 * H ** 3)) / kappa
    for element in ELEMENTS:
        model, mesh, flow = channel(element, 40, 4)
        left, right = mesh.boundary_edges(lambda x, y: x < 1e-9), mesh.boundary_edges(lambda x, y: x > L - 1e-9)
        flow.addBC_VelocityEdges(left, lambda x, y: (6 * Q * y * (H - y) / H ** 3, 0.0))
        flow.addBC_OutletEdges(right, 0.0)
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
        model.solve()
        xy = xy_of(mesh)
        centre = np.abs(xy[:, 1] - 0.5 * H) < 1e-9
        inner = centre & (xy[:, 0] > 2) & (xy[:, 0] < L - 2)
        p = np.asarray(model.sol['p'])
        error = np.max(np.abs(p[inner] - exact(xy[inner, 0]))) / p_in
        u = np.asarray(model.sol['u'])
        rows = [np.flatnonzero(np.abs(xy[:, 0] - x0) < 1e-9) for x0 in (0.0, L)]
        flux = []
        for r in rows:
            r = r[np.argsort(xy[r, 1])]
            y, f = xy[r, 1], rho0 * np.exp(kappa * p[r]) * u[r, 0]
            h = y[2::2] - y[:-2:2]
            flux.append(np.sum(h / 6 * (f[:-2:2] + 4 * f[1::2] + f[2::2])))
        note(f"{element.name}: p_in {p[centre & (xy[:, 0] < 1e-9)].mean():.3f} (lubrication {p_in:.3f}), largest error of the centreline "
             f"pressure {error:.1e} of p_in; mass flow in {flux[0]:.6f}, out {flux[1]:.6f} (m {m:.6f}); {model.newton_iterations} Newton iterations")
        assert model.nonlinear_converged and error < 2e-3 and abs(flux[1] / flux[0] - 1) < 1e-3


def packing(element, heating=False, nx=40, ny=2, dt=0.005, total=0.5):

    """Channel closed at x = L, the pressure p0 imposed at x = 0 from rest: the melt is pushed in and compressed. rho H^2 / mu = 1e-3, far below
    the time of the pressure (0.48), as in a melt (about 1e-6 s): the momentum is quasi-steady. With rho = 1 it would take 1, and the channel
    rings like a mass on a spring instead of diffusing."""

    L, H, kappa, rho0, mu, p0 = 20.0, 1.0, 1e-4, 1e-3, 1.0, 1.0
    mesh = rect((L, H), (nx, ny), element)
    psc = [ff, ht] if heating else ff
    model = Model(name='packing', dim=2, mesh=mesh, mat=material('X', miu=mu, rho=rho0, Cp=1.0 / rho0, k=1e-12), psc=psc)
    flow = model.physics.fluid if heating else model.physics
    if heating:
        beta, T0 = 1e-4, 200.0          # beta^2 T / (kappa rho Cp) = 0.047 < 1, as thermodynamics asks (it is 1 - Cv / Cp; a melt has ~0.09)
        flow.setDensityLaw(lambda P, T: rho0 * jnp.exp(kappa * (P - 1e5) - beta * (T - T0)), compression_heating=True)
        model.physics.heat.setVelocity('u')
        model.physics.heat.var['T'].updateField(np.full(mesh.getNoN(), T0))
        model.physics.heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: x < 1e-9), T0)
    else:
        flow.setDensityLaw(exponential(kappa, rho0), T=0.0)
    flow.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or y > H - 1e-9 or x > L - 1e-9))
    left = mesh.boundary_edges(lambda x, y: x < 1e-9)
    flow.addBC_OutletEdges(left, p0)
    model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, timeStep=dt, totalTime=total, timeScheme='BDF2')
    model.solve()
    return model, mesh, (L, H, kappa, rho0, mu, p0)


def series(x, t, L, D, terms=200):
    """p / p0 of dp/dt = D d2p/dx2, p(0) = p0, dp/dx(L) = 0, p(x, 0) = 0."""
    k = (2 * np.arange(terms) + 1) * np.pi / (2 * L)
    return 1.0 - np.sum(4.0 / ((2 * np.arange(terms) + 1) * np.pi) * np.sin(np.outer(x, k)) * np.exp(-D * k ** 2 * t), axis=1)


def test_packing_against_the_pressure_diffusion():

    """A closed channel pressurized at its inlet: in the lubrication limit the pressure diffuses, dp/dt = D d2p/dx2 with
    D = H^2 / (12 mu kappa) (time scale L^2 / D = 0.48 here), against its series; and the mass in the channel against what came in."""

    for element in ELEMENTS:
        model, mesh, (L, H, kappa, rho0, mu, p0) = packing(element)
        D = H ** 2 / (12 * mu * kappa)
        xy = xy_of(mesh)
        centre = np.flatnonzero(np.abs(xy[:, 1] - 0.5 * H) < 1e-9)
        centre = centre[np.argsort(xy[centre, 0])]
        times = np.asarray(model.timeVector)
        worst = 0.0
        for t_check in (0.05, 0.1, 0.25, 0.5):
            k = int(np.argmin(np.abs(times - t_check)))
            p = np.asarray(model.sol['p'])[:, k]                          # the history: time along the last axis
            err = np.max(np.abs(p[centre] / p0 - series(xy[centre, 0], times[k], L, D)))
            worst = max(worst, err)
            note(f"{element.name}, t = {times[k]:.3f} (t D / L^2 = {times[k] * D / L ** 2:.3f}): largest error of p / p0 {err:.1e}, "
                 f"p(L) / p0 {p[centre[-1]] / p0:.4f} (series {series(xy[centre[-1:], 0], times[k], L, D)[0]:.4f})")
        assert worst < 0.02


def test_compression_heating():

    """The same packing with heat and no conduction: rho Cp DT/Dt = beta T Dp/Dt, so T - T0 = beta T0 p / (rho Cp) to first order where
    the melt moves little."""

    for element in ELEMENTS:
        model, mesh, (L, H, kappa, rho0, mu, p0) = packing(element, heating=True, total=0.3)
        T, p = np.asarray(model.sol['T'])[:, -1], np.asarray(model.sol['p'])[:, -1]
        xy = xy_of(mesh)
        far = xy[:, 0] > 0.5 * L
        expected = 1e-4 * (200.0 + 273.15) * p / 1.0                       # rho Cp = 1
        err = np.max(np.abs((T - 200.0)[far] - expected[far])) / np.max(np.abs(expected[far]))
        note(f"{element.name}: T - T0 at the closed end {T[far].max() - 200.0:.4f} K, beta T0 p / rho Cp {expected[far].max():.4f} K, "
             f"largest relative difference over the far half {err:.1e}")
        assert err < 0.02


if __name__ == "__main__":
    tests = [test_tait_reduces_to_its_branches, test_tait_fit_recovers_its_parameters, test_tangent, test_constant_density_is_a_grad_div_term,
             test_steady_channel_against_lubrication, test_packing_against_the_pressure_diffusion, test_compression_heating]
    failed = 0
    for t in tests:
        note(f"start {t.__name__}")
        try:
            t()
            print(f"PASS  {t.__name__}", flush=True)
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__} {e}", flush=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
