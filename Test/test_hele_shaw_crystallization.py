"""Crystallization in the Hele-Shaw filling (Source/Physics/Crystallization.py, HeleShawFilling.setCrystallization): the Nakamura /
Hoffman-Lauritzen rate, isothermal Avrami, adiabatic crystallization with its latent heat against energy conservation and an independent
integration, the no-flow by crystallinity, the pvT that follows X, and a fill and pack with crystallization. Each test prints its progress.

Kinetics of an iPP (Sabic PP505P, Flash-DSC fit at 10 K/s, Ferrer Fabon et al., PAMM 22, e202200193, 2022): ln((1/t_1/2)_0) = 33.6,
Kg = 0.58e6 K^2, U* = 17200 J/mol, T_inf = 231.15 K, Tm0 = 467.15 K, n = 3.

Run from the project root:  python -u Test/test_hele_shaw_crystallization.py
"""

import os
import sys
import time

import numpy as np
import jax.numpy as jnp
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.Physics.Crystallization import NakamuraHoffman
from Source.Physics.HeleShaw import HeleShawFilling, _thermal_kernel
from Source.Physics.Rheology import TaitPVT
from Source.enums import ElementType

START = time.perf_counter()
PP = NakamuraHoffman(33.6, 0.58e6, 17200.0, 231.15, 467.15, 3.0)
TAIT = TaitPVT(1.006e-3, 4.936e-7, 7.989e7, 3.53e-3, 9.073e-4, 1.49e-7, 2.601e8, 2.353e-3, 453.9, 1.709e-7, 1.009e-4, 4.788e-2, 1.553e-8)
L, W, H = 0.1, 0.02, 2e-3
RHO, CP = 900.0, 2000.0


def note(text):
    print(f"   [{time.perf_counter() - START:6.1f} s] {text}", flush=True)


def newtonian(g2, T):
    return 1000.0 + 0 * g2


def still(T0, latent, cells=1):
    """A filled strip with no flow and adiabatic walls: every node is a closed cell that only crystallizes."""
    mesh = Mesh()
    mesh.generate2DMesh((0.01, 0.01), (2, 2), ElementType.quadrilateral, order=1)
    fill = HeleShawFilling(mesh, H, newtonian)
    fill.setThermal(T0, T0, RHO, CP, 0.2, htc=0.0, cells=cells, viscous_heating=False)
    fill.setCrystallization(PP, latent)
    fill.T[:] = T0
    fill.fill_factor[:] = 1.0
    fill.solve_pressure()
    return fill


def test_rate():

    """K(T) against the formula written out, zero above Tm0 and below T_inf; the isothermal half-time gives X = 0.5."""

    T = np.array([60.0, 100.0, 150.0])
    TK = T + 273.15
    ref = np.log(2) ** (1 / 3) * np.exp(33.6 - 17200 / (8.314462618 * (TK - 231.15)) - 0.58e6 * (467.15 + TK) / (2 * TK ** 2 * (467.15 - TK)))
    err = np.max(np.abs(PP.rate(T) / ref - 1))
    X_half = PP.crystallinity(PP.rate(100.0) * PP.half_time(100.0))
    note(f"rate against the formula {err:.1e}; half-time at 100 °C {PP.half_time(100.0):.3f} s, X there {X_half:.15f}; rate at 200 °C and "
         f"-50 °C: {PP.rate(200.0)}, {PP.rate(-50.0)}")
    assert err < 1e-13 and abs(X_half - 0.5) < 1e-14 and PP.rate(200.0) == 0.0 and PP.rate(-50.0) == 0.0


def test_isothermal_avrami():

    """No latent heat, adiabatic, T held at 100 °C: X(t) = 1 - exp(-(K t)^3) exactly, whatever the steps."""

    fill = still(100.0, 0.0)
    K, t, worst = PP.rate(100.0), 0.0, 0.0
    for dt in np.linspace(0.01, 0.1, 15):
        fill._advance_temperature(dt, fill.volume, fill.volume)
        t += dt
        worst = max(worst, np.max(np.abs(fill.crystallinity() - (1 - np.exp(-(K * t) ** 3)))))
    note(f"X after {t:.3f} s: {fill.crystallinity().mean():.6f}; against Avrami {worst:.1e}; temperature still {np.ptp(fill.T):.1e} K apart")
    assert worst < 1e-13 and np.max(np.abs(fill.T - 100.0)) < 1e-12


def test_adiabatic_latent_heat():

    """Adiabatic crystallization from 100 °C with the latent heat of 0.49 x 1.9e5 J/kg: the melt heats by L X / cp at every step (energy
    kept to round-off) and slows its own crystallization; X(t) against an independent solve_ivp of dtau/dt = K(T0 + L X(tau) / cp), the
    error falls at second order in the step (the rate at the mean temperature of the step)."""

    latent = 0.49 * 1.9e5
    t_end = 3.0
    ref = solve_ivp(lambda t, y: [PP.rate(100.0 + latent * PP.crystallinity(y[0]) / CP)], (0, t_end), [0.0], rtol=1e-11, atol=1e-13,
                    dense_output=True)
    errors = []
    for steps in (60, 120):
        fill = still(100.0, latent)
        energy, dt = 0.0, t_end / steps
        for k in range(steps):
            fill._advance_temperature(dt, fill.volume, fill.volume)
            energy = max(energy, np.max(np.abs(CP * (fill.T - 100.0) - latent * fill.crystallinity())))
        X_ref = float(PP.crystallinity(ref.sol(t_end)[0]))
        errors.append(abs(fill.crystallinity().mean() - X_ref))
        note(f"{steps} steps: X at {t_end} s {fill.crystallinity().mean():.6f} against {X_ref:.6f} ({errors[-1]:.1e}); T {fill.T.mean():.3f} °C "
             f"(T0 + L X / cp = {100 + latent * X_ref / CP:.3f}); energy kept to {energy / latent:.1e} of L")
        assert energy < 1e-9 * latent
    assert errors[1] < 2e-3 and errors[0] / errors[1] > 3.0


def test_noflow_crystallinity():

    """Cells beyond X_noflow carry no flow: a Newtonian gap with the outer 8 of 20 cells crystallized has S = 2 z_f^3 / (3 mu), z_f = 0.6 b,
    at a uniform temperature (no thermal freezing)."""

    kernel = _thermal_kernel(newtonian, np.linspace(0, 1, 21), None)
    solid = (np.arange(20) >= 12)[None, :]
    S = float(np.asarray(kernel(np.array([1e-3]), np.array([1e6]), np.full((1, 20), 200.0), np.zeros((1, 20, 2)), solid)[0])[0])
    err = abs(S / (2 * 0.6e-3 ** 3 / 3000.0) - 1)
    note(f"fluidity with a crystallized skin against the exact {err:.1e}")
    assert err < 1e-12


def test_pvt_crystallinity():

    """The pvT that follows X: X = 0 is the melt branch of the Tait law (the law itself above its transition), X = 1 its solid branch
    without vt; a sealed cell that crystallizes at constant mass and temperature (120 °C) loses pressure by the volume of crystallization."""

    v0 = float(TAIT.specific_volume_crystallinity(1e5 + 30e6, 230.0, 0.0))
    v_tait = float(TAIT.specific_volume(1e5 + 30e6, 230.0))
    b = TAIT.b
    dT = 25.0 + 273.15 - b['b5']
    v1 = float(TAIT.specific_volume_crystallinity(1e5, 25.0, 1.0))
    v1_ref = (b['b1s'] + b['b2s'] * dT) * (1 - 0.0894 * np.log1p(1e5 / (b['b3s'] * np.exp(-b['b4s'] * dT))))
    fill = still(120.0, 0.0)
    fill.setDensity(TAIT.density_crystallinity, crystallinity=True)
    fill.p[:] = 60e6
    fill.mass = fill.volume * fill.nodal_density()[0]
    fill.fill_time[:] = 0.0
    p_X = []
    for dt in [1.0] * 30:
        fill.tau += dt * PP.rate(120.0)                                             # crystallize at 120 °C, no flow, sealed
        m, inflow, it, clamped = fill._pack_step(dt, None)
        fill.mass = m
        p_X.append((fill.crystallinity().mean(), fill.p.mean()))
    p_X = np.array(p_X)
    note(f"X = 0 against the Tait law above its transition {abs(v0 / v_tait - 1):.1e}; X = 1 at 25 °C against the solid branch "
         f"{abs(v1 / v1_ref - 1):.1e}; sealed at 120 °C: p from {60:.0f} MPa to {p_X[-1, 1] / 1e6:.2f} MPa as X goes to {p_X[-1, 0]:.3f}")
    assert abs(v0 / v_tait - 1) < 1e-9 and abs(v1 / v1_ref - 1) < 1e-14
    assert np.all(np.diff(p_X[:, 1]) <= 1e-6) and p_X[-1, 1] < 30e6


def test_fill_and_pack_crystallizing():

    """A strip filled and packed with crystallization (latent heat, pvT with X, no flow beyond X = 0.5): the mass gained equals the gate
    inflow to round-off, X stays within [0, 1], the skin at the cold wall ends more crystallized than at the start of the packing, and the
    shrinkage uses the crystallinity reached."""

    mesh = Mesh()
    mesh.generate2DMesh((L, W), (30, 2), ElementType.quadrilateral, order=1)
    fill = HeleShawFilling(mesh, H, lambda g2, T: 1000.0 * jnp.exp(-0.02 * (T - 230.0)) + 0 * g2)
    fill.addGate(mesh.boundary_edges(lambda x, y: x < 1e-12), 1e-6)
    fill.setThermal(230.0, 50.0, RHO, CP, 0.2, htc=5000.0)
    fill.setCrystallization(PP, 0.49 * 1.9e5, X_noflow=0.5, pvt=TAIT)
    fill.fill(progress=False, switchover=0.98)
    X_switch = fill.crystallinity().copy()
    history = np.array(fill.pack(0.8 * fill.history[-1][2], 1.0, t_cool=2.0, progress=False))
    dt = np.diff(np.concatenate([[fill._switchover], history[:, 0]]))
    m_switch = history[0, 1] - history[0, 3] * dt[0]
    balance = abs(history[-1, 1] - m_switch - np.sum(history[:, 3] * dt)) / (history[-1, 1] - m_switch)
    X = fill.crystallinity()
    sh = fill.shrinkage()
    note(f"switchover at {fill._switchover:.3f} s, X at the wall then {X_switch[:, -1].mean():.3f}, at the end {X[:, -1].mean():.3f} (midplane "
         f"{X[:, 0].mean():.3f}); mass against the gate inflow {balance:.1e}; shrinkage {np.nanmin(sh):.2%} to {np.nanmax(sh):.2%}; frozen "
         f"{fill.frozen_fraction().mean():.0%}")
    assert balance < 1e-10 and X.min() >= 0.0 and X.max() <= 1.0 and X[:, -1].mean() > X_switch[:, -1].mean()
    assert np.all(np.isfinite(sh)) and np.all((sh > 0) & (sh < 0.2))


if __name__ == "__main__":
    tests = [test_rate, test_isothermal_avrami, test_adiabatic_latent_heat, test_noflow_crystallinity, test_pvt_crystallinity,
             test_fill_and_pack_crystallizing]
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
