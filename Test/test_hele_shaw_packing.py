"""Packing in the Hele-Shaw filling (HeleShawFilling.setDensity, pack): the pressure diffusion of a compressible melt into a closed
strip against its series solution, a sealed cavity that cools at constant mass (the Tait state, the ambient floor and the shrinkage
exactly), and the mass balance of a non-isothermal fill and pack. Each test prints its progress.

Run from the project root:  python -u Test/test_hele_shaw_packing.py
"""

import os
import sys
import time

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.Physics.HeleShaw import HeleShawFilling
from Source.Physics.Rheology import TaitPVT
from Source.enums import ElementType

START = time.perf_counter()
L, W, H = 0.1, 0.02, 2e-3
MU = 1000.0
TAIT = TaitPVT(1.006e-3, 4.936e-7, 7.989e7, 3.53e-3, 9.073e-4, 1.49e-7, 2.601e8, 2.353e-3, 453.9, 1.709e-7, 1.009e-4, 4.788e-2,
               1.553e-8)                                            # POLYFORT FPP 30 GFC K1079, fitted to its pvT data


def note(text):
    print(f"   [{time.perf_counter() - START:6.1f} s] {text}", flush=True)


def strip(nx, law, ny=2):
    mesh = Mesh()
    mesh.generate2DMesh((L, W), (nx, ny), ElementType.quadrilateral, order=1)
    fill = HeleShawFilling(mesh, H, law)
    fill.addGate(mesh.boundary_edges(lambda x, y: x < 1e-12), 1e-6)
    return fill


def test_pressure_diffusion():

    """A closed strip full of a melt of density rho0 exp(kappa p) packed at p_g from the gate: rho obeys h d(rho)/dt = (S / kappa) d2(rho)/dx2
    exactly (linear diffusion, D = S / (h kappa)), compared with its series at t = 0.3 L^2 / D; the error falls at first order with the step
    and the mesh together; the mass gained equals the integral of the gate inflow to round-off."""

    kappa, rho0, pg = 1e-9, 900.0, 50e6
    S = H ** 3 / (12 * MU)
    D = S / (H * kappa)
    t_end = 0.3 * L ** 2 / D
    errors = []
    for nx, steps in ((20, 40), (40, 80)):
        fill = strip(nx, lambda g2: MU + 0 * g2)
        fill.setFilled(lambda x, y: x >= 0)
        fill.setDensity(lambda p, T: rho0 * jnp.exp(kappa * (p - 1e5)))
        history = np.array(fill.pack(pg, t_end, dt_first=t_end / steps, dt_max=t_end / steps, growth=1.0, progress=False))
        dt = np.diff(np.concatenate([[0.0], history[:, 0]]))
        balance = abs(history[-1, 1] - fill.volume.sum() * rho0 - np.sum(history[:, 3] * dt)) / (history[-1, 1] - fill.volume.sum() * rho0)
        x = fill.xy[:, 0]
        rg = rho0 * np.exp(kappa * pg)
        lam = (2 * np.arange(200) + 1) * np.pi / (2 * L)
        exact = rg + (rho0 - rg) * np.sum(4 / (lam * 2 * L) * np.sin(lam * x[:, None]) * np.exp(-lam ** 2 * D * t_end), axis=1)
        rho, _ = fill.nodal_density()
        errors.append(np.max(np.abs(rho - exact)) / (rg - rho0))
        note(f"{nx} elements, {steps} steps: density against the series {errors[-1]:.2e} of the rise; mass against the gate inflow {balance:.1e}; "
             f"pressure at the closed end {fill.p[np.argmax(x)] / 1e6:.2f} MPa")
        assert balance < 1e-10
    assert errors[1] < 0.01 and 1.6 < errors[0] / errors[1] < 2.4


def test_sealed_cooling():

    """A sealed cavity (gate closed from the start, uniform state, one cell across the gap) cooling from 230 °C and 50 MPa through the mold
    contact: the mass stays fixed, the pressure follows the Tait state V rho(p, T) = m until it reaches ambient and then stays there (the
    part leaves the wall), and the shrinkage is 1 - v(ambient, T_room) / v_frozen with v_frozen = V / m, the volume at the start."""

    fill = strip(4, lambda g2, T: MU + 0 * g2)
    fill.setFilled(lambda x, y: x >= 0)
    fill.setThermal(230.0, 50.0, 970.0, 2000.0, 0.2, htc=2000.0, T_noflow=150.0, cells=1, viscous_heating=False)
    fill.setDensity(TAIT.density)
    fill.p[:] = 50e6
    m0 = None
    states, previous = [], [fill.T.copy()]

    def on_step(f):
        # the pressure of a step is solved with the temperature at its start (the temperature follows, with the fluxes of the step)
        rho = np.asarray(f.density['nodal'](f.p, previous[0])[0])
        states.append((f.p.mean(), np.max(np.abs(f.volume * rho - f.mass) / f.mass), f.T[:, 0].mean(), np.mean(f.p == 0.0)))
        previous[0] = f.T.copy()

    fill.pack(0.0, 0.0, t_cool=20.0, dt_max=0.2, progress=False, on_step=on_step)
    m0 = fill.volume * float(TAIT.density(50e6 + 1e5, 230.0))
    states = np.array(states)
    state_err = np.max(np.where(states[:, 3] == 0, states[:, 1], 0.0))
    first_floor = np.argmax(states[:, 3] > 0)
    expected = 1 - (1 / float(TAIT.density(1e5, 25.0))) / (1 / float(TAIT.density(50e6 + 1e5, 230.0)))
    note(f"mass against the start {np.max(np.abs(fill.mass / m0 - 1)):.1e}; Tait state while above ambient {state_err:.1e}; ambient reached at "
         f"{states[first_floor, 2]:.1f} °C; shrinkage {fill.shrinkage().mean():.4%} (exact {expected:.4%})")
    assert np.max(np.abs(fill.mass / m0 - 1)) < 1e-12 and state_err < 1e-9
    assert states[-1, 3] == 1.0 and np.max(np.abs(fill.shrinkage() - expected)) < 1e-9


def test_fill_and_pack_mass():

    """Non-isothermal fill then pack (gate held, then closed) of a strip with the Tait melt: the mass gained in the packing equals the
    integral of the gate inflow to round-off (at 80 % of the fill pressure some melt flows back at first), mass still enters at the end
    of the hold, and after the gate closes the pressure falls."""

    fill = strip(30, lambda g2, T: MU * jnp.exp(-0.02 * (T - 230.0)) + 0 * g2)
    fill.setThermal(230.0, 50.0, 970.0, 2000.0, 0.2, htc=5000.0, T_noflow=150.0)
    fill.setDensity(TAIT.density)
    fill.fill(progress=False)
    p_fill = fill.history[-1][2]
    history = np.array(fill.pack(0.8 * p_fill, 1.0, t_cool=1.0, progress=False))
    dt = np.diff(np.concatenate([[fill._switchover], history[:, 0]]))
    m_switch = history[0, 1] - history[0, 3] * dt[0]
    balance = abs(history[-1, 1] - m_switch - np.sum(history[:, 3] * dt)) / (history[-1, 1] - m_switch)
    held = history[:, 0] <= fill._switchover + 1.0 + 1e-12
    note(f"fill pressure {p_fill / 1e6:.2f} MPa; mass gained in the packing {(history[-1, 1] / m_switch - 1):.2%}, against the gate inflow "
         f"{balance:.1e}; mean pressure at the end of the hold {np.mean(history[held][-1, 2]) / 1e6:.1f} MPa, at the end {fill.p.mean() / 1e6:.2f} MPa; "
         f"shrinkage {fill.shrinkage().min():.2%} to {fill.shrinkage().max():.2%}")
    assert balance < 1e-10 and history[held][-1, 3] > 0 and history[-1, 1] > m_switch
    assert fill.p.mean() < 0.5 * history[held][-1, 2]


if __name__ == "__main__":
    tests = [test_pressure_diffusion, test_sealed_cooling, test_fill_and_pack_mass]
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
