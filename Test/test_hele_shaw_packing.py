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
    contact: the mass stays fixed; the pressure follows the Tait state V rho(p, T) = m (solved with the temperature predicted for the end
    of each step: against the temperature reached, the error falls with the step, largest at the transition) until it reaches ambient and
    then stays there (the part leaves the wall); the shrinkage is 1 - v(ambient, T_room) / v_frozen with v_frozen = V / m, the volume at
    the start."""

    m0 = None
    errors = []
    for dt_max in (0.1, 0.05):
        fill = strip(4, lambda g2, T: MU + 0 * g2)
        fill.setFilled(lambda x, y: x >= 0)
        fill.setThermal(230.0, 50.0, 970.0, 2000.0, 0.2, htc=2000.0, T_noflow=150.0, cells=1, viscous_heating=False)
        fill.setDensity(TAIT.density)
        fill.p[:] = 50e6
        states = []

        def on_step(f):
            rho = np.asarray(f.density['nodal'](f.p, f.T)[0])
            states.append((f.p.mean(), np.max(np.abs(f.volume * rho - f.mass) / f.mass), f.T[:, 0].mean(), np.mean(f.p == 0.0)))

        fill.pack(0.0, 0.0, t_cool=20.0, dt_max=dt_max, progress=False, on_step=on_step)
        m0 = fill.volume * float(TAIT.density(50e6 + 1e5, 230.0))
        states = np.array(states)
        errors.append(np.max(np.where(states[:, 3] == 0, states[:, 1], 0.0)))
        first_floor = np.argmax(states[:, 3] > 0)
        expected = 1 - (1 / float(TAIT.density(1e5, 25.0))) / (1 / float(TAIT.density(50e6 + 1e5, 230.0)))
        note(f"step {dt_max} s: mass against the start {np.max(np.abs(fill.mass / m0 - 1)):.1e}; Tait state while above ambient {errors[-1]:.1e}; "
             f"ambient reached at {states[first_floor, 2]:.1f} °C; shrinkage {fill.shrinkage().mean():.4%} (exact {expected:.4%})")
        assert np.max(np.abs(fill.mass / m0 - 1)) < 1e-12
        assert states[-1, 3] == 1.0 and np.max(np.abs(fill.shrinkage() - expected)) < 1e-9
    assert errors[1] < 5e-4 and errors[0] / errors[1] > 1.5


def test_fill_and_pack_mass():

    """Non-isothermal fill then pack (gate held, then closed) of a strip with the Tait melt: the mass gained in the packing equals the
    integral of the gate inflow to round-off (at 80 % of the fill pressure some melt flows back at first), mass still enters at the end
    of the hold, and after the gate closes the pressure falls."""

    fill = strip(30, lambda g2, T: MU * jnp.exp(-0.02 * (T - 230.0)) + 0 * g2)
    fill.setThermal(230.0, 50.0, 970.0, 2000.0, 0.2, htc=5000.0, T_noflow=150.0)
    fill.setDensity(TAIT.density)
    fill.fill(progress=False, switchover=0.98)                                    # compressible: V/P switchover at 98 %
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


def test_compressible_fill_limit():

    """A nearly incompressible melt (rho0 exp(1e-14 p)), both fillings switched over at 95 % of the volume: the compressible filling of a
    strip takes the filled volume / Q and reaches the gate pressure of the incompressible filling (stage 1); its mass grows by the gate
    inflow q rho(p_gate) dt to round-off."""

    fill0 = strip(40, lambda g2: MU + 0 * g2)
    fill0.fill(progress=False, switchover=0.95)
    fill = strip(40, lambda g2: MU + 0 * g2)
    fill.setDensity(lambda p, T: 900.0 * jnp.exp(1e-14 * (p - 1e5)))
    acc = {'m': 0.0, 't': 0.0}

    def on_step(f):
        acc['m'] += (f.time - acc['t']) * float(f._step[1].sum())
        acc['t'] = f.time

    m0 = None
    fill.mass = fill.fill_factor * fill.volume * fill.full_density()
    m0 = fill.mass.sum()
    fill.fill(progress=False, on_step=on_step, switchover=0.95)
    balance = abs(fill.mass.sum() - m0 - acc['m']) / acc['m']
    t_exact = (fill.fill_factor @ fill.volume - m0 / 900.0) / 1e-6
    p0, p1 = fill0.history[-1][2], fill.history[-1][2]
    note(f"fill time {fill.time:.6f} s (V/Q {t_exact:.6f} s), gate pressure {p1 / 1e6:.5f} MPa against {p0 / 1e6:.5f} MPa incompressible "
         f"({p1 / p0 - 1:+.1e}); mass against the gate inflow {balance:.1e}; {len(fill.history)} steps")
    assert abs(fill.time / t_exact - 1) < 1e-6 and abs(p1 / p0 - 1) < 1e-3 and balance < 1e-12


def test_compression_heating():

    """Adiabatic compression: a closed strip of a melt rho0 exp(kappa p - beta (T - T0)) with a low viscosity (uniform pressure), the gate
    raised from 0 to 100 MPa in 1 s, one cell across the gap, no viscous heating. Each particle heats by dT/T = beta dp / (rho cp); the melt
    that enters later at T_melt heats from its own entry. The mass-weighted temperature rise against an independent integration of
    batches of particles that fill the volume V = sum m_b / rho(p, T_b)."""

    kappa, beta, rho0, cp, T0 = 1e-9, 6e-4, 900.0, 2000.0, 230.0
    rc = rho0 * cp
    law = lambda p, T: rho0 * jnp.exp(kappa * (p - 1e5) - beta * (T - T0))
    fill = strip(10, lambda g2, T: 1.0 + 0 * g2)
    fill.setFilled(lambda x, y: x >= 0)
    fill.setThermal(T0, T0, rho0, cp, 0.2, htc=0.0, cells=1, viscous_heating=False)
    fill.setDensity(law)
    fill.pack(lambda s: 1e8 * min(s, 1.0), 1.0, dt_first=0.01, dt_max=0.01, growth=1.0, progress=False)
    K = 273.15
    rise = float(np.sum(fill.mass * (fill.T[:, 0] + K)) / fill.mass.sum() - (T0 + K))
    # independent: batches of particles, each heated by its own compression
    V = fill.volume.sum()
    p, steps = 0.0, 20000
    m = [V * rho0]
    T = [T0 + K]
    for k in range(steps):
        dp = 1e8 / steps
        p += dp
        T = [Ti * np.exp(beta * dp / rc) for Ti in T]
        occupied = sum(mi / float(law(p + 1e5, Ti - K)) for mi, Ti in zip(m, T))
        m.append((V - occupied) * float(law(p + 1e5, T0)))
        T.append(T0 + K)
        if k % 50 == 0 and len(m) > 2:                                              # merge the old batches (same law: exact up to beta^2)
            mm = sum(m[:-1]); TT = sum(a * b for a, b in zip(m[:-1], T[:-1])) / mm
            m, T = [mm, m[-1]], [TT, T[-1]]
    ref = sum(a * b for a, b in zip(m, T)) / sum(m) - (T0 + K)
    note(f"mass-weighted rise {rise:.4f} K against {ref:.4f} K ({rise / ref - 1:+.1e}); the first melt alone {(T0 + K) * (np.exp(beta * 1e8 / rc) - 1):.3f} K; "
         f"mass {fill.mass.sum() / (V * rho0) - 1:.2%} more")
    assert abs(rise / ref - 1) < 0.02


if __name__ == "__main__":
    tests = [test_pressure_diffusion, test_sealed_cooling, test_fill_and_pack_mass, test_compressible_fill_limit, test_compression_heating]
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
