"""Hele-Shaw filling with temperature across the gap (HeleShawFilling.setThermal): the fluidity kernel with temperature and frozen layer,
conduction of a still slab against its series solution, the Graetz problem against an independent marching solution, the energy balance
with adiabatic walls, and the isothermal limit. Each test prints its progress.

Run from the project root:  python -u Test/test_hele_shaw_thermal.py
"""

import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.Physics.HeleShaw import GapFluidity, HeleShawFilling, _thermal_kernel
from Source.Physics.Rheology import Cross
from Source.enums import ElementType

START = time.perf_counter()
L, W, H, Q = 0.1, 0.02, 2e-3, 1e-6                     # strip 100 x 20 mm, gap 2 mm, mean velocity 25 mm/s
CROSS = Cross(4565.0, 1.778e4, 0.2619)


def note(text):
    print(f"   [{time.perf_counter() - START:6.1f} s] {text}", flush=True)


def newtonian(g2, T):
    return 1000.0 + 0 * g2


def strip(nx, law, **thermal):
    mesh = Mesh()
    mesh.generate2DMesh((L, W), (nx, 2), ElementType.quadrilateral, order=1)
    fill = HeleShawFilling(mesh, H, law, T=230.0)
    if thermal:
        fill.setThermal(**thermal)
    fill.addGate(mesh.boundary_edges(lambda x, y: x < 1e-12), Q)
    return fill


def test_fluidity_kernel():

    """With a uniform temperature the cell-by-cell fluidity matches GapFluidity (Cross, 1e-6) and its derivative finite differences;
    the cell flows add up to q / 2 and the cell heating to S G^2 / 2 (to round-off); a Newtonian gap frozen beyond z_f has S = 2 z_f^3 / (3 mu)."""

    kernel = _thermal_kernel(lambda g2, T: CROSS(g2, T), np.linspace(0, 1, 21), None)
    b, G, T = np.full(4, 1e-3), np.array([1e4, 1e6, 1e7, 1e8]), np.full((4, 20), 230.0)
    S, dS, flow, heat, _ = (np.asarray(a) for a in kernel(b, G, T, np.zeros((4, 20, 2))))
    err = np.max(np.abs(S / GapFluidity(CROSS, T=230.0, layers=32)(b, G) - 1))
    fd = (np.asarray(kernel(b, G * (1 + 1e-6), T, np.zeros((4, 20, 2)))[0]) - S) / (G * 1e-6)
    err_d = np.max(np.abs(dS / fd - 1))
    sums = max(np.max(np.abs(flow.sum(1) / (S * G / 2) - 1)), np.max(np.abs((heat * 1e-3 / 20).sum(1) / (S * G ** 2 / 2) - 1)))
    frozen = _thermal_kernel(newtonian, np.linspace(0, 1, 21), 100.0)
    Sf = float(np.asarray(frozen(np.array([1e-3]), np.array([1e6]), np.where(np.arange(20) < 12, 200.0, 50.0)[None, :], np.zeros((1, 20, 2)))[0])[0])
    err_f = abs(Sf / (2 * 0.6e-3 ** 3 / 3000.0) - 1)
    note(f"fluidity against GapFluidity {err:.1e}, derivative against differences {err_d:.1e}, cell sums {sums:.1e}, frozen layer {err_f:.1e}")
    assert err < 1e-6 and err_d < 1e-5 and sums < 1e-12 and err_f < 1e-12


def test_slab_cooling():

    """A still gap (no gate): each node cools as a slab, cell averages against the series solution, wall at T_mold and through a heat
    transfer coefficient (Bi = 2), implicit Euler at first order in time."""

    b, alpha, rhocp = 1e-3, 1e-7, 1.8e6
    for htc in (None, 2 * 0.18 / b):
        errors = []
        for steps in (100, 200):
            mesh = Mesh()
            mesh.generate2DMesh((0.01, 0.01), (2, 2), ElementType.quadrilateral, order=1)
            fill = HeleShawFilling(mesh, 2 * b, newtonian)
            fill.setThermal(230.0, 50.0, rhocp, 1.0, alpha * rhocp, htc=htc, cells=40, viscous_heating=False)
            fill.fill_factor[:] = 1.0
            fill.solve_pressure()
            t_end = 0.2 * b ** 2 / alpha
            for _ in range(steps):
                fill._advance_temperature(t_end / steps, fill.volume, fill.volume)
            Bi = np.inf if htc is None else htc * b / (alpha * rhocp)
            roots = [(n + 0.5) * np.pi if htc is None else brentq(lambda m: m * np.tan(m) - Bi, n * np.pi + 1e-9, (n + 0.5) * np.pi - 1e-9)
                     for n in range(200)]
            e = fill.thermal['edges']
            theta = sum(4 * np.sin(m) / (2 * m + np.sin(2 * m)) * (np.sin(m * e[1:]) - np.sin(m * e[:-1])) / (m * np.diff(e))
                        * np.exp(-m ** 2 * 0.2) for m in roots)
            errors.append(np.max(np.abs(fill.T[0] - (50.0 + 180.0 * theta))))
        note(f"{'wall at T_mold' if htc is None else 'Bi = 2'}: max error {errors[0]:.3f} K, {errors[1]:.3f} K (ratio {errors[0] / errors[1]:.2f})")
        assert errors[1] < 0.6 and 1.6 < errors[0] / errors[1] < 2.4


def test_graetz():

    """Steady cooling of the flowing melt behind the front (cold wall at T_mold, Newtonian, no heating) against a fine marching solution of
    u(z) dT/dx = alpha d2T/dz2 (scipy, independent of the module), at 0.5, 1 and 2 Graetz lengths: first order in the control volume."""

    b, U, rhocp, k = H / 2, Q / (W * H), 1.8e6, 3.6
    alpha = k / rhocp
    LG = U * b ** 2 / alpha
    nf, nc = 800, 20
    zc = (np.arange(nf) + 0.5) / nf * b
    u = 1.5 * U * (1 - (zc / b) ** 2)

    def rhs(x, T):
        Tg = np.concatenate([T, [2 * 50.0 - T[-1]]])
        lap = np.empty_like(T)
        lap[0] = (Tg[1] - Tg[0]) / (b / nf) ** 2
        lap[1:] = (Tg[2:] - 2 * Tg[1:-1] + Tg[:-2]) / (b / nf) ** 2
        return alpha * lap / u

    xs = np.array([0.5, 1.0, 2.0]) * LG
    ref = solve_ivp(rhs, (0, xs[-1]), np.full(nf, 230.0), method='BDF', t_eval=xs, rtol=1e-9, atol=1e-9).y.T.reshape(3, nc, -1).mean(axis=2)
    errors = []
    for nx in (80, 160):
        fill = strip(nx, newtonian, T_melt=230.0, T_mold=50.0, rho=rhocp, cp=1.0, k=k, cells=nc, viscous_heating=False)
        fill.fill(progress=False)
        nodes = [np.argmin(np.hypot(fill.xy[:, 0] - x, fill.xy[:, 1])) for x in xs]
        errors.append(np.array([abs((fill.T[i] - r) @ fill.thermal['dz']) for i, r in zip(nodes, ref)]))
        note(f"{nx} elements: error of the mean temperature at 0.5, 1, 2 L_G: " + ", ".join(f"{e:.2f} K" for e in errors[-1])
             + f" (drop {230 - ref[-1].mean():.0f} K)")
    assert np.all(errors[1] < 0.015 * 180) and np.all(errors[0] / errors[1] > 1.4)


def test_energy_adiabatic():

    """Adiabatic walls with viscous heating (Cross): the heat in the melt equals the work of the gate, integral of p . F dt, to round-off;
    the convection with the vertical exchange conserves energy."""

    fill = strip(40, CROSS, T_melt=230.0, T_mold=50.0, rho=900.0, cp=2000.0, k=0.2, htc=0.0)
    acc = {'W': 0.0, 't': 0.0}

    def on_step(f):
        acc['W'] += (f.time - acc['t']) * float(f.p @ f._step[1])
        acc['t'] = f.time

    fill.fill(progress=False, on_step=on_step)
    E = 900 * 2000 * np.sum((fill.fill_factor * fill.volume)[:, None] * (fill.T - 230.0) * fill.thermal['dz'])
    note(f"work {acc['W']:.6f} J, heat {E:.6f} J ({E / acc['W'] - 1:+.1e}); hottest cell {fill.T.max():.2f} °C")
    assert abs(E / acc['W'] - 1) < 1e-10 and fill.T.max() > 232.0


def test_isothermal_limit():

    """Mold at the melt temperature and no heating: the gate pressure of the isothermal filling (stage 1, its table), to 1e-4."""

    p0 = strip(40, CROSS).fill(progress=False)[-1][2]
    fill = strip(40, CROSS, T_melt=230.0, T_mold=230.0, rho=900.0, cp=2000.0, k=0.2, viscous_heating=False)
    p1 = fill.fill(progress=False)[-1][2]
    note(f"gate pressure {p0 / 1e6:.5f} MPa isothermal, {p1 / 1e6:.5f} MPa with temperature ({p1 / p0 - 1:+.1e})")
    assert abs(p1 / p0 - 1) < 1e-4 and np.ptp(fill.T) < 1e-9


def test_short_shot():

    """A slow fill into a cold mold (2.5 mm/s, the Graetz length 25 mm): the frozen layer is thin at the gate (hot melt keeps coming) and
    at the front (just filled), thickest between them, where it closes the channel; the filling stops with short_shot set (no exception),
    the melt volume still Q t to round-off."""

    mesh = Mesh()
    mesh.generate2DMesh((L, W), (40, 2), ElementType.quadrilateral, order=1)
    fill = HeleShawFilling(mesh, H, newtonian)
    fill.setThermal(230.0, 50.0, 900.0, 2000.0, 0.18, T_noflow=150.0, viscous_heating=False)
    fill.addGate(mesh.boundary_edges(lambda x, y: x < 1e-12), Q / 10)
    history = np.array(fill.fill(progress=False))
    t, volume = history[:, 0], history[:, 1]
    growth = np.max(np.abs(volume - volume[0] - Q / 10 * t)) / (Q / 10 * t[-1])
    frozen = fill.frozen_fraction()
    x = np.unique(np.round(fill.xy[fill.fill_factor > 0, 0], 9))
    column = np.array([frozen[np.abs(fill.xy[:, 0] - xi) < 1e-9].mean() for xi in x])
    note(f"short shot at {fill.short_shot:.2f} s with {fill.melt_volume() / (L * W * H):.1%} of the strip; mass against Q t {growth:.1e}; "
         f"frozen fraction from the gate to the front: " + " ".join(f"{c:.2f}" for c in column))
    assert fill.short_shot is not None and fill.melt_volume() < 0.5 * L * W * H and growth < 1e-12
    assert column.max() == 1.0 and column[0] < 0.7 and column[-1] < 1.0


if __name__ == "__main__":
    tests = [test_fluidity_kernel, test_slab_cooling, test_graetz, test_energy_adiabatic, test_isothermal_limit, test_short_shot]
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
