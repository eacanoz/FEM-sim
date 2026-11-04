"""Non-isothermal viscoelastic flow: relaxation times shifted with the temperature (time-temperature superposition) and the
heating by the polymer stress. Each test prints its progress (every nonlinear iteration and the time) to follow it while it runs.

Run from the project root:  python -u Test/test_viscoelastic_thermal.py
"""

import logging
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_bvp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.HeatTransfer import ht
from Source.Physics.Viscoelastic import ve
from Source.Physics.Rheology import wlf_shift, arrhenius_shift
from Source.enums import ElementType, StudyType, ProblemType
from Source.logger_config import sim_logger

ELEMENTS = (ElementType.quadrilateral, ElementType.triangle)
START = time.perf_counter()


class Progress(logging.Handler):

    """Prints every iteration of the nonlinear solvers with the elapsed time, so that a stall shows while the test runs."""

    tag = ''

    def emit(self, record):
        message = record.getMessage()
        if (message.startswith('|') and ('||R||' in message or 'rejected' in message or message[1:8].strip().isdigit())) \
                or 'converged' in message or 'did not' in message or 'stopped' in message:
            print(f"      [{time.perf_counter() - START:7.1f} s] {self.tag} {message.strip()[:90]}", flush=True)


PROGRESS = Progress()
for handler in list(sim_logger.handlers):
    sim_logger.removeHandler(handler)
sim_logger.addHandler(PROGRESS)
sim_logger.setLevel(logging.INFO)


def note(text):
    print(f"   [{time.perf_counter() - START:7.1f} s] {text}", flush=True)


def mesh2(size, div, element=ElementType.quadrilateral):
    mesh = Mesh()
    mesh.generate2DMesh(size, div, element, order=2)
    return mesh


def xy_of(mesh):
    return np.array([n.coor[:2] for n in mesh.NL], dtype=float)


def tangent_error(model):
    system = model._system()
    layout = system.layout
    rng = np.random.default_rng(3)
    x = rng.normal(scale=0.2, size=layout.ndof)
    for name in layout.names:
        if name.startswith('b'):
            d = layout.dofs(name)
            x[d[:, 0]] += 1.5
            x[d[:, 2]] += 1.5
            if d.shape[1] == 4:
                x[d[:, 3]] += 1.5
    x[layout.dofs('T').ravel()] += 5.0
    for term in system._terms:
        if hasattr(term, 'tau_state'):
            term.tau_state = x.copy()
    J, _ = system.residual_and_tangent(x)
    FD = np.zeros((layout.ndof, layout.ndof))
    for k in range(layout.ndof):
        e = np.zeros(layout.ndof)
        e[k] = 1e-6
        FD[:, k] = (system.residual(x + e) - system.residual(x - e)) / 2e-6
    return np.max(np.abs(J.toarray() - FD)) / np.max(np.abs(FD))


def test_tangent():

    """d/dT of the relaxation, d/du and d/db of the heating, against finite differences: plane and axisymmetric, with and without DEVSS."""

    for axisymmetric in (False, True):
        for element in ELEMENTS:
            for devss in (0.0, 0.5):
                mesh = mesh2((1.0, 1.0), (2, 2), element)
                if axisymmetric:
                    for node in mesh.NL:
                        node.coor = (node.coor[0] + 0.5, node.coor[1])
                model = Model(name='tangent', dim=2, mesh=mesh, mat=material('X', miu=0.5, rho=1.0, Cp=1.0, k=0.3), psc=[ff, ht, ve],
                              axisymmetric=axisymmetric)
                polymer = model.physics.viscoelastic
                polymer.setModes([(1.0, 0.7, 0.2)])
                if devss:
                    polymer.setDEVSS(devss)
                polymer.setTemperatureShift(wlf_shift(3.0, 60.0, 5.0))
                polymer.setViscousHeating(fraction=0.8)
                model.physics.heat.setVelocity('u')
                model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
                err = tangent_error(model)
                note(f"tangent, {'axisymmetric' if axisymmetric else 'plane'} {element.name}, DEVSS {devss}: {err:.1e}")
                assert err < 1e-7, (axisymmetric, element, devss, err)


def channel(element, T_wall, shift=None, mode=(0.5, 0.5), inlet_T=None, n=8, eta=None):

    """Pressure-driven channel with the developed stress at the inlet; with heat (all walls at T_wall) if shift is given."""

    mu_p, lam = mode
    mesh = mesh2((1.0, 1.0), (n // 2, n), element)
    psc = [ff, ht, ve] if shift is not None else [ff, ve]
    model = Model(name='channel', dim=2, mesh=mesh, mat=material('X', miu=0.5, rho=1.0, Cp=1.0, k=1.0), psc=psc)
    flow, polymer = model.physics.fluid, model.physics.viscoelastic
    polymer.setModes([(mu_p, lam, 0.0)])
    G = 2.0
    flow.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or y > 1 - 1e-9))
    left, right = mesh.boundary_edges(lambda x, y: x < 1e-9), mesh.boundary_edges(lambda x, y: x > 1 - 1e-9)
    flow.addBC_OutletEdges(left, G)
    flow.addBC_OutletEdges(right, 0.0)
    polymer.addBC_OutletEdges(left)
    polymer.addBC_OutletEdges(right)
    if shift is not None:
        heat = model.physics.heat
        heat.addBC_TemperatureEdges(mesh.boundary_edges(None), T_wall)
        heat.var['T'].updateField(np.full(mesh.getNoN(), T_wall))
        polymer.setTemperatureShift(shift)
    eta = 0.5 + mu_p if eta is None else eta                      # the shear viscosity that sets the gradient at the inlet
    polymer.addBC_InflowEdges(left, lambda x, y: [[0.0, G / eta * (0.5 - y)], [0.0, 0.0]], inlet_T)
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
    PROGRESS.tag = f"channel {element.name}"
    model.solve()
    return model


def test_uniform_temperature_is_the_shifted_isothermal_flow():

    """T = T_ref everywhere gives the isothermal flow; T = T1 gives the isothermal flow with lam a_T(T1) and mu_p a_T(T1): exactly,
    the same discrete equations."""

    shift = arrhenius_shift(4000.0, 200.0)
    for element in ELEMENTS:
        for T1 in (200.0, 230.0):
            a = float(shift(jnp.asarray(T1)))
            thermal = channel(element, T1, shift, eta=0.5 + 0.5 * a)
            iso = channel(element, None, None, (0.5 * a, 0.5 * a), eta=0.5 + 0.5 * a)
            du = np.max(np.abs(thermal.sol['u'] - iso.sol['u']))
            db = np.max(np.abs(thermal.sol['b0'] - iso.sol['b0']))
            note(f"{element.name}, T = {T1} C (a_T = {a:.4f}): |u - u_iso| {du:.1e}, |b - b_iso| {db:.1e}")
            assert thermal.nonlinear_converged and iso.nonlinear_converged
            assert du < 1e-10 and db < 1e-10


# ----- heated Couette against an independent boundary value problem -----

MU_S, MU_P, LAM, K, T_W = 0.5, 0.5, 0.5, 0.0125, 0.0
SHIFT = (2.0, 50.0, 0.0)                                         # C1, C2, T_ref of the WLF shift


def couette_reference():

    """Steady Couette between y = 0 (at rest) and y = 1 (U = 1), walls at T_W, Oldroyd-B with a shifted lam and mu_p, all the
    work into heat: tau constant, eta(T) = mu_s + mu_p a_T(T), u' = tau / eta, k T'' = -tau^2 / eta. scipy's solve_bvp, no
    code of the project."""

    C1, C2, T_ref = SHIFT
    a = lambda T: 10.0 ** (-C1 * (T - T_ref) / (C2 + T - T_ref))
    eta = lambda T: MU_S + MU_P * a(T)

    def rhs(y, z, p):
        T, q, u = z
        return np.vstack([q, -p[0] ** 2 / eta(T) / K, p[0] / eta(T)])

    def bc(za, zb, p):
        return np.array([za[0] - T_W, zb[0] - T_W, za[2], zb[2] - 1.0])

    y = np.linspace(0.0, 1.0, 201)
    z0 = np.vstack([4 * y * (1 - y), 4 * (1 - 2 * y), y])
    solution = solve_bvp(rhs, bc, y, z0, p=[1.0], tol=1e-10, max_nodes=100000)
    assert solution.success
    tau = solution.p[0]
    T = lambda yy: solution.sol(yy)[0]
    gamma = lambda yy: tau / eta(T(yy))
    return T, gamma, lambda yy: solution.sol(yy)[2], a, tau


def heated_couette(element, n, reference):

    T_ref_fn, gamma, _, a, _ = reference
    mesh = mesh2((1.0, 1.0), (2, n), element)
    model = Model(name='heated couette', dim=2, mesh=mesh, mat=material('X', miu=MU_S, rho=1.0, Cp=1.0, k=K), psc=[ff, ht, ve])
    flow, heat, polymer = model.physics.fluid, model.physics.heat, model.physics.viscoelastic
    polymer.setModes([(MU_P, LAM, 0.0)])
    polymer.setTemperatureShift(wlf_shift(*SHIFT))
    polymer.setViscousHeating()
    flow.setViscousHeating()
    heat.setVelocity('u')
    flow.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9))
    flow.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), (1.0, 0.0))
    left, right = mesh.boundary_edges(lambda x, y: x < 1e-9), mesh.boundary_edges(lambda x, y: x > 1 - 1e-9)
    for edges in (left, right):
        flow.addBC_OutletEdges(edges)
        polymer.addBC_OutletEdges(edges)
    heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: y < 1e-9 or y > 1 - 1e-9), T_W)
    # the inflow takes b of the reference: data on the boundary, the interior is what is compared
    polymer.addBC_InflowEdges(left, lambda x, y: [[0.0, float(gamma(y))], [0.0, 0.0]], lambda x, y: float(T_ref_fn(y)))
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, globalization='pseudo-transient', cfl0=1.0)
    PROGRESS.tag = f"couette {element.name} n {n}"
    model.solve()
    return model, mesh


def test_heated_couette_against_a_boundary_value_problem():

    """Temperature, velocity and polymer stress of the heated Couette flow against solve_bvp: second order (b is linear in each
    element), and a temperature rise that changes the polymer viscosity by a factor of about two."""

    reference = couette_reference()
    T_fn, gamma, u_fn, a, tau = reference
    y = np.linspace(0, 1, 101)
    note(f"reference: tau {tau:.6f}, T max {T_fn(y).max():.4f} C, a_T at the centre {a(T_fn(0.5)):.4f}")
    for element in ELEMENTS:
        errors = []
        for n in (8, 16, 32):
            model, mesh = heated_couette(element, n, reference)
            assert model.nonlinear_converged
            xy = xy_of(mesh)
            yy = xy[:, 1]
            s = model.sol['polymer_stress']
            G = MU_P / LAM
            e = (np.max(np.abs(model.sol['T'] - T_fn(yy))) / T_fn(y).max(), np.max(np.abs(model.sol['u'][:, 0] - u_fn(yy))),
                 np.max(np.abs(s[:, 1] - MU_P * a(T_fn(yy)) * gamma(yy))) / tau,
                 np.max(np.abs(s[:, 0] - 2 * G * (LAM * a(T_fn(yy))) ** 2 * gamma(yy) ** 2)) / (2 * G * LAM ** 2 * tau ** 2))
            errors.append(e)
            note(f"{element.name} n {n}: {model.newton_iterations} iterations; relative errors T {e[0]:.1e}, u {e[1]:.1e}, "
                 f"tau_xy {e[2]:.1e}, tau_xx {e[3]:.1e}")
        e = np.array(errors)
        rates = np.log2(e[-2] / e[-1])
        note(f"{element.name}: rates {np.round(rates, 2)}")
        assert e[-1].max() < 2e-3 and np.all(rates > 1.7)


if __name__ == "__main__":
    tests = [test_tangent, test_uniform_temperature_is_the_shifted_isothermal_flow, test_heated_couette_against_a_boundary_value_problem]
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
