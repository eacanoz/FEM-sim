"""Heat transfer coupled with species transport: Model(psc=[ht, mt]).

An exothermic reaction A -> C whose rate depends on the temperature field (Arrhenius) and releases
heat. Run from the project root:  python Test/test_coupling.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
from Source.Physics.MassTransfer import mt
from Source.enums import ElementType, ShapeFunctionType, StudyType, ProblemType, SolverType, DirectSolvers

KELVIN = 273.15


def reactor(n=20, nx_2d=None, ny=3, k=2.0, rho_cp=5.0, D=1.0, Q=40.0, k0=3.0e4, E_R=2000.0, ends=None, species=('A', 'C')):

    """Reactor A -> C. 1D of length 1 with n elements, or 2D (n x ny Q4) when nx_2d is given.

    k: conductivity, rho_cp: volumetric heat capacity, D: diffusivity of every species,
    Q = -dH: heat released per mole reacted, k0 and E_R: Arrhenius law with T in Celsius (field).
    ends = ((T, A, C) at x = 0, (T, A, C) at x = 1) sets Dirichlet conditions on both ends.
    """

    mesh = Mesh()
    if nx_2d is None:
        mesh.Generate_Mesh(1, 1.0, n, ElementType.line)
        mesh.setElementShapeFunction(ShapeFunctionType.linear)
    else:
        mesh.generate2DMesh((1.0, 0.25), (n, ny), ElementType.quadrilateral)

    model = Model(name='reactor', dim=1 if nx_2d is None else 2, mesh=mesh,
                  mat=material('X', k=k, rho=rho_cp, Cp=1.0), psc=[ht, mt])

    for s in species:
        model.physics.mass.setChemSpecies(s, s)
        model.physics.mass.setDiffusivity(s, D)

    model.physics.mass.addReaction({'A': -1, 'C': 1}, k0=k0, E_R=E_R, T='T', order={'A': 1}, dH=-Q)

    if ends is not None:
        left, right = ends
        names = ('T',) + tuple(species)
        for node in mesh.NL:
            x = node.coor[0]
            if x < 1e-12:
                values = left
            elif x > 1 - 1e-12:
                values = right
            else:
                continue
            node.BC = {name: {'type': 'Dirichlet', 'value': v} for name, v in zip(names, values)}

    return model, mesh


def configure_steady(model):
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear,
                              Method=SolverType.direct, Solver=DirectSolvers.PARDISO)


def configure_transient(model, theta, dt, total_time):
    model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear,
                              Method=SolverType.direct, Solver=DirectSolvers.PARDISO,
                              timeDisc=theta, timeStep=dt, totalTime=total_time)


ENDS = ((25.0, 1.0, 0.0), (40.0, 0.2, 0.5))   # (T, A, C) at x = 0 and x = 1


def solve_steady_1d(n=20, **kwargs):
    model, mesh = reactor(n=n, ends=ENDS, **kwargs)
    model.physics.initField('T', 25.0)
    model.physics.initField('A', 0.5)
    model.physics.initField('C', 0.2)
    configure_steady(model)
    model.solve()
    return model, np.array(mesh.getXCoor())


# ---------------------------------------------------------------------------------------------
# Independent reference: the adiabatic well-mixed reactor is an ODE system
# ---------------------------------------------------------------------------------------------

ADIABATIC = dict(k=2.0, rho_cp=5.0, Q=400.0, k0=300.0, E_R=2000.0)


def adiabatic_ode(total_time):
    p = ADIABATIC

    def rhs(t, y):
        T, A, C = y
        r = p['k0'] * np.exp(-p['E_R'] / (T + KELVIN)) * A
        return [p['Q'] * r / p['rho_cp'], -r, r]

    sol = solve_ivp(rhs, [0.0, total_time], [25.0, 1.0, 0.0], method='LSODA', rtol=1e-12, atol=1e-13)
    return sol.y[:, -1]


def adiabatic_fem(theta, dt, total_time):

    """Uniform fields and no boundary conditions (zero flux): the FEM solution must stay uniform and follow the ODE."""

    model, _ = reactor(n=4, **ADIABATIC)
    model.physics.initField('T', 25.0)
    model.physics.initField('A', 1.0)
    model.physics.initField('C', 0.0)
    configure_transient(model, theta, dt, total_time)
    model.solve()
    return model


def test_adiabatic_reactor_follows_the_ode():
    total_time = 10.0
    reference = adiabatic_ode(total_time)
    model = adiabatic_fem(theta=0.5, dt=0.01, total_time=total_time)

    for i, name in enumerate(('T', 'A', 'C')):
        values = np.asarray(model.sol[name])[:, -1]
        assert np.allclose(values, values[0], atol=1e-10), f"{name} is not uniform"
        assert abs(values[0] - reference[i]) < 5e-3 * max(1.0, abs(reference[i])), (name, values[0], reference[i])

    assert reference[0] > 60.0   # the reaction really heated the reactor (the feedback is active)


def test_adiabatic_time_convergence_orders():
    """Self-convergence against the ODE: Euler order 1, Crank-Nicolson order 2."""

    total_time = 10.0
    reference = adiabatic_ode(total_time)

    for theta, expected in ((1.0, 1.0), (0.5, 2.0)):
        errors = [abs(np.asarray(adiabatic_fem(theta, total_time / n, total_time).sol['T'])[0, -1] - reference[0])
                  for n in (100, 200, 400)]
        rates = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
        print(f"   theta={theta}: errors {['%.2e' % e for e in errors]} rates {['%.2f' % r for r in rates]}")
        assert all(r > expected - 0.2 for r in rates), (theta, rates)


def test_adiabatic_conservation_is_exact():
    """rho_cp T + Q A and A + C are constants of the reaction, and the discrete system keeps them for any theta."""

    p = ADIABATIC
    for theta in (1.0, 0.5):
        model = adiabatic_fem(theta, dt=0.25, total_time=10.0)
        T, A, C = (np.asarray(model.sol[n]) for n in ('T', 'A', 'C'))

        assert np.allclose(p['rho_cp'] * T + p['Q'] * A, p['rho_cp'] * 25.0 + p['Q'] * 1.0, atol=1e-8), theta
        assert np.allclose(A + C, 1.0, atol=1e-10), theta


# ---------------------------------------------------------------------------------------------
# Steady state
# ---------------------------------------------------------------------------------------------

def test_steady_invariants_are_linear():

    """With one diffusivity D for A and C, k T - Q D C and k T + Q D A satisfy D'' = 0 (the reaction cancels),
    so they are linear. This holds for the discrete system too: it checks the stoichiometry, the heat release
    and the coupling through the Arrhenius law, in the nonlinear solution."""

    k, D, Q = 2.0, 1.0, 40.0
    model, x = solve_steady_1d(n=20, k=k, D=D, Q=Q)
    T, A, C = (np.asarray(model.sol[n]) for n in ('T', 'A', 'C'))

    for invariant in (k * T - Q * D * C, k * T + Q * D * A):
        line = invariant[0] + (invariant[-1] - invariant[0]) * x
        assert np.max(np.abs(invariant - line)) < 1e-8

    # The coupling is active: the reactor is hotter than the linear profile between its two ends
    assert np.max(T - np.linspace(25.0, 40.0, len(T))) > 3.0


def test_steady_2d_equals_1d():
    n = 20
    model1, _ = solve_steady_1d(n=n)

    model2, mesh2 = reactor(n=n, nx_2d=True, ny=3, ends=ENDS)
    for name, value in (('T', 25.0), ('A', 0.5), ('C', 0.2)):
        model2.physics.initField(name, value)
    configure_steady(model2)
    model2.solve()

    for name in ('T', 'A', 'C'):
        field2 = np.asarray(model2.sol[name]).reshape(-1, n + 1)
        assert np.allclose(field2, np.asarray(model1.sol[name])[None, :], atol=1e-8), name


def test_strong_exotherm_converges():
    """A reaction that triples the temperature rise still reaches a converged Newton solution."""

    model, _ = solve_steady_1d(n=20, Q=120.0)
    system = model._system()
    J, R = system.residual_and_tangent(model._state())

    assert np.linalg.norm(R) < 1e-5
    assert np.asarray(model.sol['T']).max() > 60.0


def test_newton_stops_cleanly_when_the_reaction_runs_away():
    """A reaction fast and exothermic enough to have no reachable solution must not leave NaN in the fields:
    Newton stops with a warning and keeps the last finite state."""

    model, _ = solve_steady_1d(n=20, Q=300.0, k0=1.0e6)

    for name in ('T', 'A', 'C'):
        assert np.all(np.isfinite(np.asarray(model.sol[name]))), name


def finite_difference_jacobian(system, x):
    J, R = system.residual_and_tangent(x)
    eps = 1e-6
    J_fd = np.zeros(J.shape)
    for j in range(len(x)):
        xp = x.copy()
        xp[j] += eps
        J_fd[:, j] = (system.residual_and_tangent(xp)[1] - R) / eps
    return J.toarray(), J_fd


def test_jacobian_with_heat_species_cross_blocks_matches_finite_differences():
    for model, label in (reactor(n=6, ends=ENDS)[0], '1D'), (reactor(n=4, nx_2d=True, ny=2, ends=ENDS)[0], '2D'):
        configure_steady(model)
        system = model._system()

        rng = np.random.default_rng(5)
        x = np.empty(model.layout.ndof)
        for name, low, span in (('T', 25.0, 30.0), ('A', 0.2, 0.8), ('C', 0.1, 0.5)):
            x[model.layout.dofs(name)[:, 0]] = low + span * rng.random(model.mesh.getNoN())

        J, J_fd = finite_difference_jacobian(system, x)
        assert np.allclose(J, J_fd, rtol=1e-4, atol=1e-4), (label, np.max(np.abs(J - J_fd)))

        # The heat <-> species cross blocks exist (the coupling is in the tangent)
        T_dofs, A_dofs = model.layout.dofs('T')[:, 0], model.layout.dofs('A')[:, 0]
        assert np.abs(J[np.ix_(T_dofs, A_dofs)]).max() > 1e-3 and np.abs(J[np.ix_(A_dofs, T_dofs)]).max() > 1e-3


# ---------------------------------------------------------------------------------------------
# Composition API and compatibility
# ---------------------------------------------------------------------------------------------

def test_species_alone_and_inside_the_composition_agree_for_isothermal_kinetics():
    """A reaction with a numeric temperature (no field, no heat) must not feel the heat physics."""

    def species_model(psc):
        mesh = Mesh()
        mesh.Generate_Mesh(1, 1.0, 10, ElementType.line)
        mesh.setElementShapeFunction(ShapeFunctionType.linear)
        model = Model(name='s', dim=1, mesh=mesh, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=psc)
        mass = model.physics if psc is mt else model.physics.mass
        for s in ('A', 'C'):
            mass.setChemSpecies(s, s)
            mass.setDiffusivity(s, 0.7)
        mass.addReaction({'A': -1, 'C': 1}, k0=5.0, E_R=0.0, T=300.0, order={'A': 1})
        model.physics.addBC_Concentration(0, 'A', 1.0)
        model.physics.addBC_Concentration(10, 'A', 0.0)
        model.physics.addBC_Concentration(0, 'C', 0.0)
        model.physics.addBC_Concentration(10, 'C', 0.5)
        if psc is not mt:
            model.physics.addBC_Temperature(0, 20.0)
            model.physics.addBC_Temperature(10, 60.0)
        configure_steady(model)
        model.solve()
        return model

    alone, combined = species_model(mt), species_model([ht, mt])

    for name in ('A', 'C'):
        assert np.allclose(alone.sol[name], combined.sol[name], atol=1e-9), name

    # The temperature is a plain conduction profile, unaffected by the species
    assert np.allclose(combined.sol['T'], np.linspace(20.0, 60.0, 11), atol=1e-9)


def test_composition_delegation_and_errors():
    model, _ = reactor(n=4)

    assert list(model.physics.var) == ['T', 'A', 'C'] and 'Heat transfer' in model.physics_status

    # Methods of each part are available on the combined physics, and the parts by name
    model.physics.addBC_Temperature(0, 30.0)
    model.physics.addBC_Concentration(0, 'A', 1.0)
    assert model.mesh.NL[0].BC['T']['value'] == 30.0 and model.mesh.NL[0].BC['A']['value'] == 1.0
    model.physics.initField('A', 0.3)
    assert np.allclose(model.physics.mass.var['A'].values, 0.3)

    # Attributes of a part must be set on the part, not on the combination
    try:
        model.physics.Convection = True
    except AttributeError as error:
        assert "model.physics.heat.Convection" in str(error)
    else:
        raise AssertionError("expected AttributeError")
    model.physics.heat.Convection = True

    try:
        model.physics.initField('Z', 1.0)
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for an unknown variable")


def test_composition_rejects_repeated_physics_and_fields():
    mesh = Mesh()
    mesh.Generate_Mesh(1, 1.0, 4, ElementType.line)
    mesh.setElementShapeFunction(ShapeFunctionType.linear)

    try:
        Model(name='x', dim=1, mesh=mesh, mat=material('X', k=1, rho=1, Cp=1), psc=[ht, ht])
    except ValueError as error:
        assert "dos veces" in str(error)
    else:
        raise AssertionError("expected ValueError for a repeated physics")

    class other_heat(ht):
        part_name = 'heat2'

    model = Model(name='x', dim=1, mesh=mesh, mat=material('X', k=1, rho=1, Cp=1), psc=[ht, other_heat])
    try:
        model.physics.var
    except ValueError as error:
        assert "'T'" in str(error)
    else:
        raise AssertionError("expected ValueError for a repeated variable")


def test_reaction_errors():
    # A reaction on the temperature field without a heat physics
    mesh = Mesh()
    mesh.Generate_Mesh(1, 1.0, 4, ElementType.line)
    mesh.setElementShapeFunction(ShapeFunctionType.linear)
    model = Model(name='x', dim=1, mesh=mesh, mat=material('X', k=1, rho=1, Cp=1), psc=mt)
    model.physics.setChemSpecies('A', 'A')
    model.physics.setDiffusivity('A', 1.0)
    model.physics.addReaction({'A': -1}, k0=1.0, E_R=0.0, T='T', order={'A': 1})
    configure_steady(model)

    try:
        model.solve()
    except ValueError as error:
        assert "campo 'T'" in str(error)
    else:
        raise AssertionError("expected ValueError for a missing temperature field")

    # Heat of reaction with an isothermal (numeric) temperature
    model2 = Model(name='x', dim=1, mesh=mesh, mat=material('X', k=1, rho=1, Cp=1), psc=mt)
    model2.physics.setChemSpecies('A', 'A')
    model2.physics.setDiffusivity('A', 1.0)
    model2.physics.addReaction({'A': -1}, k0=1.0, E_R=0.0, T=300.0, order={'A': 1}, dH=-1.0e3)
    configure_steady(model2)

    try:
        model2.solve()
    except ValueError as error:
        assert "dH" in str(error)
    else:
        raise AssertionError("expected ValueError for dH without a temperature field")


if __name__ == "__main__":
    tests = [test_adiabatic_reactor_follows_the_ode, test_adiabatic_time_convergence_orders,
             test_adiabatic_conservation_is_exact, test_steady_invariants_are_linear, test_steady_2d_equals_1d,
             test_strong_exotherm_converges, test_newton_stops_cleanly_when_the_reaction_runs_away,
             test_jacobian_with_heat_species_cross_blocks_matches_finite_differences,
             test_species_alone_and_inside_the_composition_agree_for_isothermal_kinetics,
             test_composition_delegation_and_errors, test_composition_rejects_repeated_physics_and_fields,
             test_reaction_errors]
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
