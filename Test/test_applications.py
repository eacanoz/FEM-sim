"""Application classes (Source/Applications.py) and their case types: the slit die against the Newtonian solution, the capillary
rheometer recovering the viscosity it was given (Newtonian exactly, a Cross-WLF melt within the coarse mesh), a filling against the
closed solution of a Newtonian strip (pressure and clamp force) and the trends of the clamp force; then the three case types.

Run from the project root:  python Test/test_applications.py
"""

import json
import os
import sys
import tempfile
import traceback

import matplotlib
matplotlib.use("Agg")

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source import Cases
from Source.Applications import SlitDie, CapillaryRheometer, InjectionCase, rest_viscosity
from Source.Material import material

MELT = 'POLYFORT_FPP30GFC_K1079'


def newtonian(eta, rho=1000.0, Cp=2000.0, k=0.2):
    return material.from_dict({'name': 'newtoniano', 'properties': {'rho': rho, 'Cp': Cp, 'k': k},
                               'viscosity': {'model': 'Newtonian', 'mu': eta}})


def test_slit_die_newtonian_and_curve():
    eta, H, L, q = 2000.0, 1e-3, 20e-3, 1e-5
    die = SlitDie(newtonian(eta), half_aperture=H, length=L, nx=24, ny=8, viscous_heating=False)
    result = die.solve(q)
    assert result['converged'] and abs(result['pressure_drop'] / (3 * eta * q * L / H ** 3) - 1.0) < 0.01

    curve = SlitDie(MELT, nx=24, ny=8).curve([2e-5, 4e-5, 8e-5])
    drops = [row['pressure_drop'] for row in curve]
    assert drops == sorted(drops) and all(row['converged'] for row in curve)
    assert np.log(drops[-1] / drops[0]) / np.log(4.0) < 0.95                        # shear thinning: slope of the die curve below 1
    assert rest_viscosity(material.from_file(MELT).viscosity_law, 230.0) > 1000.0


def test_capillary_rheometer_recovers_the_viscosity_of_the_law():
    eta = 1500.0
    result = CapillaryRheometer(newtonian(eta), T=230.0, mesh_scale=2.5).measure((20.0, 100.0, 500.0))
    assert np.allclose(result['n_prime'], 1.0, atol=0.02)
    assert np.max(np.abs(result['error'])) < 0.03, result['error']                  # Newtonian: eta = tau_w / gamma_w
    assert np.all(result['dp_entrance'] > 0) and np.all(result['dp_entrance'] < 0.25 * result['dp_total'])

    real = CapillaryRheometer(MELT, T=230.0, mesh_scale=2.5).measure((20.0, 200.0, 2000.0))
    assert np.all(real['n_prime'] < 1.0) and np.all(np.diff(real['eta_measured']) < 0)      # shear thinning
    assert np.max(np.abs(real['error'])) < 0.06, real['error']

    try:
        CapillaryRheometer(newtonian(eta), mesh_scale=3.0).measure((20.0, 50.0))
    except ValueError as error:
        assert 'tres' in str(error)
    else:
        raise AssertionError("two rates accepted")


def test_filling_a_newtonian_strip_matches_the_closed_solution():
    # a gate across the whole width: p(x) falls linearly to 0 at the front; at the end p_gate = 12 eta L^2 / (t h^2) and F = p_gate L W / 2
    eta, L, W, h, t = 1000.0, 0.10, 0.04, 2e-3, 1.0
    case = InjectionCase.plate(newtonian(eta), length=L, width=W, thickness=h, gate_width=W, fill_time=t, element_size=5e-3,
                               thermal=False)
    result = case.run()
    p_exact = 12 * eta * L ** 2 / (t * h ** 2)
    assert not result['short_shot'] and abs(result['filled_fraction'] - 1.0) < 1e-9
    assert abs(result['gate_pressure_peak'] / p_exact - 1.0) < 0.03, result['gate_pressure_peak'] / p_exact
    assert abs(result['clamp_force_peak'] / (0.5 * p_exact * L * W) - 1.0) < 0.03
    times = [time for time, _ in case.clamp_history]
    assert times == sorted(times) and case.clamp_history[-1][1] == result['clamp_force_peak']
    assert 0.9 * t < result['fill_time'] <= t * 1.001
    last = np.nanmax(case.filling.fill_time)
    assert np.allclose(case.isochrones(4), np.linspace(0, last, 5)[1:])


def test_filling_a_polymer_plate_trends():
    # isothermal: a faster filling needs more pressure and clamp force (shear thinning, but less than proportionally)
    fast = InjectionCase.plate(MELT, fill_time=0.5, element_size=5e-3, thermal=False).run()
    slow = InjectionCase.plate(MELT, fill_time=2.0, element_size=5e-3, thermal=False).run()
    assert fast['gate_pressure_peak'] > slow['gate_pressure_peak'] and fast['clamp_force_peak'] > slow['clamp_force_peak']
    assert fast['gate_pressure_peak'] < 4.0 * slow['gate_pressure_peak']

    # with the mold at 50 °C: the layer that freezes grows with the time against it, and a slow filling pays for it with pressure
    # (the U curve of the fill pressure against the fill time that examples/Example_Filling_NonIsothermal.py draws)
    hot_fast = InjectionCase.plate(MELT, fill_time=0.5, element_size=5e-3).run()
    hot_slow = InjectionCase.plate(MELT, fill_time=4.0, element_size=5e-3).run()
    assert 0.0 < hot_fast['frozen_fraction_mean'] < hot_slow['frozen_fraction_mean'] < 1.0
    assert hot_slow['gate_pressure_peak'] > 1.3 * hot_fast['gate_pressure_peak']
    assert hot_fast['gate_pressure_peak'] > fast['gate_pressure_peak']                   # the same filling, with heat lost: more pressure
    assert slow['gate_pressure_peak'] < hot_slow['gate_pressure_peak']


def test_case_types():
    with tempfile.TemporaryDirectory() as tmp:
        base = {'material': {'file': MELT}, 'output': {'directory': tmp}}

        record = Cases.run_case({'type': 'capillary_rheometer', **base, 'geometry': {'mesh_scale': 2.5},
                                 'process': {'shear_rates': [20.0, 200.0, 2000.0]}})[0]['results']
        assert len(record['gamma_apparent']) == 3 and record['max_abs_error'] < 0.06 and 0 < record['entrance_loss_fraction_mean'] < 0.5

        records = Cases.run_case({'type': 'injection', **base, 'geometry': {'element_size': 5e-3},
                                  'process': {'fill_time': 1.0, 'thermal': False}, 'sweep': {'fill_time': [0.5, 2.0]}})
        assert records[0]['results']['gate_pressure_peak'] > records[1]['results']['gate_pressure_peak']
        assert os.path.exists(os.path.join(tmp, 'injection_sweep.csv'))

        plotted = Cases.run_case({'type': 'injection', **base, 'geometry': {'element_size': 5e-3}, 'process': {'fill_time': 1.0},
                                  'output': {'directory': tmp, 'plot': True}})[0]['results']
        assert os.path.getsize(plotted['plot']) > 1000
        saved = json.load(open(os.path.join(tmp, 'injection_results.json'), encoding='utf-8'))
        assert saved['type'] == 'injection'

    for name in ('capillary_rheometer', 'injection', 'slit_die'):
        assert name in Cases.CASE_TYPES
    try:
        Cases.run_case({'type': 'injection', 'material': {'file': MELT}}, dry_run=True)
    except ValueError as error:
        assert 'process.fill_time' in str(error)
    else:
        raise AssertionError("fill_time is required")


if __name__ == "__main__":
    tests = [test_slit_die_newtonian_and_curve, test_capillary_rheometer_recovers_the_viscosity_of_the_law,
             test_filling_a_newtonian_strip_matches_the_closed_solution, test_filling_a_polymer_plate_trends, test_case_types]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS ", t.__name__)
        except Exception as e:
            traceback.print_exc()
            failed += 1
            print("FAIL ", t.__name__, repr(e))
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
