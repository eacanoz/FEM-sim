"""Case files and the command line (Source/Cases.py, Source/cli.py): the checks of a TOML file, the overrides, the expansion of a
sweep, a slit die run end to end (results, JSON and CSV files, serial and in processes) and the command line.

Run from the project root:  python Test/test_cases.py
"""

import csv
import json
import os
import sys
import tempfile
import traceback

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source import Cases
from Source.cli import main

CASE = """
type = "slit_die"
name = "demo"
[material]
file = "POLYFORT_FPP30GFC_K1079"
[geometry]
nx = 20
ny = 6
[process]
flow_rate = 5.0e-5
[output]
directory = "{out}"
"""


def write(folder, text, name='caso.toml'):
    path = os.path.join(folder, name)
    open(path, 'w', encoding='utf-8').write(text)
    return path


def fails(data, *texts):
    try:
        Cases.run_case(data, dry_run=True)
    except ValueError as error:
        for text in texts:
            assert text in str(error), (text, str(error))
    else:
        raise AssertionError(data)


def test_checks_name_the_key():
    base = {'type': 'slit_die', 'material': {'file': 'x'}, 'process': {'flow_rate': 1e-4}, '_file': 'caso.toml'}

    def with_(section, key, value):
        data = json.loads(json.dumps(base))
        data.setdefault(section, {})[key] = value
        return data

    fails({'name': 'a'}, "'type'")
    fails({'type': 'magia'}, 'magia', 'slit_die')
    fails({**base, 'geometria': {}}, "geometria", 'caso.toml')
    fails(with_('geometry', 'altura', 1.0), 'geometry.altura', 'half_aperture')
    fails(with_('geometry', 'nx', 'muchos'), 'geometry.nx')
    fails(with_('geometry', 'nx', 12.5), 'geometry.nx')
    fails(with_('process', 'viscous_heating', 1), 'process.viscous_heating')
    fails({'type': 'slit_die', 'material': {'file': 'x'}}, 'process.flow_rate')
    fails({'type': 'slit_die', 'process': {'flow_rate': 1.0}}, 'material.file')

    ok = Cases.run_case(with_('geometry', 'nx', 12.0), dry_run=True)                  # 12.0 is an integer value
    assert ok[0]['params']['geometry']['nx'] == 12 and ok[0]['params']['process']['T_inlet'] == 230.0

    try:
        Cases.run_case('no_existe.toml')
    except FileNotFoundError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        try:
            Cases.run_case(write(tmp, 'type = '), dry_run=True)
        except ValueError as error:
            assert 'caso.toml' in str(error) and 'TOML' in str(error)
        else:
            raise AssertionError("invalid TOML")


def test_overrides_and_sweep_expansion():
    data = {'type': 'slit_die', 'material': {'file': 'x'}, 'process': {'flow_rate': 1e-4}}
    runs = Cases.run_case(data, overrides=['process.flow_rate=2e-4', 'process.viscous_heating=false',
                                           'geometry.nx=30', 'name="otro"'], dry_run=True)
    assert runs[0]['params']['process']['flow_rate'] == 2e-4 and runs[0]['params']['process']['viscous_heating'] is False
    assert runs[0]['params']['geometry']['nx'] == 30 and runs[0]['name'] == 'otro'
    assert Cases.parse_value('texto sin comillas') == 'texto sin comillas' and Cases.parse_value('[1, 2]') == [1, 2]

    swept = dict(data, sweep={'flow_rate': [1e-5, 2e-5, 4e-5], 'T_inlet': [220.0, 240.0]})
    runs = Cases.run_case(swept, dry_run=True)
    assert len(runs) == 6 and runs[0]['sweep'] == {'flow_rate': 1e-5, 'T_inlet': 220.0}
    assert runs[-1]['params']['process']['flow_rate'] == 4e-5 and runs[-1]['params']['process']['T_inlet'] == 240.0
    assert data['process']['flow_rate'] == 1e-4                        # the input is not modified

    for sweep, text in (({'nada': [1]}, 'nada'), ({'flow_rate': []}, 'lista'), ({'flow_rate': 3}, 'lista')):
        fails(dict(data, sweep=sweep), text)


def test_slit_die_run_and_sweep_files():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'res')
        path = write(tmp, CASE.format(out=out.replace('\\', '/')) + "\n[sweep]\nflow_rate = [2.0e-5, 8.0e-5]\n")
        records = Cases.run_case(path)
        assert [r['run'] for r in records] == ['demo_000', 'demo_001']

        low, high = records[0]['results'], records[1]['results']
        for r in (low, high):
            assert r['converged'] and abs(r['flow_in'] - r['flow_out']) < 1e-3 * r['flow_in']
            assert r['pressure_drop'] > 0 and r['T_max'] >= 230.0
        ratio = high['pressure_drop'] / low['pressure_drop']
        assert 1.0 < ratio < 4.0, ratio                                # 4x the flow rate, less than 4x the pressure (shear thinning)
        assert abs(low['wall_shear_rate'] - 3 * 2e-5 / 1e-6) < 1e-9

        saved = json.load(open(os.path.join(out, 'demo_001_results.json'), encoding='utf-8'))
        assert saved['sweep'] == {'flow_rate': 8e-5} and saved['results']['pressure_drop'] == high['pressure_drop']
        rows = list(csv.reader(open(os.path.join(out, 'demo_sweep.csv'), encoding='utf-8')))
        assert rows[0][:2] == ['run', 'flow_rate'] and 'pressure_drop' in rows[0] and len(rows) == 3
        assert abs(float(rows[2][rows[0].index('pressure_drop')]) - high['pressure_drop']) < 1e-6 * high['pressure_drop']


def test_viscous_heating_and_vtk_and_material_without_viscosity():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'res')
        path = write(tmp, CASE.format(out=out.replace('\\', '/')).replace('5.0e-5', '2.0e-4'))
        hot = Cases.run_case(path, overrides=['output.vtk=true'])[0]['results']
        cold = Cases.run_case(path, overrides=['process.viscous_heating=false'])[0]['results']
        assert hot['T_max'] > cold['T_max'] + 0.1 and abs(cold['T_max'] - 230.0) < 1e-6
        assert hot['pressure_drop'] < cold['pressure_drop']                  # the heated melt is thinner
        assert hot['vtk'] and all(os.path.exists(f) for f in hot['vtk'])

        bare = os.path.join(tmp, 'sin_viscosidad.json')
        json.dump({'name': 'x', 'properties': {'rho': 1.0, 'Cp': 1.0, 'k': 1.0}}, open(bare, 'w'))
        try:
            Cases.run_case(path, overrides=[f'material.file="{bare.replace(chr(92), "/")}"'])
        except ValueError as error:
            assert 'viscosity' in str(error)
        else:
            raise AssertionError("a material without viscosity law ran")


def test_low_flow_rate_gives_the_newtonian_pressure_drop():
    # at a tiny shear rate the viscosity is that of rest (the law's own minimum shear rate included, 1e-2 1/s in this material, which
    # lowers it by 1.2 %): dp = 3 mu q L / H^3 for the half slit of aperture H
    from Source.Material import material
    mat = material.from_file('POLYFORT_FPP30GFC_K1079')
    mu0 = float(mat.viscosity_law(0.0, 230.0))
    q, H, L = 1e-9, 1e-3, 20e-3
    with tempfile.TemporaryDirectory() as tmp:
        data = {'type': 'slit_die', 'material': {'file': 'POLYFORT_FPP30GFC_K1079'}, 'process': {'flow_rate': q, 'viscous_heating': False},
                'geometry': {'nx': 30, 'ny': 8}, 'output': {'directory': tmp}}
        dp = Cases.run_case(data)[0]['results']['pressure_drop']
    assert abs(dp / (3.0 * mu0 * q * L / H ** 3) - 1.0) < 0.003, dp / (3.0 * mu0 * q * L / H ** 3)


def test_runs_in_processes_match_the_serial_ones():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'res').replace('\\', '/')
        path = write(tmp, CASE.format(out=out) + "\n[sweep]\nflow_rate = [2.0e-5, 8.0e-5]\n")
        serial = Cases.run_case(path, jobs=1)
        parallel = Cases.run_case(path, jobs=2)
        for a, b in zip(serial, parallel):
            assert abs(a['results']['pressure_drop'] - b['results']['pressure_drop']) < 1e-6 * a['results']['pressure_drop']


def test_command_line(capsys=None):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'res')
        path = write(tmp, CASE.format(out='ignorada'))
        assert main([path, '--dry-run']) == 0
        assert main([path, '-q', '-o', out, '--set', 'process.flow_rate=3e-5']) == 0
        assert os.path.exists(os.path.join(out, 'demo_results.json'))
        assert not os.path.exists('ignorada')                              # --output replaces [output].directory
        assert main([write(tmp, 'type = "nada"', 'mal.toml')]) == 2
        assert main(['--list-types']) == 0


if __name__ == "__main__":
    tests = [test_checks_name_the_key, test_overrides_and_sweep_expansion, test_slit_die_run_and_sweep_files,
             test_viscous_heating_and_vtk_and_material_without_viscosity, test_low_flow_rate_gives_the_newtonian_pressure_drop,
             test_runs_in_processes_match_the_serial_ones,
             test_command_line]
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
