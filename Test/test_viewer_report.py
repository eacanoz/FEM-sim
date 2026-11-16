"""Viewer (pyvista, optional) and HTML report: the grid arrays shared with the VTK export, the self-contained report of a solved model and
of the cases, and the pyvista pieces when it is installed (without it, the error says how to install it).

Run from the project root:  python Test/test_viewer_report.py
"""

import os
import re
import sys
import tempfile
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source import Cases
from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
from Source.Post_processing import Viewer
from Source.Post_processing.Report import Report
from Source.Post_processing.VTK import grid_arrays, write_vtu, VTK_CELL
from Source.enums import ElementType

MELT = 'POLYFORT_FPP30GFC_K1079'


def plate(transient=False):
    mesh = Mesh()
    mesh.generate2DMesh((1.0, 1.0), (8, 8), ElementType.quadrilateral)
    model = Model(name='placa', dim=2, mesh=mesh, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
    model.physics.Convection = False
    model.physics.F_const = 10.0
    model.physics.setConductivityLaw(lambda T: 1.0 + 0.3 * T)
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(), 0.0)
    model.physics.initField('T', 0.0)
    model.addProbe('centro', 'T', point=(0.5, 0.5))
    if transient:
        model.solverConfiguration(Study='transient', timeScheme='BDF2', timeStep=0.05, totalTime=0.2)
    return model


def test_grid_arrays_are_what_the_vtu_file_holds():
    model = plate()
    points, conn, kind = grid_arrays(model.mesh)
    assert points.shape == (81, 3) and conn.shape == (64, 4) and kind == 'quad4' and VTK_CELL[kind] == 9
    assert np.all(points[:, 2] == 0.0) and conn.max() == 80

    with tempfile.TemporaryDirectory() as tmp:
        path = write_vtu(os.path.join(tmp, 'a.vtu'), model.mesh, {'T': np.arange(81.0)})
        text = open(path, encoding='utf-8').read()
        assert 'NumberOfPoints="81"' in text and 'NumberOfCells="64"' in text


def test_report_is_one_self_contained_file():
    report = Report('Titulo <con> marcas', subtitle='sub & titulo')
    report.add_text('Texto', 'a < b')
    report.add_table('Tabla', {'rho': 970.0, 'n': 3, 'ok': True, 'lista': [1.0, 2.0], 'larga': list(range(20))})
    report.add_table('Filas', [{'a': 1.5, 'b': 'x'}, {'a': 2.5, 'b': 'y'}], caption='pie')
    figure, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    report.add_figure('Figura', figure)
    with tempfile.TemporaryDirectory() as tmp:
        text = open(report.save(os.path.join(tmp, 'r.html')), encoding='utf-8').read()

    assert text.startswith('<!doctype html>') and '<title>Titulo &lt;con&gt; marcas</title>' in text and 'sub &amp; titulo' in text
    assert 'a &lt; b' in text and '<td class="num">970</td>' in text and '(20 valores)' in text and '1, 2' in text
    assert 'data:image/png;base64,' in text and 'pie' in text
    assert not re.search(r'(src|href)="https?://', text)                      # nothing external
    assert 'prefers-color-scheme: dark' in text


def test_report_of_a_steady_and_a_transient_model():
    steady = plate()
    steady.solve()
    report = Report('Placa')
    report.add_model(steady)
    text = report.html()
    for heading in ('Modelo', 'Campos', 'Convergencia', 'Sondas'):
        assert f'<h2>{heading}</h2>' in text, heading
    assert 'placa' in text and text.count('data:image/png') == 3                # fields, convergence, probes
    assert 'convergio' in text and 'True' in text

    transient = plate(transient=True)
    transient.solve()
    report = Report('Placa transitoria')
    report.add_model(transient)
    assert '<h2>Iteraciones por paso</h2>' in report.html() and 'pasos' in report.html()

    try:
        Report('x').add_model(plate())
    except RuntimeError as error:
        assert 'solve' in str(error)
    else:
        raise AssertionError("an unsolved model got a report")


def test_cases_write_their_reports():
    with tempfile.TemporaryDirectory() as tmp:
        base = {'material': {'file': MELT}, 'output': {'directory': tmp, 'report': True}}

        slit = Cases.run_case({'type': 'slit_die', **base, 'geometry': {'nx': 20, 'ny': 6}, 'process': {'flow_rate': 5e-5}})[0]['results']
        text = open(slit['report'], encoding='utf-8').read()
        assert 'Dado de rendija' in text and '<h2>Resultados</h2>' in text and '<h2>Campos</h2>' in text and 'pressure_drop' in text

        injection = Cases.run_case({'type': 'injection', **base, 'geometry': {'element_size': 5e-3}, 'process': {'fill_time': 1.0}})[0]['results']
        text = open(injection['report'], encoding='utf-8').read()
        assert 'Isocronas del llenado' in text and 'fuerza de cierre' in text and 'clamp_force_peak' in text

        rheometer = Cases.run_case({'type': 'capillary_rheometer', **base, 'geometry': {'mesh_scale': 2.5},
                                    'process': {'shear_rates': [20.0, 200.0, 2000.0]}})[0]['results']
        text = open(rheometer['report'], encoding='utf-8').read()
        assert 'Curva de viscosidad medida' in text and '<h2>Tabla de resultados</h2>' in text and 'eta_measured' in text


def test_viewer_without_and_with_pyvista():
    model = plate()
    model.solve()

    if not Viewer.available():
        for call in (lambda: Viewer.to_pyvista(model.mesh), lambda: Viewer.show(model, 'T', screenshot='x.png')):
            try:
                call()
            except ImportError as error:
                assert 'pip install pyvista' in str(error)
            else:
                raise AssertionError("pyvista missing but no error")
        print("      (pyvista no esta instalado: probado solo el aviso)")
        return

    grid = Viewer.to_pyvista(model.mesh, {'T': model.sol['T']})
    assert grid.n_points == 81 and grid.n_cells == 64 and 'T' in grid.point_data
    with tempfile.TemporaryDirectory() as tmp:
        png = os.path.join(tmp, 'T.png')
        Viewer.show(model, 'T', screenshot=png)
        assert os.path.getsize(png) > 1000
    try:
        Viewer.show(model, 'nada', screenshot='x.png')
    except KeyError:
        pass
    else:
        raise AssertionError("unknown field")


if __name__ == "__main__":
    tests = [test_grid_arrays_are_what_the_vtu_file_holds, test_report_is_one_self_contained_file,
             test_report_of_a_steady_and_a_transient_model, test_cases_write_their_reports, test_viewer_without_and_with_pyvista]
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
