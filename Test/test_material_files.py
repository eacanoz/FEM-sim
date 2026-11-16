"""Materials from JSON files and from the library (Source/Material.py): the laws are the same objects the examples build by hand,
nested laws, errors that name the file and the section, round trip through save, search folders, and a flow that uses a material
read from a file.

Run from the project root:  python Test/test_material_files.py
"""

import json
import os
import sys
import tempfile
import traceback

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Material import material, ENV_VARIABLE
from Source.Physics.Rheology import CrossWLF, TaitPVT, Arrhenius, Cross, Newtonian
from Source.Physics.Crystallization import NakamuraHoffman


def test_library_material_equals_the_hand_built_laws():
    assert {'POLYFORT_FPP30GFC_K1079', 'Sabic_PP505P_cinetica'} <= set(material.library())

    mat = material.from_file('POLYFORT_FPP30GFC_K1079')
    assert (mat.rho, mat.Cp, mat.k) == (970.0, 2000.0, 0.22) and mat.properties['rho'] == 970.0 and mat.source
    by_hand = CrossWLF(n=0.2619, tau_star=1.778e4, D1=4.378e17, D2=266.7, A1=39.22, A2=51.6, gamma_min=1e-2)
    for gamma2, T in ((0.0, 230.0), (1e4, 230.0), (1e6, 200.0)):
        assert float(mat.viscosity_law(gamma2, T)) == float(by_hand(gamma2, T))
    tait = TaitPVT(1.006e-3, 4.936e-7, 7.989e7, 3.53e-3, 9.073e-4, 1.49e-7, 2.601e8, 2.353e-3, 453.9, 1.709e-7, 1.009e-4, 4.788e-2, 1.553e-8)
    assert float(mat.pvt.density(5e7, 200.0)) == float(tait.density(5e7, 200.0))
    assert mat.kinetics is None

    kinetics = material.from_file('Sabic_PP505P_cinetica')
    by_hand = NakamuraHoffman(33.6, 0.58e6, 17200.0, 231.15, 467.15, 3.0)
    assert float(kinetics.kinetics.rate(120.0)) == float(by_hand.rate(120.0))
    assert kinetics.viscosity_law is None and kinetics.pvt is None


def test_nested_laws_and_round_trip():
    spec = {'name': 'demo', 'properties': {'rho': 1000.0, 'k': 0.2},
            'viscosity': {'model': 'Arrhenius', 'E_R': 3000.0, 'T_ref': 200.0,
                          'law': {'model': 'Cross', 'mu0': 3000.0, 'tau_star': 2e4, 'n': 0.3}}}
    mat = material.from_dict(spec)
    assert isinstance(mat.viscosity_law, Arrhenius) and isinstance(mat.viscosity_law.law, Cross)
    by_hand = Arrhenius(Cross(3000.0, 2e4, 0.3), 3000.0, 200.0)
    assert float(mat.viscosity_law(100.0, 180.0)) == float(by_hand(100.0, 180.0))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'demo.json')
        mat.k = 0.25                                     # a changed property is what gets saved
        mat.save(path)
        again = material.from_file(path)
        assert again.k == 0.25 and again.rho == 1000.0 and again.name == 'demo'
        assert float(again.viscosity_law(100.0, 180.0)) == float(by_hand(100.0, 180.0))
        assert json.load(open(path, encoding='utf-8'))['viscosity']['law']['model'] == 'Cross'

    plain = material('X', k=1.0, rho=2.0)
    assert plain.to_dict() == {'name': 'X', 'properties': {'k': 1.0, 'rho': 2.0}}


def test_errors_name_the_file_and_the_section():
    def fails(data, text):
        try:
            material.from_dict(data, where='caso.json')
        except ValueError as error:
            assert 'caso.json' in str(error) and text in str(error), str(error)
        else:
            raise AssertionError(data)

    fails({'properties': {}}, "'name'")
    fails({'name': 'a', 'viscosidad': {}}, 'claves desconocidas')
    fails({'name': 'a', 'properties': {'rho': 'denso'}}, 'rho')
    fails({'name': 'a', 'viscosity': {'n': 1}}, "'model'")
    fails({'name': 'a', 'viscosity': {'model': 'Magic'}}, 'Magic')
    fails({'name': 'a', 'viscosity': {'model': 'Newtonian', 'eta': 1.0}}, 'Newtonian')
    fails({'name': 'a', 'viscosity': {'model': 'Arrhenius', 'E_R': 1.0, 'T_ref': 1.0}}, "'law'")
    fails({'name': 'a', 'pvt': {'model': 'CrossWLF'}}, 'CrossWLF')

    try:
        material.from_file('no_existe')
    except FileNotFoundError as error:
        assert 'POLYFORT_FPP30GFC_K1079' in str(error)
    else:
        raise AssertionError("missing material")

    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, 'bad.json')
        open(bad, 'w').write('{ not json')
        try:
            material.from_file(bad)
        except ValueError as error:
            assert 'bad.json' in str(error)
        else:
            raise AssertionError("invalid JSON")


def test_user_folder_takes_precedence():
    with tempfile.TemporaryDirectory() as tmp:
        json.dump({'name': 'propio', 'properties': {'rho': 1.0}}, open(os.path.join(tmp, 'mio.json'), 'w'))
        json.dump({'name': 'cambiado', 'properties': {'rho': 5.0}}, open(os.path.join(tmp, 'POLYFORT_FPP30GFC_K1079.json'), 'w'))
        saved = os.environ.get(ENV_VARIABLE)
        os.environ[ENV_VARIABLE] = tmp
        try:
            assert 'mio' in material.library()
            assert material.from_file('mio').rho == 1.0
            assert material.from_file('POLYFORT_FPP30GFC_K1079').rho == 5.0           # the user's folder first
        finally:
            if saved is None:
                del os.environ[ENV_VARIABLE]
            else:
                os.environ[ENV_VARIABLE] = saved
    assert material.from_file('POLYFORT_FPP30GFC_K1079').rho == 970.0


def test_a_slit_flow_with_a_material_from_a_file_matches_the_hand_built_one():
    from Source.Pre_processing.Mesh import Mesh
    from Source.Physics.HeleShaw import HeleShawFilling
    from Source.enums import ElementType

    def fill(law):
        mesh = Mesh()
        mesh.generate2DMesh((0.05, 0.02), (12, 5), ElementType.quadrilateral)
        filling = HeleShawFilling(mesh, 2e-3, law, T=230.0)
        filling.addGate(mesh.boundary_edges(lambda x, y: x < 1e-9), 0.05 * 0.02 * 2e-3 / 1.0)
        filling.fill(progress=False)
        return filling.history[-1][2]

    from_file = fill(material.from_file('POLYFORT_FPP30GFC_K1079').viscosity_law)
    by_hand = fill(CrossWLF(n=0.2619, tau_star=1.778e4, D1=4.378e17, D2=266.7, A1=39.22, A2=51.6, gamma_min=1e-2))
    assert from_file > 0 and from_file == by_hand


if __name__ == "__main__":
    tests = [test_library_material_equals_the_hand_built_laws, test_nested_laws_and_round_trip,
             test_errors_name_the_file_and_the_section, test_user_folder_takes_precedence,
             test_a_slit_flow_with_a_material_from_a_file_matches_the_hand_built_one]
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
