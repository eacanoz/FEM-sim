"""Monitors, probes and checkpoints (Source/Simulation/Monitoring.py): the callbacks of the solvers, the stop requested by a
callback, the probes against the stored solution, and the restart of a transient study from its checkpoint (linear, nonlinear
and adaptive time loops) with the same result as the uninterrupted run.

Run from the project root:  python Test/test_monitoring.py
"""

import os
import sys
import tempfile
import traceback

import matplotlib
matplotlib.use("Agg")

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
from Source.enums import ElementType


def plate(nonlinear=False, n=10):
    mesh = Mesh()
    mesh.generate2DMesh((1.0, 1.0), (n, n), ElementType.quadrilateral)
    model = Model(name='plate', dim=2, mesh=mesh, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
    model.physics.Convection = False
    model.physics.F_const = 10.0
    if nonlinear:
        model.physics.setConductivityLaw(lambda T: 1.0 + 0.3 * T)
    model.physics.addBC_TemperatureEdges(mesh.boundary_edges(), 0.0)
    model.physics.initField('T', 0.0)
    return model


def transient(nonlinear, total=0.2, **options):
    model = plate(nonlinear)
    model.solverConfiguration(Study='transient', timeStep=0.02, totalTime=total, **options)
    return model


def test_steady_monitor_table_probe_and_stop():
    model = plate(nonlinear=True)
    model.addProbe('center', 'T', point=(0.5, 0.5))
    seen = []
    model.solve(on_iteration=lambda info: seen.append((info['iteration'], info['residual'])))
    assert model.nonlinear_converged and len(seen) == model.newton_iterations
    assert [row['iteration'] for row in model.convergence] == [i for i, _ in seen]
    assert seen[-1][1] < seen[0][1]
    node = model.probes['center']['node']
    assert abs(model.probe_values['center'][-1] - model.sol['T'][node]) < 1e-12
    assert len(model.probe_values['center']) == len(model.probe_times) >= len(seen)

    stopped = plate(nonlinear=True)
    stopped.solve(on_iteration=lambda info: info['iteration'] >= 1)
    assert stopped.newton_iterations == 1 and not stopped.nonlinear_converged


def test_linear_transient_probe_and_stop():
    model = transient(False)
    model.addProbe('center', 'T', point=(0.5, 0.5))
    steps = []
    model.solve(on_step=lambda info: steps.append(info['step']))
    node = model.probes['center']['node']
    history = np.asarray(model.sol['T'])
    assert steps == list(range(1, 11)) and len(model.probe_times) == 11
    assert np.max(np.abs(np.array(model.probe_values['center']) - history[node, :])) < 1e-14
    assert np.allclose(model.probe_times, model.timeVector)

    cut = transient(False)
    cut.solve(on_step=lambda info: info['step'] == 4)
    assert len(cut.timeVector) == 5 and np.asarray(cut.sol['T']).shape[1] == 5
    assert np.allclose(np.asarray(cut.sol['T']), history[:, :5], atol=1e-14)


def test_linear_transient_resumes_from_its_checkpoint():
    reference = transient(False)
    reference.solve()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'restart')
        first = transient(False)
        first.addProbe('c', 'T', point=(0.5, 0.5))
        first.solve(checkpoint=(path, 3), on_step=lambda info: info['step'] == 7)      # saved at 3, 6 and at the stop (7)
        assert os.path.exists(path + '.npz') and len(first.timeVector) == 8

        second = transient(False)
        second.addProbe('c', 'T', point=(0.5, 0.5))
        second.solve(resume=path)
        assert np.allclose(second.timeVector, reference.timeVector)
        assert np.max(np.abs(np.asarray(second.sol['T']) - np.asarray(reference.sol['T']))) < 1e-12
        assert len(second.probe_values['c']) == 11                      # the records of the first run came along


def test_nonlinear_transient_resumes_and_checks_the_configuration():
    for options in (dict(timeScheme='BDF2'), dict(timeScheme='theta', timeDisc=0.5), dict(timeScheme='BDF2', adaptiveTol=1e-3)):
        reference = transient(True, **options)
        reference.solve()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'case.npz')
            first = transient(True, **options)
            first.solve(checkpoint=path, on_step=lambda info: info['step'] == 4)
            assert len(first.timeVector) == 5

            second = transient(True, **options)
            second.solve(resume=path)
            assert np.allclose(second.timeVector, reference.timeVector, atol=1e-12), options
            assert np.max(np.abs(np.asarray(second.sol['T']) - np.asarray(reference.sol['T']))) < 1e-7, options

            other = transient(True, total=0.4, **options)
            try:
                other.solve(resume=path)
            except ValueError as error:
                assert 'total_time' in str(error)
            else:
                raise AssertionError("a checkpoint of another total time was accepted")


def test_stop_inside_a_newton_iteration_of_a_step_drops_the_step():
    model = transient(True, timeScheme='BDF2')
    model.solve(on_iteration=lambda info: info['kind'] == 'step-newton' and info['step'] == 3 and info['iteration'] == 1)
    assert len(model.timeVector) == 3                                   # t = 0 and two complete steps


def test_probe_arguments_are_checked():
    model = plate()
    for kwargs in (dict(), dict(node=1, point=(0.0, 0.0))):
        try:
            model.addProbe('x', 'T', **kwargs)
        except ValueError:
            continue
        raise AssertionError(kwargs)
    model.addProbe('bad', 'u', node=0)
    try:
        model.solve()
    except ValueError as error:
        assert 'campo desconocido' in str(error)
    else:
        raise AssertionError("unknown field")


if __name__ == "__main__":
    tests = [test_steady_monitor_table_probe_and_stop, test_linear_transient_probe_and_stop,
             test_linear_transient_resumes_from_its_checkpoint, test_nonlinear_transient_resumes_and_checks_the_configuration,
             test_stop_inside_a_newton_iteration_of_a_step_drops_the_step, test_probe_arguments_are_checked]
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
