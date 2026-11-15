"""Monitors, probes and checkpoints of a solve (phase E.4 of docs/PLAN_RENDIMIENTO_GPU_ML.md).

Model.solve(on_iteration=..., on_step=..., checkpoint=..., resume=...) builds a Monitor that the solvers call:

    on_iteration(info): every nonlinear iteration (Newton, pseudo-transient, and the Newton iterations inside a time step).
        info: kind ('newton', 'pseudo-transient' or 'step-newton'), iteration, residual (scaled norm), relative_residual, x,
        and step / time for the iterations of a time step.
    on_step(info): every accepted time step (and the end of a steady solve): step, time, dt, newton_iterations, x.

A callback that returns True stops the solve. A steady solve keeps its current state (model.nonlinear_converged is False); a time
loop keeps the steps already done. x is the state vector of the model (all fields; Model.layout splits it): read it, do not change it.

Probes (Model.addProbe) are sampled at t = 0 and after every accepted step; a steady solve samples them at every nonlinear
iteration (their "time" is the iteration number) and at the end. Results: model.probe_times, model.probe_values[name].
model.convergence holds the table of the nonlinear iterations (steady) or of the time steps (transient).

Checkpoints (transient studies): checkpoint=(path, every) or checkpoint=path (every 10 steps) writes an .npz with everything the
time loop needs, replacing the previous one; resume=path continues from it with the same configuration (the mesh, physics, time scheme and
total time are checked, as far as the file can tell).
"""

import os

import numpy as np

from Source.logger_config import sim_logger

CHECKPOINT_VERSION = 1
DEFAULT_EVERY = 10


class Monitor:

    def __init__(self, model, on_step=None, on_iteration=None, checkpoint=None, resume=None):

        self.model = model
        self.on_step = on_step
        self.on_iteration = on_iteration

        def npz(path):
            path = os.fspath(path)
            return path if path.endswith('.npz') else path + '.npz'

        if isinstance(checkpoint, (str, os.PathLike)):
            checkpoint = (checkpoint, DEFAULT_EVERY)
        self.checkpoint = None if checkpoint is None else (npz(checkpoint[0]), max(1, int(checkpoint[1])))
        self.resume = None if resume is None else npz(resume)

        self.stopped = False
        self.convergence = []
        self.probe_times = []
        self.probe_values = {name: [] for name in model.probes}
        self._dofs = {name: self._probe_dofs(probe) for name, probe in model.probes.items()}

    # ----- probes -----

    def _probe_dofs(self, probe):

        layout = self.model.layout
        if probe['field'] not in layout.names:
            raise ValueError(f"Sonda {probe['name']!r}: campo desconocido {probe['field']!r} (campos: {list(layout.names)}).")

        dofs = layout.dofs(probe['field'])
        return dofs[probe['node']] if probe['component'] is None else dofs[probe['node'], probe['component']]

    def record(self, x, t):

        """Samples every probe in the state x at time t."""

        if not self._dofs:
            return

        self.probe_times.append(float(t))
        for name, dofs in self._dofs.items():
            value = np.array(x[dofs], dtype=float)
            self.probe_values[name].append(float(value) if value.size == 1 else value)    # scalar field: a number

    # ----- callbacks -----

    def iteration(self, info):

        """Called by the nonlinear solvers at each iteration. Returns True if the run must stop."""

        info.setdefault('relative_residual', float('nan'))
        if info.get('kind') != 'step-newton':
            self.convergence.append({key: value for key, value in info.items() if key != 'x'})
            self.record(info['x'], info['iteration'])         # steady: the "time" of a sample is the iteration

        if self.on_iteration is not None and self.on_iteration(info):
            self.stopped = True

        return self.stopped

    def step(self, info, state=None):

        """Called after each accepted time step (and at the end of a steady solve). state: function that returns the arrays of
        a checkpoint. Returns True if the run must stop."""

        if 'time' in info:
            self.record(info['x'], info['time'])
            self.convergence.append({key: value for key, value in info.items() if key != 'x'})

        if self.on_step is not None and self.on_step(info):
            self.stopped = True

        if self.checkpoint is not None and state is not None and 'step' in info:
            path, every = self.checkpoint
            if info['step'] % every == 0 or self.stopped:
                self.save(path, state())

        return self.stopped

    # ----- checkpoints -----

    def save(self, path, arrays):

        """Writes the checkpoint atomically (a file is replaced only when the new one is complete)."""

        arrays = dict(arrays)
        arrays['version'] = np.array(CHECKPOINT_VERSION)
        arrays['ndof'] = np.array(self.model.layout.ndof)
        arrays['probe_times'] = np.array(self.probe_times)
        for name, values in self.probe_values.items():
            arrays['probe::' + name] = np.array(values)

        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        temporary = os.path.join(folder, '.' + os.path.basename(path) + '.tmp.npz')
        np.savez(temporary, **arrays)
        os.replace(temporary, path)

    def load(self, expected):

        """Reads the checkpoint of resume. expected: values that must agree with the file (scheme, total time...). Restores the
        probe records and returns the arrays."""

        with np.load(self.resume, allow_pickle=False) as file:
            data = {key: file[key] for key in file.files}

        if int(data['version']) != CHECKPOINT_VERSION:
            raise ValueError(f"Punto de control {self.resume!r}: version {int(data['version'])}, se esperaba {CHECKPOINT_VERSION}.")
        if int(data['ndof']) != self.model.layout.ndof:
            raise ValueError(f"Punto de control {self.resume!r}: {int(data['ndof'])} incognitas, el modelo tiene {self.model.layout.ndof}.")

        for key, value in expected.items():
            found = data[key].item() if key in data else None
            if found is None or (found != value if not isinstance(value, float) else not np.isclose(found, value, rtol=1e-12)):
                raise ValueError(f"Punto de control {self.resume!r}: {key} = {found!r}, la configuracion actual tiene {value!r}.")

        self.probe_times = [float(t) for t in data['probe_times']]
        for name in self.probe_values:
            key = 'probe::' + name
            self.probe_values[name] = list(data[key]) if key in data else []

        sim_logger.info(f"Reanudando desde {self.resume} (paso {int(data['step'])}, t = {float(data['time']):.6g}).")
        return data
