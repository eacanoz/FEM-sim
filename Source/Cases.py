"""Case files (TOML) and their runner (phase E.6 of docs/PLAN_RENDIMIENTO_GPU_ML.md).

A case file says WHAT to compute (a type of case, its material, geometry and process) and the program builds and solves the model:

    python -m femsim caso.toml                              # run it
    python -m femsim caso.toml --set process.flow_rate=2e-4 --output resultados
    python -m femsim caso.toml --jobs 4                     # the points of a [sweep] in 4 processes
    python -m femsim caso.toml --dry-run                    # check the file and list the runs, without solving

    # caso.toml
    type = "slit_die"
    name = "dado_pp"
    [material]
    file = "POLYFORT_FPP30GFC_K1079"      # library name or path of a material JSON (Source/Material.py)
    [geometry]
    half_aperture = 1.0e-3
    length = 20.0e-3
    [process]
    flow_rate = 1.0e-4
    T_inlet = 230.0
    [output]
    directory = "resultados"
    vtk = true
    [sweep]                                # optional: one run per combination
    flow_rate = [1e-5, 5e-5, 1e-4]

Keys of [sweep] are the names of keys of any section (they must be unambiguous) or the full "section.key". Types: slit_die (flat slit die), capillary_rheometer
(Bagley and Rabinowitsch) and injection (filling of a plate: isochrones, pressure, clamp force); see their classes below and
Source/Applications.py. A sweep writes <name>_sweep.csv; every run writes its own <name>[_i]_results.json with the resolved case and its results.

Types of case are registered with @case_type(name) on a class with `defaults` (section -> {key: default}; a key without default is
required, written as REQUIRED) and a method run(params, output_dir) -> dict of results. Unknown sections or keys, wrong types and
missing keys are errors that name the file and the key.
"""

import copy
import csv
import itertools
import json
import os
import time
import tomllib

import numpy as np

REQUIRED = object()
CASE_TYPES = {}


def case_type(name):

    def register(cls):
        cls.type_name = name
        CASE_TYPES[name] = cls
        return cls

    return register


# ----- reading, checking and overriding -----

def load_case(source):

    """The dictionary of a TOML file (path) or the dictionary itself."""

    if isinstance(source, dict):
        return copy.deepcopy(source)

    path = os.fspath(source)
    try:
        with open(path, 'rb') as file:
            data = tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{path}: TOML invalido ({error}).") from None

    data.setdefault('_file', path)
    return data


def parse_value(text):

    """Value of a command-line override: a TOML value (number, true/false, "text", [list]) or, if it is not one, the text."""

    try:
        return tomllib.loads(f"v = {text}")['v']
    except tomllib.TOMLDecodeError:
        return text


def apply_override(data, assignment):

    """data['section']['key'] = value from 'section.key=value' (a key of the top level: 'name=value')."""

    if '=' not in assignment:
        raise ValueError(f"--set {assignment!r}: use seccion.clave=valor.")

    dotted, text = assignment.split('=', 1)
    parts = dotted.strip().split('.')
    target = data
    for part in parts[:-1]:
        target = target.setdefault(part, {})
        if not isinstance(target, dict):
            raise ValueError(f"--set {assignment!r}: {part!r} no es una seccion.")
    target[parts[-1]] = parse_value(text.strip())


def _check_type(where, value, default):

    if default is REQUIRED or default is None:
        return value

    if isinstance(default, bool):
        ok = isinstance(value, bool)
    elif isinstance(default, (int, float)):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        if ok and isinstance(default, int) and not isinstance(value, int):
            ok = float(value).is_integer()
    elif isinstance(default, str):
        ok = isinstance(value, str)
    else:
        ok = True

    if not ok:
        raise ValueError(f"{where}: se esperaba {type(default).__name__}, se encontro {value!r}.")

    return int(value) if isinstance(default, int) and not isinstance(default, bool) else value


def resolve(data):

    """The case with every default filled in and checked: returns (case class, params {section: {key: value}}, rest) where rest is
    {'name', 'sweep', 'file'}."""

    where = data.get('_file', 'caso')

    if 'type' not in data:
        raise ValueError(f"{where}: falta la clave 'type' (tipos: {sorted(CASE_TYPES)}).")
    if data['type'] not in CASE_TYPES:
        raise ValueError(f"{where}: tipo de caso {data['type']!r} desconocido (tipos: {sorted(CASE_TYPES)}).")

    cls = CASE_TYPES[data['type']]
    top = {'type', 'name', 'sweep', '_file'}
    unknown = sorted(set(data) - top - set(cls.defaults))
    if unknown:
        raise ValueError(f"{where}: secciones desconocidas {unknown} para el tipo {cls.type_name!r} "
                         f"(validas: {sorted(set(cls.defaults) | top - {'_file'})}).")

    params = {}
    for section, defaults in cls.defaults.items():
        given = data.get(section, {})
        if not isinstance(given, dict):
            raise ValueError(f"{where}: [{section}] debe ser una seccion, no {given!r}.")
        extra = sorted(set(given) - set(defaults))
        if extra:
            raise ValueError(f"{where}: claves desconocidas {section}.{extra[0]} (validas en [{section}]: {sorted(defaults)}).")

        params[section] = {}
        for key, default in defaults.items():
            if key in given:
                params[section][key] = _check_type(f"{where}: {section}.{key}", given[key], default)
            elif default is REQUIRED:
                raise ValueError(f"{where}: falta {section}.{key}.")
            else:
                params[section][key] = default

    return cls, params, {'name': data.get('name', cls.type_name), 'sweep': data.get('sweep', {}), 'file': where}


def _locate(params, key, where):

    """(section, key) of a sweep key, 'key' or 'section.key'."""

    if '.' in key:
        section, name = key.split('.', 1)
        if section in params and name in params[section]:
            return section, name
    else:
        found = [(s, key) for s in params if key in params[s]]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise ValueError(f"{where}: la clave de barrido {key!r} esta en varias secciones {[s for s, _ in found]}: "
                             f"escriba seccion.clave.")

    raise ValueError(f"{where}: la clave de barrido {key!r} no existe en el caso.")


def expand(params, sweep, where='caso'):

    """List of (label dict, params) for the combinations of a sweep (one run without it)."""

    if not sweep:
        return [({}, params)]

    axes = []
    for key, values in sweep.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"{where}: sweep.{key} debe ser una lista con valores.")
        axes.append((key, _locate(params, key, where), values))

    runs = []
    for combination in itertools.product(*[values for _, _, values in axes]):
        point = copy.deepcopy(params)
        label = {}
        for (key, (section, name), _), value in zip(axes, combination):
            point[section][name] = value
            label[key] = value
        runs.append((label, point))

    return runs


# ----- running -----

def _run_one(job):

    """One run (top level so that processes can call it)."""

    case_name, params, directory, stem = job
    cls = CASE_TYPES[case_name]
    start = time.perf_counter()
    results = cls().run(params, directory, stem)
    results['seconds'] = round(time.perf_counter() - start, 3)
    return results


def run_case(source, overrides=(), output=None, jobs=1, dry_run=False):

    """Runs a case file (path or dictionary). overrides: 'section.key=value' strings. output: folder of the results (it replaces
    [output].directory). jobs > 1: the runs of a sweep in that many processes. Returns the list of results, one per run (with the
    sweep values under 'sweep'); with dry_run, the list of runs without solving them."""

    data = load_case(source)
    for assignment in overrides:
        apply_override(data, assignment)

    cls, params, rest = resolve(data)
    name = rest['name']
    runs = expand(params, rest['sweep'], rest['file'])

    directory = output if output is not None else params.get('output', {}).get('directory', '.')
    if dry_run:
        return [{'name': name, 'sweep': label, 'params': point} for label, point in runs]

    os.makedirs(directory, exist_ok=True)
    stems = [name if len(runs) == 1 else f"{name}_{i:03d}" for i in range(len(runs))]
    cases = [(cls.type_name, point, directory, stem) for (_, point), stem in zip(runs, stems)]

    if jobs > 1 and len(cases) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            outcomes = list(pool.map(_run_one, cases))
    else:
        outcomes = [_run_one(job) for job in cases]

    results = []
    for (label, point), stem, outcome in zip(runs, stems, outcomes):
        record = {'name': name, 'run': stem, 'type': cls.type_name, 'sweep': label, 'params': point, 'results': outcome}
        with open(os.path.join(directory, f"{stem}_results.json"), 'w', encoding='utf-8') as file:
            json.dump(record, file, indent=2, ensure_ascii=False, default=_json_default)
            file.write('\n')
        results.append(record)

    if rest['sweep']:
        _write_sweep(os.path.join(directory, f"{name}_sweep.csv"), results)

    return results


def _json_default(value):

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_sweep(path, results):

    sweep_keys = list(results[0]['sweep'])
    result_keys = [key for key, value in results[0]['results'].items() if isinstance(value, (int, float, bool, str))]

    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['run'] + sweep_keys + result_keys)
        for record in results:
            writer.writerow([record['run']] + [record['sweep'][key] for key in sweep_keys]
                            + [record['results'].get(key, '') for key in result_keys])


# ----- types of case (thin layers over Source/Applications.py) -----

def _report(title, params, results):

    from Source.Post_processing.Report import case_report
    return case_report(title, {k: v for k, v in params.items()}, results)


def _isochrones_figure(case):

    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(6.5, 4.2))
    case.plot_isochrones(ax=ax)
    return figure


def _scalars(results):
    return {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in results.items()}


@case_type('slit_die')
class SlitDieCase:

    """Melt in a flat slit die, the half slit by symmetry (Applications.SlitDie, as in examples/Example_Polymer_Slit_Die.py): a parabolic
    profile of the given flow rate at the inlet, no slip at the wall, free outlet; velocity, pressure and temperature together (viscous
    heating on request); the viscosity is the material's law.

    Results: pressure_drop [Pa], T_max and T_outlet_mixed [°C], flow_in and flow_out [m2/s], wall_shear_rate [1/s] (apparent, 3 q / H^2),
    converged, newton_iterations."""

    defaults = {
        'material': {'file': REQUIRED, 'rho': None, 'Cp': None, 'k': None},
        'geometry': {'half_aperture': 1.0e-3, 'length': 20.0e-3, 'nx': 80, 'ny': 16},
        'process': {'flow_rate': REQUIRED, 'T_inlet': 230.0, 'T_wall': None, 'viscous_heating': True},
        'output': {'directory': '.', 'vtk': False, 'report': False},
    }

    def run(self, params, directory, stem):

        from Source.Applications import SlitDie

        m, g, p = params['material'], params['geometry'], params['process']
        die = SlitDie(m['file'], half_aperture=g['half_aperture'], length=g['length'], nx=g['nx'], ny=g['ny'], T_inlet=p['T_inlet'],
                      T_wall=p['T_wall'], viscous_heating=p['viscous_heating'], rho=m['rho'], Cp=m['Cp'], k=m['k'])
        results = die.solve(p['flow_rate'])

        if params['output']['vtk']:
            results['vtk'] = die.model.exportVTK(os.path.join(directory, stem))

        if params['output']['report']:
            report = _report(f"Dado de rendija: {stem}", params, results)
            report.add_model(die.model)
            results['report'] = report.save(os.path.join(directory, f"{stem}_report.html"))

        return results


@case_type('capillary_rheometer')
class CapillaryRheometerCase:

    """Simulated capillary rheometer (Applications.CapillaryRheometer): axisymmetric barrel and capillary, isothermal at T_inlet,
    with the Bagley and Rabinowitsch corrections. The apparent shear rates of process.shear_rates run in increasing order (at least
    three); mesh_scale multiplies the element sizes.

    Results: the arrays gamma_apparent, tau_wall, dp_total, dp_entrance, n_prime, gamma_wall, eta_measured, eta_law, error and
    newton_iterations (lists), and max_abs_error and entrance_loss_fraction_mean."""

    defaults = {
        'material': {'file': REQUIRED},
        'geometry': {'barrel_radius': 4.75e-3, 'barrel_length': 8.0e-3, 'capillary_radius': 0.5e-3, 'capillary_length': 16.0e-3,
                     'mesh_scale': 1.0},
        'process': {'T': 230.0, 'shear_rates': [20.0, 60.0, 200.0, 600.0, 2000.0]},
        'output': {'directory': '.', 'vtk': False, 'report': False},
    }

    def run(self, params, directory, stem):

        from Source.Applications import CapillaryRheometer

        g, p = params['geometry'], params['process']
        rheometer = CapillaryRheometer(params['material']['file'], T=p['T'], mesh_scale=g['mesh_scale'],
                                       **{k: g[k] for k in ('barrel_radius', 'barrel_length', 'capillary_radius', 'capillary_length')})
        table = rheometer.measure(p['shear_rates'])

        results = _scalars(table)
        results['max_abs_error'] = float(np.max(np.abs(table['error'])))
        results['entrance_loss_fraction_mean'] = float(np.mean(table['dp_entrance'] / table['dp_total']))

        if params['output']['vtk']:
            results['vtk'] = rheometer.model.exportVTK(os.path.join(directory, stem))

        if params['output']['report']:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            report = _report(f"Reometro capilar: {stem}", params, results)
            figure, ax = plt.subplots(figsize=(6.5, 3.6))
            ax.loglog(table['gamma_wall'], table['eta_law'], 'k-', label='ley del material')
            ax.loglog(table['gamma_apparent'], table['tau_wall'] / table['gamma_apparent'], 's', mfc='none', label='aparente')
            ax.loglog(table['gamma_wall'], table['eta_measured'], 'o', label='corregida (Rabinowitsch)')
            ax.set_xlabel('tasa de corte [1/s]'); ax.set_ylabel('viscosidad [Pa s]'); ax.legend(); ax.grid(True, which='both', alpha=0.3)
            report.add_figure('Curva de viscosidad medida', figure)
            report.add_model(rheometer.model, fields=['p', 'viscosity'])
            results['report'] = report.save(os.path.join(directory, f"{stem}_report.html"))

        return results


@case_type('injection')
class InjectionCaseCase:

    """Filling of a rectangular plate with the Hele-Shaw model (Applications.InjectionCase.plate): a gate at the middle of the edge
    x = 0, constant flow rate (volume / fill_time), with the temperature across the gap if process.thermal. output.plot writes the
    isochrones as <name>_isochrones.png.

    Results: fill_time [s], gate_pressure_peak and gate_pressure_end [Pa], clamp_force_peak [N], short_shot, filled_fraction,
    frozen_fraction_mean (thermal), steps."""

    defaults = {
        'material': {'file': REQUIRED, 'rho': None, 'Cp': None, 'k': None},
        'geometry': {'length': 0.10, 'width': 0.06, 'thickness': 2.0e-3, 'gate_width': 0.01, 'element_size': 2.5e-3},
        'process': {'fill_time': REQUIRED, 'T_melt': 230.0, 'T_mold': 50.0, 'thermal': True, 'htc': 5000.0, 'batch': 0.1},
        'output': {'directory': '.', 'plot': False, 'report': False},
    }

    def run(self, params, directory, stem):

        from Source.Applications import InjectionCase

        m, g, p = params['material'], params['geometry'], params['process']
        case = InjectionCase.plate(m['file'], fill_time=p['fill_time'], T_melt=p['T_melt'], T_mold=p['T_mold'], thermal=p['thermal'],
                                   htc=p['htc'], batch=p['batch'], rho=m['rho'], Cp=m['Cp'], k=m['k'], **g)
        results = case.run()

        if params['output']['plot']:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            ax = case.plot_isochrones()
            path = os.path.join(directory, f"{stem}_isochrones.png")
            ax.figure.savefig(path, dpi=120, bbox_inches='tight')
            plt.close(ax.figure)
            results['plot'] = path

        if params['output']['report']:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            report = _report(f"Llenado: {stem}", params, results)
            report.add_figure('Isocronas del llenado', _isochrones_figure(case))

            history = np.array(case.filling.history)
            figure, (left, right) = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
            left.plot(history[:, 0], history[:, 2] / 1e6); left.set_xlabel('t [s]'); left.set_ylabel('presion de entrada [MPa]')
            clamp = np.array(case.clamp_history)
            right.plot(clamp[:, 0], clamp[:, 1] / 1e3); right.set_xlabel('t [s]'); right.set_ylabel('fuerza de cierre [kN]')
            for ax in (left, right):
                ax.grid(alpha=0.3)
            report.add_figure('Presion y fuerza de cierre', figure)
            results['report'] = report.save(os.path.join(directory, f"{stem}_report.html"))

        return results
