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

Keys of [sweep] are the names of keys of any section (they must be unambiguous) or the full "section.key". A sweep writes
<name>_sweep.csv; every run writes its own <name>[_i]_results.json with the resolved case and its results.

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


# ----- types of case -----

@case_type('slit_die')
class SlitDie:

    """Melt in a flat slit die (the half of the slit by symmetry, as in examples/Example_Polymer_Slit_Die.py): a parabolic
    profile of the given flow rate at the inlet, no slip at the wall, free outlet. Velocity, pressure and temperature are solved
    together (viscous heating on request); the viscosity is the material's law (a material with a 'viscosity' section).

    Results: pressure_drop [Pa] (mean inlet pressure, outlet at 0), T_max and T_outlet_mixed [°C], flow_in and flow_out [m2/s],
    wall_shear_rate [1/s] (the apparent, Newtonian one: 3 q / H^2 for the flow rate q of the half slit), converged, newton_iterations."""

    defaults = {
        'material': {'file': REQUIRED, 'rho': None, 'Cp': None, 'k': None},
        'geometry': {'half_aperture': 1.0e-3, 'length': 20.0e-3, 'nx': 80, 'ny': 16},
        'process': {'flow_rate': REQUIRED, 'T_inlet': 230.0, 'T_wall': None, 'viscous_heating': True},
        'output': {'directory': '.', 'vtk': False},
    }

    def run(self, params, directory, stem):

        from Source.Pre_processing.Mesh import Mesh
        from Source.core import Model
        from Source.Material import material
        from Source.Physics.FluidFlow import ff
        from Source.Physics.HeatTransfer import ht
        from Source.enums import ElementType

        m, g, p = params['material'], params['geometry'], params['process']
        mat = material.from_file(m['file'])
        if getattr(mat, 'viscosity_law', None) is None:
            raise ValueError(f"material {m['file']!r}: sin seccion 'viscosity'; el dado necesita una ley de viscosidad.")

        thermal = {}
        for key in ('rho', 'Cp', 'k'):
            value = m[key] if m[key] is not None else getattr(mat, key, None)
            if value is None:
                raise ValueError(f"material {m['file']!r}: falta la propiedad {key!r} (en el archivo o en [material]).")
            thermal[key] = float(value)

        H, L, q = float(g['half_aperture']), float(g['length']), float(p['flow_rate'])
        T_in = float(p['T_inlet'])
        T_wall = T_in if p['T_wall'] is None else float(p['T_wall'])

        mesh = Mesh()
        mesh.generate2DMesh((L, H), (g['nx'], g['ny']), ElementType.quadrilateral)
        inlet = mesh.boundary_edges(lambda x, y: x < 1e-12 * L)
        outlet = mesh.boundary_edges(lambda x, y: x > L * (1 - 1e-12))
        axis = mesh.boundary_edges(lambda x, y: y < 1e-12 * H)
        wall = mesh.boundary_edges(lambda x, y: y > H * (1 - 1e-12))
        xy = np.array([node.coor for node in mesh.NL])

        mu0 = float(mat.viscosity_law.zero_shear(T_in)) if hasattr(mat.viscosity_law, 'zero_shear') else 1.0
        model = Model(name=stem, dim=2, mesh=mesh, mat=material('melt', miu=mu0, **thermal), psc=[ff, ht])
        fluid, heat = model.physics.fluid, model.physics.heat

        u_max = 1.5 * q / H
        fluid.addBC_VelocityEdges(inlet, lambda x, y: (u_max * (1 - (y / H) ** 2), 0.0))
        fluid.addBC_SymmetryEdges(axis)
        fluid.addBC_NoSlipEdges(wall)
        fluid.addBC_OutletEdges(outlet, 0.0)
        heat.addBC_TemperatureEdges(inlet, T_in)
        heat.addBC_TemperatureEdges(wall, T_wall)
        model.physics.setVelocity('u')
        model.physics.setStabilization('SUPG')
        model.physics.initField('T', T_in)

        model.solverConfiguration(Study='steady', Type='nonlinear', Method='direct')
        model.solve()                                                   # Newtonian (mu0): the starting point
        fluid.setViscosityLaw(mat.viscosity_law)
        fluid.setViscousHeating(bool(p['viscous_heating']))
        model.solve()

        nodes = np.unique(np.ravel(inlet))
        outlet_column = np.abs(xy[:, 0] - L) < 1e-9 * L
        order = np.argsort(xy[outlet_column, 1])
        y = xy[outlet_column, 1][order]
        u_out, T_out = model.sol['u'][outlet_column, 0][order], model.sol['T'][outlet_column][order]

        results = {
            'pressure_drop': float(model.sol['p'][nodes].mean()),
            'T_max': float(model.sol['T'].max()),
            'T_outlet_mixed': float(np.trapezoid(u_out * T_out, y) / np.trapezoid(u_out, y)),
            'flow_in': float(-fluid.flowRate(inlet)),
            'flow_out': float(fluid.flowRate(outlet)),
            'wall_shear_rate': float(3.0 * q / H ** 2),
            'converged': bool(model.nonlinear_converged),
            'newton_iterations': int(model.newton_iterations),
        }

        if params['output']['vtk']:
            results['vtk'] = model.exportVTK(os.path.join(directory, stem))

        return results
