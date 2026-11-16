# Class Material -> Store material properties
#
# A material can also be read from a JSON file or from the library (phase E.5 of docs/PLAN_RENDIMIENTO_GPU_ML.md):
#
#     mat = material.from_file('PP505P')            # a name of the library, or the path of a .json file
#     mat.rho, mat.k, mat.Cp                        # scalar properties (SI, temperatures in °C as in the laws)
#     mat.viscosity_law, mat.pvt, mat.kinetics      # the laws, built from their sections (None if the file has no section)
#     mat.save('mi_material.json')                  # writes the same format
#     material.library()                            # names available
#
# Format of a file (all sections but "name" are optional):
#
#     {"name": "...", "source": "where the numbers come from",
#      "properties": {"rho": 970.0, "k": 0.22, "Cp": 2000.0},
#      "viscosity": {"model": "CrossWLF", "n": 0.26, "tau_star": 1.8e4, "D1": 4.4e17, "D2": 266.7, "A1": 39.2, "A2": 51.6},
#      "pvt": {"model": "TaitPVT", "b1m": ..., ..., "b9": ...},
#      "kinetics": {"model": "NakamuraHoffman", "ln_inv_t_half0": ..., "Kg": ..., "U_star": ..., "T_inf": ..., "Tm0": ..., "n": 3.0}}
#
# "model" is the name of the class in Source/Physics (Rheology: Newtonian, PowerLaw, CarreauYasuda, Cross, CrossWLF, Bingham,
# HerschelBulkley, Arrhenius, Exponential; TaitPVT; Crystallization: NakamuraHoffman) and the other keys are its arguments.
# Arrhenius and Exponential take a nested viscosity law under "law". The library is Source/data/materials plus the folders
# of the environment variable FEMSIM_MATERIALS (separated by os.pathsep), which take precedence.

import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'materials')
ENV_VARIABLE = 'FEMSIM_MATERIALS'

# section -> (attribute of the material, {model name: (module, class)})
_SECTIONS = {
    'viscosity': ('viscosity_law', {name: ('Source.Physics.Rheology', name) for name in
                                    ('Newtonian', 'PowerLaw', 'CarreauYasuda', 'Cross', 'CrossWLF', 'Bingham', 'HerschelBulkley',
                                     'Arrhenius', 'Exponential')}),
    'pvt': ('pvt', {'TaitPVT': ('Source.Physics.Rheology', 'TaitPVT')}),
    'kinetics': ('kinetics', {'NakamuraHoffman': ('Source.Physics.Crystallization', 'NakamuraHoffman')}),
}
_NESTED_LAW = ('Arrhenius', 'Exponential')           # their argument "law" is itself a viscosity section


def _build(section, spec, where):

    """Law of the section `section` from its dictionary (argument errors name the file and the section)."""

    import importlib

    attribute, models = _SECTIONS[section]

    if not isinstance(spec, dict) or 'model' not in spec:
        raise ValueError(f"{where}: la seccion {section!r} debe ser un objeto con la clave 'model' (modelos: {sorted(models)}).")

    name = spec['model']
    if name not in models:
        raise ValueError(f"{where}: modelo {name!r} desconocido en {section!r} (modelos: {sorted(models)}).")

    arguments = {key: value for key, value in spec.items() if key != 'model'}
    if name in _NESTED_LAW:
        if 'law' not in arguments:
            raise ValueError(f"{where}: {name} necesita la clave 'law' con la ley de viscosidad que modifica.")
        arguments['law'] = _build('viscosity', arguments['law'], where + ' (law)')

    module, class_name = models[name]
    constructor = getattr(importlib.import_module(module), class_name)

    try:
        return constructor(**arguments)
    except TypeError as error:
        raise ValueError(f"{where}: argumentos invalidos para {name} en {section!r}: {error}") from None


def library_folders():

    """Folders searched by from_file (the ones of FEMSIM_MATERIALS first, then the library of the package)."""

    extra = [folder for folder in os.environ.get(ENV_VARIABLE, '').split(os.pathsep) if folder]
    return extra + [DATA_DIR]


class material:

    def __init__(self, name, **kwargs):

        """

        Parameters
        ------------

        name:       Name of material

        ++kwargs:

        k       :       Thermal conductivity [W m-1 K-1]
        miu     :       Dynamic viscosity [kg s-1 m-1]
        rho     :       Density [kg m-3]
        ...

        """

        self.name = name
        self.__dict__.update(kwargs)

    # ----- files and library -----

    @classmethod
    def library(cls):

        """Names of the materials of the library (file names without .json), sorted."""

        names = set()
        for folder in library_folders():
            if os.path.isdir(folder):
                names.update(os.path.splitext(f)[0] for f in os.listdir(folder) if f.lower().endswith('.json'))
        return sorted(names)

    @classmethod
    def _find(cls, source):

        source = os.fspath(source)

        if os.path.isfile(source):
            return source

        stem = source if not source.lower().endswith('.json') else source[:-5]
        for folder in library_folders():
            candidate = os.path.join(folder, stem + '.json')
            if os.path.isfile(candidate):
                return candidate

        raise FileNotFoundError(f"Material {source!r}: no es un archivo ni esta en la biblioteca (disponibles: {cls.library()}).")

    @classmethod
    def from_file(cls, source):

        """Material of a JSON file (path) or of the library (name). See the header of this file for the format."""

        path = cls._find(source)

        with open(path, encoding='utf-8') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}: JSON invalido ({error}).") from None

        return cls.from_dict(data, where=path)

    @classmethod
    def from_dict(cls, data, where='material'):

        """Material of a dictionary with the format of the files."""

        if not isinstance(data, dict) or 'name' not in data:
            raise ValueError(f"{where}: el material necesita la clave 'name'.")

        known = {'name', 'source', 'properties'} | set(_SECTIONS)
        unknown = sorted(set(data) - known - {key for key in data if key.startswith('_')})
        if unknown:
            raise ValueError(f"{where}: claves desconocidas {unknown} (validas: {sorted(known)}).")

        properties = data.get('properties', {})
        for key, value in properties.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{where}: la propiedad {key!r} debe ser un numero, no {value!r}.")

        mat = cls(data['name'], **properties)
        mat.properties = dict(properties)
        mat.source = data.get('source')

        for section, (attribute, _) in _SECTIONS.items():
            setattr(mat, attribute, _build(section, data[section], where) if section in data else None)

        mat._spec = {key: value for key, value in data.items()}
        return mat

    def to_dict(self):

        """Dictionary with the format of the files (what the material was read from, with the current scalar properties)."""

        if not hasattr(self, '_spec'):
            reserved = {'name', 'properties', 'source', '_spec', 'viscosity_law', 'pvt', 'kinetics'}
            properties = {k: v for k, v in self.__dict__.items() if k not in reserved and isinstance(v, (int, float))
                          and not isinstance(v, bool)}
            return {'name': self.name, 'properties': properties}

        data = dict(self._spec)
        data['properties'] = {key: getattr(self, key) for key in self.properties}
        return data

    def save(self, path):

        """Writes the material as a JSON file."""

        with open(path, 'w', encoding='utf-8') as file:
            json.dump(self.to_dict(), file, indent=2, ensure_ascii=False)
            file.write('\n')
