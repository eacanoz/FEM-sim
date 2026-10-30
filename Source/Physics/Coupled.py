"""Several physics solved together in one model.

CoupledPhysics gathers physics (heat transfer, species transport, ...) that share the mesh. Their
fields live in one DofLayout and their terms are added to one System, so any term can depend on
fields of the other physics (a reaction rate on the temperature, a heat source on a concentration).

    model = Model(name='reactor', dim=1, mesh=mesh, mat=material(...), psc=[ht, mt])
    model.physics.addBC_Temperature(0, 25.0)            # delegated to the heat part
    model.physics.addBC_Concentration(0, 'A', 1.0)      # delegated to the mass part
    model.physics.heat.Convection = True                # attributes are set on the part

Methods and attributes that the combined physics does not define are looked up in its parts, in
order. Assigning an attribute that belongs to a part is refused: it must be set on the part
(model.physics.heat, model.physics.mass), otherwise the value would never reach the physics.
"""

from Source.Physics.Physics import physics


class CoupledPhysics(physics):

    _own = ('parts', 'modelRef', 'mat')

    def __init__(self, model, parts):

        self.modelRef = model
        self.mat = model.mat

        self.parts = {}

        for cls in parts:

            name = getattr(cls, 'part_name', cls.__name__)

            if name in self.parts:
                raise ValueError(f"La fisica '{name}' aparece dos veces.")

            self.parts[name] = cls(model)

    @property
    def physics_description(self):

        """Descriptions of the parts, read live (a part may change it, e.g. ff.setInertia)."""

        return ' + '.join(p.physics_description for p in self.parts.values())

    # ----- fields and conditions of all the parts -----

    @property
    def var(self):

        merged = {}

        for part in self.parts.values():
            for variable, field in part.var.items():

                if variable in merged:
                    raise ValueError(f"La variable '{variable}' esta definida en dos fisicas.")

                merged[variable] = field

        return merged

    @property
    def facet_bcs(self):
        return [bc for part in self.parts.values() for bc in part.facet_bcs]

    def initField(self, variable, value):

        for part in self.parts.values():
            if variable in part.var:
                part.initField(variable, value)
                return

        raise KeyError(f"La variable '{variable}' no existe en ninguna fisica del modelo.")

    def add_terms(self, system, model, disc, layout):

        for part in self.parts.values():
            part.add_terms(system, model, disc, layout)

    def derived_fields(self, model):

        merged = {}

        for part in self.parts.values():
            merged.update(part.derived_fields(model))

        return merged

    def setBCTime(self, field, edges, f):

        """physics.setBCTime on the part that owns the field."""

        for part in self.parts.values():
            if field in part.var:
                return part.setBCTime(field, edges, f)

        raise KeyError(f"La variable '{field}' no existe en ninguna fisica del modelo.")

    def setVelocity(self, v):

        """One velocity for every transported part: they all convect with the same field. A flow physics (whose
        velocity is an unknown) is skipped, so model.physics.setVelocity('u') couples heat and species with it."""

        for part in self.parts.values():
            if not getattr(part, 'computes_velocity', False):
                part.setVelocity(v)

    def setStabilization(self, method='SUPG'):

        """Convection stabilization of the transported parts (the flow physics keeps its own, set on the part)."""

        for part in self.parts.values():
            if not getattr(part, 'computes_velocity', False):
                part.setStabilization(method)

    # ----- delegation to the parts -----

    def __getattr__(self, name):

        # Only called when the normal lookup fails. Avoid recursion while `parts` does not exist yet.
        if name.startswith('__') or name == 'parts':
            raise AttributeError(name)

        if name in self.parts:
            return self.parts[name]

        for part in self.parts.values():
            if hasattr(part, name):
                return getattr(part, name)

        raise AttributeError(f"'{type(self).__name__}' no tiene el atributo '{name}'.")

    def __setattr__(self, name, value):

        parts = self.__dict__.get('parts')

        if parts and name not in self._own:
            for part_name, part in parts.items():
                if hasattr(part, name):
                    raise AttributeError(f"Asigne '{name}' en la parte: model.physics.{part_name}.{name} = ...")

        super().__setattr__(name, value)
