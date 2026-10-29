import numpy as np

from Source.enums import StudyType


class vectorField:

    """Nodal field with several components (velocity, displacement): values (n_nodes, ncomp).

    Transient values are stored as (n_nodes, ncomp, n_times).
    """

    def __init__(self, name: str = None, desc: str = None, unit: str = None, basisFunction=None, mesh=None, ncomp: int = 2):
        self.name = name
        self.desc = desc
        self.unit = unit
        self.shape = basisFunction
        self.ncomp = int(ncomp)
        self.values = np.zeros((mesh.getNoN(), self.ncomp))

        self.timeValues = None

    def initField(self, value):

        """A number (every component) or a vector with one value per component."""

        self.values = np.broadcast_to(np.asarray(value, dtype=float), self.values.shape).copy()

    def updateField(self, values):
        self.values = np.asarray(values, dtype=float).reshape(-1, self.ncomp)

    def updateTimeValues(self, values=None):
        self.timeValues = values

    def getFieldValues(self, study):
        if study == StudyType.steady_state:
            return self.values
        elif study == StudyType.transient:
            return self.timeValues
