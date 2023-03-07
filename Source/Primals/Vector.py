import numpy as np
from Source.Pre_processing.BasisFunctions import basisFunctions

class vectorField:

    def __init__(self, name=None, desc=None, unit = None, basisFunction=None, value: list=None):
        self.name = name
        self.desc = desc
        self.unit = unit
        self.bf = basisFunction
        self.value = value



    # +++++++++++++++++++++++++++++++++++++++++
    # Overloading operators
    # +++++++++++++++++++++++++++++++++++++++++

    def __add__(self, other):
        if self.unit == other.unit:
            return vectorField(name = self.name, desc=self.desc)
