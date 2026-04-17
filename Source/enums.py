from enum import Enum

class ElementType(Enum):
    line = 1
    triangle = 2
    quadrilateral = 3

class ShapeFunctionType(Enum):
    linear = 'Linear'
    quadratic = 'Quadratic'


class SolverType(Enum):
    direct = 'Direct'
    iterative = 'Iterative'

class StudyType(Enum):
    steady_state = 'Steady state'
    transient = 'Transient'

class ProblemType(Enum):
    linear = 'Linear'
    nonlinear = 'Nonlinear'

