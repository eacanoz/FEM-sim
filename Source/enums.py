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


class DirectSolvers(Enum):
    PARDISO = 'PARDISO'
    UMFPACK = 'UMFPACK'
    MUMPS = 'MUMPS'
    SuperLU = 'SuperLU'

class IterativeSolvers(Enum):
    CG = 'Conjugate Gradient'
    BiCGSTAB = 'Bi-Conjugate Gradient Stabilized'
    GMRES = 'Generalized Minimal Residual'