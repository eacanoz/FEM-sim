"""Compatibility module: the vectorized assembly now lives in Discretization, Operators and System.

LinearGeometry is the old name of Discretization (same attributes: conn, N, B, detJw, x_gp).
"""

from Source.Simulation.Discretization import Discretization as LinearGeometry  # noqa: F401
from Source.Pre_processing.Elements import lagrange_1d  # noqa: F401
