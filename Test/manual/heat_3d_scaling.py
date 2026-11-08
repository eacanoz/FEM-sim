"""Time and memory of the 3D heat equation (Poisson in the unit cube) as the mesh grows: mesh, Discretization, solve.

Measures what limits the 3D runs before optimizing (docs/PLAN_3D.md, phase 1). Run from the project root:
    python Test/manual/heat_3d_scaling.py [element] [n1 n2 ...]
"""

import os
import sys
import threading
import time

import numpy as np
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers

KIND = {'tet4': (ElementType.tetrahedron, 1), 'tet10': (ElementType.tetrahedron, 2),
        'hex8': (ElementType.hexahedron, 1), 'hex27': (ElementType.hexahedron, 2)}


class PeakMemory:

    """Peak resident memory of this process while the block runs (sampled every 20 ms)."""

    def __enter__(self):
        self.process, self.peak, self.running = psutil.Process(), 0, True
        self.base = self.process.memory_info().rss
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while self.running:
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(0.02)

    def __exit__(self, *exc):
        self.running = False
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def run(name, n):
    exact = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)

    with PeakMemory() as memory:
        t0 = time.perf_counter()
        mesh = Mesh()
        mesh.generate3DMesh((1.0, 1.0, 1.0), (n, n, n), *KIND[name])
        faces = mesh.boundary_faces()
        t1 = time.perf_counter()
        model = Model(name='scaling', dim=3, mesh=mesh, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
        model.physics.Convection = False
        model.physics.F_const = lambda x, y, z: 3 * np.pi ** 2 * exact(x, y, z)
        model.physics.addBC_TemperatureEdges(faces, 0.0)
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear, Method=SolverType.direct,
                                  Solver=DirectSolvers.PARDISO)
        model.solve()
        t2 = time.perf_counter()

    xyz = np.array([node.coor for node in mesh.NL])
    error = np.max(np.abs(np.asarray(model.sol['T']) - exact(*xyz.T)))
    print(f"{name:6s} n = {n:3d}: {mesh.getNoE():8d} elements {mesh.getNoN():8d} nodes | mesh + faces {t1 - t0:6.1f} s, "
          f"model + solve {t2 - t1:6.1f} s | peak memory +{(memory.peak - memory.base) / 2 ** 30:5.2f} GB | "
          f"nodal error {error:.1e}", flush=True)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else 'tet10'
    sizes = [int(a) for a in sys.argv[2:]] or [8, 12, 16, 20]
    for n in sizes:
        run(name, n)
