# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++

# Main Class for FEA model

import numpy as np
import matplotlib.pyplot as plt

from Source.Pre_processing.Mesh import Mesh
from Source.Material import material
from Source.Physics.Coupled import CoupledPhysics
import Source.Simulation.Solvers as Solution
from Source.enums import StudyType, ProblemType, SolverType, DirectSolvers, Preconditioners
from Source.logger_config import sim_logger
from Source.Simulation.Discretization import Discretization
from Source.Simulation.DofLayout import DofLayout


class Model(object):
    """
    Superclass for all FEA models 
    """

    area = 1

    def __init__(self, name: str, mtype=None, dim: int = 1, mesh: Mesh = None, mat: material = None, psc=None):
        """
        Initializes the FEA model with the given parameters.

        Parameters
        -----------
        :param name:  Name of model
        :param mtype: Type of Model (Not defined)
        :param dim: Problem dimension: 1->1D; 2->2D(!); 3->3D(!)
        :param mesh Mesh (Class: Mesh)
        """

        self._mesh = mesh  # Model mesh
        self._PD = dim  # Model dimension
        self.name = name  # Name of the model
        self.mtype = mtype  # Type of model
        self.mat = mat  # Material domain
        # Model physics: one physics class, or a list of classes solved together (CoupledPhysics)
        self.physics = CoupledPhysics(self, psc) if isinstance(psc, (list, tuple)) else psc(self)

        if self.physics is None:
            self.physics_status = 'Not defined'

        else:
            self.physics_status = self.physics.physics_description

        self.A = None
        self.b = None

        self.solverOptions = None

        self.sol = None

        self.timeVector = None

        self._geometry = None      # Discretization, built on first use by the vectorized path
        self._layout = None        # DofLayout of the physics fields
        self._system_cache = None  # System, rebuilt at the start of every solve()

    @property
    def mesh(self):

        """Returns the mesh of the model."""
        return self._mesh

    def add_material(self):
        pass

    def set_physics(self, physics):
        pass

    def __str__(self):

        """String representation of the model."""


        custom_str = ("Model: " + self.name + 
                      "\nDimension: " + str(self._PD) + "D" + 
                      "\nNodes: " + str(self.mesh.getNoN()) + 
                      "\nElements: " + str(self.mesh.getNoE()) + 
                      "\nPhysics: " + self.physics_status)

        return custom_str

    @property
    def layout(self):

        """Degrees of freedom of the model (fields and components)."""

        if self._layout is None:
            self._layout = DofLayout(self.physics.fields(), self._mesh.getNoN())

        return self._layout

    def _system(self):

        """System of the model (built by the physics, once per solve)."""

        if self._system_cache is None:

            if self._geometry is None:
                self._geometry = Discretization(self._mesh)

            self._system_cache = self.physics.build_system(self, self._geometry, self.layout)

        return self._system_cache

    def dirichletDofs(self):

        """Boolean mask over all dofs: True where a field has a Dirichlet condition."""

        layout = self.layout
        mask = np.zeros(layout.ndof, dtype=bool)

        for node in self._mesh.NL:
            if not node.BC:
                continue

            for name in layout.names:
                if node.BC.get(name, {}).get('type') == 'Dirichlet':
                    mask[layout.node_dofs(name, node.id)] = True

        return mask

    def _state(self):

        """Solution vector with the current values of every field."""

        return self.layout.join({name: self.physics.var[name].values for name in self.layout.names})

    def assembleGlobalSystem(self, solverOptions = None):

        """Linear system A x = b of the model (Newton-linearized at the current state if the physics is nonlinear)."""

        return self._system().linear(self._state())

    def assembleGlobalSystemNonLinear(self, solverOptions = None):

        """Jacobian and residual of the model at the current state, for Newton-Raphson."""

        return self._system().residual_and_tangent(self._state())

    def assembleMassMatrix(self, solverOptions = None):

        """Global mass matrix (transient studies)."""

        return self._system().mass()

    def solverConfiguration(self, Study=StudyType.steady_state, Type=ProblemType.linear, Method=SolverType.direct, Solver = DirectSolvers.PARDISO, 
                            timeDisc = 1, timeStep=0.05, totalTime = 3, prec = Preconditioners.iLU):
        
        """Configures the solver for the model based on the provided options."""

        self.solverOptions = {
            'Study': Study,
            'Type': Type,
            'Method': Method,
            'Solver': Solver,
            'Preconditioner': prec,
            'Time Discretization': timeDisc,
            'Time Step': timeStep,
            'Time': totalTime
        }


    def solve(self):

        """Solves the model based on the configured solver options."""

        self._system_cache = None  # boundary conditions and coefficients may have changed

        if self.solverOptions is None:
            self.solverConfiguration()

        sim_logger.info("------- Starting simulation -------")
        sim_logger.info("Model: " + self.name)
        sim_logger.info("Dimension: " + str(self._PD) + "D")
        sim_logger.info("Nodes: " + str(self.mesh.getNoN()))
        sim_logger.info("Elements: " + str(self.mesh.getNoE()))
        sim_logger.info("Physics: " + self.physics_status)

        sim_logger.info("------- Solver configuration -------")
        sim_logger.info("Study: " + str(self.solverOptions['Study'].value))
        sim_logger.info("Type: " + str(self.solverOptions['Type'].value))
        sim_logger.info("Method: " + str(self.solverOptions['Method'].value))
        sim_logger.info("Solver: " + str(self.solverOptions['Solver'].value))

        if self.solverOptions['Method'] == SolverType.iterative:
            sim_logger.info("Preconditioner: " + str(self.solverOptions['Preconditioner'].value))

        if self.solverOptions['Study'] == StudyType.transient:
            sim_logger.info("Time Discretization: " + str(self.solverOptions['Time Discretization']))
            sim_logger.info("Time Step: " + str(self.solverOptions['Time Step']))
            sim_logger.info("Total Time: " + str(self.solverOptions['Time']))


        solver = Solution.modelSolver(self)

        solver.solve()

        self.timeVector = solver.timeVector

        sim_logger.info("------- Simulation finished -------")


    def postProcess(self):

        plt.plot(self.mesh.getXCoor(), self.sol['T'], marker='o')
        plt.xlabel("x-axis [m]")
        plt.ylabel("Temperature[°C]")
        plt.show()

    def postProcess2D(self, variable=None, levels=30, step=None, ax=None, vmin=None, vmax=None):

        """Filled contour of the nodal solution on a 2D mesh (each Q4 is split into two triangles).

        step: index of a time step of a transient solution (default: the final state). ax: draw on an
        existing axes instead of opening a figure. vmin, vmax: fixed color range (to compare steps).
        """

        import matplotlib.tri as mtri

        variable = variable or next(iter(self.physics.var))

        xy = np.array([node.coor[:2] for node in self._mesh.NL], dtype=float)

        triangles = []
        for element in self._mesh.EL:
            a, b, c, d = element.getNodesId()
            triangles += [[a, b, c], [a, c, d]]

        triang = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles)

        values = self.physics.var[variable].values if step is None else self.sol[variable][:, step]

        own_figure = ax is None
        if own_figure:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if vmin is not None and vmax is not None:
            levels = np.linspace(vmin, vmax, levels)

        contour = ax.tricontourf(triang, values, levels=levels, vmin=vmin, vmax=vmax)
        fig.colorbar(contour, ax=ax, label=variable)
        ax.set_aspect('equal')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        if own_figure:
            plt.show()

        return contour



#@jit


