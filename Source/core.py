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

    def __init__(self, name: str, mtype=None, dim: int = 1, mesh: Mesh = None, mat: material = None, psc=None,
                 axisymmetric: bool = False):
        """
        Initializes the FEA model with the given parameters.

        Parameters
        -----------
        :param name:  Name of model
        :param mtype: Type of Model (Not defined)
        :param dim: Problem dimension: 1->1D; 2->2D(!); 3->3D(!)
        :param mesh Mesh (Class: Mesh)
        :param axisymmetric: 2D mesh of the meridian plane of a body of revolution (x = r, y = z, axis at x = 0)
        """

        self.axisymmetric = bool(axisymmetric)

        self._mesh = mesh  # Model mesh
        self._PD = dim  # Model dimension
        self.name = name  # Name of the model
        self.mtype = mtype  # Type of model
        self.mat = mat  # Material domain
        # Model physics: one physics class, or a list of classes solved together (CoupledPhysics)
        self.physics = CoupledPhysics(self, psc) if isinstance(psc, (list, tuple)) else psc(self)

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

    @property
    def physics_status(self):

        """Current description of the physics (read live: setInertia and similar calls change it)."""

        return 'Not defined' if self.physics is None else self.physics.physics_description

    def add_material(self):
        pass

    def set_physics(self, physics):
        pass

    def __str__(self):

        """String representation of the model."""


        custom_str = ("Model: " + self.name + 
                      "\nDimension: " + str(self._PD) + "D" + (" axisymmetric (r, z)" if self.axisymmetric else "") +
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

    @property
    def geometry(self):

        """Discretization of the mesh (shape functions and gradients at the Gauss points), built once."""

        if self._geometry is None:
            self._geometry = Discretization(self._mesh, axisymmetric=self.axisymmetric)

        return self._geometry

    def region_field(self, values, default=None):

        """A coefficient by region of an imported mesh, {'name': value}, for any property given at the Gauss points
        (flow.setViscosity, heat.K_const, ...): (nel, 1), broadcast to the points of each element. A coefficient that jumps
        between regions needs an interface along element edges. default: value of the elements outside the given regions."""

        field = np.full((self._mesh.getNoE(), 1), np.nan if default is None else float(default))

        for name, value in values.items():
            if name not in self._mesh.regions:
                raise KeyError(f"La region '{name}' no existe en la malla (regiones: {sorted(self._mesh.regions)}).")
            field[np.asarray(self._mesh.regions[name], dtype=int)] = float(value)

        if np.any(np.isnan(field)):
            raise ValueError("Hay elementos fuera de las regiones dadas: indique default o el resto de regiones.")

        return field

    def _system(self):

        """System of the model (built by the physics, once per solve)."""

        if self._system_cache is None:
            self._system_cache = self.physics.build_system(self, self.geometry, self.layout)

        return self._system_cache

    def dirichletDofs(self):

        """Boolean mask over all dofs: True where a Dirichlet condition fixes the dof (node conditions, single
        components of a vector field and values fixed by the physics, such as a reference pressure)."""

        return self._system().is_dirichlet.copy()

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
                            timeDisc = 1, timeStep=0.05, totalTime = 3, prec = Preconditioners.iLU, tol = 1e-10, maxiter = None,
                            timeScheme='theta', globalization='newton', cfl0=10.0, relaxation=None, reuseTangent=True, adaptiveTol=None,
                            adaptiveMaxStep=None, nonlinearMaxIter=None):

        """Configures the solver for the model based on the provided options.

        Transient studies: timeScheme 'theta' (timeDisc = theta: 1 implicit Euler, 0.5 Crank-Nicolson) or 'BDF2'
        (second order and L-stable: no oscillating algebraic variables such as the pressure; the first step is
        implicit Euler). Systems with nonlinear terms iterate Newton inside every step, with a modified Newton that keeps
        the tangent and its factorization while the residual falls fast (reuseTangent=False rebuilds it every iteration).
        adaptiveTol (BDF2): adaptive time steps with that local error per step, relative to each field, for example 1e-3;
        timeStep is then the first step, to be chosen small, and timeVector has variable spacing. adaptiveMaxStep limits dt:
        the control only sees what has happened, so a long step can jump over a pulse of a time function that has not started.

        Steady nonlinear studies: globalization 'newton' (line search) or 'pseudo-transient' (local pseudo-time step with
        a CFL that starts at cfl0 and grows as the residual falls, for hard problems or poor initial guesses; it reaches
        the same steady solution). relaxation = {field: factor} takes only that fraction of the update of a field.
        nonlinearMaxIter: limit of the pseudo-transient march (default 500)."""

        if timeScheme not in ('theta', 'BDF2'):
            raise ValueError(f"Esquema temporal desconocido: {timeScheme!r} (use 'theta' o 'BDF2').")

        if adaptiveTol is not None and timeScheme != 'BDF2':
            raise ValueError("El paso adaptativo necesita timeScheme='BDF2'.")

        if globalization not in ('newton', 'pseudo-transient'):
            raise ValueError(f"Globalizacion desconocida: {globalization!r} (use 'newton' o 'pseudo-transient').")

        self.solverOptions = {
            'Study': Study,
            'Type': Type,
            'Method': Method,
            'Solver': Solver,
            'Preconditioner': prec,
            'Time Discretization': timeDisc,
            'Time Scheme': timeScheme,
            'Time Step': timeStep,
            'Time': totalTime,
            'Tolerance': tol,           # relative residual of the iterative solvers
            'Max Iterations': maxiter,  # iterative solvers (None: scipy default)
            'Globalization': globalization,
            'CFL0': cfl0,
            'Relaxation': relaxation,
            'Reuse Tangent': reuseTangent,
            'Adaptive Tolerance': adaptiveTol,
            'Adaptive Max Step': adaptiveMaxStep,
            'Nonlinear Max Iterations': nonlinearMaxIter
        }


    def solve(self):

        """Solves the model based on the configured solver options."""

        self._system_cache = None  # boundary conditions and coefficients may have changed

        if self.solverOptions is None:
            self.solverConfiguration()

        sim_logger.info("------- Starting simulation -------")
        sim_logger.info("Model: " + self.name)
        sim_logger.info("Dimension: " + str(self._PD) + "D" + (" axisymmetric (r, z)" if self.axisymmetric else ""))
        sim_logger.info("Nodes: " + str(self.mesh.getNoN()))
        sim_logger.info("Elements: " + str(self.mesh.getNoE()))
        sim_logger.info("Physics: " + self.physics_status)

        sim_logger.info("------- Solver configuration -------")
        sim_logger.info("Study: " + str(self.solverOptions['Study'].value))
        sim_logger.info("Type: " + str(self.solverOptions['Type'].value))
        sim_logger.info("Method: " + str(self.solverOptions['Method'].value))
        sim_logger.info("Solver: " + str(self.solverOptions['Solver'].value))

        if self.solverOptions['Method'] == SolverType.iterative:
            prec = self.solverOptions['Preconditioner']
            sim_logger.info("Preconditioner: " + (prec.value if prec is not None else "none"))

        if self.solverOptions['Study'] == StudyType.transient:
            sim_logger.info("Time Discretization: " + str(self.solverOptions['Time Discretization']) +
                            ", scheme: " + str(self.solverOptions.get('Time Scheme', 'theta')))
            sim_logger.info("Time Step: " + str(self.solverOptions['Time Step']))
            sim_logger.info("Total Time: " + str(self.solverOptions['Time']))


        solver = Solution.modelSolver(self)

        solver.solve()

        self.timeVector = solver.timeVector
        self.solver_iterations = solver.iterations   # Krylov iterations of every linear solve (empty with direct solvers)
        self.newton_iterations = getattr(solver, 'newton_iterations', None)   # steady nonlinear solves
        self.nonlinear_converged = getattr(solver, 'nonlinear_converged', None)   # steady nonlinear solves
        self.step_iterations = getattr(solver, 'step_iterations', None)       # Newton iterations of each time step
        self.tangent_assemblies = getattr(solver, 'tangent_assemblies', None) # tangents built in a nonlinear transient
        self.rejected_steps = getattr(solver, 'rejected_steps', None)         # steps repeated by the adaptive control

        if self.solverOptions['Study'] == StudyType.steady_state:
            self.sol.update(self.physics.derived_fields(self))

        sim_logger.info("------- Simulation finished -------")


    def exportVTK(self, base, every=1, fields=None):

        """Writes the solution for ParaView: base.vtu (steady) or base_NNNN.vtu + base.pvd (transient). Returns the files."""

        from Source.Post_processing.VTK import export_model

        return export_model(self, base, every=every, fields=fields)

    def postProcess(self):

        plt.plot(self.mesh.getXCoor(), self.sol['T'], marker='o')
        plt.xlabel("x-axis [m]")
        plt.ylabel("Temperature[°C]")
        plt.show()

    def postProcess2D(self, variable=None, levels=30, step=None, ax=None, vmin=None, vmax=None):

        """Filled contour of the nodal solution on a 2D mesh (the triangles of Mesh.plot_triangles; a Q4 is split into two triangles; T3 are drawn as they are).

        step: index of a time step of a transient solution (default: the final state). ax: draw on an
        existing axes instead of opening a figure. vmin, vmax: fixed color range (to compare steps).
        """

        variable = variable or next(iter(self.physics.var))

        triang = self._mesh.triangulation()

        values = self.physics.var[variable].values if step is None else np.asarray(self.sol[variable])[..., step]
        if np.ndim(values) == 2:
            values = np.linalg.norm(values, axis=1)   # vector field: magnitude

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


