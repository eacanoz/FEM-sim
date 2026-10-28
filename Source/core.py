# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++

# Main Class for FEA model

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from Source.Pre_processing.Mesh import Mesh
from Source.Material import material
from Source.Pre_processing.BasisFunctions import basisFunctions
import Source.Simulation.Solvers as Solution
from Source.enums import ShapeFunctionType, StudyType, ProblemType, SolverType, DirectSolvers, Preconditioners
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
        self.w = basisFunctions(self._mesh, ShapeFunctionType.linear) # Test Function
        self.w_shape = ShapeFunctionType.linear # Test Function
        self.mat = mat  # Material domain
        self.physics = psc(self)  # Model physics

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

        """Vectorized System for the physics that build one, or None for the element-based path."""

        if not getattr(self.physics, 'supports_vectorized_assembly', False):
            return None

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

        """Assembles the global system of equations for the model based on the physics and mesh.

        Parameters
        -----------
        :param solverOptions: Dictionary containing solver configuration options.
        """

        system = self._system()
        if system is not None:
            return system.linear(self._state())

        Var = self.physics.getVariables()
        nVar = self.physics.getNumOfVar()
        nNodes = self._mesh.getNoN()
        nDOF = nNodes*nVar

        rows_A = []
        cols_A = []
        values_A = []

        b = np.zeros(nDOF)

        for idxVar, Variable in enumerate(Var):
            for element in self._mesh.EL:
                K_e = self.physics.getElementMatrix(element, Variable, solverOptions)
                f_e = self.physics.getElementVector(element, Variable, solverOptions)

                g_indices = [idxVar + nVar * node.id for node in element.nodes]

                for i, g_i in enumerate(g_indices):

                    node_i = element.nodes[i]
                    bc = node_i.BC.get(Variable, {}) if node_i.BC else {}

                    if bc.get('type') == 'Dirichlet':
                        rows_A.append(g_i)
                        cols_A.append(g_i)
                        values_A.append(1.0)
                        b[g_i] = bc.get('value', 0.0)
                    else:
                        b[g_i] += f_e[i]

                        for j, g_j in enumerate(g_indices):
                            rows_A.append(g_i)
                            cols_A.append(g_j)
                            values_A.append(K_e[i, j])


        A = sparse.coo_matrix((values_A, (rows_A, cols_A)), shape=(nDOF, nDOF)).tocsr()

        return A, b

    def assembleGlobalSystemNonLinear(self, solverOptions = None):

        """Assembles the global system of equations for a nonlinear problem, based on the physics and mesh.
        
        Parameters
        -----------
        :param solverOptions: Dictionary containing solver configuration options.
       
        """

        system = self._system()
        if system is not None:
            return system.residual_and_tangent(self._state())

        Var = self.physics.getVariables()
        nVar = self.physics.getNumOfVar()
        nNodes = self._mesh.getNoN()
        nDOF = nNodes*nVar

        rows_K = []
        cols_K = []
        values_K = []

        self.physics.normalize_constants()

        F = np.zeros(nDOF)

        for element in self._mesh.EL:

            nodes_id = element.getNodesId()

            for idxVar, Variable in enumerate(Var):
                K_e = self.physics.getElementTangentMatrix(element, Variable, solverOptions)
                f_e = self.physics.getResidualVector(element, Variable, solverOptions)

                x_e = self.physics.get_variable_element_values(element, Variable)

                for i_local, i_global in enumerate(nodes_id):

                    dof_r = i_global*nVar + idxVar
                    node_i = element.nodes[i_local]
                    bc = node_i.BC.get(Variable, {}) if node_i.BC else {}

                    if bc.get('type') == 'Dirichlet':

                        rows_K.append(dof_r)
                        cols_K.append(dof_r)
                        values_K.append(1.0)
                        F[dof_r] += (x_e[i_local] - bc.get('value', 0.0))
                    else:
                        F[dof_r] += f_e[i_local]

                        for jdxVar, Variable in enumerate(Var):
                            K_block = np.asarray(K_e[Variable])

                            for j_local, j_global in enumerate(nodes_id):
                                dof_c = j_global*nVar + jdxVar
                                rows_K.append(dof_r)
                                cols_K.append(dof_c)
                                values_K.append(K_block[i_local, j_local])

        K = sparse.coo_matrix((values_K, (rows_K, cols_K)), shape=(nDOF, nDOF)).tocsr()

        return K, F

    def assembleMassMatrix(self, solverOptions = None):

        """Assembles the global mass matrix for the model based on the physics and mesh.
        
        Parameters
        -----------
        :param solverOptions: Dictionary containing solver configuration options.
               
        """

        system = self._system()
        if system is not None:
            return system.mass()

        M = sparse.lil_matrix((self._mesh.getNoN(), self._mesh.getNoN()))

        for element in self._mesh.EL:

            """Compute the element mass matrix and add its contributions to the global mass matrix."""

            M_e = self.physics.getElementMassMatrix(element, solverOptions)

            for idx, node_i in enumerate(element.nodes):
                for jdx, node_j in enumerate(element.nodes):

                    M[node_i.id, node_j.id] += M_e[idx, jdx]

        return M

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

    def postProcess2D(self, variable=None, levels=30):

        """Filled contour of the nodal solution on a 2D mesh (each Q4 is split into two triangles)."""

        import matplotlib.tri as mtri

        variable = variable or next(iter(self.physics.var))

        xy = np.array([node.coor[:2] for node in self._mesh.NL], dtype=float)

        triangles = []
        for element in self._mesh.EL:
            a, b, c, d = element.getNodesId()
            triangles += [[a, b, c], [a, c, d]]

        triang = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles)

        fig, ax = plt.subplots()
        contour = ax.tricontourf(triang, self.physics.var[variable].values, levels=levels)
        fig.colorbar(contour, ax=ax, label=variable)
        ax.set_aspect('equal')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        plt.show()



#@jit


