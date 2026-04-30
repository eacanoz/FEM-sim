# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++++

# Main Class for FEA model

import numpy as np
import sympy as sp
import scipy as sc
import matplotlib.pyplot as plt
from itertools import product

from Source.Pre_processing.Mesh import Mesh, Element, Node
from Source.Physics.Physics import physics
from Source.Material import material
from Source.Pre_processing.BasisFunctions import basisFunctions
import Source.Simulation.Solvers as Solution
from Source.enums import ElementType, ShapeFunctionType, StudyType, ProblemType, SolverType, DirectSolvers, IterativeSolvers, Preconditioners
from Source.logger_config import sim_logger



from progress.bar import Bar
from joblib import Parallel, delayed

import jax

from numba import jit
# Model


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

    def assembleGlobalSystem(self, solverOptions = None):

        """Assembles the global system of equations for the model based on the physics and mesh.

        Parameters
        -----------
        :param solverOptions: Dictionary containing solver configuration options.
        """

        #print('-------Assembling global system-------')

        Var = self.physics.getVariables()
        nVar = self.physics.getNumOfVar()
        nNodes = self._mesh.getNoN()
        nDOF = nNodes*nVar

        # Preallocate lists for local contributions
        rows_A = []
        cols_A = []
        values_A = []


        b = np.zeros(nDOF)

        for idxVar, Variable in enumerate(Var):
            for element in self._mesh.EL:
                K_e = self.physics.getElementMatrix(element, Variable, solverOptions)
                f_e = self.physics.getElementVector(element, Variable, solverOptions)

                #r, c, v, rhs = self._element_contributions(element, idxVar, Variable, K_e, f_e)

                #rows_A.extend(r)
                #cols_A.extend(c)
                #values_A.extend(v)

                #for gi, value in rhs:
                #    b[gi] = value  # Dirichlet sobrescribe

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


        A = sc.sparse.coo_matrix((values_A, (rows_A, cols_A)), shape=(nDOF, nDOF)).tocsr()

        return A, b

    def assembleResidualVector(self, solverOptions = None):

        """Assembles the global residual vector for the model based on the physics and mesh.
        
        Parameters
        -----------
        :param solverOptions: Dictionary containing solver configuration options.
       
        """

        Var = self.physics.getVariables()
        nVar = self.physics.getNumOfVar()
        nNodes = self._mesh.getNoN()
   
        nDOF = nNodes*nVar
        results = []

        penalty = 1e15
        

        self.physics.normalize_constants()

        #rows_F = []
        #values_F = []


        F = np.zeros(nDOF)

        for element in self._mesh.EL:

            nodes_id = element.getNodesId()

            for idxVar, Variable in enumerate(Var):
                
                f_e = self.physics.getResidualVector(element, Variable, solverOptions)

                x_e = self.physics.get_variable_element_values(element, Variable)

                for i_local, i_global in enumerate(nodes_id):

                    dof_r = i_global*nVar + idxVar
                    node_i = element.nodes[i_local]
                    bc = node_i.BC.get(Variable, {}) if node_i.BC else {}

                    if bc.get('type') == 'Dirichlet':

                        F[dof_r] += (x_e[i_local] - bc.get('value', 0.0))
                        #F[g_i] = bc.get('value', 0.0)
                    else:
                        F[dof_r] += f_e[i_local]

        return F
    
    def assembleGlobalSystemNonLinear(self, solverOptions = None):

        """Assembles the global system of equations for a nonlinear problem, based on the physics and mesh.
        
        Parameters
        -----------
        :param solverOptions: Dictionary containing solver configuration options.
       
        """

        Var = self.physics.getVariables()
        nVar = self.physics.getNumOfVar()
        nNodes = self._mesh.getNoN()
        nDOF = nNodes*nVar
        #results_K = []
        #results_F = []      

        #K = sc.sparse.dok_matrix((nDOF, nDOF))

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
                        #values_K.append(K_e[i, i]+penalty)
                        values_K.append(1.0)
                        F[dof_r] += (x_e[i_local] - bc.get('value', 0.0))
                        #F[g_i] = bc.get('value', 0.0)
                    else:
                        F[dof_r] += f_e[i_local]

                        for jdxVar, Variable in enumerate(Var):
                            K_block = np.asarray(K_e[Variable])

                            for j_local, j_global in enumerate(nodes_id):
                                dof_c = j_global*nVar + jdxVar
                                rows_K.append(dof_r)
                                cols_K.append(dof_c)
                                values_K.append(K_block[i_local, j_local])

                        
        K = sc.sparse.coo_matrix((values_K, (rows_K, cols_K)), shape=(nDOF, nDOF)).tocsr()

        return K, F

    def assembleMassMatrix(self, solverOptions = None):

        """Assembles the global mass matrix for the model based on the physics and mesh.
        
        Parameters
        -----------
        :param solverOptions: Dictionary containing solver configuration options.
               
        """

        M = sc.sparse.lil_matrix((self._mesh.getNoN(), self._mesh.getNoN()))

        for element in self._mesh.EL:

            """Compute the element mass matrix and add its contributions to the global mass matrix."""

            M_e = self.physics.getElementMassMatrix(element, solverOptions)

            for idx, node_i in enumerate(element.nodes):
                for jdx, node_j in enumerate(element.nodes):

                    M[node_i.id, node_j.id] += M_e[idx, jdx]

        return M
    
    def _element_global_dof_indices(self, element: Element, idxVar: int, nVar: int) -> list[int]:
        return [idxVar + nVar * node.id for node in element.nodes]
    
    def _get_dirichlet_value(self, node: Node, variable: str):
        if not node.BC:
            return None
        bc = node.BC.get(variable)
        if bc and bc.get('type') == 'Dirichlet':
            return bc.get('value', 0.0)
        return None

    def _element_contributions(self, element, idxVar, Variable, K_e, f_e):
        rows = []
        cols = []
        vals = []
        rhs = []
        g_indices = self._element_global_dof_indices(element, idxVar, self.physics.getNumOfVar())

        for i, g_i in enumerate(g_indices):
            value_dirichlet = self._get_dirichlet_value(element.nodes[i], Variable)
            if value_dirichlet is not None:
                rows.append(g_i)
                cols.append(g_i)
                vals.append(1.0)
                rhs.append((g_i, value_dirichlet))
            else:
                rhs.append((g_i, f_e[i]))
                for j, g_j in enumerate(g_indices):
                    rows.append(g_i)
                    cols.append(g_j)
                    vals.append(K_e[i, j])

        return rows, cols, vals, rhs


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
            sim_logger.info("Time Discretization: " + str(self.solverOptions['Time Discretization'].value))
            sim_logger.info("Time Step: " + str(self.solverOptions['Time Step'].value))
            sim_logger.info("Total Time: " + str(self.solverOptions['Time'].value))


        #print('-------Simulation started-------')

        if self.solverOptions is None:
            self.solverConfiguration()

        solver = Solution.modelSolver(self)

        solver.solve()

        self.timeVector = solver.timeVector

        sim_logger.info("------- Simulation finished -------")


    def postProcess(self):

        plt.plot(self.mesh.getXCoor(), self.sol['T'], marker='o')
        plt.xlabel("x-axis [m]")
        plt.ylabel("Temperature[°C]")
        plt.show()



#@jit
def contributions_determination(A_e: np.ndarray, b_e: np.ndarray, nVar: int, nodes_id: list[int], nodes_bc: list[dict],
                                idxVar: int, Variable: str) -> tuple[list[tuple[int, int, float]], list[tuple[int, float]]]:

    """Determines the contributions of an element to the global system, taking into account boundary conditions.
    
    Parameters
    -----------
    :param A_e: Element matrix for the variable being processed.
    :param b_e: Element vector for the variable being processed.
    :param nVar: Total number of variables in the system.
    :param nodes_id: List of node IDs for the current element.
    :param nodes_bc: List of boundary condition dictionaries for each node.
    :param idxVar: Index of the variable in the global system.
    :param Variable: Name of the variable being assembled.

    """

    local_contributions = []

    force_contributions = []

    for idx, node_i in enumerate(nodes_id):
        global_idx_i = idxVar + nVar * node_i
        
        if nodes_bc[idx] and nodes_bc[idx].get(Variable, {}).get('type') == 'Dirichlet':
            local_contributions.append((global_idx_i, global_idx_i, 1))
            force_contributions.append((global_idx_i, nodes_bc[idx][Variable]['value']))
        else:
            force_contributions.append((global_idx_i, b_e[idx]))
            for jdx, node_j in enumerate(nodes_id):
                global_idx_j = idxVar + nVar * node_j
                local_contributions.append((global_idx_i, global_idx_j, A_e[idx, jdx]))

    return local_contributions, force_contributions


def contributions_determination_nonlinear(K_e: np.ndarray, f_e: np.ndarray, nVar: int, nodes_id: list[int], nodes_bc: list[dict],
                                idxVar: int, Variable: str) -> tuple[list[tuple[int, int, float]], list[tuple[int, float]]]:

    """Determines the contributions of an element to the global system, taking into account boundary conditions.
    
    Parameters
    -----------
    :param K_e: Element tangent matrix for the variable being processed.
    :param f_e: Element residual vector for the variable being processed.
    :param nVar: Total number of variables in the system.
    :param nodes_id: List of node IDs for the current element.
    :param nodes_bc: List of boundary condition dictionaries for each node.
    :param idxVar: Index of the variable in the global system.
    :param Variable: Name of the variable being assembled.

    """

    local_contributions = []

    force_contributions = []

    for idx, node_i in enumerate(nodes_id):
        global_idx_i = idxVar + nVar * node_i
        
        if nodes_bc[idx] and nodes_bc[idx].get(Variable, {}).get('type') == 'Dirichlet':
            local_contributions.append((global_idx_i, global_idx_i, 1))
            force_contributions.append((global_idx_i, 0))
        else:
            force_contributions.append((global_idx_i, f_e[idx]))
            for jdx, node_j in enumerate(nodes_id):
                global_idx_j = idxVar + nVar * node_j
                local_contributions.append((global_idx_i, global_idx_j, K_e[idx, jdx]))

    return local_contributions, force_contributions