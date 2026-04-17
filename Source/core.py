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
from Source.Pre_processing.BasisFunctions import BasisFunctions
import Source.Simulation.Solvers as Solution

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
        self.w = BasisFunctions(self._mesh, 'Linear') # Test Function
        self.mat = mat  # Material domain
        self.physics = psc(self)  # Model physics

        self.A = None
        self.b = None

        self.solverOptions = None

        #self.solverOptions = {'Study': 'Steady state', 'Type': 'Nonlinear', 'Method': 'Direct', 'Solver':'PARDISO'}
        # self.solverOptions = {'Type': 'Linear', 'Method': 'Direct', 'Solver':'PARDISO'}
        # self.solverOptions = {'Type': 'Linear', 'Method': 'Iterative', 'Solver':'BicgStab', 'Preconditioner': 'iLU Factorization'}
        # self.solverOptions = {'Type': 'Linear', 'Method': 'Iterative', 'Solver':'BicgStab', 'Preconditioner': None}

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

        if self.physics is None:
            physics_status = 'Not defined'

        custom_str = ("Model: " + self.name + 
                      "\nDimension: " + str(self._PD) + "D" + 
                      "\nNodes: " + str(self.mesh.getNoN()) + 
                      "\nElements: " + str(self.mesh.getNoE()) + 
                      "\nPhysics: " + physics_status)

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

                r, c, v, rhs = self._element_contributions(element, idxVar, Variable, K_e, f_e)

                rows_A.extend(r)
                cols_A.extend(c)
                values_A.extend(v)

                for gi, value in rhs:
                    b[gi] = value  # Dirichlet sobrescribe

                # g_indices = [idxVar + nVar * node.id for node in element.nodes]

                # for i, g_i in enumerate(g_indices):

                #     node_i = element.nodes[i]
                #     bc = node_i.BC.get(Variable, {}) if node_i.BC else {}

                #     if bc.get('type') == 'Dirichlet':
                #         rows_A.append(g_i)
                #         cols_A.append(g_i)
                #         values_A.append(1.0)
                #         b[g_i] = bc.get('value', 0.0)
                #     else:
                #         b[g_i] += f_e[i]

                #         for j, g_j in enumerate(g_indices):
                #             rows_A.append(g_i)
                #             cols_A.append(g_j)
                #             values_A.append(K_e[i, j])


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
        

        F = np.zeros(nDOF)

        # F = sc.sparse.dok_array((nDOF, 1))

        def process_element(element, idxVar, Variable):
            """
            Process a single element to compute its contributions to the global system.
            """
            F_e = self.physics.getResidualVector(element, Variable, solverOptions)
            local_contributions = []

            for idx, node_i in enumerate(element.nodes):
                global_idx_i = idxVar + nVar * node_i.id

                if node_i.BC and node_i.BC[Variable]['type'] == 'Dirichlet':
                    local_contributions.append((global_idx_i, 0))
                    
                else:
                    local_contributions.append((global_idx_i, F_e[idx]))
                    
            return local_contributions

        for idxVar, Variable in enumerate(Var):

            results.extend(sum([process_element(element, idxVar, Variable) for element in self._mesh.EL], []))


        for local_contributions in results:
            global_idx_i, value = local_contributions
            F[global_idx_i] += value

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

        F = np.zeros(nDOF)

        def process_element_parallel(element: Element, idxVar: int, Variable: str):
            K_e = self.physics.getElementTangentMatrix(element, Variable, solverOptions)
            f_e = self.physics.getResidualVector(element, Variable, solverOptions)
            nodes_id = [node.id for node in element.nodes]
            nodes_bc = [node.BC for node in element.nodes]

            return contributions_determination_nonlinear(K_e, f_e, nVar, nodes_id, nodes_bc, idxVar, Variable)   
        
        for idxVar, Variable in enumerate(Var):
            #results.extend(sum([process_element(element, idxVar, Variable) for element in self._mesh.EL], []))
            #results_K.extend(sum([process_element_parallel(element, idxVar, Variable)[0] for element in self._mesh.EL], []))
            #results_F.extend(sum([process_element_parallel(element, idxVar, Variable)[1] for element in self._mesh.EL], []))

            for element in self._mesh.EL:
                K_e = self.physics.getElementTangentMatrix(element, Variable, solverOptions)
                f_e = self.physics.getResidualVector(element, Variable, solverOptions)

                #r, c, v, rhs = self._element_contributions(element, idxVar, Variable, K_e, f_e)

                #rows_K.extend(r)
                #cols_K.extend(c)
                #values_K.extend(v)

                #for gi, value in rhs:
                #    if self._get_dirichlet_value(element.nodes[gi], Variable) is None:
                #        F[gi] += value

                # Indices globales para ensamblaje
                g_indices = [idxVar + nVar * node.id for node in element.nodes]

                for i, g_i in enumerate(g_indices):

                    node_i = element.nodes[i]
                    bc = node_i.BC.get(Variable, {}) if node_i.BC else {}

                    if bc.get('type') == 'Dirichlet':

            
                        rows_K.append(g_i)
                        cols_K.append(g_i)
                        values_K.append(1.0)
                        #F[g_i] = bc.get('value', 0.0)
                    else:
                        F[g_i] += f_e[i]

                        for j, g_j in enumerate(g_indices):
                            rows_K.append(g_i)
                            cols_K.append(g_j)
                            values_K.append(K_e[i, j])

        # # Combine local contributions into the global system
        # for local_contributions in results_K:
           
        #     global_idx_i, global_idx_j, value = local_contributions
        #     K[global_idx_i, global_idx_j] += value

        # for force_contributions in results_F:
        #     global_idx_i, value = force_contributions
        #     F[global_idx_i] += value

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


    def solverConfiguration(self, Study='Steady state', Type='Linear', Method = 'Direct', Solver = 'PARDISO', 
                            timeDisc = 1, timeStep=0.05, totalTime = 3, prec = 'iLU Factorization'):
        
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

        print('-------Simulation started-------')

        if self.solverOptions is None:
            self.solverConfiguration()

        solver = Solution.modelSolver(self)

        solver.solve()

        self.timeVector = solver.timeVector

        print('Simulation finished')

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