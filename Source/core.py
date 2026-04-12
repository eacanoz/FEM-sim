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
        self.w = basisFunctions(self._mesh, 'Linear') # Test Function
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
        results_A = []
        results_b = []

        A = sc.sparse.dok_matrix((nDOF, nDOF))
        b = np.zeros(nDOF)

        def process_element_parallel(element: Element, idxVar: int, Variable: str):
            A_e = self.physics.getElementMatrix(element, Variable, solverOptions)
            b_e = self.physics.getElementVector(element, Variable, solverOptions)
            nodes_id = [node.id for node in element.nodes]
            nodes_bc = [node.BC for node in element.nodes]

            return contributions_determination(A_e, b_e, nVar, nodes_id, nodes_bc, idxVar, Variable)


        for idxVar, Variable in enumerate(Var):
            #results.extend(sum([process_element(element, idxVar, Variable) for element in self._mesh.EL], []))
            results_A.extend(sum([process_element_parallel(element, idxVar, Variable)[0] for element in self._mesh.EL], []))
            results_b.extend(sum([process_element_parallel(element, idxVar, Variable)[1] for element in self._mesh.EL], []))

        # Combine local contributions into the global system
        for local_contributions in results_A:
           
            global_idx_i, global_idx_j, value = local_contributions
            A[global_idx_i, global_idx_j] += value

        for force_contributions in results_b:
            global_idx_i, value = force_contributions
            b[global_idx_i] += value

        return A.tocsr(), b

    def assembleResidualVector(self, solverOptions = None):

        """Assembles the global residual vector for the model based on the physics and mesh.
        
        Parameters
        -----------
        :param solverOptions: Dictionary containing solver configuration options.
       
        
        """

        Var = self.physics.getVariables()
        nVar = self.physics.getNumOfVar()
        nNodes = self._mesh.getNoN()
        # nElements = self._mesh.getNoE()
        nDOF = nNodes*nVar
        results = []
        # idxBC = []
        # bCB = []
        

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
            # dummy = []
            # dummy.extend(process_element(element, idxVar, Variable) for element in self._mesh.EL)

            results.extend(sum([process_element(element, idxVar, Variable) for element in self._mesh.EL], []))


        for local_contributions in results:
            # for global_idx_i, global_idx_j, value in local_contributions:
            global_idx_i, value = local_contributions
            F[global_idx_i] += value
        # for idxVar, Variable in enumerate(Var):



        return F

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