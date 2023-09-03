# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++++

# Class physics: Contains all the functions necessary to create the conservative relations


import numpy as np
import sympy as sp
import math

from Source.Pre_processing.BasisFunctions import basisFunctions
from Source.Pre_processing.Mesh import Mesh, Element, Node
from Source.Material import material
from Source.Primals.Scalar import scalarField
from Source.Primals.Vector import vectorField

# from Source.core import Model

# Define basis functions variables
e1, e2, e3 = sp.symbols('e1 e2 e3')


class physics:

    def __init__(self, model):

        self.modelRef = model
        self.w = model.w
        self.mat = model.mat

        self.var = {}

        self.source = 0

        self.Convection = False
        self.Stab = None

        self.C = 0
        self.K = None
        self.B = 0
        self.F = None
        self.G = None

        self.Pe = 0  # Peclet number

    def getVariables(self):
        return self.var.keys()


    def getNumOfVar(self):
        return len(self.var.keys())

    def initField(self, variable, value):
        self.var[variable].initField(value)
        

    def getElementMatrix(self, element, Variable, solverOptions=None):

        self.initializeMatrices(element, Variable)
        
        return self.C + self.K + self.B

    def getElementVector(self, element, Variable, solverOptions=None):

        self.initializeVectors(element, Variable)
        
        return self.F - self.G   
    
    def getElementMassMatrix(self, element, solverOptions=None):
        
        self.initializeMassMatrix(element)

        return self.M
    
    def getResidualVector(self, element, Variable, solverOptions=None):

        A_e = self.getElementMatrix(element, Variable, solverOptions)
        b_e = self.getElementVector(element, Variable, solverOptions)

        x_e = self.var[Variable].getElementValues(element)

        return A_e.dot(x_e) - b_e

    def laplacian(self, const: float, var: scalarField, element: Element):
        """
        Define the element matrix from the weak form for the Laplacian term.


        e. g.
        ∇.(const * ∇T)  -> Integral(dNi/dx^Trans*const*dNj/dx*det(J^-1)de1, -1, 1)


        Note: Stills for 1D. Next updates will show the code for multiple dimensions

        :param var:
        :param const: material proporcionality constant
        :param element: Element
        :return: Matrix of laplacian term
        """

            # diff_A = (self.mbF.jacobian(sp.Matrix(list(self.mbF.free_symbols))) * self.Jacobian(
            #     element).inv()) * const * \
            #          (var.bfGrad() * element.Jacobian().inv()).transpose() * \
            #          element.Jacobian().det()

        diff_A = (self.w.bfGrad() * element.Jacobian().inv()) * const * \
                (var.bfGrad() * element.Jacobian().inv()).transpose() * \
                element.Jacobian().det()

        A = sp.integrate(diff_A, (e1, -1, 1)).tolist()

        return np.array(A).astype(np.float64)

    def Grad(self, var: scalarField, element: Element):

        """
        Define de element matrix from the weak form for the Gradient term.

        e. g.
        ∇T -> Integral(Ni*dNj/dx*det(J^-1)de1, -1, 1)

        :param element: Element
        :return: Matrix form of the Gradient term
        """        
        if self.Stab == 'PG':
            self.w.addStab(self.Stab, self.stabilization(element))


        diff_Grad = (self.w.N) * \
                    (var.bfGrad() * element.Jacobian().inv()).transpose() * \
                    element.Jacobian().det()

        Grad = sp.integrate(diff_Grad, (e1, -1, 1)).tolist()

        return np.array(Grad).astype(np.float64)

    def div(self, var: scalarField, element: Element, const, Vel):

        """
        Define de element matrix from the weak form for the Divergence term.

        e. g.
        const*_u∇T -> Integral(const*Ni*dNj/dx*vel*det(J^-1)de1, -1, 1)

        :param element: Element
        :return: Matrix form of the Gradient term
        """

        if self.Stab == 'PG':
            self.w.addStab(self.Stab, self.stabilization(element))

        diff_Div = const * (self.w.N) * \
                   (var.bfGrad() * element.Jacobian().inv()).transpose() * \
                   Vel * element.Jacobian().det()

        Div = sp.integrate(diff_Div, (e1, -1, 1)).tolist()

        return np.array(Div).astype(np.float64)

    def mass(self, var: scalarField, element: Element, constM):

        diff_M = constM * (self.w.N) * (var.bf.N).transpose() * element.Jacobian().det()

        Mass_Matrix = sp.integrate(diff_M, (e1, -1, 1)).tolist()

        return np.array(Mass_Matrix).astype(np.float64)

    def forceVector(self, element: Element, Variable):

        if callable(self.source):

            f = self.source(element, Variable)
        else:

            f = self.source


        diff_F = self.w.N * f * element.Jacobian()

        F = sp.integrate(diff_F, (e1, -1, 1)).tolist()

        return np.array(F).astype(np.float64).reshape((element.getNumberNodes(), 1))[0]  ## To fix!!!


    def addBMatrix(self, element, Variable):

        B = np.zeros((element.getNumberNodes(), element.getNumberNodes()))

        for i, node in enumerate(element.nodes):

            if node.BC and node.BC[Variable]['type'] == 'Newton':

                if callable(node.BC[Variable]['h']):
                    B[i, i] = node.BC[Variable]['h'](i)
                    # print('Radiaction BC used')
                else:
                    B[i, i] = node.BC[Variable]['h']

        return B

    def addGVector(self, element, Variable):

        G = np.zeros((element.getNumberNodes())).transpose()

        for i, node in enumerate(element.nodes):

            if node.BC:

                if node.BC[Variable]['type'] == 'Newton':

                    if callable(node.BC[Variable]['h']):
                        G[i] = - node.BC[Variable]['h'](i) * node.BC[Variable]['var_ext']
                        # print('Radiaction BC used')
                    else:
                        G[i] = - node.BC[Variable]['h'] * node.BC[Variable]['var_ext']

                elif node.BC[Variable]['type'] == 'Neumann':

                    G[i] = node.BC[Variable]['flux']

        return G

    def stabilization(self, element: Element):

        Pe_h = self.Pe * element.getLength()
        alpha = (1 / math.tanh(Pe_h / 2)) - 2 / Pe_h
        
        stab = alpha * element.getLength() / 2 * self.w.bfGrad() \
            * element.Jacobian().inv()

        return stab
    
    def setDirichletBC(self, val):

        """
        Define a Dirichlet Boundary Condition for the node

        :param T0: Temperature at the boundary
        :return: Dictionary with boundary information for the physics
        """

        return {'type': 'Dirichlet', 'value': val}

    def setNeumannBC(self, flux):

        """
        Define a Neumann Boundary Condition for the node

        :param q_flux: Heat flux at the boundary
        :return: Dictionary with boundary information for the physics
        """

        return {'type': 'Neumann', 'flux': flux}

    def setNewtonBC(self, h, var_ext):

        """
        Define a Newton Boundary Condition for the node

        :param h_c: Heat transfer coefficient
        :param T_ext: External temperature
        :return: Dictionary with boundary information for the physics
        """

        return {'type': 'Newton', 'h': h, 'var_ext': var_ext}

    def initializeMatrices(self, element, Variable):
        pass

    def addStabilization(self, element):
        pass

    def setVariables(self):
        pass
