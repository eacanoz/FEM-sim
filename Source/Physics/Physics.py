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
from Source.Pre_processing.Mesh import Mesh
from Source.Material import material
from Source.Primals.Scalar import scalarField
from Source.Primals.Vector import vectorField

# from Source.core import Model

# Define basis functions variables
e1, e2, e3 = sp.symbols('e1 e2 e3')


class physics:

    def __init__(self, model):
        self.w = model.w
        self.mat = model.mat

        self.var = {}

        self.source = 0

        self.Convection = False
        self.Stab = None

        self.C = 0
        self.K = None
        self.B = None
        self.F = None
        self.G = None

        self.Pe = 0  # Peclet number


        

    def getElementMatrix(self, element):
        pass

    def getElementVector(self, element):
        pass

    def laplacian(self, const: float, var: scalarField, element):
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
            #          (var.bfGrad() * self.Jacobian(element).inv()).transpose() * \
            #          self.Jacobian(element).det()

        diff_A = (self.w.bfGrad() * self.Jacobian(element).inv()) * const * \
                (var.bfGrad() * self.Jacobian(element).inv()).transpose() * \
                self.Jacobian(element).det()

        A = sp.integrate(diff_A, (e1, -1, 1)).tolist()

        return np.array(A).astype(np.float64)

    def Grad(self, var: scalarField, element):

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
                    (var.bfGrad() * self.Jacobian(element).inv()).transpose() * \
                    self.Jacobian(element).det()

        Grad = sp.integrate(diff_Grad, (e1, -1, 1)).tolist()

        return np.array(Grad).astype(np.float64)

    def div(self, var: scalarField, element, const, Vel):

        if self.Stab == 'PG':
            self.w.addStab(self.Stab, self.stabilization(element))

        diff_Div = const * (self.w.N) * \
                   (var.bfGrad() * self.Jacobian(element).inv()).transpose() * \
                   Vel * self.Jacobian(element).det()

        Div = sp.integrate(diff_Div, (e1, -1, 1)).tolist()

        return np.array(Div).astype(np.float64)

    def Jacobian(self, element):
        x_map = element.sF.N.transpose() * sp.Matrix(element.getCoor())

        # This definition applies to Gradients
        J = x_map.jacobian(sp.Matrix(list(x_map.free_symbols)))

        return J

    def forceVector(self, element):
        diff_F = self.w.N * self.source * self.Jacobian(element)

        F = sp.integrate(diff_F, (e1, -1, 1)).tolist()

        return np.array(F).astype(np.float64).reshape((1, 2))[0]  ## To fix!!!

    def stabilization(self, element):

        Pe_h = self.Pe * element.getLength()
        alpha = (1 / math.tanh(Pe_h / 2)) - 2 / Pe_h
        
        stab = alpha * element.getLength() / 2 * self.w.bfGrad() \
            * self.Jacobian(element).inv()

        return stab


    def addStabilization(self, element):
        pass

    def setDirichletBC(self, **kwargs):
        pass

    def setNeumannBC(self, **kwargs):
        pass

    def setNewtonBC(self, **kwargs):
        pass

    def setVariables(self):
        pass

    def addBMatrix(self, element):
        pass

    def addGVector(self, element):
        pass
