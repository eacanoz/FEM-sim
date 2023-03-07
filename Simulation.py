## FEM Simualtion

import sympy as sp
import numpy as np

from Source.Pre_processing.Mesh import Mesh
from Source.Pre_processing.BasisFunctions import basisFunctions
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht


# Define basis functions variables
# e1 = sp.symbols('e1')

PD = 1  # Problem Dimension: 1->1D; 2->2D(!); 3->3D(!)

# Create mesh from linear element

L = 2  # Length size of the domain [m]
NoE = 3  # Number of Elements
MeshType = "1DROD2P"  # Type of Mesh: LIN1D2P-> 1D Linear with 2 points per element
shapeFunction = 'Linear'

Mesh1 = Mesh()
Mesh1.Generate_Mesh(PD, L, NoE, MeshType)
Mesh1.setElementShapeFunction(shapeFunction)

Mat1 = material('Aluminio', k = 1, miu=1, rho=1, Cp= 1)

Model1 = Model(name='1D_Conductivity', mtype=None, dim=PD, mesh=Mesh1, mat=Mat1, psc=ht)

Model1.setBC(id = 0, type='Dirichlet', T0=2)
Model1.setBC(id = 3, type='Newton',  h_c=1, T_ext = 10)
# Model1.setBC(id = 3, type='Dirichlet', T0=4)

Model1.physics.Convection = True
Model1.physics.Stab = None

Model1.solve()

Model1.postProcess()
# print(int(Mesh1.EL[1, 1]))

# print(Mesh1.he)

### Problem Definition

## Steady state 1D conduction PDE

## d/dx(k*d/dx(T)) = Q

## Problem definition style

# ConstRel = "lap(k,T) = 0"


# k = 1
# A = 1       # Cross sectional area [m^2]