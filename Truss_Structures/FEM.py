## Trust Element using Finite Element Method (FEM)

import numpy as np
from Truss_Structures.functions import *
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Mesh

NL = np.array([[0,0],
               [1,0],
               [0.5, 1]])  # Point coordinates

EL = np.array([[1, 2],
               [2, 3],
               [3, 1]])    # Elements' points

# Boundary conditions

# Dirichlet BC -> -1
# Neumann BC -> 1

DorN = np.array([[-1, -1],
                 [1, -1],
                [1, 1]])    # Boundary conditions for each node

Fu = np.array([[0, 0],
              [0, 0],
              [0, -20]])     # Forces on each node

U_u = np.array([[0, 0],
              [0, 0],
              [0, 0]])       # Displacements

# Material properties

E = 10**6       # Young modulus
A = 0.01        # Cross sectional area m^2

# Defining dimension

PD = np.size(NL, 1)         # Problem Dimension
NoN = np.size(NL, 0)        # Number of nodes

ENL = np.zeros([NoN, 6*PD])     # Extended Node List

ENL[:, 0:PD] = NL[:,:]
ENL[:, PD:2*PD] = DorN[:,:]

# Assigning boundary conditions

(ENL, DOFs, DOCs) = assign_BCs(NL, ENL)

# Stiffness matrix

K = assemble_stiffness(ENL, EL, NL, E, A)

# Update displacements and Forces

ENL[:, 4*PD:5*PD] = U_u[:, :]
ENL[:, 5*PD:6*PD] = Fu[:, :]


U_u = U_u.flatten()
Fu = Fu.flatten()

Fp = assemble_forces(ENL, NL)
Up = assemble_displacements(ENL, NL)

# Solution

K_UU = K[0:DOFs, 0:DOFs]
K_UP = K[0:DOFs, DOFs:DOFs+DOCs]
K_PU = K[DOFs:DOFs+DOCs, 0:DOFs]
K_PP = K[DOFs:DOFs+DOCs, DOFs:DOFs+DOCs]

F = Fp - np.matmul(K_UP, Up)
U_u = np.matmul(np.linalg.inv(K_UU), F)
Fu = np.matmul(K_PU, U_u) + np.matmul(K_PP, Up)

ENL = update_nodes(ENL, U_u, NL, Fu)

###

scale = 100       # Exaggeration
coor = []
dispx_array = []

for i in range(np.size(NL, 0)):
    dispx = ENL[i, 8]
    dispy = ENL[i, 9]

    x = ENL[i, 0] + dispx*scale
    y = ENL[i, 1] + dispy*scale

    dispx_array.append(dispx)
    coor.append(np.array([x, y]))

coor = np.vstack(coor)
dispx_array = np.vstack(dispx_array)


x_scatter = []
y_scatter = []

color_x = []

for i in range(0, np.size(EL, 0)):
    x1 = coor[EL[i,0]-1, 0]
    x2 = coor[EL[i,1]-1, 0]
    y1 = coor[EL[i,0]-1, 1]
    y2 = coor[EL[i,1]-1, 1]

    dispx_EL = np.array([dispx_array[EL[i,0]-1], dispx_array[EL[i,1]-1]])

    if x1 == x2:
        x = np.linspace(x1, x2, 200)
        y = np.linspace(y1, y2, 200)
    else:
        m = (y2-y1)/(x2-x1)
        x = np.linspace(x1, x2, 200)
        y = m*(x-x1)+y1

    x_scatter.append(x)
    y_scatter.append(y)

    color_x.append(np.linspace(np.abs(dispx_EL[0]), np.abs(dispx_EL[1]), 200))

x_scatter = np.vstack([x_scatter]).flatten()
y_scatter = np.vstack([y_scatter]).flatten()
color_x = np.vstack([color_x]).flatten()

dipsFigure =plt.figure(1)
ax_dispx = dipsFigure.add_subplot(111)

cmap = plt.get_cmap('jet')
ax_dispx.scatter(x_scatter, y_scatter, c = color_x, cmap=cmap, s=10, edgecolor='none')

norm_x = Normalize(np.abs(dispx_array.min()), np.abs(dispx_array.max()))

dipsFigure.colorbar(ScalarMappable(norm=norm_x, cmap=cmap))

plt.show()

