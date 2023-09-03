"""
Demostration of the FEM-SIM module following the example 9.1 from 
'Polymer Processing. Modeling and Simulation - Tim A. Osswald and
Juan P. Hernandez

Dimension = 1
Physics = Steady state Heat Conduction

∇.(k∇T) + Q_viscous_heating = 0

Q_viscous_heating = n(du_y/dx)^2 ~ 25000 W/m3

h = 0.04m, n = 1000 Pa-s, k= 0.2 J/m/K, u0 = 0.2 m/s, Tm = 200°C

----------- Analytical solution ----------------
T(x) = Tm + (n/2k)*(u0/h)^2 * (2hx-x^2)

"""

import matplotlib.pyplot as plt
import numpy as np

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht

# Defining problem dimension: 1D
PD = 1 

# Create mesh from linear element

L = 0.04  # Length size of the domain [m]
NoE = 40  # Number of Elements
MeshType = "1DROD2P"  # Type of element
shapeFunction = 'Linear' # Shape function for spatial discretization

Mesh1 = Mesh()
Mesh1.Generate_Mesh(PD, L, NoE, MeshType)
Mesh1.setElementShapeFunction(shapeFunction)

Mesh1.checkBoundaries()

# Defining material properties
Mat1 = material('Polymer', k = 0.2, miu=1, rho=1, Cp= 1)

# Defining main model
Model1 = Model(name='Couette_device', mtype=None, dim=PD, mesh=Mesh1, mat=Mat1, psc=ht)

# Adding source term (Q_viscous_heating)
Model1.physics.source = 25000

# Neglecting convection term
Model1.physics.Convection = False
Model1.physics.Stab = None

# Adding boundary conditions
Model1.physics.addBC_Temperature(id=0, T=200)
Model1.physics.addBC_HeatFlux(id=5, q_flux = 0)


# Initialize field
Model1.physics.initField('T', 200)

# Setting solver options

solverOptions = {'Study': 'Steady state', 'Type': 'Linear', 'Method': 'Direct', 'Solver':'PARDISO'}

Model1.solverConfiguration(**solverOptions)

# Solving PDE
Model1.solve()

# Model1.postProcess()


# Analytical solution

h = 0.04
n = 1000
k= 0.2
u0 = 0.2
Tm = 200

x_axis = np.array([n*(L/20) for n in range(20+1)])

T_x = Tm + (n/(2*k))*(u0/h)**2 * (2*h*x_axis-x_axis**2)


plt.plot(x_axis, T_x)
plt.plot(Model1.mesh.getXCoor(), Model1.sol['T'], 'or')

plt.legend(['Analytical', 'FEM'])

plt.xlabel("x-axis [m]")
plt.ylabel("Temperature [°C]")
