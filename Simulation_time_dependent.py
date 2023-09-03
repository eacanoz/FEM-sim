## FEM Simualtion

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
import matplotlib.pyplot as plt

# Define basis functions variables
# e1 = sp.symbols('e1')

PD = 1  # Problem Dimension: 1->1D; 2->2D(!); 3->3D(!)

# Create mesh from linear element

L = 2  # Length size of the domain [m]
NoE = 5  # Number of Elements
MeshType = "1DROD2P"  # Type of Mesh: LIN1D2P-> 1D Linear with 2 points per element
shapeFunction = 'Linear'

Mesh1 = Mesh()
Mesh1.Generate_Mesh(PD, L, NoE, MeshType)
Mesh1.setElementShapeFunction(shapeFunction)

Mesh1.defineBoundary('inlet', 0)
Mesh1.defineBoundary('outlet', NoE) # Hardcoded

Mat1 = material('Aluminio', k = 1, miu=1, rho=1, Cp= 1)

Model1 = Model(name='1D_Conductivity', mtype=None, dim=PD, mesh=Mesh1, mat=Mat1, psc=ht)

Model1.physics.addBC_Temperature(id=Mesh1.boundaries['inlet'], T=2)
Model1.physics.addBC_Convection(Mesh1.boundaries['outlet'], 1, 10)
#Model1.physics.addBC_Radiation(3, 0.7, 26)

Model1.physics.Convection = True
Model1.physics.Stab = None

Model1.physics.initField('T', 2)

options = {'Study': 'Transient', 'Type': 'Linear', 'Method': 'Iterative', 'Solver':'BicgStab'}

Model1.solverConfiguration(**options)

Model1.solve()

# Model1.postProcess()

times = Model1.timeVector.size

plt.plot(Model1.mesh.getXCoor(), Model1.sol['T'][:, 0], 'or')
plt.plot(Model1.mesh.getXCoor(), Model1.sol['T'][:, 2], 'ob')
plt.plot(Model1.mesh.getXCoor(), Model1.sol['T'][:, 20], 'og')
plt.plot(Model1.mesh.getXCoor(), Model1.sol['T'][:, times-1], 'ok')
plt.legend([str(Model1.timeVector[0]), str(Model1.timeVector[2]), str(Model1.timeVector[25]), str(Model1.timeVector[times-1])])
plt.xlabel("x-axis [m]")
plt.ylabel("Temperature [°C]")
