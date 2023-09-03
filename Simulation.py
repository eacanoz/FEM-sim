## FEM Simualtion

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht


# Define basis functions variables
# e1 = sp.symbols('e1')

PD = 1  # Problem Dimension: 1->1D; 2->2D(!); 3->3D(!)

# Create mesh from linear element

L = 5  # Length size of the domain [m]
NoE = 10  # Number of Elements
MeshType = "1DROD2P"  # Type of Mesh: LIN1D2P-> 1D Linear with 2 points per element
shapeFunction = 'Linear'

Mesh1 = Mesh()
Mesh1.Generate_Mesh(PD, L, NoE, MeshType)
Mesh1.setElementShapeFunction(shapeFunction)

Mesh1.defineBoundary('inlet', 0)
Mesh1.defineBoundary('outlet', NoE) # Hardcoded

Mat1 = material('Aluminio', k = 1, miu=1, rho=1, Cp= 1)

Model1 = Model(name='1D_Conductivity', mtype=None, dim=PD, mesh=Mesh1, mat=Mat1, psc=ht)

Model1.physics.addBC_Temperature(id=Mesh1.boundaries['inlet'], T=50)
# Model1.physics.addBC_Convection(Mesh1.boundaries['outlet'], 1, 10)
Model1.physics.addBC_Radiation(Mesh1.boundaries['outlet'], 0.7, 26)

Model1.physics.Convection = True
Model1.physics.Stab = None

Model1.physics.initField('T', 50)

options = {'Study': 'Steady state', 'Type': 'Nonlinear', 'Method': 'Direct', 'Solver':'PARDISO'}

Model1.solverConfiguration(**options)

Model1.solve()

Model1.postProcess()
