## FEM Simualtion

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
import matplotlib.pyplot as plt


from Source.enums import ElementType, ShapeFunctionType, StudyType, ProblemType, SolverType, DirectSolvers, IterativeSolvers


# Define basis functions variables
# e1 = sp.symbols('e1')

PD = 1  # Problem Dimension: 1->1D; 2->2D(!); 3->3D(!)

# Create mesh from linear element

L = 2  # Length size of the domain [mm]
NoE = 10  # Number of Elements
MeshType = ElementType.line  # Type of element
shapeFunction = ShapeFunctionType.linear  # Shape function for spatial discretization

Mesh1 = Mesh()
Mesh1.Generate_Mesh(PD, L, NoE, MeshType)
Mesh1.setElementShapeFunction(shapeFunction)

Mesh1.defineBoundary('inlet', 0)
Mesh1.defineBoundary('outlet', NoE) # Hardcoded

Mat1 = material('HDPE', k = 0.07, miu=1, rho=1, Cp= 1)

Model1 = Model(name='1D_Conductivity_HDPE', mtype=None, dim=PD, mesh=Mesh1, mat=Mat1, psc=ht)

Model1.physics.addBC_Temperature(id=Mesh1.boundaries['inlet'], T=40)
#Model1.physics.addBC_Convection(Mesh1.boundaries['outlet'], 1, 10)
#Model1.physics.addBC_Temperature(id=Mesh1.boundaries['outlet'], T=40)
Model1.physics.addBC_HeatFlux(id=Mesh1.boundaries['outlet'], q_flux=0)
#Model1.physics.addBC_Radiation(3, 0.7, 26)

Model1.physics.Convection = False
Model1.physics.Stab = None

Model1.physics.initField('T', 150)

options = {'Study': 'Transient', 
           'Type': 'Linear', 
           'Method': 'Iterative', 
           'Solver':'BicgStab', 
           'totalTime': 16}

solverOptions = {'Study': StudyType.transient, 'Type': ProblemType.linear, 'Method': SolverType.iterative, 'Solver': IterativeSolvers.BiCGSTAB, 'totalTime': 16}

Model1.solverConfiguration(**solverOptions)

Model1.solve()

# Model1.postProcess()

times = Model1.timeVector.size

print(times)

plt.plot(Model1.mesh.getXCoor(), Model1.sol['T'][:, 0], 'r')
plt.plot(Model1.mesh.getXCoor(), Model1.sol['T'][:, 30], 'b')
plt.plot(Model1.mesh.getXCoor(), Model1.sol['T'][:, 40], 'c')
plt.plot(Model1.mesh.getXCoor(), Model1.sol['T'][:, 44], 'g')
plt.plot(Model1.mesh.getXCoor(), Model1.sol['T'][:, times-1], 'k')
plt.legend(title = 'Time [s]', 
           labels=[str(round(Model1.timeVector[0], 2)), 
                   str(round(Model1.timeVector[30], 2)), 
                   str(round(Model1.timeVector[40], 2)), 
                   str(round(Model1.timeVector[44], 2)), 
                   str(round(Model1.timeVector[times-1], 2))])
plt.xlabel("x-axis [mm]")
plt.ylabel("Temperature [°C]")
