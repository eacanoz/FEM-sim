# FEM-sim: simulador de elementos finitos

FEM-sim es un simulador de elementos finitos en Python para problemas de conducción de calor.
Hoy resuelve problemas **1D** (elementos lineales) y **2D estacionarios lineales** (cuadriláteros Q4).
Está pensado para crecer por capas: geometría, física, condiciones de contorno y solvers se pueden ampliar por separado.

> Documentación detallada:
> - [docs/ARQUITECTURA.md](docs/ARQUITECTURA.md): cómo está organizado el código y cómo extenderlo.
> - [docs/ESTADO.md](docs/ESTADO.md): qué funciona, qué está validado, limitaciones y próximos pasos.

## Estado actual

| Área | Estado |
|---|---|
| Calor 1D estacionario, lineal | Operativo, vectorizado, validado con soluciones analíticas |
| Calor 1D no lineal (radiación, Newton-Raphson con line search) | Operativo |
| Calor 1D transitorio (θ-método, paso fijo) | Operativo |
| Calor 2D estacionario lineal (Q4, BC en aristas) | Operativo, validado con soluciones manufacturadas (orden 2) |
| Transporte de especies (`mt`, `MassTransferSimulation.py`) | Ruta antigua por elemento, **sin validar** |
| 2D no lineal, 2D transitorio, triángulos, mallas gmsh | No implementado |

## Requisitos

Las versiones son las que usa el entorno `.venv` del proyecto (ver `requirements.txt`):

- Python 3.13
- numpy, scipy, sympy, matplotlib
- jax (solo para la ruta antigua por elemento y `BasisFunctions`)
- numba, numdifftools, progress, joblib
- pypardiso (solver directo PARDISO). Es obligatorio: `Source/Simulation/DirectSolver.py` lo importa al cargar el módulo.
- gmsh (solo para `readMesh.py`)

Instalación en Windows (PowerShell):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uso rápido

### Problema 1D

```python
from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
from Source.enums import ElementType, ShapeFunctionType, StudyType, ProblemType, SolverType, DirectSolvers

mesh = Mesh()
mesh.Generate_Mesh(1, 2.0, 10, ElementType.line)          # longitud 2 m, 10 elementos
mesh.setElementShapeFunction(ShapeFunctionType.linear)

model = Model(name='barra', dim=1, mesh=mesh, mat=material('Acero', k=1, rho=1, Cp=1), psc=ht)
model.physics.addBC_Temperature(id=0, T=60)                # Dirichlet en el nodo 0
model.physics.addBC_Convection(id=10, h_c=5, T_ext=20)     # Newton en el nodo 10
model.physics.initField('T', 40.0)

model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear,
                          Method=SolverType.direct, Solver=DirectSolvers.PARDISO)
model.solve()
print(model.sol['T'])
model.postProcess()
```

Para el caso no lineal, use `Type=ProblemType.nonlinear` y `model.physics.addBC_Radiation(id, epsilon, T_ext)`
(temperaturas en °C; la radiación se calcula internamente en Kelvin).
Para transitorio, use `Study=StudyType.transient` con `timeDisc` (1 = Euler implícito, 0.5 = Crank-Nicolson),
`timeStep` y `totalTime`. Ver `Simulation_time_dependent.py`.

### Problema 2D

```python
from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.HeatTransfer import ht
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers

mesh = Mesh()
mesh.generate2DMesh((1.0, 1.0), (16, 16), ElementType.quadrilateral)   # 1 m x 1 m, 16 x 16 elementos Q4

model = Model(name='placa', dim=2, mesh=mesh, mat=material('Acero', k=1, rho=1, Cp=1), psc=ht)
model.physics.Convection = False
model.physics.F_const = lambda x, y: 0.0                               # fuente Q(x, y), opcional

izquierda = mesh.boundary_edges(lambda x, y: x < 1e-12)
derecha = mesh.boundary_edges(lambda x, y: x > 1 - 1e-12)
model.physics.addBC_TemperatureEdges(izquierda, 100.0)                  # Dirichlet en aristas
model.physics.addBC_ConvectionEdges(derecha, h_c=5.0, T_ext=20.0)      # Newton en aristas

model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.linear,
                          Method=SolverType.direct, Solver=DirectSolvers.PARDISO)
model.solve()
model.postProcess2D()
```

Las aristas se seleccionan con un predicado sobre su punto medio: `mesh.boundary_edges(lambda x, y: ...)`.

## Ejemplos

| Archivo | Qué muestra |
|---|---|
| `Simulation.py` | Barra 1D no lineal con Dirichlet, convección y radiación |
| `Simulation_time_dependent.py` | Barra 1D transitoria |
| `Example_2D_Poisson.py` | Poisson 2D con solución exacta, tabla de convergencia y gráficos |
| `2DFEM.py` | Generación de malla 2D (referencia de numeración) |
| `readMesh.py` | Lectura de información de un archivo `.msh` con gmsh |

Algunos ejemplos abren ventanas de Matplotlib (`plt.show()`). Para ejecutarlos sin ventanas use `MPLBACKEND=Agg`.

## Pruebas

Para correr todas las suites de una vez (devuelve código de salida distinto de cero si alguna falla):

```bash
python run_tests.py
```

Cada archivo de `Test/` también se ejecuta directamente y es compatible con pytest:

```bash
python Test/test_analytic.py       # 1D: soluciones analíticas (5 pruebas)
python Test/test_vectorized.py     # ensamblado vectorizado frente a la ruta por elemento (5)
python Test/test_solvers.py        # Newton con line search y transitorio θ-método (4)
python Test/test_2d_poisson.py     # 2D: topología, parche, BC en aristas y convergencia (5)
```

Todas deben pasar antes de hacer un commit que toque el núcleo (`Source/`).

## Estructura del proyecto

```
Source/
  core.py                      Model: ensamblado, solve(), postprocesado
  Material.py                  propiedades del material (k, rho, Cp, ...)
  enums.py                     tipos de elemento, solvers, estudios
  logger_config.py             registro en Logs/ (un archivo por ejecución)
  Pre_processing/
    Mesh.py                    Mesh, Node, Element; generación 1D y 2D (Q4), boundary_edges
    BasisFunctions.py          funciones de forma 1D (jax/sympy) y Q4 (numpy)
  Physics/
    Physics.py                 clase base de física (ruta por elemento con jax)
    HeatTransfer.py            ht: conducción, convección, radiación, BC en nodos y aristas
    MassTransfer.py            mt: transporte de especies (sin validar)
  Primals/                     campos escalares y vectoriales (valores nodales)
  Simulation/
    VectorizedAssembler.py     geometría 1D/2D y ensamblado vectorizado (NumPy/einsum)
    Solvers.py                 modelSolver: lineal, Newton, θ-método transitorio
    DirectSolver.py            PARDISO / SuperLU
    IterativeSolver.py         BiCGSTAB, GMRES, CG con iLU
Test/                          pruebas automáticas (ver arriba) y cuadernos antiguos
Logs/                          registros de cada ejecución
```

## Convenciones

- **Unidades:** las temperaturas se introducen en °C. La radiación convierte a Kelvin internamente.
- **Flujos de contorno:** `addBC_HeatFlux` y `addBC_HeatFluxEdges` reciben el flujo que **sale** del dominio,
  `q = -k dT/dn` con `n` hacia afuera (valor positivo = pierde calor).
- **Convección:** `Newton` usa `q_saliente = h (T - T_ext)`.
- **Numeración:** los nodos se numeran desde 0 en el orden de creación de la malla. En 2D, a lo largo de x primero.

## Licencia

GNU General Public License v3.0. Ver `LICENSE.md`.
