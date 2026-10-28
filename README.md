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
| Transporte de especies 1D y 2D (`mt`): difusión, convección, varias reacciones acopladas, BC en nodos y aristas | Operativo sobre la base multifísica, validado con soluciones analíticas y manufacturadas |
| Calor 2D no lineal (radiación y convección `h(T)` en aristas, Newton con line search) | Operativo, validado frente a 1D y por balance de energía |
| Calor 2D transitorio (θ-método, también con BC no lineales) | Operativo, validado con soluciones analíticas, orden temporal y conservación de energía |
| Conductividad dependiente de la temperatura `k(T)` (1D y 2D, estacionario y transitorio) | Operativo, validado con la solución exacta de Kirchhoff y una solución manufacturada |
| Calor + especies acopladas (`Model(psc=[ht, mt])`): Arrhenius con la T del campo y calor de reacción `dH` | Operativo, validado con una EDO independiente, invariantes exactos y jacobiano cruzado |
| Coeficientes de difusión que dependen de otros campos: `k(T)`, `k(c)`, `D(T)`, `D(c)` (`depends_on`) | Operativo, validado con soluciones exactas, convergencia de segundo orden y jacobiano |
| Triángulos, mallas gmsh, `rho Cp(T)`, convección vectorial estabilizada | No implementado |

## Requisitos

Las versiones son las que usa el entorno `.venv` del proyecto (ver `requirements.txt`):

- Python 3.13
- numpy, scipy, matplotlib
- jax y jaxlib (solo para las leyes puntuales no lineales: reacciones, `h(T)` de radiación, ...)
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

Para una conductividad que dependa de la temperatura use `model.physics.setConductivityLaw(lambda T: 2.0 * (1 + 0.01 * T))`
(escrita con `jax.numpy`, porque se deriva automáticamente) y `ProblemType.nonlinear`.

El θ-método usa un paso fijo. Euler implícito (θ = 1) es robusto y de primer orden; Crank-Nicolson (θ = 0.5) es de
segundo orden pero no es L-estable: con estados iniciales discontinuos y pasos grandes los modos rígidos oscilan
durante bastante tiempo (use un paso menor o Euler). Con BC no lineales, cada paso se linealiza con Newton al inicio
del paso. El estado inicial puede ser un número o una función de las coordenadas: `initField('T', lambda x, y: ...)`.

### Calor y especies acopladas

Varias físicas pueden resolverse juntas en un modelo: comparten malla y campos, y un término de una puede depender de
los campos de la otra.

```python
model = Model(name='reactor', dim=1, mesh=mesh, mat=material('Medio', k=2, rho=5, Cp=1), psc=[ht, mt])
heat, mass = model.physics.heat, model.physics.mass          # cada física, por nombre

for especie in ('A', 'C'):
    mass.setChemSpecies(especie, especie)
    mass.setDiffusivity(especie, 1.0)
# T='T': la velocidad de reacción usa la temperatura del campo (en °C); dH < 0: exotérmica [J/mol]
mass.addReaction({'A': -1, 'C': 1}, k0=3e4, E_R=2000.0, T='T', order={'A': 1}, dH=-40.0)

model.physics.addBC_Temperature(0, 25.0)                      # los métodos de cada parte se delegan
model.physics.addBC_Concentration(0, 'A', 1.0)
heat.Convection = False                                       # los atributos se asignan en la parte
```

Los coeficientes pueden depender de cualquier campo del modelo (funciones de `jax.numpy`, con `depends_on` en el orden de los argumentos):

```python
heat.setConductivityLaw(lambda T, A: 2.0 * (1 + 0.5 * A), depends_on=('T', 'A'))          # k(c)
mass.setDiffusivity('A', lambda T: 1e-3 * jnp.exp(0.02 * T), depends_on=('T',))          # D(T)
```

Con `T` numérico (Kelvin) la reacción es isotérmica y no depende del calor. Con campo de temperatura puede haber varios
estados estacionarios (ignición): si Newton no converge o cae en otra rama, marche en el tiempo hasta el estacionario y
use ese estado como inicial. Ver `Example_Reactive_Heat.py`.

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
# Radiación (no lineal): model.physics.addBC_RadiationEdges(derecha, epsilon=0.8, T_ext=20.0)

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
| `Example_2D_Transient.py` | Placa 2D con punto caliente, radiación y convección: transitorio no lineal, convergencia en el tiempo y gráficos |
| `Example_Reactive_Heat.py` | Reacción exotérmica acoplada con el calor: verificación frente a una EDO, barrido del calor de reacción y arranque transitorio |
| `Ejemplo_Fisicas_Acopladas.ipynb` | **Cuaderno guiado** del caso reactivo: monta el modelo paso a paso, explica cada función y parámetro, muestra qué ocurre dentro del solver (geometría, DOF, jacobiano, Newton), los resultados, la verificación y el transitorio |
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
python Test/test_vectorized.py     # calor: matrices, masa, residuo, tangente y transitorio frente a referencias (6)
python Test/test_solvers.py        # Newton con line search y transitorio θ-método (4)
python Test/test_2d_poisson.py     # 2D: topología, parche, BC en aristas, convección y convergencia (6)
python Test/test_foundation.py     # elementos, facetas, DOF, operadores y leyes puntuales (11)
python Test/test_species.py        # especies: analíticas 1D, conservación, jacobiano, 2D manufacturada (8)
python Test/test_nonlinear_transient.py  # 2D: radiación, Newton, transitorio, orden temporal y energía (9)
python Test/test_nonlinear_diffusion.py  # k(T): Kirchhoff exacta, manufacturada 2D, jacobiano, transitorio (8)
python Test/test_coupling.py       # calor + especies: EDO adiabática, conservación, invariantes, jacobiano cruzado, API (12)
python Test/test_multifield_diffusion.py  # D(c), D(T), k(c): soluciones exactas, jacobiano, conservación, errores (8)
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
    Mesh.py                    Mesh, Node, Element; generación 1D y 2D (Q4), boundary_facets
    Elements.py                registro de elementos de referencia (line2, line3, quad4)
    BasisFunctions.py          funciones de forma Q4 (numpy)
  Physics/
    Physics.py                 clase base de física (campos, BC, add_terms y build_system)
    Coupled.py                 CoupledPhysics: varias físicas en un modelo (Model(psc=[ht, mt]))
    HeatTransfer.py            ht: conducción, convección, radiación, BC en nodos y aristas
    MassTransfer.py            mt: especies, reacciones acopladas, BC en nodos y aristas
  Primals/                     campos escalares y vectoriales (valores nodales)
  Simulation/
    Discretization.py          geometría en los puntos de Gauss y facetas (cualquier dimensión)
    DofLayout.py               campos, componentes y numeración de DOF
    Operators.py               operadores de forma débil vectorizados (einsum)
    PointwiseLaw.py            leyes no lineales con jacobianos automáticos (JAX)
    System.py                  matrices, términos no lineales y BC independientes de la física
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
