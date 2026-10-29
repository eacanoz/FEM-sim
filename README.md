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
| Convección con velocidad compartida entre físicas y estabilización SUPG (1D, 2D) | Operativo, nodalmente exacto en 1D a Pe alto y de segundo orden con fuente |
| Triángulos T3 y mallas de gmsh (geometría no rectangular, grupos físicos con nombre) | Operativo: parche exacto, orden 2, anillo frente a solución logarítmica |
| Solvers iterativos (CG, BiCGSTAB, GMRES) con AMG (pyamg), iLU o Jacobi sobre el sistema reducido; reutilización de factorizaciones | Operativo: AMG con iteraciones casi independientes de la malla |
| Exportación a ParaView (`.vtu` y series `.pvd`) | Operativo; XDMF no implementado |
| Base de fluidos (Etapa 9): campos vectoriales, operadores viscoso y divergencia, leyes con gradientes `f(u, ∇u)`, Dirichlet por componente, tracciones, VTK vectorial | Operativo: MMS vectorial de orden 2, jacobiano frente a diferencias finitas, ley de potencia en canal |
| Flujo de Darcy (`darcy`): permeabilidad escalar, función o tensor, gravedad, fuente, almacenamiento; velocidad hacia la convección de `ht`/`mt` | Operativo, exacto en 1D y 2D, hidrostática, inyección |
| Flujo reptante (Stokes, `ff`): velocidad y presión de igual orden (Q4/T3) con PSPG y esfuerzo viscoso recuperado; no deslizamiento, entrada, salida, simetría, tracción; transitorio | Operativo: Poiseuille nodalmente exacto (Q4), solución exacta (u orden 2, p orden 1–1.5), cavidad frente a la referencia, arranque frente a la serie de Fourier |
| Navier–Stokes estacionario (`ff.setInertia(True)`): convección con Newton o Picard, SUPG/PSPG/LSIC sobre el residuo completo | Operativo: Kovasznay (orden 2 en u y p), cavidad Re = 100 frente a Ghia et al., Newton en 5–7 iteraciones |
| Fluidos no newtonianos (`ff.setViscosityLaw`): ley de potencia, Carreau–Yasuda, Cross, Cross–WLF, Bingham y Herschel–Bulkley regularizados, desplazamiento de Arrhenius | Operativo: perfiles analíticos de ley de potencia (orden 2) y Bingham (tapón), jacobiano exacto con dμ/dγ̇ y dμ/dT |
| Acoplamiento flujo–calor (`Model(psc=[ff, ht])`, `setVelocity('u')`, `setViscousHeating`): convección por la velocidad calculada con SUPG, disipación viscosa, μ(T) | Operativo: Brinkman exacto, μ(T) en Couette, equivalencia con velocidad prescrita, jacobiano acoplado completo |
| Axisimétrico r-z (`Model(axisymmetric=True)`): calor, especies, Darcy y flujo (términos de aro en deformación, divergencia y esfuerzo) | Operativo: conducción radial, Hagen–Poiseuille, extensión uniaxial exacta, ley de potencia en tubo, jacobiano acoplado |
| Transitorio no lineal (`solverConfiguration(timeScheme='BDF2')` o θ): Newton en cada paso, estabilización con el residuo discreto en el tiempo, presión implícita; fuerzas por reacciones (`ff.boundaryForce`) | Operativo: Taylor–Green (BDF2 orden 2, Euler orden 1), arranque frente a la serie de Fourier, límite estacionario exacto, cilindro de Schäfer–Turek |
| Convección natural (`ff.setBoussinesq`, térmica y solutal) y especies arrastradas por el flujo; flujos de contorno por reacciones (`boundaryFlux`, Nusselt) | Operativo: cavidad de de Vahl Davis Ra = 1e3–1e6 (Nusselt < 0.2 %), estratificación estable en reposo, especies = calor con Le = 1 |
| Turbulencia RANS Spalart–Allmaras (`Model(psc=[ff, sa])`, SA-neg, distancia a la pared de Spalding, sin funciones de pared) | Operativo: canal Re_τ = 180–590 (U_c⁺ a 1–2 % de la DNS, ley de pared), límite laminar, jacobiano acoplado. k-ω SST pendiente |
| `rho Cp(T)`, elementos de orden superior, 3D | No implementado |

## Requisitos

Las versiones son las que usa el entorno `.venv` del proyecto (ver `requirements.txt`):

- Python 3.13
- numpy, scipy, matplotlib
- jax y jaxlib (solo para las leyes puntuales no lineales: reacciones, `h(T)` de radiación, ...)
- pypardiso (solver directo PARDISO). Es obligatorio: `Source/Simulation/DirectSolver.py` lo importa al cargar el módulo.
- pyamg (precondicionador AMG, opcional)
- gmsh (importar mallas y construir geometrías; opcional si solo se usan las mallas internas)

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

### Solvers y salida

```python
model.solverConfiguration(Method=SolverType.iterative, Solver=IterativeSolvers.CG,
                          prec=Preconditioners.AMG, tol=1e-10)      # AMG necesita pyamg
model.solve()
model.solver_iterations                                              # iteraciones de cada solve lineal
model.exportVTK('resultados/placa')                                  # placa.vtu; transitorio: placa_0000.vtu... + placa.pvd (every=n)
```

Los solvers iterativos eliminan antes los nodos con Dirichlet (el sistema reducido es simétrico en difusión, lo que permite CG y AMG). Un sistema lineal se factoriza una vez, también entre llamadas sucesivas a `solve()` mientras la matriz no cambie.

### Triángulos y mallas de gmsh

```python
mesh = Mesh(); mesh.generate2DMesh((2.0, 1.0), (20, 10), ElementType.triangle)   # T3 estructurados, sin gmsh
mesh = Mesh.read_msh('pieza.msh')            # archivo de gmsh (T3 o Q4)
mesh = Mesh.from_gmsh_model()                # modelo de gmsh abierto y ya mallado
physics.addBC_TemperatureEdges(mesh.boundary_edges('inner'), 100.0)     # grupo físico de dimensión 1, por nombre
```

La malla debe tener un solo tipo de elemento 2D. Ver `Example_Gmsh_Geometry.py`.

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

Una sola velocidad para todas las físicas, con estabilización para números de Péclet altos:

```python
model.physics.setVelocity([1.0, 0.0])     # número en 1D, vector o función de (x, y) en 2D; activa la convección en todas las partes
model.physics.setStabilization('SUPG')    # None para Galerkin puro
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
| `Example_Solver_Scaling.py` | Poisson 2D hasta 160 mil incógnitas: PARDISO frente a CG y CG+AMG, factorización reutilizada y exportación a ParaView |
| `Example_Gmsh_Geometry.py` | Cuarto de anillo (convergencia frente a la solución logarítmica) y placa con orificio, con mallas T3 de gmsh |
| `Example_Darcy_Transport.py` | Darcy en un medio con una inclusión de baja permeabilidad, conservación del caudal y pluma de un trazador arrastrada por la velocidad de Darcy |
| `Example_Stokes_Contraction.py` | Contracción plana 4:1 (entrada a un dado) en flujo reptante: conservación del caudal, pérdida de presión de entrada y corrección de Couette en tres mallas, vórtice de esquina |
| `Example_NavierStokes_Cavity.py` | Cavidad con tapa móvil a Re = 100, 400 y 1000 con continuación en Re; perfil central frente a Ghia et al. y líneas de corriente |
| `Example_Polymer_Slit_Die.py` | Fundido de polímero en un dado de rendija: Cross–WLF, calentamiento viscoso y acoplamiento térmico; perfil de velocidad, temperatura, balance de energía y curva caudal–presión |
| `Example_Capillary_Rheometer.py` | Reómetro capilar simulado (axisimétrico, Cross): tasa de corte aparente, esfuerzo en pared, pérdida de entrada (Bagley), corrección de Rabinowitsch; recupera la curva de viscosidad del modelo |
| `Example_Cylinder_Vortex_Shedding.py` | Desprendimiento de vórtices detrás de un cilindro (Schäfer–Turek 2D-2, Re = 100): Navier–Stokes transitorio con BDF2, C_D, C_L y Strouhal frente a la referencia (varios minutos) |
| `Example_Natural_Convection.py` | Cavidad calentada lateralmente (de Vahl Davis), Ra = 1e3–1e6 con continuación: Nusselt, velocidades máximas frente a la referencia, isotermas y líneas de corriente |
| `Example_Turbulent_Channel.py` | Canal turbulento con Spalart–Allmaras a Re_τ = 180, 395 y 590: u⁺(y⁺) frente a la ley de pared, U_c⁺ frente a la DNS, C_f frente a Dean, viscosidad turbulenta |
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
python Test/test_solver_scaling_output.py # AMG, solvers iterativos, reutilización de factorización y salida VTK (10)
python Test/test_triangles_gmsh.py        # triángulos T3 y gmsh: parche, manufacturada, anillo, grupos, Q4 de gmsh (10)
python Test/test_convection_supg.py       # velocidad compartida y SUPG: exactitud 1D, 2D = 1D, manufacturada, transitorio (9)
python Test/test_fluids_foundation.py     # Etapa 9: operadores vectoriales, leyes con gradientes, ley de potencia, Darcy (11)
python Test/test_stokes.py                # Etapa 10: Stokes con PSPG: parche, Poiseuille, solución exacta, cavidad, simetría, arranque (8)
python Test/test_navier_stokes.py         # Etapa 11: Navier–Stokes: tangente, Kovasznay, cavidad frente a Ghia, canal, límite de Stokes, Picard (8)
python Test/test_non_newtonian.py         # Etapa 12: leyes de viscosidad, ley de potencia, Bingham, Brinkman, μ(T), convección acoplada, jacobiano (8)
python Test/test_axisymmetric.py          # Etapa 13: medidas r-z, conducción radial, Hagen–Poiseuille, extensión uniaxial, tubo de ley de potencia, jacobiano (6)
python Test/test_transient_flow.py        # Etapa 14: Taylor–Green, arranque, límite estacionario, tangente con dt, calor transitorio, fuerzas (7)
python Test/test_buoyancy.py              # Etapa 15: de Vahl Davis, estratificación estable, especies y flotación solutal, tangente, flujo de calor (5)
python Test/test_turbulence.py            # Etapa 16: distancia a la pared, canal turbulento Re_τ 180/395, límite laminar, tangente, viscosidad turbulenta (5)
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
    Elements.py                registro de elementos de referencia (line2, line3, tri3, quad4)
    BasisFunctions.py          funciones de forma Q4 (numpy)
  Physics/
    Physics.py                 clase base de física (campos, BC, add_terms y build_system)
    Coupled.py                 CoupledPhysics: varias físicas en un modelo (Model(psc=[ht, mt]))
    HeatTransfer.py            ht: conducción, convección, radiación, BC en nodos y aristas
    MassTransfer.py            mt: especies, reacciones acopladas, BC en nodos y aristas
    Darcy.py                   darcy: flujo en medios porosos (presión) y velocidad de Darcy
    FluidFlow.py               ff: flujo incompresible (Stokes y Navier–Stokes, no newtoniano, calentamiento viscoso) con PSPG/SUPG/LSIC
    Rheology.py                leyes de viscosidad μ(γ̇, T): potencia, Carreau–Yasuda, Cross, Cross–WLF, Bingham, Herschel–Bulkley, Arrhenius
    Turbulence.py              sa: Spalart–Allmaras (SA-neg), distancia a la pared, viscosidad turbulenta
  Primals/                     campos escalares y vectoriales (valores nodales)
  Simulation/
    Discretization.py          geometría en los puntos de Gauss y facetas (cualquier dimensión)
    DofLayout.py               campos, componentes y numeración de DOF
    Operators.py               operadores de forma débil vectorizados (einsum), escalares y vectoriales
    PointwiseLaw.py            leyes no lineales con jacobianos automáticos (JAX)
    System.py                  matrices, términos no lineales (leyes de valores y de gradientes) y BC
    Solvers.py                 modelSolver: lineal, Newton, transitorio θ (lineal) y no lineal θ/BDF2 con Newton por paso
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
