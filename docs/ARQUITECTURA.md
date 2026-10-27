# Arquitectura de FEM-sim

Este documento explica cómo fluye un problema desde la malla hasta la solución, qué responsabilidad tiene cada módulo
y cómo añadir nuevas piezas sin romper las existentes.

## 1. Flujo de una simulación

```
Mesh  ──►  Model (core.py)  ──►  physics (ht, mt)  ──►  ensamblado  ──►  modelSolver  ──►  sol / timeVector
 nodos,      solverConfiguration    variables, BC,        vectorizado      lineal, Newton,     nodal (sol['T'])
 elementos   solve() / postProcess  constantes            (1D/2D)          θ-método
```

1. **Malla.** `Mesh` contiene `NL` (lista de `Node`, con `id`, `coor` y `BC`) y `EL` (lista de `Element`, con `nodes`).
   - 1D: `Generate_Mesh(PD, L, NoE, ElementType.line)` y `setElementShapeFunction(...)`.
   - 2D: `generate2DMesh((Lx, Ly), (nx, ny), ElementType.quadrilateral)`. Elementos Q4 antihorarios.
2. **Modelo.** `Model(name, dim, mesh, mat, psc)` crea la física (`psc`), que a su vez crea los campos (`scalarField`).
3. **Condiciones.** Los métodos `physics.addBC_*` escriben en `node.BC[variable]` (BC de nodo) o en
   `physics.edge_bcs` (BC de arista, solo 2D).
4. **Configuración.** `solverConfiguration(...)` guarda `model.solverOptions` (estudio, tipo, método, solver, θ, paso, tiempo).
5. **Solve.** `Model.solve()` crea un `modelSolver`, que decide la ruta según el estudio y el tipo de problema.
   Al final guarda `model.sol[variable]` (nodal en estacionario, `(nodos, tiempos)` en transitorio).

## 2. Módulos

### `Source/core.py` — `Model`

- `solve()`, `postProcess()` (1D), `postProcess2D()` (2D, contorno con `tricontourf`).
- `assembleGlobalSystem`, `assembleGlobalSystemNonLinear`, `assembleMassMatrix`: **cada uno comprueba primero
  `_heat_system()`**. Si la física declara `supports_vectorized_assembly = True` y tiene una sola variable, usa el
  ensamblador vectorizado. Si no, sigue con la ruta antigua por elemento (`physics.getElementMatrix`, etc.).
- `dirichletMask(variable)`: máscara booleana de nodos con Dirichlet.
- `_geometry` es un caché de `LinearGeometry`, que se construye la primera vez.

### `Source/Pre_processing/`

- `Mesh.py`: generación 1D y 2D, `boundary_edges(selector)`, `Element.getCoor()` (escalares en 1D, `(x, y)` en 2D).
- `BasisFunctions.py`:
  - Funciones de forma 1D Lagrange (jax y sympy, usadas por la ruta antigua).
  - `q4_shape_functions(xi, eta)`: Q4 en numpy, usada por el ensamblador 2D.

### `Source/Simulation/`

- `VectorizedAssembler.py` (**el núcleo de cálculo actual**):
  - `LinearGeometry(mesh)`: precalcula, una vez por malla, las funciones de forma y gradientes en los puntos de Gauss,
    `detJw`, `B = dN/dx` y las coordenadas de Gauss. Sirve para 1D y 2D (`mesh.PD`).
  - Matrices y vectores por elemento con `einsum`, después `_scatter` a `scipy.sparse`:
    `stiffness(k)`, `mass(c)`, `convection(c, vel)` (solo 1D), `load(f)` (f número o función de (x, y)).
  - Aristas (2D): `edge_lengths`, `edge_mass`.
  - `HeatSystem(model, geometry)`: ensambla el problema de calor.
    - `linear(T)` → `(A, b)`, con filas identidad en Dirichlet.
    - `residual_and_tangent(T)` → `(J, R)` para Newton (solo 1D).
- `Solvers.py`: `modelSolver`.
  - `linearSolver`: ensamblado + `DirectSolver` o `IterativeSolver`.
  - `newtonSolver`: tolerancia relativa sobre `||R||`, búsqueda de línea por *backtracking*.
  - `transientSolver`: θ-método con paso fijo, LU reutilizada si la matriz no cambia.
- `DirectSolver.py`: PARDISO (`pypardiso`) o SuperLU.
- `IterativeSolver.py`: BiCGSTAB, GMRES y CG, con precondicionador iLU. Registra un aviso si `info != 0`.

### `Source/Physics/`

- `Physics.py` — clase base `physics`:
  - Constantes (`K_const`, `C_const`, `M_const`, `F_const`), flag `Convection`, velocidad `vel`.
  - `edge_bcs`: lista de BC de arista.
  - `supports_vectorized_assembly = False` por defecto.
  - Ruta antigua por elemento, basada en JAX (`_calc_*`, `getElementMatrix`, `getResidualVector`, ...).
- `HeatTransfer.py` — `ht`:
  - `supports_vectorized_assembly = True`.
  - BC de nodo: `addBC_Temperature`, `addBC_Convection`, `addBC_HeatFlux`, `addBC_Radiation`.
  - BC de arista (2D): `addBC_TemperatureEdges`, `addBC_HeatFluxEdges`, `addBC_ConvectionEdges`.
- `MassTransfer.py` — `mt`: transporte de especies con reacción. Usa la ruta antigua. **Sin validar.**

### `Source/Primals/`, `Source/Material.py`, `Source/enums.py`, `Source/logger_config.py`

- `scalarField`: valores nodales, `timeValues` (transitorio), `getFieldValues(study)`.
- `material(name, **kwargs)`: atributos arbitrarios (`k`, `rho`, `Cp`, ...).
- `enums.py`: `ElementType`, `ShapeFunctionType`, `StudyType`, `ProblemType`, `SolverType`, `DirectSolvers`, `IterativeSolvers`, `Preconditioners`.
- `logger_config.py`: crea un archivo `Logs/sim_<fecha>.log` por ejecución. Cada importación de `Source` crea uno nuevo.

## 3. Convenciones

| Tema | Convención |
|---|---|
| Ecuación | `-∇·(k ∇T) = Q` (conducción), `ρ Cp ∂T/∂t` en transitorio |
| Forma débil | `K T + C T + h T = F + ...`; el residuo es `R = (K + C) T - F + flujos` |
| Dirichlet | fila identidad en `A` y valor prescrito en `b` (las columnas no se modifican) |
| Newton | calor que sale `q = h (T - T_ext)`. Término `+ h (T - T_ext)` en el residuo |
| Neumann | flujo que sale `q = -k ∂T/∂n`. Término `+ q` en el residuo |
| Radiación | `h(T)` con temperaturas en Kelvin; `T` de entrada en °C |
| Nodos | `node.id` es el índice global de la variable (`nVar = 1` en calor) |
| Elementos | Q4 en orden antihorario: BL, BR, TR, TL. Lineales: de menor a mayor coordenada |
| Puntos de Gauss | 2 en 1D, 2×2 en 2D (`GAUSS_XI`, `GAUSS_W` en `VectorizedAssembler.py`) |

## 4. Cómo extender el proyecto

### 4.1 Añadir una física con coeficientes constantes

1. Crear una clase que herede de `physics` (como `ht`), con sus constantes y variables (`scalarField`).
2. Si la forma débil es lineal en la variable y los coeficientes son números, activar
   `supports_vectorized_assembly = True` y usar `HeatSystem` como modelo. Para una física nueva, lo habitual es
   generalizar `HeatSystem` (hoy está especializado en calor).
3. Si la física tiene coeficientes dependientes del estado (como las reacciones de `mt`), seguir por la ruta antigua
   o implementar `residual_and_tangent` sin contar con la ruta por elemento.
4. Escribir un test con solución analítica (ver `Test/test_analytic.py`) y, si hay dos rutas, un test de equivalencia
   (ver `Test/test_vectorized.py`).

### 4.2 Añadir un tipo de condición de contorno

- **Nodo (1D):** en `HeatSystem.__init__`, leer el nuevo tipo de `node.BC[variable]` y aplicarlo en `linear()` y
  `residual_and_tangent()`.
- **Arista (2D):** añadir una rama en `HeatSystem.__init__` (lista `edge_*`) y su contribución en `linear()`.
  Las integrales de arista están en `LinearGeometry.edge_mass` y en el reparto `L/2` por nodo.
- Exponer el método en la física (`addBC_*Edges`) que añade la entrada a `edge_bcs`.

### 4.3 Añadir un elemento o una dimensión

- Crear la referencia del elemento (funciones de forma y gradientes en los puntos de Gauss) y añadir su rama en
  `LinearGeometry.__init__` (hoy: `dim == 1` y `dim == 2`).
- El resto del ensamblado (`stiffness`, `mass`, `load`, `_scatter`) es genérico en `dim` mientras `B` tenga forma
  `(nel, ngp, nne, dim)`.
- Añadir el generador de malla en `Mesh` y verificar que `Element.getCoor()` devuelve la forma esperada.
- Test de parche (un campo lineal debe reproducirse exacto) y test de convergencia (`Test/test_2d_poisson.py`).

### 4.4 Añadir un solver o un precondicionador

- Solver directo: nuevo valor en `DirectSolvers` y rama en `DirectSolver.solve()`.
- Iterativo: nuevo valor en `IterativeSolvers`, método en `IterativeSolver` y rama en `solve()`.
  Si `solve()` no reconoce el valor, lanza `ValueError`.
- Nuevo estudio o tipo de problema: rama en `modelSolver.solve()` (`steadyStateSolver`, `transientSolver`).

### 4.5 Añadir una prueba

- Todas las pruebas están en `Test/` y son scripts ejecutables con `if __name__ == "__main__"`.
- Usar `matplotlib.use("Agg")` al principio para no abrir ventanas.
- Cada prueba debe comparar con un valor analítico, un valor de referencia documentado o con la ruta antigua.

## 5. Dónde está cada cosa en el historial

- `backup/vmap-wip`: intento de ensamblado por lotes con `jax.vmap` (`Physics.get_batched_*`). No se usa; contiene
  errores conocidos (ver `docs/ESTADO.md`).
- `test-branch`: rama de trabajo. Las etapas 1 a 4 están commiteadas en orden (`d464fa9`, `d7388e3`, `141d15e`, `fa50dbe`).
