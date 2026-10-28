# Arquitectura de FEM-sim

Este documento explica cómo fluye un problema desde la malla hasta la solución, qué responsabilidad tiene cada módulo
y cómo añadir nuevas piezas (otras físicas, condiciones de contorno, elementos o dimensiones) sin romper las existentes.

La base de cálculo es **independiente de la física y de la dimensión**: calor y transporte de especies ya la usan, y
fluidos o mecánica de sólidos (1D, 2D y 3D) se construirán sobre las mismas capas.

## 1. Capas

```
Mesh ─► Elements ─► Discretization ─► Operators ─► System ─► Solvers (modelSolver)
 nodos,   registro     N, B, detJw,      forma débil   matrices,    lineal, Newton,
 elementos de elementos facetas con      por punto     leyes,       θ-método
                        normales          de Gauss      BC
                          │                  ▲            ▲
                          └── DofLayout ─────┴── PointwiseLaw (JAX, jacobianos automáticos)
                                   ▲
                    Physics.build_system()  (ht, mt, ...)
```

| Capa | Módulo | Qué hace | Depende de la física |
|---|---|---|---|
| Elementos | `Pre_processing/Elements.py` | Registro de elementos de referencia (`line2`, `line3`, `quad4`): funciones de forma, Gauss y facetas | no |
| Geometría | `Simulation/Discretization.py` | `N`, `B = dN/dx`, `detJw`, `x_gp` en los puntos de Gauss; `FacetSet` (puntos en 1D, aristas en 2D) con medida y normales salientes | no |
| Grados de libertad | `Simulation/DofLayout.py` | Campos con `ncomp` componentes, numeración intercalada por nodo, reparto por bloques | no |
| Operadores | `Simulation/Operators.py` | `diffusion`, `diffusion_tensor`, `convection`, `mass`, `source`, `facet_mass`, `facet_source` con coeficientes por punto de Gauss | no |
| Leyes puntuales | `Simulation/PointwiseLaw.py` | Función JAX → `vmap` + `jacfwd` compilados una vez, aplicados a todos los puntos | no |
| Sistema | `Simulation/System.py` | Acumula matrices, términos no lineales y BC; da `linear(x)`, `residual_and_tangent(x)`, `mass()` | no |
| Física | `Physics/HeatTransfer.py`, `Physics/MassTransfer.py` | Construye el `System` con sus operadores y leyes | sí |
| Solvers | `Simulation/Solvers.py` | Lineal, Newton con line search, θ-método transitorio, todo sobre vectores de DOF | no |

## 2. Flujo de una simulación

1. **Malla.** `Mesh` contiene `NL` (`Node` con `id`, `coor`, `BC`) y `EL` (`Element`). Guarda `element_type` (`'line2'`, `'quad4'`).
   - 1D: `Generate_Mesh(PD, L, NoE, ElementType.line)`. 2D: `generate2DMesh((Lx, Ly), (nx, ny), ElementType.quadrilateral)`.
   - `boundary_facets(selector)` (alias `boundary_edges`) devuelve las facetas del contorno cuyo punto medio cumple el predicado.
2. **Modelo.** `Model(name, dim, mesh, mat, psc)` crea la física, que crea los campos (`scalarField`).
3. **Condiciones.** `physics.addBC_*` escribe en `node.BC[campo]` (Dirichlet, y Newton/Neumann en un nodo en 1D) o en
   `physics.facet_bcs` (Newton/Neumann sobre facetas: aristas en 2D).
4. **Configuración.** `solverConfiguration(...)` guarda `model.solverOptions`.
5. **Solve.** `Model.solve()` reinicia el caché del sistema y crea un `modelSolver`. Éste pide a `Model` el `System`
   (`physics.build_system(model, disc, layout)`, construido una sola vez por solve) y lo resuelve.
   El resultado queda en `model.sol[campo]` (nodal en estacionario, `(nodos, tiempos)` en transitorio).

## 3. Convenciones

| Tema | Convención |
|---|---|
| Residuo | `R = A x - b + términos no lineales`; el solver de Newton resuelve `J dx = -R` |
| Dirichlet | fila identidad en `A`, `b = valor` (las columnas no se modifican); residuo `x - valor` |
| Flujo de contorno | siempre **saliente**: Neumann `q = -k ∂T/∂n`; Newton/Robin `q = h (T - T_ext)` |
| Fuentes | `F` positivo produce el campo: el residuo lleva `- ∫ N F` |
| Reacciones | `r_s = Σ_k ν_{s,k} k_k(T) Π c_j^{n_j}` con `ν < 0` en reactivos; el residuo lleva `- ∫ N r_s` |
| Radiación | `h(T)` con Kelvin internamente; las temperaturas de entrada son °C |
| DOF | `dof = nodo * n_total + desplazamiento_del_campo + componente` (intercalado) |
| Elementos | Q4 antihorarios: BL, BR, TR, TL. Líneas de menor a mayor coordenada |
| Gauss | 2 puntos por dirección (`Discretization(mesh, order=2)`) |
| Coeficientes | número, array en puntos de Gauss, o función de las coordenadas `f(x)`, `f(x, y)` |
| `h` y `flux` como función | en una BC, una función recibe el **valor del campo** (p. ej. `h(T)` de radiación) |

## 4. Módulos en detalle

### `Source/core.py` — `Model`
`solve()`, `postProcess()` (1D), `postProcess2D()`, `layout` (el `DofLayout` de los campos de la física), `_system()`
(construye y guarda el `System`), `dirichletDofs()`, `_state()` (vector de DOF con los valores de los campos).
Sigue existiendo la ruta antigua por elemento para físicas sin `build_system`; se retira en la Etapa 6c.

### `Source/Simulation/System.py`
- `add_matrix`, `add_vector`, `add_mass`: partes lineales (el residuo es `A x - b`).
- `add_volume_law(law, entradas, salidas, sign)`: término volumétrico no lineal (reacciones, k(T), ...).
- `add_flux(campo, facetas, q)` y `add_robin(campo, facetas, h, ext)`: BC sobre facetas, con `h` o `q` numéricos o
  funciones del campo (leyes puntuales de faceta).
- `add_boundary_conditions(mesh, facet_bcs)`: lee Dirichlet de los nodos y las BC Newton/Neumann (nodos en 1D, facetas en general).
- `linear(x)`: `(A, b)`. Con términos no lineales devuelve la linealización de Newton en `x` (`A = J`, `b = J x - R`).
- `residual_and_tangent(x)`, `mass()`, `is_dirichlet`, `dirichlet_value`.

### `Source/Physics/`
- `Physics.py`: clase base. `fields()` (un escalar por variable) y `build_system(...)` (a implementar por cada física).
- `HeatTransfer.py` (`ht`): difusión `k`, convección `rho Cp v`, fuente, masa `rho Cp`, Dirichlet, Neumann, Newton y radiación.
- `MassTransfer.py` (`mt`): varias especies con difusividad propia (`setDiffusivity`), convección, fuentes externas
  (`setSource`), **varias reacciones** (`addReaction`, se acumulan) acopladas por una sola ley puntual, y BC por nodos
  o aristas (`addBC_Concentration`, `addBC_Outflow`, `addBC_ConcentrationEdges`, `addBC_FluxEdges`, `addBC_MassTransferEdges`).
  La velocidad `physics.vel` es un escalar en 1D y un vector en 2D.

### `Source/Simulation/Solvers.py`
- `linearSolver`: ensamblado + `DirectSolver` o `IterativeSolver`.
- `newtonSolver`: tolerancia relativa sobre `||R||`, búsqueda de línea por *backtracking*.
- `transientSolver`: θ-método con paso fijo, LU reutilizada si la matriz no cambia, DOF fijados por Dirichlet,
  `timeValues` guardados por campo.

## 5. Cómo extender el proyecto

### 5.1 Añadir una física
1. Crear una clase que herede de `physics`, con sus variables (`scalarField`) y constantes.
2. Implementar `build_system(model, disc, layout)`: crear un `System`, añadir los operadores (`ops.diffusion`, ...)
   repartidos con `layout.scatter_matrix`, y llamar a `system.add_boundary_conditions(model.mesh, self.facet_bcs)`.
3. Para términos no lineales, definir una función JAX `f(u)` y envolverla con `PointwiseLaw`; añadirla con
   `system.add_volume_law(...)`. El jacobiano sale por autodiff (conserve el objeto `PointwiseLaw` entre ensamblados).
4. Añadir métodos `addBC_*` que escriban en `node.BC` o en `facet_bcs` (`{'type', 'field', 'facets', ...}`).
5. Probar con soluciones analíticas y manufacturadas (ver `Test/test_species.py`) y un chequeo de jacobiano por
   diferencias finitas.

### 5.2 Fluidos y mecánica de sólidos (lo que ya está y lo que falta)
- **Ya está:** campos con varias componentes en `DofLayout` (`FieldSpec('u', ncomp=dim)`), reparto por bloques entre
  campos distintos, facetas con normales salientes (presiones y tracciones), leyes puntuales con jacobianos automáticos
  (viscosidad, tensión-deformación), Newton y transitorio por DOF.
- **Falta:** operadores vectoriales en `Operators.py` (deformación-desplazamiento `BᵀCB`, divergencia-presión),
  elementos de orden mixto para velocidad-presión (`quad9`, `tri6`) en `Elements.py`, estabilización de convección,
  y estado interno para plasticidad.

### 5.3 Añadir un elemento o una dimensión
- Añadir una entrada a `ELEMENTS` en `Elements.py` (nodos de referencia, `shape`, `quadrature`, facetas locales y
  elemento de faceta). `Discretization` y los operadores son genéricos en `dim`.
- Añadir el generador de malla (y `element_type`) en `Mesh.py`. Las normales de faceta en 3D se completan en
  `Discretization.facets`.
- Pruebas: partición de la unidad (`Test/test_foundation.py`), parche lineal y convergencia (`Test/test_2d_poisson.py`).

### 5.4 Añadir un tipo de condición de contorno
- Sobre facetas: método en `System` (como `add_flux` o `add_robin`) y rama en `add_boundary_conditions`.
- Dirichlet por componente: `System.set_dirichlet(dofs, valores)`.

### 5.5 Añadir un solver o un precondicionador
- Directo: valor en `DirectSolvers` y rama en `DirectSolver.solve()`. Iterativo: valor en `IterativeSolvers`, método en
  `IterativeSolver` y rama en `solve()` (lanza `ValueError` si no la reconoce).

### 5.6 Añadir una prueba
- Scripts ejecutables en `Test/` con `matplotlib.use("Agg")`; registrarlos en `run_tests.py`.
- Comparar con valores analíticos, soluciones manufacturadas, equivalencias entre rutas o diferencias finitas del jacobiano.

## 6. Historial relevante
- `backup/vmap-wip`: intento de ensamblado por lotes con `jax.vmap`. Ese código ya no existe en `test-branch`.
- `test-branch`: etapas 1 a 5 commiteadas; la Etapa 6 (base multifísica y especies) está en curso.
