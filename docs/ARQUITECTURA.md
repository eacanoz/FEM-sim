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
                    Physics.add_terms() / build_system()  (ht, mt, CoupledPhysics)
```

| Capa | Módulo | Qué hace | Depende de la física |
|---|---|---|---|
| Elementos | `Pre_processing/Elements.py` | Registro de elementos de referencia (`line2`, `line3`, `tri3`, `quad4`): funciones de forma, Gauss y facetas | no |
| Geometría | `Simulation/Discretization.py` | `N`, `B = dN/dx`, `detJw` (con el factor `r` si es axisimétrica), `area`, `hoop = N/r`, `x_gp` en los puntos de Gauss; `FacetSet` (puntos en 1D, aristas en 2D) con medida y normales salientes | no |
| Grados de libertad | `Simulation/DofLayout.py` | Campos con `ncomp` componentes, numeración intercalada por nodo, reparto por bloques | no |
| Operadores | `Simulation/Operators.py` | Escalares: `diffusion`, `diffusion_tensor`, `convection`, `mass`, `source`, `gradient_source`, `facet_mass`, `facet_source`. Vectoriales: `vector_mass`, `vector_diffusion`, `viscous`, `divergence`, `vector_source`, `facet_vector_source`. PSPG: `element_size`, `pspg_tau`, `pspg_mass`, `recovered_stress`, `recovered_stress_divergence`, `gauss_stress_divergence`, `hoop_stress_terms`, `pspg_weight`, `lsic_tau`. Coeficientes por punto de Gauss | no |
| Leyes puntuales | `Simulation/PointwiseLaw.py` | Función JAX → `vmap` + `jacfwd` compilados una vez, aplicados a todos los puntos | no |
| Sistema | `Simulation/System.py` | Acumula matrices, términos no lineales y BC; da `linear(x)`, `residual_and_tangent(x)`, `mass()` | no |
| Física | `Physics/HeatTransfer.py`, `Physics/MassTransfer.py`, `Physics/Darcy.py`, `Physics/FluidFlow.py` (+ `Physics/Rheology.py`) | Añade sus operadores y leyes a un `System` (`add_terms`) | sí |
| Composición | `Physics/Coupled.py` | Reúne varias físicas en un modelo: un `DofLayout` y un `System` compartidos | no |
| Solvers | `Simulation/Solvers.py` | Lineal, Newton con line search, θ-método transitorio, todo sobre vectores de DOF | no |

## 2. Flujo de una simulación

1. **Malla.** `Mesh` contiene `NL` (`Node` con `id`, `coor`, `BC`) y `EL` (`Element`). Guarda `element_type` (`'line2'`, `'tri3'`, `'quad4'`). Se genera con `generate2DMesh` (Q4 o T3) o se importa de gmsh con
   `Mesh.read_msh(path)` / `Mesh.from_gmsh_model()`: los grupos físicos de dimensión 1 quedan en `mesh.groups` y se usan por
   nombre en `boundary_edges('nombre')`; los de dimensión 2 en `mesh.regions`. Solo un tipo de elemento por malla.
   - 1D: `Generate_Mesh(PD, L, NoE, ElementType.line)`. 2D: `generate2DMesh((Lx, Ly), (nx, ny), ElementType.quadrilateral)`.
   - `boundary_facets(selector)` (alias `boundary_edges`) devuelve las facetas del contorno cuyo punto medio cumple el predicado.
2. **Modelo.** `Model(name, dim, mesh, mat, psc)` crea la física, que crea los campos (`scalarField`).
3. **Condiciones.** `physics.addBC_*` escribe en `node.BC[campo]` (Dirichlet, y Newton/Neumann en un nodo en 1D) o en
   `physics.facet_bcs` (Newton/Neumann sobre facetas: aristas en 2D). `addBC_ConvectionEdges` y `addBC_RadiationEdges`
   aceptan un `h` que dependa de la temperatura (ley de faceta no lineal con jacobiano automático).
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
| Campos vectoriales | `vectorField(..., ncomp)`: valores `(nodos, ncomp)`, transitorio `(nodos, ncomp, tiempos)`; DOF de elemento ordenados por nodo `(i, a) -> i * nc + a` |
| Dirichlet vectorial | `node.BC['u'] = {'type': 'Dirichlet', 'value': [ux, None]}`: `None` deja libre la componente (simetría, deslizamiento) |
| Flujo incompresible | `R_u = ∫2μ ε(u):ε(v) - ∫p div v + ∫ρ u_t·v - ∫f·v - ∮t·v`; `R_p = ∫q div u + ∫τ ∇q·(ρ u_t + ∇p - f - div σ̃)`. La forma de esfuerzo exige, en una salida con perfil desarrollado, fijar la velocidad tangencial (`addBC_OutletEdges(fully_developed=True)`) |
| Tracción | BC de faceta `{'type': 'Traction', 'field': 'u', 'facets': ..., 'traction': t}`: fuerza por unidad de área **sobre** el dominio (`b += ∫ t·v`) |
| Elementos | Q4 antihorarios: BL, BR, TR, TL. Líneas de menor a mayor coordenada |
| Gauss | 2 puntos por dirección (`Discretization(mesh, order=2)`) |
| Coeficientes | número, array en puntos de Gauss, o función de las coordenadas `f(x)`, `f(x, y)` |
| `h` y `flux` como función | en una BC, una función recibe el **valor del campo** (p. ej. `h(T)` de radiación) |
| Transitorio | θ-método de paso fijo sobre `M x' + A x = b`; con términos no lineales cada paso usa la linealización de Newton: `(M + θ dt J) Δx = -dt R(x_n)` |
| Estado inicial | `physics.initField(campo, valor)`: número o función de las coordenadas |

## 4. Módulos en detalle

### `Source/core.py` — `Model`
`solve()`, `postProcess()` (1D), `postProcess2D()`, `layout` (el `DofLayout` de los campos de la física), `_system()`
(construye y guarda el `System`), `dirichletDofs()`, `_state()` (vector de DOF con los valores de los campos).
Toda física debe implementar `build_system`; la antigua ruta por elemento (JAX por elemento) se retiró en la Etapa 6c.

### `Source/Simulation/System.py`
- `add_matrix`, `add_vector`, `add_mass`: partes lineales (el residuo es `A x - b`).
- `add_volume_law(law, entradas, salidas, sign)`: término volumétrico no lineal (reacciones, fuentes `Q(T, c)`, ...).
- `add_diffusion_law(law, campo, entradas)`: difusión no lineal `∫ k(u) ∇campo · ∇N`, con `k` una ley puntual de los
  campos de entrada (k(T), D(c, T), viscosidad). El jacobiano incluye la rigidez `∫ k ∇Ni·∇Nj` y la dependencia de `k`
  respecto a cada entrada, que acopla el campo con ellas.
- `add_gradient_law(ley, entradas, salidas, sign=1)`: residuo `∫ (N s + ∇N · F)` con `s` y `F` funciones de los **valores y
  gradientes** de las entradas (`GradientLawTerm`). Por cada entrada la ley recibe sus valores (`nc`) y su gradiente
  (`nc * dim`, fila a fila); por cada salida devuelve una fuente `s` (`nc`) y un flujo `F` (`nc * dim`). El jacobiano
  sale por autodiff respecto a valores y gradientes. Sirve para `p`-Laplaciano/ley de potencia, convección por un campo
  calculado, `μ(γ̇)` y modelos de turbulencia. `value_gradient_basis` da a la vez la interpolación y las funciones de prueba.
- `add_term(term)`: cualquier término no lineal con `contribution(x) -> (R, J)` (p. ej. `FlowTerm`, `FieldConvectionTerm`).
- `add_traction(campo, facetas, t)`: tracción sobre un campo vectorial. `add_normal_pressure(campo, facetas, p)`:
  tracción `-p n` (BC `NormalTraction`, salidas).
- `add_flux(campo, facetas, q)` y `add_robin(campo, facetas, h, ext)`: BC sobre facetas, con `h` o `q` numéricos o
  funciones del campo (leyes puntuales de faceta).
- `add_boundary_conditions(mesh, facet_bcs)`: lee Dirichlet de los nodos y las BC Newton/Neumann (nodos en 1D, facetas en general).
- `Model.dirichletDofs()` lee `System.is_dirichlet` (incluye componentes sueltas y valores fijados por la física).
- `linear(x)`: `(A, b)`. Con términos no lineales devuelve la linealización de Newton en `x` (`A = J`, `b = J x - R`).
- `residual_and_tangent(x)`, `mass()`, `is_dirichlet`, `dirichlet_value`.

### `Source/Physics/`
- `Physics.py`: clase base. `fields()` (un escalar por variable), `add_terms(system, model, disc, layout)` (a implementar por
  cada física: operadores y leyes, sin condiciones de contorno) y `build_system(...)` (crea el `System`, llama a
  `add_terms` y añade las BC). `KELVIN_OFFSET` es la constante de conversión °C → K compartida.
- Solvers (`Simulation/Solvers.py`, `DirectSolver.py`, `IterativeSolver.py`): `modelSolver._solve_system(A, b)` es el único punto
  de resolución lineal (estacionario, pasos de Newton y transitorio). Directo: `FactorizedSolver` (PARDISO o SuperLU) guarda la
  factorización de la última matriz y solo refactoriza si cambia; vive en el modelo (`model._factorizer`), de modo que `solve()`
  repetidos con la misma matriz no vuelven a factorizar. Iterativo: se eliminan los nodos con Dirichlet (filas identidad) y se
  resuelve el sistema reducido con CG/BiCGSTAB/GMRES y `build_preconditioner` (`iLU`, `Jacobi`, `AMG` de pyamg), reutilizando el
  precondicionador mientras la matriz no cambie. Un transitorio sin términos no lineales (`System.is_linear`) ensambla y
  factoriza una sola vez.
- Salida (`Post_processing/VTK.py`): `write_vtu`, `write_pvd`, `export_model` (`Model.exportVTK`). VTK XML en ASCII, sin
  dependencias; line2, tri3 y quad4.
- Convección: `physics.setVelocity(v)` y `setStabilization('SUPG')` (en `CoupledPhysics` se aplican a todas las partes: una sola
  velocidad compartida). `physics.add_convection(...)` añade `Operators.convection` y, con SUPG, `supg_tau` (parámetro
  `tau = h xi(Pe) / (2|a|)`, `xi = coth(Pe) - 1/Pe`, `h` de Hughes) y `supg_terms` (rigidez, masa y fuente consistentes).
  La estabilización es lineal: no alcanza a las reacciones no lineales, y `tau` usa la difusión constante si existe
  (con `k(T)` o `D(T)` usa advección pura, algo más difusivo).
- `Coupled.py` (`CoupledPhysics`): `Model(psc=[ht, mt])` instancia cada física y las expone por nombre (`part_name`:
  `model.physics.heat`, `model.physics.mass`). Une las variables (error si dos físicas repiten una), concatena los campos
  y llama a `add_terms` de cada parte sobre **un solo** `System`; las BC se añaden una vez con las de todas. Los métodos
  de las partes se delegan (`model.physics.addBC_Temperature`), pero **asignar** un atributo de una parte en la
  combinación se rechaza con un error que indica la parte (`model.physics.heat.Convection = ...`).
- `HeatTransfer.py` (`ht`): difusión `k` (constante, función de las coordenadas o `setConductivityLaw(k(T))`), convección
  `rho Cp v`, fuente, masa `rho Cp`, Dirichlet, Neumann, Newton y radiación.
- `MassTransfer.py` (`mt`): varias especies con difusividad propia (`setDiffusivity`), convección, fuentes externas
  (`setSource`), **varias reacciones** (`addReaction`, se acumulan) acopladas por una sola ley puntual, y BC por nodos
  o aristas (`addBC_Concentration`, `addBC_Outflow`, `addBC_ConcentrationEdges`, `addBC_FluxEdges`, `addBC_MassTransferEdges`).
  La velocidad `physics.vel` es un escalar en 1D y un vector en 2D.
  `addReaction(..., T, ..., dH)`: `T` es un número en Kelvin (isotérmica) o el nombre del campo de temperatura (`'T'`, en
  °C, convertido a Kelvin dentro de la ley); `dH` [J/mol] (negativo = exotérmica) libera `-dH * rate` [W/m³] en ese
  campo. Una sola ley puntual tiene como entradas las especies y la temperatura y como salidas la producción de cada
  especie y el calor, de modo que los bloques cruzados del jacobiano salen por autodiff.

### `Source/Simulation/Solvers.py`
- `linearSolver`: ensamblado + `DirectSolver` o `IterativeSolver`.
- `newtonSolver`: tolerancia relativa sobre `||R||` escalada fila a fila con la diagonal del jacobiano inicial (cada ecuación
  queda en las unidades de su incógnita, de modo que T ~ 300 no domina sobre c ~ 1), búsqueda de línea por *backtracking*.
  Si el residuo deja de ser finito con el paso mínimo, se detiene con un aviso y conserva el último estado finito.
- `transientSolver`: θ-método con paso fijo, LU reutilizada si la matriz no cambia, DOF fijados por Dirichlet,
  `timeValues` guardados por campo.

## 5. Cómo extender el proyecto

### 5.1 Añadir una física
1. Crear una clase que herede de `physics`, con sus variables (`scalarField`) y constantes.
2. Implementar `add_terms(system, model, disc, layout)`: añadir al `System` los operadores (`ops.diffusion`, ...)
   repartidos con `layout.scatter_matrix`, y llamar a `system.add_boundary_conditions(model.mesh, self.facet_bcs)`.
3. Para términos no lineales, definir una función JAX `f(u)` y envolverla con `PointwiseLaw`; añadirla con
   `system.add_volume_law(...)`. El jacobiano sale por autodiff (conserve el objeto `PointwiseLaw` entre ensamblados).
4. Añadir métodos `addBC_*` que escriban en `node.BC` o en `facet_bcs` (`{'type', 'field', 'facets', ...}`).
5. Probar con soluciones analíticas y manufacturadas (ver `Test/test_species.py`) y un chequeo de jacobiano por
   diferencias finitas.

### 5.1b Acoplar físicas
- Declarar el modelo con varias físicas: `Model(psc=[ht, mt, ...])`. Cada una debe tener `part_name` y `add_terms`.
- El acoplamiento es un término cuya ley puntual lee campos de varias físicas y escribe en las de otras: se añade con
  `system.add_volume_law(ley, entradas, salidas)` desde la física que posee la ley (como las reacciones de `mt`, que
  escriben en el campo de temperatura) o con `system.add_diffusion_law(ley, campo, entradas)` si lo que depende de otros
  campos es un coeficiente de difusión (`D(T)`, `k(c)`). Desde las físicas: `ht.setConductivityLaw(fn, depends_on=(...))` y
  `mt.setDiffusivity(especie, fn, depends_on=(...))` crean la ley (compilada una vez con `cached_law`) y llaman a
  `add_diffusion_law`; el tangente incluye `dk/du` para cada entrada, así que aparecen los bloques cruzados.
- Probar con una referencia independiente (una EDO del caso uniforme), con invariantes exactos del sistema acoplado y con
  el jacobiano frente a diferencias finitas, incluidos los bloques cruzados (ver `Test/test_coupling.py`).

### 5.2 Fluidos y mecánica de sólidos (lo que ya está y lo que falta)
El plan completo de fluidos está en [PLAN_FLUIDOS.md](PLAN_FLUIDOS.md).
- **Ya está (Etapa 9):** campos vectoriales (`Primals/Vector.py`, `physics.fields()` lee `ncomp`), operadores
  vectoriales (viscoso `2μ ε:ε`, divergencia y su traspuesta para la presión, masa, fuente y tracción), leyes con
  gradientes (`add_gradient_law`), Dirichlet por componente, `Discretization.gradient` y `project` (proyección L2
  concentrada), `physics.derived_fields(model)` (cantidades derivadas que `solve()` añade a `model.sol`), VTK vectorial.
  `darcy` (`Physics/Darcy.py`): presión como incógnita, `velocity('gauss')` lista para `setVelocity` de otra física en
  la misma malla, `velocity('nodes')` para salida.
- **Ya está (Etapa 10):** `ff` (`Physics/FluidFlow.py`), Stokes con velocidad y presión en los mismos nodos y PSPG.
  `tau = 1/sqrt((2ρ/dt)² + (2ρ|u|/h)² + (12μ/h²)²)` (el de Tezduyar/Shakib dividido por ρ, definido con ρ = 0). Como
  `div(2μ ε(u_h))` se anula dentro de un elemento lineal, la PSPG pierde consistencia; `viscous_projection=True`
  (por defecto) usa el esfuerzo recuperado `σ̃` (`Discretization.recovery`: ajuste lineal por mínimos cuadrados de los
  promedios por elemento de un parche; en los nodos de contorno el parche incluye el de sus vecinos interiores) y
  hace exacto el Poiseuille en Q4. Con todas las velocidades prescritas la presión se fija en un nodo
  (`setPressureReference`, o el nodo 0 automáticamente). `flowRate(aristas)` integra `u·n`.
- **Ya está (Etapa 11):** `ff.setInertia(True, linearization='newton'|'picard')`. `NavierStokesTerm` (hoy `FlowTerm`, en
  `FluidFlow.py`, añadido con `System.add_term`) ensambla en bloque la convección de Galerkin y los términos de
  estabilización sobre el residuo completo `R_M = ρ(u·∇)u + ∇p - f - div σ̃`: SUPG `∫τρ(u·∇v)·R_M`, PSPG `∫τ∇q·R_M` y
  grad-div `∫τ_c div u div v` (`τ_c = ρ|u|h/2·min(Re_h/3, 1)`). El tangente es exacto con `τ` congelado (se omite
  `dτ/du`); `div σ̃` es una matriz dispersa `V` que acopla los elementos de cada parche de recuperación. `model.newton_iterations`
  guarda las iteraciones del último solve no lineal. Continuación en Re: cambiar `setViscosity` y volver a resolver
  (los campos conservan la solución anterior).
- **Ya está (Etapa 12):** `ff.setViscosityLaw(ley)` con las leyes de `Physics/Rheology.py` (`ViscosityLaw`: `μ(γ̇², T)`
  en `jax.numpy`, `γ̇ = sqrt(2 ε:ε)`, temperatura en °C). `NavierStokesTerm` pasó a llamarse `FlowTerm` y reúne lo
  no lineal del flujo: viscosidad por ley (tangente con `dμ/dγ̇²` y `dμ/dT`), inercia, PSPG/SUPG/LSIC y disipación
  viscosa `Φ = μγ̇²` (`setViscousHeating`) como fuente del calor. Con una ley, el esfuerzo recuperado se calcula a partir
  del esfuerzo en Gauss con `Operators.gauss_stress_divergence` (solo geometría) y su tangente es `Q @ dσ/dx`. Sin ley,
  inercia ni calentamiento, Stokes sigue lineal. `physics.setVelocity('u')` (nombre de un campo) añade
  `FieldConvectionTerm` (`Physics.py`): convección monolítica por la velocidad calculada, con SUPG y bloques cruzados;
  `CoupledPhysics.setVelocity/setStabilization` omiten la parte de flujo (`computes_velocity = True`).
  `derived_fields` de `ff`: `shear_rate` y, con ley, `viscosity`.
- **Ya está (Etapa 13):** `Model(..., axisymmetric=True)` → `Discretization(axisymmetric=True)`: plano meridiano con
  `x = r`, `y = z` y el eje en `x = 0`. `detJw` y las medidas de faceta llevan el factor `r` (sin `2π`), así que todos los
  operadores escalares (calor, especies, Darcy, BC, convección) integran sobre el volumen de revolución sin cambios;
  `area` conserva la medida plana para `element_size` y los parches de recuperación. El flujo añade la deformación de
  aro `ε_θθ = u_r/r` (`disc.hoop = N/r`) en `viscous`, `vector_diffusion`, `divergence`, `γ̇²` y la disipación, y los
  términos `(σ_rr - σ_θθ)/r`, `σ_rz/r` de `div σ` (`Operators.hoop_stress_terms`, en el residuo de la PSPG/SUPG). La
  PSPG lineal de Stokes usa ahora `pspg_weight(disc, τ) @ recovered_stress_divergence(disc, μ)`. El eje es una
  condición de simetría (`addBC_SymmetryEdges`, `u_r = 0`); `flowRate` devuelve el caudal volumétrico `2π∫u·n r ds`.
- **Falta:** transitorio de Navier–Stokes (Etapa 14), elementos de orden mixto (`quad9`, `tri6`) y
  `DofLayout` con nodos distintos por campo (Taylor–Hood), estado interno para plasticidad.

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
- `test-branch`: etapas 1 a 6 (la 6c retiró la ruta por elemento). JAX queda solo en `PointwiseLaw` y en las funciones
  de usuario trazables (reacciones, `h(T)`).
