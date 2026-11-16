# Plan: rendimiento, GPU, aprendizaje automático e interfaz

Revisión completa del código hecha el 2026-11-12 sobre `test-branch` (commit `0562c81`), con perfiles medidos en esta máquina
(Ryzen de 12 hilos, 33 GB, RTX 4060 de 8 GB, Windows 11, Python 3.13 de `.venv`). Este documento es el traspaso: qué se midió, qué
se cambió ya, y las fases siguientes con su criterio para seguir. Convenciones de trabajo: las de `docs/PLAN_INYECCION.md`.

## Veredicto

1. **Lo más rentable sigue estando en la CPU.** A ~190 mil incógnitas el tiempo se reparte entre PARDISO (35–55 %), `einsum`
   (20–60 % según el estudio) y álgebra dispersa (10–25 %). Dos cambios pequeños, ya hechos, dan ×1.2 a ×1.7; pasar los núcleos de
   elemento a funciones compiladas (JAX `jit`) da ×10 en esa parte **sin GPU** (medido, abajo).
2. **GPU: sí, pero por pasos y con criterio de salida.** Con 8 GB de memoria de vídeo y doble precisión, la GPU sirve para los núcleos
   de elemento, las leyes puntuales y los productos matriz–vector de los solvers iterativos; el solver directo en GPU (cuDSS) es una
   prueba, no una promesa. El mismo refactor de la fase B (núcleos compilados) es el que deja el código listo para GPU.
3. **Redes neuronales: no como sustituto del solver.** Una PINN clásica resolviendo el problema directo es más lenta y menos exacta
   que este código. Lo que sí aporta, y aprovecha lo que ya existe (residuo y tangente exactos, leyes en JAX): gradientes por
   adjunto para calibrar e invertir, leyes constitutivas neuronales dentro de `PointwiseLaw`, y sustitutos (surrogates) entrenados
   con el propio simulador para ventanas de proceso.
4. **Interfaz:** la física es amplia y está bien validada; el uso es lo más mejorable (configuración verbosa, resultados en memoria,
   sin base de materiales ni archivo de caso). Varias mejoras son baratas y no tocan la numérica.

## Perfil medido (antes de los cambios)

Casos de `prof.py` (guardado fuera del repositorio; reproducible con `cProfile` sobre los ejemplos):

| Caso | Incógnitas | Tiempo | PARDISO | `einsum` | Disperso (productos, sumas, COO) |
|---|---|---|---|---|---|
| Rendija Cross–WLF + calor, Q4, Newton estacionario (5 iter.) | 186 628 | 28.3 s | 37 % | 25 % | 21 % |
| La misma en Taylor–Hood Q9 | 186 628 | 34.9 s | 60 % | 25 % | 5 % |
| Cavidad Re = 400, Q4 PSPG/SUPG, Newton (7 iter.) | 111 747 | 26.8 s | 43 % | 20 % | 12 % |
| Cavidad transitoria BDF2, 40 pasos, Newton modificado | 49 923 | 71.3 s | 19 % | **59 %** | 1 % |

Qué había dentro:

- **PARDISO repetía el análisis simbólico** (reordenación y factorización simbólica, fase 11) en cada iteración de Newton, aunque el
  patrón de la tangente no cambia: 1.1 de 2.0 s por factorización en Q4 y **3.8 de 4.4 s en Taylor–Hood**.
- **El residuo del flujo calculaba la cinemática tres veces** (`FlowTerm._kinematics` en `contribution`, en `viscosity` y para `|u|`
  de τ), y con el bucle genérico de `einsum`: 35 % del transitorio.
- `einsum` de tres operandos sin BLAS: `np.matmul` hace lo mismo 2–9 veces más rápido en las contracciones del residuo (Q4, Q9 y
  hex27 medidos); con 12 hilos sobre trozos de elementos, ×5; **con `jax.jit` en CPU, ×11–12** en la contracción más cara de la
  tangente (`eg,egib,egabjc->eiajc`: 0.110 → 0.009 s en Q4, 1.79 → 0.16 s en hex27).
- `Q @ dσ/dx` (esfuerzo recuperado de la PSPG con ley de viscosidad): 11–13 % de la rendija Q4, un producto disperso por iteración.
- `Discretization.recovery`: bucle de Python con una pseudoinversa por nodo, 3.3 s con 37 mil nodos (una vez por malla).

## Fase A: CPU, cambios pequeños

**Hecho el 2026-11-12:**

- `FactorizedSolver` (`Source/Simulation/DirectSolver.py`) reutiliza el análisis de PARDISO cuando la matriz nueva tiene el mismo
  patrón (solo fase 22). El emparejamiento ponderado del análisis sale de los valores de su matriz, así que la primera solución con
  cada análisis reutilizado se comprueba: residuo por encima de 1e-10 tras refinar → factorización completa; si esta mejora 10 veces,
  el solver deja de reutilizar; si no, la matriz es mal condicionada de por sí y se sigue reutilizando. `reuse_analysis = False` lo
  desactiva; `n_analyses` cuenta los análisis. De paso: los índices se ordenan antes de comparar matrices (sin ordenar, una matriz
  igual parecía nueva).
- `FlowTerm`: la cinemática se calcula una vez por evaluación y con `matmul`; también `Discretization.interpolate`, `grad p` y los
  términos de convección y SUPG del residuo.

| Caso | Antes | Después | |
|---|---|---|---|
| Rendija Q4 | 28.3 s | 22.1 s | ×1.28 |
| Rendija Taylor–Hood | 34.9 s | 23.8 s | ×1.47 |
| Cavidad estacionaria | 26.8 s | 20.2 s | ×1.33 |
| Cavidad transitoria | 71.3 s | 41.6 s | ×1.71 |

**Hecho el 2026-11-12, segunda tanda:**

- **`Discretization.recovery` vectorizado:** parches por productos de matrices de incidencia y pseudoinversas por lotes de igual
  tamaño. Resultado idéntico bit a bit en Q4, T3, Q9, T6, hex8, tet4, axisimétrico y una tira sin nodos interiores; 2.68 → 0.33 s
  con 37 mil nodos.

**Probado y descartado:**

- **Conservar el análisis de PARDISO cuando el patrón cambia** (escribir la matriz nueva sobre el patrón analizado con ceros
  explícitos, o analizar sobre la unión). En 3D la tangente pierde o gana entradas entre iteraciones (scipy elimina las que se
  cancelan) y el análisis se repite. Funcionaba (conducto hex27 con Cross, 44 772 incógnitas: 7 análisis → 3, 36.9 → 32.8 s), pero
  **rompió dos pruebas**: `test_continuation_in_weissenberg` (Couette a Wi = 4: Newton sin converger en 200 iteraciones) y
  `test_turbulence_on_quadratic_elements`. Los residuos lineales eran buenos (1e-15), así que no es un fallo de la solución sino
  de su error en sistemas mal condicionados: el escalado y el emparejamiento del análisis salen de los valores de la primera
  matriz (el estado de reposo), y aplicados a tangentes muy distintas cambian el redondeo lo bastante para sacar a Newton de su
  camino. El residuo no lo detecta. Retirado. **Aviso:** el mismo riesgo existe en principio con la reutilización entre patrones
  idénticos que sí está activa (las 47 suites pasan con ella); si un caso mal condicionado se comporta distinto, probar
  `FactorizedSolver.reuse_analysis = False`. Una vía sin ese riesgo: análisis sin escalado ni emparejamiento (depende solo del
  patrón), que habría que validar caso a caso porque cambia las trayectorias ya validadas.

- **Caché de compilación de JAX en disco.** El 1.4 s por ley del primer perfil era un arranque en frío: en caliente compilar una
  ley cuesta ~0.05 s, y con `jax_compilation_cache_dir` JAX no escribió ninguna entrada en este equipo (Windows, CPU). Sin ganancia.
- **Búsqueda de línea con solo el residuo** en Newton estacionario: cuando el paso completo se acepta (lo habitual) habría que
  ensamblar después la tangente en el mismo punto, un 20 % más por iteración; solo gana si hay muchos retrocesos.

**Pendiente de la fase A** (en orden de rendimiento esperado):

1. `Q @ dσ/dx` (11–13 % de la rendija Q4): producto con el patrón simbólico calculado una vez, o ensamblado por elementos del parche.
2. **Patrón disperso fijo por término** (`np.bincount` sobre posiciones precalculadas, como `HeleShaw._scatter`): 5–8 % directo.
3. Historia del transitorio: hoy es una matriz densa (incógnitas × pasos) en memoria. Guardar cada n pasos o volcar a disco.
4. Barridos de parámetros en paralelo por procesos (ventana de proceso, caudal–presión): 12 hilos sin tocar el solver.

En 3D pesa tanto la tangente como el solver: en el conducto hex27, `einsum` son 11 de 37 s (la contracción
`eg,egib,egabjc->eiajc`, 4.8 s). Eso es la fase B.

Criterio para cerrar: las 47 suites pasan y los cuatro casos de la tabla no empeoran.

## Fase B: núcleos compilados (el paso que habilita la GPU)

Hoy `FlowTerm`, `FieldConvectionTerm`, `GradientLawTerm` y `ConformationTerm` son cadenas de `einsum` de numpy. Propuesta: cada
término expone un **núcleo de elemento** puro, `(x_e, geometría_e, parámetros) → (R_e, K_e)`, escrito en `jax.numpy` y compilado con
`jit(vmap(...))`; el ensamblaje (dispersión con patrón fijo) queda en numpy/scipy.

- Ganancia medida en CPU: ×10 en las contracciones. Sobre el transitorio (lo que queda de `einsum` tras la fase A, ~15–25 %) y sobre
  la tangente estacionaria (20–35 %).
- La tangente puede salir de `jacfwd` del residuo del elemento: menos código que mantener (hoy `FlowTerm.contribution` son 200 líneas
  de derivadas a mano) y sirve para comprobar la tangente manual. Riesgo: `jacfwd` de un hex27 con 19 incógnitas por nodo es una
  matriz de 513 × 513 por elemento; hay que medir memoria y trocear por bloques de elementos (`CHUNK_ENTRIES` ya existe).
- Hacerlo por término, empezando por el residuo de `FlowTerm` (lo que domina el Newton modificado), con la versión numpy como
  referencia en las pruebas (igualdad a 1e-12) hasta retirarla.
- τ congelado, el residuo del nivel anterior y el contexto de paso (`System.time`) pasan como argumentos del núcleo.

Criterio: mismo resultado a 1e-10 en `test_navier_stokes`, `test_non_newtonian`, `test_transient_flow`, `test_3d_flow` y al menos
×2 total en la cavidad transitoria y ×1.3 en la rendija.

**Hecho el 2026-11-13 (B.1, residuo de `FlowTerm`):**

- `Source/Physics/FlowKernel.py`: el residuo sin tangente de `FlowTerm` con `jax.numpy`, compilado una vez por configuración
  (ley de viscosidad, inercia, SUPG, PSPG, LSIC, calentamiento, flotación, axisimétrico, contexto de paso). Solo la parte local:
  lo que acopla elementos (`V @ x`, `Q @ σ`) y el residuo del nivel anterior entran como datos en los puntos de Gauss, y la
  dispersión es un `bincount`. No es por elemento con `vmap`, sino las mismas expresiones sobre todos los elementos: basta para
  que XLA las funda. `FlowTerm.compiled = False` deja la ruta numpy, que es la referencia.
- Igual a numpy a 1e-12 en Q4, T3, Q9, T6, hex8, tet4, hex27 y tet10, con las cuatro formas del contexto de paso
  (`Test/test_flow_kernel.py`, 7 pruebas; también `Test/test_modified_newton.py`, que ya comparaba el residuo con el de la tangente).
- **Refinamiento de PARDISO a demanda** (encontrado al perfilar lo que quedaba): PARDISO hace dos pasos de su refinamiento
  iterativo en cada solución (`iparm(8) = 2`), sea cual sea el residuo: 39 ms por solución con 50 mil incógnitas frente a 15 ms sin
  ellos. `FactorizedSolver` los mide en la **primera solución de cada factorización**: esa solución es la de PARDISO de siempre,
  y una segunda sin refinar dice cuánto cambia; si es menos de 1e-11 de su tamaño, las siguientes soluciones con esa
  factorización van sin él y solo refinan por encima de un residuo de 1e-13. Gana el Newton modificado (muchas soluciones por
  factorización); el Newton estacionario queda como antes. `FactorizedSolver.refine_on_demand = False` lo desactiva.
  **Dos intentos fallidos antes de este:** (1) decidir solo por el residuo rompió `test_continuation_in_weissenberg` y
  `test_sst_flat_plate_closes_its_residual` (el Couette viscoelástico a Wi = 4 tiene residuos de 1e-14 sin refinar y aun así
  lo necesita). (2) Apagar el refinamiento de PARDISO y hacerlo aquí con la misma factorización, **que es lo que entró en el
  commit `641ad6b`**: en puntos de silla de Taylor–Hood la respuesta sin refinar llega a tener residuo relativo 1, que ese
  refinamiento no repara. Corregido el 2026-11-14 (`6a9de71`).
- **Estancamiento de Newton en el piso de redondeo (2026-11-14, defecto anterior a esta revisión):** al buscar una prueba para
  lo anterior apareció la causa de fondo. En la rendija Taylor–Hood con calor, el arranque newtoniano (lineal en u y p) se
  resuelve en una iteración, pero las soluciones lineales del punto de silla dejan el residuo en 3e-9 del inicial, por encima
  de la tolerancia relativa (1e-10). Newton seguía entonces con el paso mínimo de la búsqueda de línea (1/1024, una docena
  de ensamblajes cada vez) hasta que un paso completo pasaba por azar: **de 2 a 175 iteraciones de una ejecución a otra** (PARDISO
  en paralelo no es reproducible al redondeo), entre 5 y 50 s para 21 mil incógnitas. Con el código anterior a la revisión, igual
  (3 a 24 en nueve ejecuciones). Era dentro de ese estancamiento donde el refinamiento de `641ad6b` no terminaba nunca; desde un
  arranque limpio (`Type=linear`) aquel solver también convergía, más despacio (23 s frente a 9 con 83 mil incógnitas).
  `newtonSolver` da ahora por convergido el caso en que la búsqueda de línea no encuentra descenso dos iteraciones seguidas con
  un residuo ya por debajo de 1e-6 del inicial (o de 1e-7): 2 a 4 iteraciones en ocho ejecuciones.
  `test_newton_stops_at_the_round_off_floor` lo cubre (sin el cambio falla dos de cada tres veces: 18 y 79 iteraciones).

| Cavidad transitoria BDF2 (49 923 incógnitas, 40 pasos) | Tiempo |
|---|---|
| Inicio de la revisión | 71.3 s |
| Fase A (cinemática una vez, `matmul`) | 42.2 s |
| Residuo compilado | 27.2 s |
| + refinamiento a demanda | 18.7 s |

El residuo pasa de 57 a 18 ms por evaluación (16 384 Q4); de esos, unos 12 ms son ya el núcleo y el resto reunir y dispersar.
Los estudios estacionarios apenas cambian: allí pesa la tangente.

**Hecho el 2026-11-14 (B.2, tangente de `FlowTerm`):**

- La tangente es la **derivada automática del residuo compilado**, elemento a elemento (`jacfwd` + `vmap`), respecto a las
  velocidades, presiones y campos de la ley de viscosidad y de la flotación del elemento. τ y τ_c quedan fuera de la derivada
  (`stop_gradient`, o el estado de `tau_state`), como en la tangente a mano; con Picard, también la velocidad convectora. La
  parte no local sigue en scipy: `−(pesos de SUPG y PSPG) @ d(div σ̃)/dx`, con los pesos que devuelve el núcleo y `dσ/dx` de la
  derivada de `stress`. Por trozos de elementos (30 millones de entradas) para acotar la memoria.
- Igual a la tangente a mano a 1e-11 (relativo a su mayor entrada) en las mismas configuraciones que el residuo, con τ en el
  propio estado y en otro (`Test/test_flow_kernel.py`). Las 200 líneas de derivadas a mano de `FlowTerm.contribution` quedan
  como referencia (`FlowTerm.compiled = False`).

| Estacionario (Newton completo) | Incógnitas | Antes de B.2 | Con B.2 |
|---|---|---|---|
| Cavidad Re = 400, Q4 | 111 747 | 20.2 s | 16.2 s |
| Rendija Cross–WLF + calor, Taylor–Hood Q9 | 186 628 | 23.8 s | 20.8 s |
| Conducto 3D hex27 con Cross | 44 772 | 36.9 s | 24.4 s |
| Rendija Cross–WLF + calor, Q4 | 186 628 | 22.1 s | 23.2 s |

La rendija Q4 no gana: de los 1.6 s de cada tangente, lo que pesa son los productos dispersos del esfuerzo recuperado
(`Q @ dσ/dx`, `W @ d_divs`) y las sumas de matrices, no los núcleos. La cavidad transitoria queda en 20.8 s (18.7 s con el
refinamiento defectuoso de `641ad6b`).

**Pendiente de la fase B:**

1. Productos dispersos de la tangente con esfuerzo recuperado (`Q @ dσ/dx`: patrón simbólico una vez) y ensamblaje de patrón
   fijo: es lo que domina ya la rendija Q4.
2. `FieldConvectionTerm` (calor y especies convectados: 2.5 de 22 s en la rendija), `GradientLawTerm` (turbulencia) y
   `ConformationTerm` (viscoelástico).
3. Reunir y dispersar dentro del núcleo (índices fijos) para quitar las copias entre numpy y JAX.
4. Retirar las derivadas a mano de `FlowTerm` cuando la ruta compilada lleve un tiempo en uso.

## Fase C: GPU (RTX 4060, 8 GB)

Limitaciones que conviene tener claras antes de invertir:

- **JAX con CUDA no se distribuye para Windows nativo** (los paquetes con GPU son de Linux): para usar los núcleos de la fase B en la
  GPU hace falta WSL2 con Ubuntu (hoy solo está la distribución de Docker Desktop). **A confirmar al empezar la fase.** CuPy
  (`cupy-cuda12x`/`13x`) y PyTorch sí publican ruedas; no se ha comprobado que instalen con Python 3.13 en este equipo.
- **Doble precisión:** las tarjetas de consumo rinden mucho menos en `float64` que en `float32`; el código necesita `float64` en el
  residuo. Hay que medir, y considerar precisión mixta (precondicionador en simple, residuo en doble).
- **8 GB:** una factorización directa de 190 mil incógnitas 2D ya ocupa varios GB en CPU. El solver directo en GPU queda para
  mallas medianas; las grandes van por Krylov.

Pasos, cada uno con una medida que decide si se sigue:

| Paso | Qué | Sigue si |
|---|---|---|
| C0 | Entorno: WSL2 + `jax[cuda12]` (o CuPy nativo). Reproducir `einsum_par.py` en GPU, en `float64` | ≥ ×5 frente a `jit` en CPU con ≥ 50 mil elementos |
| C1 | Núcleos de la fase B y leyes puntuales en GPU; ensamblaje en CPU | el tiempo de traer `K_e` a la CPU no se come la ganancia |
| C2 | Krylov en GPU: `SpMV` y precondicionador (Jacobi/Chebyshev; jerarquía de pyamg copiada a la GPU) para CG + AMG escalar 3D y GMRES + bloques | más rápido que la CPU desde 300 mil incógnitas |
| C3 | Solver directo en GPU (NVIDIA cuDSS, `nvidia-cudss-cu12` existe en PyPI) frente a PARDISO | más rápido y cabe en 8 GB en los casos de la tabla |
| C4 | Hele-Shaw: el núcleo de fluidez y el perfil T(z) por nodos (ya son JAX) en GPU | reduce el tiempo de la pieza de 16 mil nodos (947 s) |

Lo que no se propone: MPI ni varias GPU (fuera de alcance, como en `PLAN_INYECCION.md`), ni reescribir en otro lenguaje.

## Fase D: diferenciable y aprendizaje automático

Ordenado por valor para procesamiento de polímeros y por lo cerca que está del código actual:

1. **Sensibilidades por adjunto discreto.** Con la tangente convergida `J` ya factorizada, el gradiente de una cantidad de interés
   `q(x, θ)` respecto a parámetros `θ` (Cross–WLF, Tait, caudal, temperatura de pared) cuesta una solución con `Jᵀ` más `∂R/∂θ`, que
   sale de `jax.grad` sobre las leyes. Sirve para calibrar con medidas de proceso (presión en cavidad, caída de un dado) y para
   optimizar; es más exacto y mucho más barato que una PINN inversa. Prueba: gradiente frente a diferencias finitas.
2. **Leyes constitutivas neuronales.** `PointwiseLaw` acepta cualquier función JAX: una red pequeña (viscosidad monótona en γ̇,
   pvT, cinética) entra sin tocar el ensamblaje y con jacobiano automático. `Fitting.py` ganaría `fit_neural_viscosity` con las
   restricciones físicas en la arquitectura. Riesgo bajo; útil cuando Cross–WLF no ajusta (cargas, fluencia).
3. **Sustitutos entrenados con el simulador.** Barridos en paralelo (fase A.7) → base de datos → modelo reducido (POD + regresión
   gaussiana primero; DeepONet o red de grafos sobre la malla después) que da presión de llenado, tiempos de llenado y contracción en
   milisegundos para la ventana de proceso (fase 7 de `PLAN_INYECCION.md`). Siempre con error medido frente al solver en casos no
   vistos.
4. **Entrenamiento informado por la física con el residuo de elementos finitos.** `System.residual(x)` como función de pérdida de una
   red que predice el campo nodal (sin etiquetas), o como término de regularización del sustituto del punto 3. Es la versión de
   "PINN" que aprovecha la forma débil, las condiciones de contorno y la estabilización ya validadas. Requiere la fase B (residuo
   diferenciable de extremo a extremo en JAX).
5. **Arranque de Newton aprendido:** el sustituto da el estado inicial y Newton lo corrige; el resultado conserva la exactitud del
   solver.
6. **PINN clásica (colocación, sin malla):** solo como demostración en un caso 1D/2D pequeño para comparar coste y error con el
   código; no se recomienda como vía de trabajo.

Dependencias nuevas: `optax` (y `equinox` o `flax`) sobre JAX, sin PyTorch, para no tener dos pilas.

## Fase E: interfaz y uso

Observaciones de la revisión:

- `solverConfiguration(Study=StudyType..., Type=ProblemType..., Method=SolverType..., Solver=DirectSolvers...)`: cuatro enumerados
  para lo habitual. `Type` lo sabe el sistema (`System.is_linear`) y `Method='auto'` ya elige solver.
- Las condiciones se guardan en objetos `Node` (`node.BC`) y `System.add_boundary_conditions` recorre nodos × campos en Python; la
  malla como lista de objetos pesa en mallas de 10⁵–10⁶ nodos.
- `Source` no es un paquete instalable; hay ~45 scripts y cuadernos en la raíz; el encabezado del `README` aún describe un
  simulador de calor 1D/2D.
- Cada `import Source` crea un archivo en `Logs/` del directorio actual.
- Resultados: `model.sol` en memoria; sin puntos de control ni reanudación; el posproceso de inyección vive en un cuaderno.
- Nombres mezclados (`addBC_VelocityEdges` vale para caras 3D, `miu`, español e inglés).

Propuesta, de menos a más esfuerzo:

1. `model.solve()` sin configuración: estudio estacionario, tipo detectado, solver `auto`. `solverConfiguration` acepta cadenas
   (`study='transient'`, `scheme='BDF2'`) y conserva los enumerados.
2. Registro: un solo archivo por sesión creado al primer `solve()`, carpeta configurable, nivel de consola configurable.
3. `pyproject.toml` (paquete `femsim` con `Source` como alias), ejemplos a `examples/`, `README` al día.
4. Monitores y parada: `on_step`/`on_iteration` en todos los solvers (hoy solo en Hele-Shaw), tabla de convergencia y sondas
   (presión en un punto, caudal) durante el cálculo; puntos de control (`.npz`) y reanudación del transitorio.
5. Base de materiales en JSON (reología, pvT, térmicas, cinética; fase 6 de inyección) y `material.from_file`.
6. Archivo de caso (YAML/TOML) + línea de órdenes (`python -m femsim caso.toml`) para los flujos típicos: dado, reómetro,
   llenado–empaque; es también el formato de los barridos de la fase D.3.
7. Asistentes por aplicación (clases finas sobre `Model`): `SlitDie`, `CapillaryRheometer`, `InjectionCase` con sus condiciones,
   mallas y salidas típicas (curva del dado, corrección de Bagley, isócronas, fuerza de cierre).
8. Visor: `pyvista` opcional para 3D y superficies (hoy matplotlib y VTK para ParaView); informe HTML de un caso.
9. Malla en arrays (coordenadas, conectividad, grupos) con los objetos `Node`/`Element` como vista de compatibilidad.

**Hecho el 2026-11-14 (E.1, E.2 y E.3):**

- `Model.solve()` sin configuración: estudio estacionario, tipo detectado de `System.is_linear` y `Method='auto'`.
  `solverConfiguration` toma `Type=None` por defecto (se detecta al resolver y se vuelve a detectar si el modelo cambia; un
  `Type` explícito se respeta tal cual) y acepta cadenas en `Study`, `Type`, `Method` y `Solver` (`'transient'`, `'steady'`,
  `'nonlinear'`, `'PARDISO'`...; los enumerados siguen valiendo; un nombre desconocido da `ValueError` con las opciones).
  `timeScheme='BDF2'` y compañía ya eran cadenas. Pruebas en `Test/test_auto_solver.py`.
- Registro (`Source/logger_config.py`): el archivo de la sesión se crea con el primer mensaje, no al importar `Source` (se
  acabaron los archivos vacíos en `Logs/`), y `configure_logging(log_dir=..., console_level=...)` cambia carpeta (`None`: sin
  archivo) y nivel de consola (`None`: silencio). Pruebas en `Test/test_interface.py`.
- E.3: `pyproject.toml` (paquete `femsim`, extras `amg` y `mesh`; instalar con `pip install -e .`). `femsim.X` es `Source.X`
  (mismos objetos de módulo, por un buscador de importación en `femsim/__init__.py`, no copias: los enumerados comparan igual).
  Los 33 ejemplos y cuadernos versionados pasan a `examples/` (con `git mv`); `README` con instalación, uso y rutas al día. Los
  archivos sin versionar de la raíz quedan donde estaban. Hizo falta instalar `setuptools` en `.venv`.

**Hecho el 2026-11-15 (E.4, monitores, sondas y puntos de control):** `Source/Simulation/Monitoring.py`.

- `Model.solve(on_iteration=..., on_step=..., checkpoint=..., resume=...)`. `on_iteration(info)` se llama en cada iteración de
  Newton estacionario, de la marcha pseudotransitoria y de Newton dentro de cada paso temporal (`info`: `kind`, `iteration`,
  `residual`, `relative_residual`, `x`, y `step`/`time` en un paso). `on_step(info)` en cada paso aceptado y al final de un estudio
  estacionario. Si un monitor devuelve `True` el cálculo se detiene: el estacionario conserva su estado actual
  (`nonlinear_converged` es `False`) y el transitorio conserva los pasos hechos (si se detiene dentro de un Newton, ese paso se
  descarta). Las iteraciones de Krylov de los solvers lineales no tienen monitor.
- Sondas: `model.addProbe(nombre, campo, node=... | point=..., component=...)` (el nodo más cercano a `point`); se muestrean en
  t = 0 y tras cada paso, o en cada iteración en un estudio estacionario. Resultados: `model.probe_times`,
  `model.probe_values[nombre]`. Tabla de convergencia: `model.convergence` (una fila por iteración estacionaria o por paso).
- Puntos de control: `checkpoint=ruta` o `(ruta, cada)` (10 pasos por defecto) escribe un `.npz` de forma atómica; `resume=ruta`
  continúa un estudio transitorio (theta lineal, y los lazos no lineales: theta, BDF2 y paso adaptativo) comprobando esquema,
  tiempo total, theta, número de pasos (paso fijo), número de incógnitas y versión. Las sondas ya registradas se restauran. Tras
  reanudar se reconstruye la tangente (el Newton modificado la reutilizaba), así que el resultado coincide con el del cálculo
  seguido a la tolerancia del Newton (probado a 1e-7; lineal a 1e-12).
- `Test/test_monitoring.py`, 6 pruebas. Pendiente de E.4: `on_step` en el llenado Hele-Shaw (ya tenía el suyo) unificado con este.

**Hecho el 2026-11-16 (E.5, base de materiales en JSON):** `Source/Material.py`.

- `material.from_file(ruta o nombre)`, `material.from_dict`, `mat.save(ruta)`, `material.library()`. Un archivo tiene `name`,
  `source` (de dónde salen los números), `properties` (números: `rho`, `k`, `Cp`...) y secciones `viscosity`, `pvt` y `kinetics`
  con `model` = nombre de la clase (Newtonian, PowerLaw, CarreauYasuda, Cross, CrossWLF, Bingham, HerschelBulkley, Arrhenius y
  Exponential con `law` anidada; TaitPVT; NakamuraHoffman) y el resto de claves como sus argumentos. El material expone
  `viscosity_law`, `pvt` y `kinetics` (`None` si falta la sección) y las propiedades como atributos.
- Biblioteca: `Source/data/materials/` más las carpetas de la variable de entorno `FEMSIM_MATERIALS`, que mandan. Los errores
  (clave desconocida, modelo inexistente, argumento que sobra, JSON roto) nombran el archivo y la sección.
- Contenido inicial, solo lo que ya estaba en los ejemplos y con su procedencia en `source`: `POLYFORT_FPP30GFC_K1079` (Cross–WLF
  y Tait de ese grado; `rho`, `Cp` y `k` son valores de trabajo de los ejemplos, no medidas) y `Sabic_PP505P_cinetica` (solo la
  cinética, que los ejemplos usan con otro material). No se añadieron datos de materiales que no estuvieran ya en el repositorio.
- Falta de E.5: térmicas dependientes de la temperatura (`rho Cp(T)`) y modos viscoelásticos en el formato; entran cuando haya
  leyes que los consuman. `Test/test_material_files.py`, 5 pruebas.

**Hecho el 2026-11-16 (E.6, archivo de caso y línea de órdenes; primer tipo de caso):** `Source/Cases.py`, `Source/cli.py`.

- `python -m femsim caso.toml [--set seccion.clave=valor] [--output DIR] [--jobs N] [--dry-run] [--quiet]` (también el
  guion `femsim` de `pyproject.toml`, y `--list-types`). Un caso es `type`, `name` y secciones propias del tipo; `[sweep]` con listas
  da una ejecución por combinación (producto cartesiano), en procesos con `--jobs` (la fase A.4 para barridos de proceso).
  Cada ejecución escribe `<name>[_i]_results.json` con el caso resuelto y sus resultados, y un barrido `<name>_sweep.csv`.
- Comprobación estricta: tipo, secciones y claves desconocidas, tipo del valor (un entero acepta `12.0`, no `12.5`), claves
  obligatorias; los errores nombran el archivo y la clave. `--set` toma valores TOML (`2e-4`, `false`, `"texto"`, `[1, 2]`).
- Los tipos se registran con `@case_type(nombre)` sobre una clase con `defaults` y `run(params, directorio, nombre)`. **Solo
  existe `slit_die`** (dado de rendija, el modelo de `Example_Polymer_Slit_Die.py` con el material de la biblioteca: caída de
  presión, T máxima y de mezcla a la salida, caudales, balance de caudal, velocidad de corte aparente, VTK opcional). Validado
  contra la solución newtoniana (`dp = 3 mu q L / H^3`, 0.3 %) y por la monotonía de la curva del dado (adelgazamiento). Ejemplo:
  `examples/casos/dado_rendija.toml`.
- **Pendiente:** los tipos `capillary_rheometer` e `injection` (llenado–empaque) entran con E.7, que construye las clases
  `CapillaryRheometer` e `InjectionCase`; el tipo de caso será una capa fina sobre ellas. `Test/test_cases.py`, 7 pruebas.

**Hecho el 2026-11-16 (E.7, clases por aplicación y sus tipos de caso):** `Source/Applications.py`.

- `SlitDie(material, ...)`: `.solve(q)` y `.curve(caudales)` (continuación en caudal, la curva del dado). El tipo de caso
  `slit_die` pasa a ser una capa sobre esta clase.
- `CapillaryRheometer(material, T, ...)`: malla de gmsh del barril y el capilar (tamaños relativos al radio del capilar,
  `mesh_scale`), `.measure(tasas_aparentes)` con las correcciones de Bagley (pérdida de entrada) y Rabinowitsch y la viscosidad
  medida frente a la ley. Es el flujo de `Example_Capillary_Rheometer.py` con la ley de la biblioteca a temperatura fija
  (`Rheology.AtTemperature`, nueva). Tipo de caso `capillary_rheometer`.
- `InjectionCase(material, malla, entrada, espesor, tiempo_de_llenado, ...)` y `InjectionCase.plate(...)`: llenado Hele-Shaw a caudal
  constante, con temperatura en el espesor si `thermal`. `.run()` da tiempo de llenado, presión de entrada máxima y final, **fuerza
  de cierre máxima** (integral de la presión sobre el área proyectada en cada paso; solo mallas planas), llenado corto, fracción
  llena y congelada; `.isochrones(n)` y `.plot_isochrones()`. Tipo de caso `injection` (con `output.plot` guarda el PNG de las isócronas).
- Validación: dado newtoniano contra `3 mu q L / H^3` (1 %); reómetro newtoniano (n' = 1, viscosidad recuperada a 3 %, pérdida de entrada
  positiva y menor del 25 %) y con Cross–WLF (error por debajo del 6 % en una malla gruesa, 1.7 % con la del caso de ejemplo);
  tira newtoniana llenada contra la solución cerrada (`p = 12 eta L^2 / (t h^2)` y `F = p L W / 2`, 3 %). Con molde frío aparece la
  **curva U** (presión de entrada mínima hacia 0.5 s en la placa de ejemplo; a 4 s es 1.7 veces mayor por la capa congelada), como
  en `Example_Filling_NonIsothermal.py`. Ejemplos: `examples/casos/reometro_capilar.toml` y `llenado_placa.toml`.
- Pendiente de E.7: empaque y enfriamiento (`pack`) y la contracción en `InjectionCase`; piezas desde CAD (`from_dual_domain`) y
  la red de alimentación (`addRunners`) solo por el modelo de `.filling`; fuerza de cierre en superficies plegadas (área proyectada).
  El reómetro hace la corrección de Bagley por extrapolación del gradiente (la del ejemplo), no con varias relaciones L/D.
  `Test/test_applications.py`, 5 pruebas.

**Hecho el 2026-11-16 (E.8, visor pyvista opcional e informe HTML):**

- `Source/Post_processing/Report.py`: `Report` escribe **un solo archivo HTML** autocontenido (figuras PNG incrustadas, sin recursos
  externos, tema claro/oscuro): `add_text`, `add_table` (diccionario o filas), `add_figure` (matplotlib), `add_model(model)` (resumen,
  estadísticas por campo, contornos en mallas 2D, curva de convergencia o iteraciones por paso, sondas) y `case_report` (parámetros y
  resultados). Los tres tipos de caso lo escriben con `output.report = true` (`<nombre>_report.html`; el llenado añade isócronas, presión
  de entrada y fuerza de cierre; el reómetro, la curva de viscosidad medida frente a la ley). Funciona sin dependencias nuevas.
- `Source/Post_processing/Viewer.py`: `to_pyvista(mesh, campos)` (la malla como `UnstructuredGrid`, con los mismos arrays que el
  `.vtu`: `VTK.grid_arrays`, extraído de `write_vtu` sin cambiar su salida) y `show(modelo | llenado, campo, step, warp, screenshot)`.
  pyvista es **opcional** (`pip install femsim[view]`): sin él da un error que dice cómo instalarlo. En mallas 3D el informe usa
  `Viewer.show` para las vistas si pyvista está; si no, solo las estadísticas.
- Verificado con pyvista 0.49.2 (2026-11-16): `to_pyvista`, capturas PNG fuera de pantalla, llenado (`fill_time`, `p`) y vistas 3D del
  informe. Al mirar las imágenes, un sólido solo mostraba su piel (el núcleo caliente de un cubo era invisible): `show(..., cut='auto')`
  quita media pieza con un plano por el centro (normal a y; `'x'`, `'z'` o `None` para la pieza entera) y la cara de corte da el campo
  interior; la barra de color pasó a vertical porque su título pisaba los números. `Test/test_viewer_report.py`, 6 pruebas.

**Medido el 2026-11-16 (E.9, coste de la malla de objetos; sin cambios de código):** un proceso por caso, Q4 2D y hex8 3D, conducción
lineal con PARDISO. `Node`/`Element` solo se construyen en `Source/Pre_processing/Mesh.py`; se leen en ~35 sitios (casi todos
`np.array([n.coor for n in mesh.NL])` y `getNodesId`) y las condiciones se escriben en `node.BC` en ~20.

| Malla | Nodos | Generar | Memoria de la malla | Objetos → arrays | Discretización | Solución | Pico de memoria |
|---|---|---|---|---|---|---|---|
| 2D Q4 | 100 mil | 0.24 s | 66 MB | 0.07 s | 0.38 s | 1.1 s | 0.55 GB |
| 2D Q4 | 400 mil | 1.1 s | 259 MB | 0.45 s | 1.7 s | 3.4 s | 1.8 GB |
| 2D Q4 | 1 M | 2.8 s | 645 MB | 1.0 s | 4.2 s | 7.3 s | 4.3 GB |
| 3D hex8 | 130 mil | 1.5 s | 92 MB | 0.12 s | 1.4 s | 6.3 s | 2.4 GB |
| 3D hex8 | 530 mil | 6.8 s | 373 MB | 0.78 s | 6.5 s | 76 s | 12.6 GB |

- Los objetos cuestan ~650 B por nodo frente a 48 B de los arrays (13 veces), pero son el **3–15 % del pico de memoria**; en 2D son
  el 10–20 % del tiempo de un problema lineal (generar la malla y pasar los objetos a arrays, lo que se repite en `Discretization`, `boundary_facets`,
  la exportación VTK y la turbulencia) y en 3D o en estudios no lineales y transitorios, donde la malla se construye una vez y se resuelve
  muchas, bajan del 5 %. El bucle sobre `NL` para las condiciones de contorno es despreciable (0.02 s con 400 mil nodos).
  El grueso del tiempo de preparación (einsum de la geometría, inversas, ensamblaje) no depende del modelo de objetos.
- Conclusión: la malla de objetos **no es el cuello de botella**. Opciones, de menos a más: (1) guardar `mesh.coords` y `mesh.conn`
  (arrays calculados una vez, invalidados por `moveNodes`) y que los ~10 sitios que los reconstruyen los lean: ahorra 0.1–1 s por
  pasada, sin tocar la semántica; (2) generadores estructurados en numpy y `NL`/`EL` como vistas perezosas (`NodeView` con `id`, `coor`
  y `BC` disperso por nodo de contorno): 1 M de nodos pasaría de 2.8 s y 645 MB a ~0.3 s y ~60 MB; (3) la malla solo en arrays sin vista.

## Físicas: robustez y especialización

Lo que la revisión confirma como siguiente (ya inventariado en `ESTADO.md` y `PLAN_INYECCION.md`): orientación de fibras, molde y
ciclo, mecánica de sólidos para el alabeo, `rho Cp(T)`, llenado compresible por lotes. Nada de esto depende de la GPU; sí se
beneficia de las fases A y B.

## Orden recomendado

A (resto) → B (residuo de `FlowTerm`) → E.1–E.3 (baratas) → D.1 y D.2 → C0 (medir la GPU antes de comprometerse) → resto según lo
medido. La fase 1 de inyección (geometría desde CAD) puede ir en paralelo: no toca los mismos archivos.

## Avance

- **2026-11-16:** fase E.8 (informe HTML y visor pyvista opcional).
- **2026-11-16:** fase E.7 (`SlitDie`, `CapillaryRheometer`, `InjectionCase` y sus tipos de caso).
- **2026-11-16:** fase E.6 (archivo de caso TOML, línea de órdenes y barridos; tipo `slit_die`).
- **2026-11-16:** fase E.5 (materiales en JSON y biblioteca).
- **2026-11-15:** fase E.4 (monitores, sondas, puntos de control y reanudación del transitorio).
- **2026-11-14:** fase E.1 (`solve()` sin configuración, cadenas), E.2 (registro de la sesión) y E.3 (paquete `femsim`, `examples/`).
- **2026-11-12:** revisión y perfiles; fase A, dos primeros cambios (análisis de PARDISO reutilizado, cinemática del flujo), que
  entraron en el commit `a0d5e6e` junto con la fase 1 de inyección.
- **2026-11-14:** Newton se para en el piso de redondeo (el arranque Taylor–Hood tardaba de 2 a 175 iteraciones según la ejecución).
- **2026-11-14:** fase B.2, tangente de `FlowTerm` por derivada automática (conducto 3D 36.9 → 24.4 s); corregido el refinamiento
  de PARDISO de `641ad6b`, que no convergía en Taylor–Hood grande.
- **2026-11-13:** fase B.1, residuo compilado de `FlowTerm` y refinamiento de PARDISO a demanda: cavidad transitoria 42.2 → 18.7 s.
- **2026-11-12, segunda tanda:** `recovery` vectorizado; patrón conservado en 3D, caché de JAX y búsqueda de línea descartados.
