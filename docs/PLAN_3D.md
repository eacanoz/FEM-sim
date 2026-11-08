# Plan: ampliación de las físicas a 3D

Documento de traspaso para una sesión dedicada al 3D. Se leen primero este archivo, `docs/ESTADO.md` (estado y validación de cada
física) y `docs/PLAN_ETAPA17.md` (historia de las etapas). Escrito el 2026-11-07, sobre `test-branch` en el commit `2505dfa`; las 39 suites
de `run_tests.py` pasaban.

## Veredicto: viable

El núcleo ya se escribió pensando en la dimensión: los elementos de referencia se registran en `ELEMENTS`
(`Source/Pre_processing/Elements.py`, «añadir tet4 o hex8 es añadir una entrada»), `Discretization` y `Operators` trabajan con arrays
`(nel, ngp, nne, dim)`, los campos vectoriales usan `ncomp=dim` (`FluidFlow`: `vectorField('u', ..., ncomp=dim)`) y las leyes puntuales en
jax reciben `nc * (1 + dim)` entradas (`System.py`). Lo que falta está localizado (abajo). El reto real no es el código sino el **coste**:
memoria de los arrays vectorizados y solvers para millones de incógnitas.

Máquina: 12 núcleos, 33.6 GB de RAM; `pyamg` y `pypardiso` instalados; no hay `petsc4py`, `pyvista`, `meshio` ni MUMPS.

## Estado actual (resumen)

- **Infraestructura:** elementos `line2/3`, `tri3/6`, `quad4/9`; Taylor–Hood (Q9/Q4, T6/T3); BDF2 adaptativo, Newton con tangente
  automática (jax), continuación pseudo-transitoria, axisimetría; VTK ASCII.
- **Físicas (2D y axisimétrico):** calor (radiación, convección), masa, Darcy, Stokes/Navier–Stokes (SUPG/PSPG/LSIC, grad-div), no
  newtonianos (Cross–WLF, Carreau, Bingham...), compresible (Tait), deslizamiento, viscoelástico multimodo (raíz de la conformación,
  DEVSS-G, Oldroyd-B/Giesekus/PTT, no isotermo), turbulencia SA y SST, superficie libre estacionaria (línea de corriente).
- **Llenado de moldes:** Hele-Shaw 2.5D (`Source/Physics/HeleShaw.py`), con temperatura en el espesor, empaque con Tait, llenado
  compresible y cristalización. Es un modelo de plano medio: **no** pasa a 3D cambiando la malla (ver fase 6).
- **Ajuste de materiales:** `Source/Physics/Fitting.py` (Cross–WLF, Tait, espectro viscoelástico).

## Lo que hoy es solo 2D (inventario)

| Dónde | Qué | Qué hace falta |
|---|---|---|
| `Source/enums.py` | `ElementType` solo `line`, `triangle`, `quadrilateral` | `tetrahedron`, `hexahedron` (y quizá `prism`) |
| `Source/Pre_processing/Elements.py` | `ELEMENTS` sin elementos 3D | `tet4`, `tet10`, `hex8`, `hex27`: funciones de forma, segundas derivadas (cuadráticos), cuadratura, caras con sus nodos locales y el `facet_element` (`tri3/tri6/quad4/quad9`) |
| `Elements.py` | cuadratura de triángulo y cuadrilátero | tetraedro (Keast o Grundmann–Möller, o Gauss colapsado) y hexaedro (producto tensorial, trivial) |
| `Discretization.py` (normales de faceta) | `NotImplementedError` en 3D | normal = producto vectorial de las dos tangentes, orientada con el centroide (como en 2D) |
| `Discretization.py` (axisimetría) | exige 2D | sin cambio: en 3D no hay axisimetría |
| `Operators.element_size` | lado del cuadrado o cateto del triángulo de igual área | en 3D, arista del cubo o del tetraedro de igual volumen (afecta a SUPG/PSPG) |
| `Source/Pre_processing/Mesh.py` | `from_gmsh_model`/`read_msh` leen `getElements(2)`, grupos de dim 1 como bordes; `boundary_edges`, `generate2DMesh` | leer dim 3, grupos de dim 2 como caras de borde, `boundary_faces(selector)`, un generador de cajas (`generate3DMesh`) |
| `Source/Post_processing/VTK.py` | `VTK_CELL` sin 3D | tet4 10, hex8 12, tet10 24, hex27 29; VTK binario (los archivos ASCII 3D pesan mucho) |
| Gráficos (`plot_triangles`, `triangulation`) | matplotlib 2D | 3D se ve en ParaView (VTK); opcional `pyvista` |
| `Source/Physics/Viscoelastic.py` | conformación de 3 (plano) o 4 (axisimétrico) componentes, `L` 2 × 2, DEVSS-G de 4 | 6 componentes simétricas, `L` 3 × 3, G de 9; la forma cerrada de la raíz de la conformación en 3D |
| `Source/Physics/Turbulence.py` (`wall_distance`) | distancia a la pared en la malla | comprobar en 3D (si es geométrica a facetas, hace falta la distancia a triángulos; alternativa: ecuación de Poisson/eikonal) |
| `Source/Physics/FreeSurface.py` | método de la línea de corriente, coordenadas `along/across` de una curva | no se generaliza: superficie en 3D es otro método (fase 6) |
| `Source/Physics/HeleShaw.py` | 2.5D por diseño | se queda; el llenado 3D es un módulo nuevo (fase 6) |
| `run_tests.py` y `Test/` | casos 2D/axisimétricos | suites 3D propias, con mallas pequeñas para que la suite siga en tiempo razonable |

Buscar además, al empezar, cada `dim == 2`, `[:, 0]`/`[:, 1]` con significado de x/y, `coor[:2]`, y los comentarios «solo 2D» en `Source/`
(`grep -rn "dim == 2\|coor\[:2\]\|2D" Source`).

## Retos del 3D

1. **Memoria de los arrays vectorizados.** `disc.B` es `(nel, ngp, nne, dim)`: con 10⁵ tet10 y una cuadratura de 15 puntos son
   10⁵ × 15 × 10 × 3 = 45·10⁶ dobles (360 MB) por array, y las matrices de elemento de Taylor–Hood `(nel, 30+4, 30+4)` y los einsum
   intermedios multiplican eso. Hace falta **ensamblar por bloques de elementos** (iterar sobre trozos de `nel`) en `Discretization`,
   `Operators` y en el ensamblaje de `System`, y no guardar arrays que se pueden recalcular.
2. **Solvers.** La factorización directa en 3D crece mucho más que en 2D (relleno ~ n^(4/3) en memoria); PARDISO sirve hasta unos
   10⁵–10⁶ incógnitas en esta máquina. Por encima: GMRES con precondicionador de bloques para el punto de silla (ya existe
   `saddle_point_preconditioner` en `IterativeSolver.py`, solo para `u`, `p`) con AMG (`pyamg`) para la velocidad y la masa de presión para
   el complemento de Schur. Medir con `SolverSelection.benchmark` en cada fase.
3. **Mallas.** Taylor–Hood en 3D: P2/P1 (tet10/tet4) o Q2/Q1 (hex27/hex8). gmsh genera tet10 (`Mesh.ElementOrder 2`); la orientación de
   los nodos de las aristas sigue la convención de gmsh y hay que mapearla a la del elemento de referencia (lo mismo que se hizo con T6/Q9).
4. **Estabilización.** El tamaño de elemento de SUPG/PSPG cambia con la dimensión; las fórmulas de τ que usan `h` deben revisarse.
   Los elementos cuadráticos curvos y las segundas derivadas (residuo fuerte de SUPG) necesitan el hessiano 3D.
5. **Tiempo de las pruebas.** Una suite 3D con mallas medianas puede tardar más que todo lo actual; diseñar pruebas pequeñas
   (soluciones manufacturadas en cubos de 4³–8³ elementos) y dejar los estudios de convergencia grandes en `Test/manual/`.
6. **Visualización y salida.** VTK binario (base64 o appended) y probablemente un VTU por paso con un `.pvd`; los campos de 6 o 9
   componentes (conformación, G) como tensores.
7. **Física 3D nueva.** El viscoelástico en 3D (6 + 9 incógnitas por nodo y modo) es el más caro; el llenado 3D de moldes (dos fases)
   es un proyecto en sí mismo.

## Plan por fases (cada una con pruebas, documentación y petición de commit)

| Fase | Qué | Criterio para seguir |
|---|---|---|
| 0. Infraestructura | `tet4`, `hex8`, `tet10`, `hex27` en `ELEMENTS` (formas, cuadratura, caras), normales 3D, `element_size` 3D, `Mesh.from_gmsh_model` en 3D con grupos de caras, `boundary_faces`, `generate3DMesh` de cajas, VTK 3D | cuadratura exacta hasta su grado; prueba de parche (campo lineal exacto); Poisson manufacturado en un cubo con orden 2 (lineales) y 3 (cuadráticos) en L2; integral de superficie y normales en un cubo y una esfera; lectura de una malla gmsh con grupos |
| 1. Escalares | calor, masa, Darcy en 3D (deberían funcionar sin cambios de física) | soluciones manufacturadas, conducción transitoria en un cubo frente a la serie, convección–difusión con SUPG |
| 2. Stokes y Navier–Stokes | Taylor–Hood P2/P1 y Q2/Q1, igual orden con PSPG; ensamblaje por bloques; solver iterativo de bloques con AMG | Poiseuille en un conducto rectangular frente a su serie de Fourier; tubería en malla 3D frente al resultado axisimétrico del propio código; Navier–Stokes con la solución exacta de Ethier y Steinman (1994) y orden de convergencia; **medir tiempo y memoria** y elegir solver |
| 3. Fluidos de polímeros | no newtonianos, acoplamiento térmico, compresible (Tait), deslizamiento | los mismos casos analíticos de 2D en un conducto; dado rectangular 3D frente al de placas paralelas en el límite ancho |
| 4. Viscoelástico 3D | conformación de 6 componentes, DEVSS-G de 9, SUPG | viscometría en corte y extensión homogéneos (frente a `solve_ivp`, como en 2D); Couette/Poiseuille en conducto; luego contracción 3D si el coste lo permite |
| 5. Turbulencia | SA/SST en 3D, distancia a la pared | canal 3D periódico o conducto frente a los resultados 2D del código y a datos publicados (tomar los valores de las fuentes, no de memoria) |
| 6. Llenado 3D (opcional) | dos fases (conjunto de nivel o VOF) con Navier–Stokes, o 2.5D mejorado (Hele-Shaw con espesores variables y uniones de placas) | decidir antes de empezar: es el bloque más grande; en 2D ya se vio que el frente con conjunto de nivel y un salto de viscosidad de 10⁴ no fue fiable en Hele-Shaw |

Fuera de alcance salvo decisión: superficie libre 3D, alabeo (necesita mecánica de sólidos: una física nueva, aunque el núcleo
lo permite), paralelismo MPI.

## Cómo implementarlo con eficacia

- **Primero la fase 0 completa y probada**: casi todo lo demás hereda de ella. Cada elemento con su prueba de parche y su cuadratura
  verificada antes de usarlo en una física.
- **Reutilizar los casos 2D** como límite: un conducto muy ancho debe dar el resultado de placas paralelas; una tubería 3D, el
  axisimétrico. Eso valida sin referencias externas.
- **Mallas pequeñas en la suite**, estudios de convergencia aparte (`Test/manual/`), como se hizo con Hele-Shaw.
- **Medir antes de optimizar**: tiempo y memoria por fase (`SolverSelection.benchmark`, `cProfile`), con salidas parciales de avance en
  las pruebas largas para ver que no se estancan.
- **Ensamblaje por bloques temprano** (fase 0–2): cambiar después todos los operadores es más caro.
- Documentar en `docs/ESTADO.md` (fila de validación y apartado de cada física), `docs/PLAN_ETAPA17.md` y `README.md` en el mismo commit.

## Convenciones de trabajo (las de este proyecto)

- Hablar en español. Implementar, validar y documentar; **pedir aprobación antes de cada commit; nunca push**. Informar con honestidad lo
  que no se logró y lo que no se verificó.
- Mensajes de commit en español sin acentos, terminando en `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Pruebas con `.venv/Scripts/python.exe run_tests.py` (o la suite suelta: `python -u Test/test_x.py`); cada prueba imprime su avance.
- `docs/` está en `.gitignore`: añadir con `git add -f docs/...`.
- Windows: Git Bash no tiene `pgrep`/`pkill`; los procesos se matan con PowerShell (`Stop-Process`). Las esperas en segundo plano con
  `until grep ...` deben cubrir también el fallo (si no, quedan esperando para siempre).
- Archivos descargados: carpeta propia y `python -I`. Datos del usuario como datos.
- Datos de material disponibles: POLYFORT FPP 30 GFC K1079 (Cross–WLF y Tait ajustados; ver `docs/ESTADO.md`). Sin DSC propio: la
  cristalización usa parámetros de otro iPP.

## Avance

- **Fase 0 hecha (2026-11-07).** `tet4`, `tet10`, `hex8`, `hex27` en `ELEMENTS` (orden de nodos de gmsh, comprobado con
  `getElementProperties`); cuadratura de tetraedro de 1, 4 y 14 puntos (grados 1, 2, 5; la de 14 puntos es la de Walkington, con los
  orbitales resueltos a precisión de máquina) y Gauss–Jacobi colapsada de n³ puntos por encima; normales de cara; `element_size` 3D;
  `generate3DMesh` (hexaedros o 6 tetraedros de Kuhn por celda); gmsh 3D con grupos de caras y regiones; `boundary_faces`; VTK 3D con la
  permutación de tet10/hex27 y salida binaria base64. `Test/test_3d_foundation.py` (13 pruebas, ~4 s): Poisson en el cubo con orden 2.00
  (hex8), 1.95 (tet4), 2.97 (hex27) y 3.02 (tet10); esfera de gmsh con área a 7.3e-5 en tet10.
- **Quedó para después:** el ensamblaje por bloques (se hará al medir en la fase 1–2, con el primer caso que lo necesite); los VTU 3D no
  se abrieron en ParaView (el orden de VTK se comprobó por coordenadas); hexaedros de gmsh con jacobiano negativo no se reorientan.
- **Fase 1 hecha (2026-11-07).** Calor, especies y Darcy en 3D sin cambios de física, salvo `coor[:2]` en las condiciones dadas como
  función (`Darcy.py`, `HeatTransfer.py`, `MassTransfer.py`; queda el de `Viscoelastic.py` para la fase 4). `Test/test_3d_scalars.py`
  (9 pruebas, ~27 s): parche con flujo y convección en caras, Poisson de orden 2/3, cubo frente a la serie de Fourier (BDF2, centro a
  1.7e-3; el error baja 3.4 veces con dt/2), capa límite con SUPG (hex8 exacto; tet4 0.33, no monótono como T3), SUPG manufacturado,
  difusión–reacción = 1D, Darcy anisótropo exacto e hidrostático, inyección que transporta calor.
- **Coste (`Test/manual/heat_3d_scaling.py`):** con PARDISO ~18 KB por nodo (358 k nodos hex8: 36 s, 6.4 GB), casi todo de la
  factorización (la discretización tet10 de 329 k nodos ocupa 1.4 GB). CG + AMG gana desde 9 k nodos (329 k tet10: 14 s y 0.8 GB frente
  a 37 s y 5.9 GB); `SolverSelection.ITERATIVE_SCALAR_3D = 10 000` lo aplica en `Method=auto` (solo escalares simétricos). El
  ensamblaje por bloques no hace falta en escalares; volver a medir en la fase 2, donde las matrices de Taylor–Hood son 3–4 veces mayores
  por elemento y el solver es de punto de silla.
- **Fase 2 hecha (2026-11-08).** Stokes y Navier–Stokes en 3D. `Test/test_3d_flow.py` (7 pruebas): Poiseuille plano exacto en
  Taylor–Hood y en hex8 con PSPG (tet4 no, como T3); conducto 1 × 0.6 frente a la serie de Fourier con orden L2 3 (hex27, tet10) y 2 (hex8);
  tubería de gmsh en tet10 curvos igual al modelo axisimétrico (caudal a 4e-5); Navier–Stokes estacionario con la solución de Ethier y
  Steinman (1994) y la fuerza ν d² u que la hace estacionaria: u de orden 3 y p de orden 2 en Taylor–Hood, u y p de orden 2 en hex8; GMRES
  + bloques igual a PARDISO en 75–95 iteraciones. Las dos soluciones exactas se comprueban en la propia suite (ecuación, paredes, caudal;
  diferencias finitas) porque se escribieron a partir de su forma, no se copiaron de una tabla.
- **Cambios de código de la fase 2:** normal de simetría/salida por el eje en que la faceta es plana; modos rígidos 3D del precondicionador
  de bloques; proyección L2 por CG + Jacobi en 3D (la LU de la masa consistente se llenaba: 12 de 32 s); `origin` en `generate3DMesh`;
  `ITERATIVE_FLOW_3D = 120 000` en `Method=auto`.
- **Coste:** con 180 k incógnitas PARDISO 27 s y 4.7 GB, bloques 19.9 s y 0.8 GB; con 332 k PARDISO 129 s y 11.6 GB. Con bloques, 551 k incógnitas en 67 s y 5.1 GB, y 1.09 millones
  (194 k tet10) en 139 s y 8.9 GB, siempre en 83 iteraciones; la generación de la malla ya tarda 24 s (objetos `Node` de Python). La
  memoria por incógnita sube de 4.5 a 8 KB al crecer: ahí sí pesan los arrays del ensamblaje, de modo que con 33 GB el techo práctico es de
  unos 3–4 millones de incógnitas sin ensamblaje por bloques; es la palanca siguiente si hiciera falta más.
- **Pendientes de la fase 2, cerrados (2026-11-08):**
  - Navier–Stokes transitorio: Ethier–Steinman con BDF2 y la velocidad de contorno escalada con `setBCTime`; a t = 1 el error es el
    espacial (u 3.0e-3 relativo en hex27 4³) y el temporal converge con orden 2.5. El Newton modificado usa 2 tangentes en 20 pasos.
  - SUPG de Taylor–Hood: **defecto encontrado y corregido, también en 2D**. SUPG solo dejaba la presión del vértice de un elemento con
    todos sus vértices en una pared de Dirichlet en manos de la ecuación de momento de un único nodo interior, donde el término SUPG casi
    anula el de Galerkin: T6 (Kovasznay Re = 1000) 0.73 de error en esa esquina, tet10 10. Ahora es SUPG/PSPG (Tezduyar): 1.3e-4 y 0.27,
    y la cavidad a Re = 5000 queda más cerca de Ghia que con Galerkin (0.085 frente a 0.188 en 16²). En tetraedros a Re moderado cuesta
    precisión (constante 3 veces la de Galerkin); sirve para Re alto, donde tet10 sin estabilizar no converge.
  - Bloques con inercia: mismos pasos de Newton y misma solución que PARDISO; las iteraciones crecen con la convección (ν = 0.1: ~160;
    ν = 0.01: hasta 1000), de modo que con convección fuerte PARDISO es el doble de rápido con 144 k incógnitas, aunque usa 2.5 veces más
    memoria. El umbral de `Method=auto` (120 k) queda igual.
  - Memoria: medida por fases, dominan las copias de la matriz global y el AMG, no los arrays de elemento. `boundary_faces` sin
    `Discretization` (malla de 1.09 M: 24 → 5.6 s), AMG `jacobi` en 3D (−35 % de pico con 332 k), acoplamiento de presiones sin LIL,
    `scatter_matrix` por tramos. Con 1.09 M incógnitas sigue en ~140 s y ~9 GB: la siguiente palanca es no guardar a la vez la matriz de
    `System`, la restringida, la de Dirichlet y la libre del solver.
- **Fase 3 hecha (2026-11-08).** `compressible_law` en cualquier dimensión; deslizamiento con ley en 3D, isótropo en el plano de la
  pared (τ = τ(|u_s|) u_s/|u_s|, liso en u_s = 0). `Test/test_3d_polymer.py` (7 pruebas, ~60 s): el mismo canal plano en 2D y como losa
  3D coincide a 1e-11 o mejor con ley de potencia, Carreau, Cross–WLF con D3 + calentamiento viscoso + calor convectado, densidad
  exponencial, Tait térmico con calentamiento por compresión (transitorio) y deslizamiento de Navier y de potencia (también frente a su
  perfil exacto); tangente 3D con todos los términos frente a diferencias finitas (3e-8); dado rectangular de W/H = 4, 8, 16 frente a
  placas paralelas: newtoniano a 6e-4 de la serie del conducto, ley de potencia con un déficit de caudal ∝ H/W.
- **Defecto de PARDISO (fase 3):** el emparejamiento ponderado por omisión arruinaba algunos puntos de silla de Taylor–Hood 3D
  (residuo 1e3 en el dado W/H = 16). `DirectSolver.checked_pardiso` comprueba el residuo de cada solución: refina por encima de 1e-10 y
  refactoriza sin emparejamiento por encima de 1e-6, usando esa factorización solo si da un residuo bueno; con una matriz casi singular
  conserva la respuesta de PARDISO (cambiarla alteraba el Newton de casos viscoelásticos validados). Prueba de regresión en
  `Test/test_3d_flow.py`.
- **Fase 4 hecha (2026-11-08).** Raíz de la conformación con 6 componentes: la antisimétrica a = [ω]× sale de
  (tr b I − b) ω = axial(b Lᵀ − L b) (identidad b [ω]× + [ω]× b = [(tr b I − b) ω]×; matriz definida positiva para b definida positiva),
  que en flujo plano es la forma cerrada 2D. DEVSS-G de 9 componentes, entrada con gradiente 3 × 3, tracción de salida, energía y esfuerzo
  en 3D; definición positiva por autovalores. `Test/test_3d_viscoelastic.py` (5 pruebas, ~4 min): ecuación de b frente a la de c (1e-15)
  y frente a la plana; flujos homogéneos 3D con BDF2 de orden 2 frente a `solve_ivp`; tangente 3D completa (2e-8); Couette y canal 3D
  iguales al 2D (1e-14, 3e-9: la losa con el grosor que iguala el h de SUPG de b, que en 2D es √área y en 3D ∛volumen); conducto
  rectangular: Oldroyd-B (N2 = 0) rectilíneo con la velocidad newtoniana de μ_s + μ_p, su flujo secundario cae con orden 3 al refinar,
  y el de Giesekus converge a ~2e-3 de u_máx (14 veces mayor en la malla fina).
- **Sin hacer en la fase 4:** contracción 3D (el plan la dejaba para «si el coste lo permite»: 19 incógnitas por nodo, 21 k incógnitas
  ya son 36 s por solución estacionaria con PARDISO), viscoelástico no isotermo en un flujo 3D, difusión artificial, varios modos y PTT en flujo 3D, y el
  solver de bloques con los campos b y G (los trataría con ILU, sin medir).
- **Fase 5, lo siguiente:** turbulencia SA/SST en 3D; revisar `Turbulence.wall_distance` (usa `coor[:2]`: distancia geométrica a
  aristas, en 3D hace falta a caras triangulares/cuadriláteras o una ecuación de Poisson/eikonal).

## Primeros pasos de la nueva sesión

1. Leer este archivo y `docs/ESTADO.md`; correr la suite completa para tener la línea base (39 suites, unos 35 min).
2. `grep` del inventario de arriba para confirmar los puntos 2D y encontrar los que falten.
3. Fase 0: `tet4` y `hex8` primero (lineales, más simples), con su cuadratura, caras y prueba de parche; luego `tet10` y `hex27`.
