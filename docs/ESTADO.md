# Estado del proyecto, validación y próximos pasos

Última actualización: 2026-10-28. Rama de trabajo: `test-branch`.

## 1. Qué está validado

| Capacidad | Cómo se valida | Archivo de prueba | Resultado |
|---|---|---|---|
| 1D lineal, Dirichlet | perfil lineal exacto | `Test/test_analytic.py` | pasa |
| 1D lineal, Newton en extremo | solución exacta con convección en un extremo | `Test/test_analytic.py` | pasa |
| 1D CG + iLU | igual que el caso lineal | `Test/test_analytic.py` | pasa |
| Radiación (unidades) | h(T)(T - T_ext) = εσ(T⁴ - T_ext⁴) | `Test/test_analytic.py` | pasa |
| Calor vectorizado | A, b, M, R, J, transitorio y Newton frente a valores de referencia (obtenidos cuando coincidían con la ruta por elemento) y J frente a diferencias finitas | `Test/test_vectorized.py` | pasa (6) |
| Newton (radiación) | valor de referencia y punto inicial malo | `Test/test_solvers.py` | pasa |
| Transitorio θ-método | decaimiento analítico sin(πx)·exp(-π²t) | `Test/test_solvers.py` | pasa (error 1.6e-3 Euler dt=1e-3; 4.9e-4 CN dt=1e-2) |
| 2D topología y orientación | 12 nodos, 6 elementos, 10 aristas de contorno | `Test/test_2d_poisson.py` | pasa |
| 2D parche lineal (Dirichlet) | T = 2x + 3y + 1 exacto | `Test/test_2d_poisson.py` | pasa (error < 1e-10) |
| 2D Neumann en arista | T = 2x + 1 exacto | `Test/test_2d_poisson.py` | pasa |
| 2D Newton en arista | T = 2x + 1 exacto | `Test/test_2d_poisson.py` | pasa |
| 2D convergencia (fuente) | Poisson con sin·sin, mallas 4 a 32 | `Test/test_2d_poisson.py`, `Example_2D_Poisson.py` | orden 2.00 |
| 2D convección = 1D | flujo en x con Dirichlet izquierda/derecha no depende de y | `Test/test_2d_poisson.py` | pasa (1e-9) |
| Base: elementos, facetas, normales, DOF, operadores | partición de la unidad, áreas, normales salientes, matrices 1D conocidas | `Test/test_foundation.py` | pasa (11) |
| Leyes puntuales | derivadas frente a diferencias finitas; se compilan una sola vez | `Test/test_foundation.py` | pasa |
| Especies: difusión + reacción de primer orden | `c = sinh(m(L-x))/sinh(mL)`, mallas 8 a 64 | `Test/test_species.py` | orden 2.00 |
| Especies: convección-difusión-reacción | exponenciales exactas, Pe moderado | `Test/test_species.py` | error < 5e-4 |
| Especies: A + B → C (segundo orden) | `A + C` y `B + C` deben ser lineales (estequiometría, acoplo, Newton) | `Test/test_species.py` | pasa (1e-7) |
| Especies: jacobiano con derivadas cruzadas | 3 especies, 2 reacciones, convección, frente a diferencias finitas | `Test/test_species.py` | pasa |
| Especies: transitorio | decaimiento de sin(πx) con difusividades distintas | `Test/test_species.py` | pasa (< 2e-3) |
| Especies 2D: reacción no lineal manufacturada | `c = 1 + sin·sin`, `k c²`, mallas 4 a 16 | `Test/test_species.py` | orden 2.0 |
| Especies 2D: flujo y transferencia en aristas | campo lineal exacto | `Test/test_species.py` | pasa (1e-10) |
| 2D no lineal: radiación | solución independiente de y = solución 1D en los mismos nodos | `Test/test_nonlinear_transient.py` | pasa (1e-8) |
| 2D no lineal: balance de energía | calor que entra por el borde fijo = calor radiado (cuadratura de faceta) | `Test/test_nonlinear_transient.py` | pasa (1e-6) |
| 2D no lineal: jacobiano y Newton | radiación + `h(T)` en otra arista frente a diferencias finitas; punto inicial malo (400 C) | `Test/test_nonlinear_transient.py` | pasa |
| 2D transitorio: decaimiento | `exp(-2π²αt) sin(πx) sin(πy)` con Euler y Crank-Nicolson | `Test/test_nonlinear_transient.py` | pasa |
| 2D transitorio: orden temporal | autoconvergencia en una malla | `Test/test_nonlinear_transient.py` | Euler 1.0, Crank-Nicolson 2.0 |
| 2D transitorio: conservación | `sum(M ΔT) = -q L t` exacto para θ = 1 y 0.5 | `Test/test_nonlinear_transient.py` | pasa (1e-9) |
| 2D transitorio no lineal: límite de tiempo largo | tiende a la solución estacionaria de Newton | `Test/test_nonlinear_transient.py` | pasa (1e-4) |
| k(T) estacionario 1D | solución exacta de Kirchhoff (`T + βT²/2` lineal) | `Test/test_nonlinear_diffusion.py` | exacta a 1e-9 en los nodos |
| k(T) 2D | y-independiente = 1D; manufacturada con `k = 1 + T²` | `Test/test_nonlinear_diffusion.py` | pasa; orden 2.00 |
| k(T) jacobiano | k(T) + convección vectorial + radiación frente a diferencias finitas | `Test/test_nonlinear_diffusion.py` | pasa |
| k(T) transitorio | conservación exacta de energía y límite estacionario | `Test/test_nonlinear_diffusion.py` | pasa |
| Calor + especies: reactor adiabático | `T`, `A`, `C` uniformes frente a `scipy.integrate.solve_ivp` | `Test/test_coupling.py` | orden temporal 2.00 (Crank-Nicolson), ~1.1 (Euler) |
| Calor + especies: conservación | `rho Cp T + Q A` y `A + C` constantes en el sistema discreto, θ = 1 y 0.5 | `Test/test_coupling.py` | pasa (1e-8) |
| Calor + especies: invariantes estacionarios | `k T - Q D C` y `k T + Q D A` lineales con la solución de Newton | `Test/test_coupling.py` | pasa (1e-8) |
| Calor + especies: 2D = 1D, jacobiano cruzado | datos independientes de y; J frente a diferencias finitas con bloques T-especie | `Test/test_coupling.py` | pasa |
| Composición de físicas | delegación, errores por física o variable repetida, campo `T` ausente, `dH` sin campo | `Test/test_coupling.py` | pasa |
| `D(c)` exacta (Kirchhoff) | P1 con 2 puntos de Gauss es exacto en los nodos para `D = D0 (1 + b c)` | `Test/test_multifield_diffusion.py` | pasa (1e-9) |
| `k(c)` y `D(T)` acoplados | perfil exacto con un campo conductor lineal; orden temporal-espacial | `Test/test_multifield_diffusion.py` | orden 2 |
| Leyes multicampo | jacobiano (1D y 2D) frente a diferencias finitas, conservación transitoria con `D(c)`, errores de campo ausente | `Test/test_multifield_diffusion.py` | pasa |
| SUPG 1D | solución nodalmente exacta de convección-difusión con Pe = 20, 50, 500; Galerkin oscila | `Test/test_convection_supg.py` | error < 1e-8 |
| SUPG 2D y fuente | 2D con flujo en x = 1D; manufacturada con fuente consistente | `Test/test_convection_supg.py` | orden 2 |
| Velocidad compartida | calor y especie con una sola `setVelocity` siguen el mismo perfil exacto; transitorio llega al estacionario estabilizado | `Test/test_convection_supg.py` | pasa |
| T3: elemento de referencia y geometría | partición de la unidad, regla de Gauss, áreas, perímetro y normales (identidades de superficie cerrada) | `Test/test_triangles_gmsh.py` | pasa (1e-12) |
| T3: parche | `T = 2x + 3y` exacto con Dirichlet y flujo | `Test/test_triangles_gmsh.py` | pasa (1e-10) |
| T3: manufacturada y reacción-difusión | Poisson `sin sin`; `cosh` analítica con reacción de primer orden | `Test/test_triangles_gmsh.py` | orden 2 |
| gmsh: anillo y grupos | solución logarítmica en un cuarto de anillo; grupos físicos cubren el contorno; Q4 de gmsh exacto | `Test/test_triangles_gmsh.py` | pasa |
| AMG y solvers iterativos | CG con AMG, iLU y Jacobi igualan a PARDISO (1e-8); iteraciones de AMG casi constantes (8-10 de 2.6 mil a 160 mil incógnitas) frente a CG sin precondicionar (72 a 293) | `Test/test_solver_scaling_output.py` | pasa |
| Newton iterativo | GMRES + AMG en cada paso de Newton igual a Newton directo con `k(T)` | `Test/test_solver_scaling_output.py` | pasa (1e-7) |
| Factorización reutilizada | transitorio lineal con una sola factorización (PARDISO y SuperLU); `solve()` repetido solo refactoriza si cambia la matriz | `Test/test_solver_scaling_output.py` | pasa |
| Salida VTK | `.vtu` (Q4, T3, línea) y serie `.pvd` releídos y comparados con malla y solución | `Test/test_solver_scaling_output.py` | pasa |
| Newton ante una reacción desbocada | nunca deja NaN: se detiene con aviso y conserva el último estado finito | `Test/test_coupling.py` | pasa |
| Operadores vectoriales | bloques = escalar ⊗ I, simetría, núcleo de movimientos rígidos, energía de corte puro, divergencia y su traspuesta | `Test/test_fluids_foundation.py` | pasa (1e-12) |
| Viscoso `-∇·(2μ ε(u)) = f` | manufacturada, Q4 y T3, mallas 4 a 32 | `Test/test_fluids_foundation.py` | orden 2.00 (Q4), 1.99 (T3) |
| Tracciones y Dirichlet por componente | deformación pura exacta con paredes de simetría y tracciones, malla distorsionada | `Test/test_fluids_foundation.py` | pasa (1e-10) |
| Campo vectorial transitorio y VTK | decaimiento `exp(-2π²t)` de dos componentes; `.pvd` con vectores de 3 componentes | `Test/test_fluids_foundation.py` | pasa |
| Leyes con gradientes | reproducen difusión, convección y operador viscoso; jacobiano vector + escalar no lineal frente a diferencias finitas (Q4, T3) | `Test/test_fluids_foundation.py` | pasa (1e-7 relativo) |
| Ley de potencia en canal (`p`-Laplaciano) | perfil analítico, Newton desde el perfil newtoniano | `Test/test_fluids_foundation.py` | n = 0.5: orden 2.00; n = 1.5: orden 1.6 (regularidad `C^{1,2/3}` del perfil exacto) |
| Darcy | 1D con fuente (exacto en nodos); 2D lineal con permeabilidad tensorial y gravedad; inyección; hidrostática sin flujo | `Test/test_fluids_foundation.py` | pasa (1e-11) |
| Darcy → convección | `ht` y `mt` con la velocidad de Darcy en Gauss = con la velocidad constante equivalente | `Test/test_fluids_foundation.py` | pasa (1e-12) |
| Malla 2D frente a referencia | coordenadas y conectividad iguales a `2DFEM.py` | comprobación manual | igual |

Para reproducir todo: ver la sección «Pruebas» del [README](../README.md).

## 2. Qué no está validado o está incompleto

### Ruta antigua por elemento: retirada
La ruta por elemento con JAX (`Physics.py`, ramas de `core.py`, `basisFunctions` con sympy y las funciones 1D de JAX)
se eliminó en la Etapa 6c (-1065 líneas). Los límites que tenía (un nodo Newton por elemento, nodos interiores contados
varias veces) ya no existen. Las BC Newton/Neumann actúan solo sobre facetas del contorno: una BC en un nodo interior
lanza `ValueError` en lugar de contarse varias veces.

### Transporte de especies (`MassTransfer.py`, clase `mt`)
- Ahora usa la base multifísica (`build_system`, `System`, `PointwiseLaw`) y está validado (ver sección 1).
  `MassTransferSimulation.py` pasó de ~49 s a ~0.3 s.
- Corrección de comportamiento: la ruta antigua no lineal **ignoraba `setDiffusivity`** (usaba difusividad 1 para todas
  las especies). La nueva respeta la de cada especie, así que los resultados del ejemplo cambian. Con difusividades
  iguales a 1 la diferencia con la ruta antigua es de 0.09 sobre valores de ~20 (10 elementos), por evaluar la reacción
  en los puntos de Gauss en lugar de interpolar tasas nodales.
- La velocidad de convección es `physics.vel` (antes había una constante `u = 0.01` en la ruta lineal y `vel = 1` en la
  no lineal).
- La temperatura como campo en la ley de Arrhenius y el calor de reacción ya están (ver «Acoplamiento calor-masa»).

### Limpieza hecha en la Etapa 5 y lo que queda
Se eliminó el código muerto y los imports sin uso (`Source/` pasó de 3463 a 2763 líneas, -20 %):
- `Solvers.py`: `fixPointSolver`, `createNonlinearSystem`, `callBackFunc`, `updConvergencePlot`, `assembleNonlinearSystem`.
- `Physics.py`: toda la ruta por lotes con `jax.vmap` (`get_batched_*`, `build_batched_*`, `_batched_*`), `Grad`,
  `stabilization`, `aux_getTangentMatrix` y el código comentado.
- `core.py`: `assembleResidualVector`, `build_batched_inputs` y las funciones `contributions_determination*`.
- `Mesh.py`: `generate1DMesh`. `BasisFunctions.py`: `addStab`.
- Imports sin uso en todos los módulos (`numba`, `joblib`, `progress`, `numdifftools`, `scipy.linalg`, ...).

Lo que sigue pendiente:
- `Mesh.checkBoundaries` usa `EpP`, que solo se fija para elementos `line` (la usa `oswald_Hernandez_9_1.py`).
- `Mesh.setElementShapeFunction` solo registra el tipo de función de forma (se mantiene por compatibilidad con los ejemplos).
- Los archivos `test_ilu.py`, `test_lagrange.py`, `test_lambda.py`, `test_pardiso.py` y `testNumDiffTools.py` son
  scripts exploratorios, no pruebas automáticas.

### Acoplamiento calor-masa
- Operativo con `Model(psc=[ht, mt])`: Arrhenius con la T del campo, calor de reacción `dH`, y coeficientes de difusión que
  dependen de otros campos (`k(T)`, `k(c)`, `D(T)`, `D(c)` con `depends_on`). La velocidad se comparte con `model.physics.setVelocity(v)` y se estabiliza con `setStabilization('SUPG')`.
  No incluye todavía `rho Cp(T)`, velocidad calculada por un solver de fluidos, ni captura de choques.
- Las leyes de difusión deben escribirse con `jax.numpy`; si se cambia la función hay que reasignarla (se compila por función).
- **Varios estados estacionarios (ignición):** con exotermia fuerte puede haber más de una solución. Newton puede caer en
  otra rama o tardar muchas iteraciones cerca de la ignición (con `Q = 300` y `k0 = 3e4` el caso de prueba necesita
  unas 160 iteraciones). Recomendación: marchar en el tiempo hasta el estacionario y usar ese estado como inicial.
- **Escalado en Newton:** el residuo se escala fila a fila con la diagonal del jacobiano inicial (cada ecuación en las
  unidades de su incógnita). La tolerancia relativa pasó de 1e-8 a 1e-10 para conservar la exactitud de los casos 1D.
- `heat.Convection = True` etc. deben asignarse en la parte (`model.physics.heat`), no en la combinación.

### Fluidos (Etapa 9)
- Solo Darcy está operativo como física de flujo; la velocidad pasa a otra física en los puntos de Gauss (acoplamiento
  en un sentido, misma malla). El acoplamiento monolítico con un campo de velocidad calculado llega con Navier–Stokes.
- `GradientLawTerm` evalúa un arreglo `W` de tamaño `nel·ngp·nne·nc·nc(1+dim)` por campo; es genérico pero más lento que
  un operador dedicado. Para Navier–Stokes en mallas grandes conviene un operador de convección vectorizado propio.
- `derived_fields` solo se añade en estudios estacionarios.
- La ley de potencia con `n < 1` necesita regularización (`(γ̇² + ε²)^((n-1)/2)`) y un punto inicial no nulo.

### Física y numérica
- **Convección:** diferencias centradas sin estabilización. Para Pe alto habrá oscilaciones. `Pe` se calcula pero no se
  usa. El código de estabilización que no estaba conectado se eliminó en la Etapa 5; habrá que implementarlo.
- **Convección 2D:** no implementada.
- **Transitorio:** paso fijo y solver directo (`splu`). Con términos no lineales cada paso se linealiza con Newton al
  inicio del paso (esquema linealmente implícito), sin iterar dentro del paso.
- **Crank-Nicolson:** no es L-estable. Con estados iniciales discontinuos o muy rígidos y pasos grandes los modos de
  alta frecuencia oscilan y decaen despacio (en `Example_2D_Transient.py` los errores con pasos grandes son irregulares
  hasta que el paso es pequeño). Use Euler implícito o un paso menor.
- **Transitorio, precisión:** `Simulation_time_dependent.py` usa `timeStep = 0.05` (valor por defecto). Con Euler
  implícito el error del caso analítico de prueba es 7.5e-2 (medido). Con Crank-Nicolson (`timeDisc = 0.5`) y el mismo
  paso el error es 7.9e-3. Para el ejemplo conviene usar `timeDisc = 0.5` o un paso menor.
- **Radiación:** las temperaturas de entrada son °C. La salida de `Simulation.py` cambió en la Etapa 1
  (extremo ~33 °C, antes ~59.6 °C) porque antes la radiación no usaba Kelvin.
- **Solvers iterativos:** solo iLU como precondicionador. No hay AMG. CG solo es válido si la matriz es SPD.
- **No linealidad en el material:** `k(T)` está soportado (`setConductivityLaw`). `rho Cp(T)` no: la masa es constante
  en el θ-método. La ley de `k(T)` debe escribirse con `jax.numpy`.
- **Aristas (2D):** `T_ext` debe ser un número (`h` puede ser un número o una función de la temperatura). Si una esquina
  pertenece a dos aristas con BC distintas, gana la última asignación de Dirichlet.

### 2D y mallas
- Elementos: Q4 y T3 (lineales). T3 usa una regla de Gauss de 3 puntos (grado 2), suficiente para masa y términos lineales.
- Mallas de gmsh: `Mesh.read_msh` / `Mesh.from_gmsh_model` admiten un solo tipo de elemento 2D (T3 o Q4); las mallas mixtas se rechazan.
  Los bordes son rectos (sin elementos curvos), de modo que los arcos se aproximan por polígonos (error geométrico O(h²)).
- Sin elementos de orden superior (T6, Q8/Q9), sin tetraedros/hexaedros ni 3D. La región física 2D (`mesh.regions`) se guarda
  pero aún no asigna materiales distintos por región.
- SUPG sobre triángulos reduce mucho las oscilaciones pero no las elimina (no es monótono con flujo oblicuo a la malla).

### Solvers y salida
- En mallas hasta 160 mil incógnitas PARDISO sigue siendo tan rápido como CG + AMG (el ensamblado domina): el beneficio de AMG
  son las iteraciones casi constantes y la memoria, que importan en mallas mayores.
- AMG (aglomeración suavizada) está pensado para difusión simétrica; con convección fuerte o varios campos acoplados puede
  necesitar más iteraciones o GMRES/BiCGSTAB. Sin precondicionador de bloques para sistemas acoplados.
- La salida es VTK XML en ASCII (grande en mallas muy finas); no hay XDMF/HDF5 ni VTK binario ni datos por celda.
- La factorización se reutiliza si la matriz es idéntica; en problemas no lineales cada iteración de Newton refactoriza.

### Entorno y archivos
- `Logs/`: cada importación de `Source` crea un archivo de log nuevo. Con muchas ejecuciones la carpeta crece.
- `plt.show()` bloquea en `postProcess` y `postProcess2D`. En scripts sin pantalla use `MPLBACKEND=Agg`.
- `Simulation.py` y otros ejemplos definen BC con IDs fijos (`NoE` para la salida). Cambiar la malla exige revisarlos.
- Los cuadernos `.ipynb` no se han actualizado a la API actual.
- `oswald_Hernandez_9_1.py`, `MassTransferSimulation.py` y `Truss_Structures/` no se han revisado.
- `venv/` y `.venv/` están en el repositorio. Conviene ignorarlos en git.

## 3. Historial de cambios relevantes

| Commit | Rama | Contenido |
|---|---|---|
| `8c1fee4` | `backup/vmap-wip` | WIP del ensamblado por lotes con vmap (respaldo, con errores conocidos) |
| `d464fa9` | `test-branch` | Etapa 1: BC correctas, radiación en Kelvin, `solve()`, CG, pruebas analíticas |
| `d7388e3` | `test-branch` | Etapa 2: ensamblado vectorizado 1D |
| `141d15e` | `test-branch` | Etapa 3: Newton con line search, transitorio θ-método |
| `fa50dbe` | `test-branch` | Etapa 4: calor 2D estacionario lineal (Q4) con BC en aristas |

| `d8b1740` | `test-branch` | Documentación, `requirements.txt` y `Example_2D_Poisson.py` |

| `9e295c7` | `test-branch` | Etapa 5: limpieza de código muerto y `run_tests.py` |

| `f2ed5bf` | `test-branch` | Etapa 6a y 6b: base multifísica y transporte de especies |

| `0221dc8` | `test-branch` | Etapa 6c: retirada de la ruta por elemento |

| `2effe82` | `test-branch` | Etapa 7: calor 2D no lineal y transitorio |

| `675e5b4` | `test-branch` | `k(T)`: difusión no lineal con conductividad dependiente de la temperatura |

El acoplamiento calor-masa está en el árbol de trabajo y todavía no está commiteado.

## 4. Próximos pasos propuestos

Ordenados por el valor que aportan al siguiente paso. Cada uno debe llevar sus pruebas.

1. ~~**Limpieza (Etapa 5).**~~ Hecha: código muerto e imports eliminados y `run_tests.py` añadido. Las 19 pruebas
   siguen en verde y `Physics.py` pasó de 980 a 660 líneas.
2. ~~**Etapa 6: base multifísica y especies.**~~ Hecha (6a base, 6b especies, 6c retirada de la ruta por elemento).
   Las siguientes etapas de física (sólidos, fluidos) deben apoyarse en `System`, `DofLayout`, `Operators` y `PointwiseLaw`.
3. ~~**2D no lineal y transitorio (Etapa 7).**~~ Hecha: radiación y `h(T)` en aristas, Newton y transitorio 2D, con
   `Test/test_nonlinear_transient.py` y `Example_2D_Transient.py`. `k(T)` también está hecho (término de difusión no
   lineal). El acoplamiento calor-masa también está hecho (primer paso: Arrhenius con la T del campo y calor de reacción). Siguiente en
   física: sólidos/fluidos (la convección compartida y estabilizada ya está) (`D(T)` y `k(c)` ya están).
4. ~~**Mallas y geometría (Etapa 8).**~~ Hecha: T3 en `Elements.py`, `generate2DMesh` con triángulos, importación de gmsh con
   grupos físicos por nombre, `Test/test_triangles_gmsh.py` y `Example_Gmsh_Geometry.py`. Pendiente: materiales por región,
   elementos de orden superior y 3D.
5. ~~**Escalado de solvers y salida.**~~ Hecha: AMG (`pyamg`), Jacobi e iLU sobre el sistema reducido, `tol`/`maxiter`,
   factorización reutilizada (transitorio lineal y `solve()` repetidos), exportación VTK/PVD. Pendiente: XDMF/HDF5, VTK binario,
   iterativo para sistemas con convección fuerte o varios campos acoplados (AMG por bloques).
6. ~~**Estabilización de convección.**~~ Hecha (SUPG en 1D y 2D).
7. **Fluidos.** Ver [PLAN_FLUIDOS.md](PLAN_FLUIDOS.md): Etapa 9 hecha (base vectorial y Darcy); siguiente, Stokes con PSPG.

## 5. Criterio para cerrar una etapa

- Todas las suites de `Test/` pasan (`python run_tests.py`).
- Un caso de referencia reproduce el resultado anterior o explica la diferencia.
- La documentación de `README.md`, `docs/ARQUITECTURA.md` y esta página se actualiza en el mismo commit.
