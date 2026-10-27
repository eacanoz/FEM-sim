# Estado del proyecto, validación y próximos pasos

Última actualización: 2026-10-26. Rama de trabajo: `test-branch`.

## 1. Qué está validado

| Capacidad | Cómo se valida | Archivo de prueba | Resultado |
|---|---|---|---|
| 1D lineal, Dirichlet | perfil lineal exacto | `Test/test_analytic.py` | pasa |
| 1D lineal, Newton en extremo | solución exacta con convección en un extremo | `Test/test_analytic.py` | pasa |
| 1D CG + iLU | igual que el caso lineal | `Test/test_analytic.py` | pasa |
| Radiación (unidades) | h(T)(T - T_ext) = εσ(T⁴ - T_ext⁴) | `Test/test_analytic.py` | pasa |
| Ensamblado vectorizado = ruta por elemento | comparación de A, b, M, R, J y transitorio corto | `Test/test_vectorized.py` | pasa |
| Newton (radiación) | valor de referencia y punto inicial malo | `Test/test_solvers.py` | pasa |
| Transitorio θ-método | decaimiento analítico sin(πx)·exp(-π²t) | `Test/test_solvers.py` | pasa (error 1.6e-3 Euler dt=1e-3; 4.9e-4 CN dt=1e-2) |
| 2D topología y orientación | 12 nodos, 6 elementos, 10 aristas de contorno | `Test/test_2d_poisson.py` | pasa |
| 2D parche lineal (Dirichlet) | T = 2x + 3y + 1 exacto | `Test/test_2d_poisson.py` | pasa (error < 1e-10) |
| 2D Neumann en arista | T = 2x + 1 exacto | `Test/test_2d_poisson.py` | pasa |
| 2D Newton en arista | T = 2x + 1 exacto | `Test/test_2d_poisson.py` | pasa |
| 2D convergencia (fuente) | Poisson con sin·sin, mallas 4 a 32 | `Test/test_2d_poisson.py`, `Example_2D_Poisson.py` | orden 2.00 |
| Malla 2D frente a referencia | coordenadas y conectividad iguales a `2DFEM.py` | comprobación manual | igual |

Para reproducir todo: ver la sección «Pruebas» del [README](../README.md).

## 2. Qué no está validado o está incompleto

### Ruta antigua por elemento (JAX)
- `Physics.py` y `BasisFunctions.py` siguen usando JAX con funciones estáticas. Cada elemento puede provocar una
  recompilación porque `func_normalization` crea funciones nuevas. Es lento. Hoy solo la usan `mt` y los casos
  no vectorizados.
- Límite: un elemento no puede tener más de un nodo con BC Newton o Neumann (`NotImplementedError`).
- En esa ruta, un nodo interior con BC Neumann o Newton se contaría varias veces. El ensamblador vectorizado sí lo
  cuenta una sola vez.

### Transporte de especies (`MassTransfer.py`, clase `mt`)
- **Sin validar.** No tiene pruebas. `MassTransferSimulation.py` y `Test/mass_transfer_reaction.ipynb` no se han
  ejecutado en esta revisión.
- `setDiffusivity` hace `raise "texto"`, lo que produce `TypeError`.
- `build_reaction_function` tiene un `rate_vector.reshape(-1, 1)` sin asignar (no hace nada).
- Usa la ruta antigua por elemento.

### Código muerto o incompleto (candidato a limpieza)
- `Solvers.py`: `fixPointSolver`, `createNonlinearSystem`, `callBackFunc` de `modelSolver`, `updConvergencePlot`.
- `Physics.py`: `get_batched_*`, `build_batched_g_params` (no devuelve nada), `_batched_*` con `in_axes` de 13 entradas
  para 14 parámetros, `_calc_stabilization_term`, `stabilization`, `Grad`, `_calc_b_matrix_2` (asume el último nodo).
- `core.py`: `assembleResidualVector` (variable `penalty` sin uso), `build_batched_inputs`.
- `Mesh.py`: `generate1DMesh` (vacía), `checkBoundaries` (usa `EpP`, solo fijado para `line`).
- Imports sin uso: `jax` y `numba` en varios módulos, `numba.jitclass` en `Mesh.py`, `numdifftools`.
- Los archivos `test_ilu.py`, `test_lagrange.py`, `test_lambda.py`, `test_pardiso.py` y `testNumDiffTools.py` son
  scripts exploratorios, no pruebas automáticas.

### Física y numérica
- **Convección:** diferencias centradas sin estabilización. Para Pe alto habrá oscilaciones. `Pe` se calcula pero no se
  usa. `_calc_stabilization_term` existe pero no está conectado.
- **Convección 2D:** no implementada.
- **Transitorio:** solo 1D. Las BC no lineales se evalúan al inicio de cada paso (esquema semi-implícito).
- **Transitorio, precisión:** `Simulation_time_dependent.py` usa `timeStep = 0.05` (valor por defecto). Con Euler
  implícito el error del caso analítico de prueba es 7.5e-2 (medido). Con Crank-Nicolson (`timeDisc = 0.5`) y el mismo
  paso el error es 7.9e-3. Para el ejemplo conviene usar `timeDisc = 0.5` o un paso menor.
- **Radiación:** las temperaturas de entrada son °C. La salida de `Simulation.py` cambió en la Etapa 1
  (extremo ~33 °C, antes ~59.6 °C) porque antes la radiación no usaba Kelvin.
- **Solvers iterativos:** solo iLU como precondicionador. No hay AMG. CG solo es válido si la matriz es SPD.
- **Newton:** solo 1D. En 2D no lineal (radiación en aristas) no hay soporte.
- **Aristas (2D):** `h` y `T_ext` deben ser números. Si una esquina pertenece a dos aristas con BC distintas, gana la
  última asignación de Dirichlet.

### 2D y mallas
- Solo cuadriláteros Q4 en rectángulo, generados internamente.
- No hay importación de gmsh (`readMesh.py` solo imprime información). `t1.msh` no se usa.
- Triángulos T3: no implementados.

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

Los cambios posteriores a `fa50dbe` (ejemplo 2D, documentación y `requirements.txt`) no están commiteados.

## 4. Próximos pasos propuestos

Ordenados por el valor que aportan al siguiente paso. Cada uno debe llevar sus pruebas.

1. **Limpieza (Etapa 5).** Eliminar el código muerto de la sección 2 y los imports sin uso. Añadir `run_tests.py` que
   ejecute las cuatro suites y devuelva código de salida. Resultado esperado: el mismo número de pruebas en verde y menos
   líneas en `Physics.py`. Es la base para escalar sin miedo a romper cosas.
2. **Transporte de especies vectorizado (Etapa 6).** Llevar `mt` a `HeatSystem` (o una clase genérica) con reacción
   dependiente del estado vía `residual_and_tangent`. Validar con difusión-reacción 1D con solución analítica
   (decaimiento lineal de primer orden) y con la ruta antigua.
3. **2D no lineal y transitorio (Etapa 7).** Radiación en aristas (Newton con h(T) y su derivada) y transitorio 2D
   reutilizando `HeatSystem.linear()` dentro del θ-método. Validar con el caso 2D de convergencia en transitorio
   (solución exacta del decaimiento de la primera autofunción).
4. **Mallas y geometría (Etapa 8).** Importar `.msh` de gmsh (grupos físicos → aristas con nombre), añadir triángulos
   T3 a `LinearGeometry` y permitir geometrías no rectangulares. Validar con el parche en la malla importada.
5. **Escalado de solvers y salida.** Precondicionador AMG (`pyamg`) para problemas grandes, cache de factorizaciones
   en el estacionario, y exportar a VTK/XDMF para ParaView.
6. **Estabilización de convección.** Petrov-Galerkin o SUPG en 1D (y 2D cuando exista convección vectorial). Validar con
   el problema de convección-difusión de Péclet alto.

## 5. Criterio para cerrar una etapa

- Las cuatro suites de `Test/` pasan (y las que se añadan).
- Un caso de referencia reproduce el resultado anterior o explica la diferencia.
- La documentación de `README.md`, `docs/ARQUITECTURA.md` y esta página se actualiza en el mismo commit.
