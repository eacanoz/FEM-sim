# Etapa 17: prioridades de mejora (calidad y robustez)

## Contexto

Las Etapas 9–16a dejaron una base de fluidos validada (Darcy, Stokes, Navier–Stokes, no newtonianos con
acoplamiento térmico, axisimétrico, transitorio BDF2, convección natural, Spalart–Allmaras). Quedó abierto **k-ω SST**:
la física está escrita y su jacobiano es exacto (4e-9 frente a diferencias finitas), pero no converge en el canal con
Newton, con Newton limitado o recortado ni en pseudo-tiempo. El diagnóstico apunta a **condicionamiento y globalización
del solver no lineal**, no a las ecuaciones: cerca de la pared ω ~ 10⁶ y su residuo ~ 10¹², frente a u ~ 10.

La Etapa 17 reúne las limitaciones documentadas en `docs/ESTADO.md` (secciones «Fluidos (Etapa 9…16a)»). Este plan las
ordena por **impacto en la calidad del proyecto** y por **aporte al reto de SST**. Criterio: primero lo que desbloquea
más casos con menos riesgo; después la calidad numérica de fondo; al final lo deseable.

Paso previo (sin código nuevo): dejar SST en el repositorio como **experimental, sin validar**, junto con la
generalización de leyes de viscosidad a varios campos (`ff.setViscosityLaw(..., fields=...)`, ya probada: 21/21
suites) y el recorte de campos positivos (`System.require_positive` / `clip_positive`).

## Prioridad 1: robustez del solver no lineal (desbloquea SST)

Impacto: **muy alto**. Es la causa directa del fallo de SST, y además hace más fiables Navier–Stokes a Re alto,
Bingham con `m` grande, Cross–WLF a baja temperatura y SA en flujos con separación, que hoy dependen de continuación
manual y de buenos puntos de partida.

1. **Escalado por campo de incógnitas y ecuaciones.** Normalizar cada campo (u, p, T, ν̃, k, ω) con una escala
   característica antes de Newton y de la búsqueda de línea. Hoy el escalado es fila a fila con la diagonal del
   jacobiano inicial (`modelSolver.newtonSolver`, `_nonlinear_transient` en `Source/Simulation/Solvers.py`). Con ω ~ 10⁶
   el residuo de ω domina la norma y la búsqueda de línea rechaza pasos buenos para el resto.
2. **Pseudo-transitorio con CFL creciente (SER: switched evolution relaxation).** Paso de pseudo-tiempo local por
   elemento, `Δt_e = CFL · h_e / (|u| + ν/h_e)`, que crece con la caída del residuo (`CFL_{n+1} = CFL_n · ‖R_{n-1}‖/‖R_n‖`).
   Reutiliza el contexto `System.time` y la masa ya existentes. Es el método estándar de los códigos RANS y sustituye a
   la «continuación manual» actual.
3. **Subrelajación por campo** como alternativa, sobre todo para la turbulencia: `x ← x + λ_campo Δx`.
4. **Resolución segregada opcional (flujo ↔ turbulencia).** Resolver primero el flujo con μ_t congelado y luego la
   turbulencia con u congelada, alternando. Es más lento por iteración pero muy robusto, y sirve de arranque para el
   Newton monolítico.

Archivos: `Source/Simulation/Solvers.py` (estrategias), `Source/Simulation/System.py` (escalas por campo),
`Source/core.py` (`solverConfiguration`: opciones `globalization='newton'|'pseudo-transient'`, `cfl0`, `relaxation`).

**Estado: puntos 1 a 3 hechos** (`Test/test_nonlinear_globalization.py`). Decisiones frente a lo previsto:

- El paso local no usa `h_e`, `|u|` ni `System.time`: es `D = |diag J| / CFL`, que mide convección, difusión y reacción de
  cada grado de libertad sin conocer la física, y solo entra en la matriz, de modo que el residuo sigue siendo el
  estacionario. Las filas de restricción (`System.require_constraint`, la presión) y las de Dirichlet no lo llevan.
- Los valores de Dirichlet se imponen antes del primer paso: si no, una pared caliente en un fundido frío hace rechazar
  todos los pasos con cualquier CFL.
- El punto 4 (segregado) queda aplazado: solo se hará si la Prioridad 2 lo necesita.

Resultado: SA desde un estado uniforme pasa de estancarse a 12 iteraciones (por el escalado) y la cavidad a Re = 5000 y
10000 desde el reposo converge en 24 y 48 iteraciones. SST no convergía solo con esto: hizo falta la Prioridad 2.

## Prioridad 2: SST robusto sobre lo anterior

Impacto: **alto para turbulencia**. Cierra el reto encontrado.

1. **Variable ln ω** en lugar de ω (Ilinca y Pelletier; Bassi et al.). Garantiza ω > 0 sin recortes, reduce el rango de
   10⁶ a ~14 y suaviza el jacobiano cerca de la pared. Cambio local en `sst_law` y en la condición de pared
   (`Source/Physics/Turbulence.py`), ya que `GradientLawTerm` da el jacobiano automáticamente.
2. **k ≥ 0 con recorte** (`System.clip_positive`, ya implementado).
3. **Validación**: canal Re_τ = 180, 395 y 590 frente a la DNS (Moser–Kim–Mansour) y a la ley de pared, con los
   mismos criterios que SA (`Test/test_turbulence.py`); después, una placa plana o un escalón (C_f, readherencia).

**Estado: canal validado; placa plana pendiente** (`Test/test_sst.py`). Resultados y decisiones:

- Con ln ω el canal converge en 9–10 iteraciones con pseudo-transitorio. Newton puro sigue divergiendo.
- **El criterio U_c⁺ ≤ 3 % no se cumple**: −3.6 %, −2.5 % y −4.3 % a Re_τ = 180, 395 y 590. Un resolvedor 1D independiente,
  con ω como incógnita, da lo mismo a < 1 %, así que es el modelo. La prueba exige < 1 % frente al 1D y < 5 % frente a la DNS.
- k se recorta con piso 0, no 1e-10, y nunca en filas de Dirichlet: el piso anterior dejaba el canal en la rama laminar.
- La placa plana obligó a cambiar el CFL: reducirlo con cada subida pequeña del residuo congelaba el avance mientras la
  turbulencia se desarrolla. Aun así no cierra el residuo (ver `docs/ESTADO.md`). La causa probable es el recorte de k en
  el borde de ataque, pero no está demostrada.

## Prioridad 3: rendimiento del ensamblado

Impacto: **alto y transversal**. Hoy cada iteración reensambla en Python; el cilindro tarda 15 min. Además acelera
los experimentos de las Prioridades 1–2.

1. **Patrón disperso precalculado.** `DofLayout.scatter_matrix` construye un COO→CSR completo en cada bloque e
   iteración. Calcular una vez los índices CSR de cada bloque (`FlowTerm`, `GradientLawTerm`, `FieldConvectionTerm`)
   y en cada iteración solo rellenar `data`. Ganancia esperada ×2–3 en el ensamblado.
2. **Newton modificado / reutilización de la factorización** durante varias iteraciones (`FactorizedSolver` en
   `Source/Simulation/DirectSolver.py` ya cachea por matriz idéntica): refactorizar solo si la convergencia se degrada.
3. **Operador de convección dedicado** en lugar de `GradientLawTerm` para escalares convectados con gradiente
   (su arreglo `W` es genérico y pesado).

**Estado: hecho el punto 2, con otro enfoque; puntos 1 y 3 descartados por lo medido** (`Test/test_modified_newton.py`).

- El perfil contradijo la estimación del punto 1. Convertir COO a CSR pesa ~4 % y todo el ensamblado disperso ~20 %, así que
  el patrón precalculado daría a lo sumo ×1.25, no ×2–3. `einsum` pesa 50–60 %, repartido sin un punto dominante, y
  `optimize=True` solo da 0–12 %.
- Lo que rinde es evitar ensamblar y factorizar: `System.residual` cuesta unas 5 veces menos que residuo más tangente, y el
  Newton modificado (`reuseTangent`, activo por defecto) conserva la factorización entre iteraciones y pasos mientras cada
  iteración reduzca el residuo al menos a la mitad.
- Resultado: ×1.8 a ×3.5 en cuatro casos transitorios, con soluciones iguales a 4e-8 o mejor; cilindro de 15 a 6.7 min con
  los mismos C_D, C_L y St. **Los estudios estacionarios no ganan**: conservan el Newton completo por su convergencia
  cuadrática. El criterio previsto de 1e-10 no aplica en transitorio, porque Newton para en 1e-8 relativo.
- Corregido de paso: un residuo NaN pasaba el test de parada del transitorio y la simulación seguía en silencio.

## Prioridad 4: calidad numérica de fondo (Taylor–Hood)

Impacto: **alto, esfuerzo alto**. Resuelve de raíz varias limitaciones de la presión:

- presión de orden 2 (hoy O(h) en norma máxima y O(h^1.5) en L2);
- presión en el eje axisimétrico;
- robustez en presión (corrientes espurias en T3 con flotación);
- sin depender del esfuerzo recuperado ni de la PSPG.

Requiere elementos `tri6` / `quad9` en `Source/Pre_processing/Elements.py`, mallas con nodos intermedios (gmsh de
orden 2) y un `DofLayout` con nodos distintos por campo (hoy todos los campos viven en los mismos nodos). Es la mejora
de mayor calidad para el foco de polímeros (caída de presión en dados y reómetros), pero no ayuda a SST.

**Estado: hecho** (`Test/test_taylor_hood.py`). Decisiones frente a lo previsto:

- **Sin `DofLayout` por campo.** La presión sigue en todos los nodos, con los no vértices atados a los vértices y las filas
  condensadas con `T^T`: es Taylor–Hood genuino sin tocar cada física y cada salida.
- **Corrientes espurias: la expectativa estaba mal.** Una presión lineal no equilibra una flotación lineal. El reposo a
  redondeo solo aparece en mallas uniformes, por cancelación entre elementos; en malla graduada quedan corrientes de orden 3.
- LSIC se conserva por robustez a Re alto, aunque en P2/P1 empeora algo la presión en mallas gruesas.

**Comparación en dados: Taylor–Hood gana en contracción y reómetro, no en todo.** Contracción 4:1 y reómetro capilar: 10 a 15
veces menos incógnitas y unas 20 a 35 veces menos tiempo para la misma caída de presión. Rendija con Cross–WLF: mejor en el
gradiente desarrollado y en la temperatura de mezcla, pero peor en la caída total, porque el perfil impuesto en la entrada
es singular para un fluido que adelgaza.

## Prioridad 5: mejoras puntuales de impacto medio

| Mejora | Por qué |
|---|---|
| Distancia a la pared eikonal (o p-Poisson) | Spalding sobreestima ≈ d/(2R) en paredes convexas (SA y SST) |
| Crank–Nicolson de segundo orden limpio | Aislar el término O(dt) residual; hoy se recomienda BDF2 |
| Paso de tiempo adaptativo (estimador BDF2/BDF1) | Arranques y transitorios con escalas muy distintas |
| Términos de aro en `GradientLawTerm` vectorial | SA y SST axisimétricos (boquillas, tuberías) |
| Condiciones de contorno dependientes del tiempo | Arranques de bomba, pulsos de entrada |
| Solvers iterativos con precondicionador de bloques (punto de silla) | Mallas de 10⁵–10⁶ incógnitas (hoy PARDISO directo) |

**Estado: hecho** (`Test/test_time_and_scale.py`). Decisiones frente a lo previsto:

- Distancia a la pared: la geométrica mínima en lugar de eikonal o p-Poisson. Es exacta en paredes rectas y más simple.
- Términos de aro: no hicieron falta en `GradientLawTerm` para SA ni para los escalares, porque la medida ya lleva r. Solo SST
  necesita la deformación de aro en S² (`gauss_inputs`).
- Crank–Nicolson: el término O(dt) no era de la PSPG, sino del instante al que pertenece la presión implícita.
- Paso adaptativo: añadido `adaptiveMaxStep`, porque el control se saltaba un pulso corto.
- Corregido de paso: el criterio de parada de Newton en transitorio, que por redondeo acortaba el paso sin fin.

## Prioridad 6: deseable / nuevas capacidades

- **Viscoelástico multimodo (Oldroyd-B, Giesekus): hecho, salvo lo que sigue** (`Source/Physics/Viscoelastic.py`,
  `Test/test_viscoelastic.py`, `Example_Viscoelastic_Contraction.py`). Raíz de la conformación, DEVSS-G y SUPG sobre Taylor–Hood.
  - Cilindro en canal a 0.05 % o mejor de la literatura hasta Wi = 0.9: se supera el punto de parada de B3.
  - Cambios frente al plan: Poiseuille no es exacto, solo Couette (`b = √c` no es polinómico), así que el criterio es el orden; una salida
    con presión necesita descontar la tracción polimérica (`addBC_OutletEdges`).
  - **Bagley a tasas de proceso:** con difusión artificial alrededor de la esquina el reómetro multimodo llega a 618 s⁻¹. Frente al fluido inelástico
    de igual viscosidad, la elasticidad multiplica la pérdida de entrada por 1.48 a esa tasa, y la reduce por debajo de ~100 s⁻¹. `Example_Viscoelastic_Rheometer.py`.
    Espectro ilustrativo: sin comparar con experimentos.
  - **PTT y deslizamiento (añadidos):** `setModes(..., 'ptt-linear' | 'ptt-exp')`, `addBC_SlipLawEdges`. PTT llega a 1583 s⁻¹ en el reómetro, y da otra razón de Bagley: 0.96 a 618 s⁻¹,
    frente a 1.48 con Giesekus. El resultado de Bagley depende del modelo, así que hace falta ajustar datos extensionales.
  - **No cumplido:** sin solución manufacturada; contracción sin tablas de Alves et al. 2003 y sin converger en malla a De alto; la detección de soluciones
    falsas a Wi alto no quedó resuelta: `energy_balance` indica error, pero no las separa de una solución buena en malla gruesa.
  - Corregidos de paso: `System.field_scales` tomaba el ruido de una presión nula como su tamaño y Newton no paraba (viene de la
    Prioridad 1); y el criterio de parada estacionario, ahora por actualización despreciable.
- Hechos: materiales por región (`Model.region_field`) y Cross–WLF con presión (`D3`), en `Test/test_regions_pressure.py`.
- VTK binario / XDMF: solo si el tamaño de los archivos lo exige.
- Hecha: limpieza de `docs/ESTADO.md`.

## Pausa del viscoelástico: huecos menores y SST

Objetivo: que el usuario pueda probar por su cuenta con ejemplos de otros programas (`example_mixing_elbow.ipynb`). Resultados medidos en `docs/ESTADO.md`.

Hecho: SUPG consistente en cuadráticos; Taylor–Hood con SUPG opcional; SA y SST en canal Q9/T6; especies en Q9/T6; GMRES de bloques con más campos; `Mesh.triangulation`; paso
adaptativo; placa SST en Q4 con residuo cerrado; escalón con SST y SA en Q4; codo mezclador con los balances de masa y energía.

Sin resolver, para decidir si se aborda:

1. Balance de energía en el codo: T media de salida 0.05–0.10 °C por debajo del balance, por ∫T div(u) con el término convectivo no conservativo. La sobreoscilación de T ya está explicada: es resolución del frente, no el SUPG (`docs/ESTADO.md`).
2. SST en mallas cuadráticas fuera del canal (placa y escalón no cierran). Por ahora, Q4.
3. Arranque del pseudo-transitorio desde u = 0 con entrada impuesta: hoy hay que dar un campo sin divergencia.
4. Difusividad turbulenta en calor y especies como opción de la física, con validación propia.

## Orden recomendado

1. Commit de SST experimental + generalización de viscosidad.
2. Prioridad 1 (escalado por campo + pseudo-transitorio con CFL; segregado como respaldo).
3. Prioridad 2 (SST con ln ω) → SST validado.
4. Prioridad 3 (rendimiento). Puede adelantarse a la 2 si las pruebas de SST resultan lentas.
5. Prioridad 4 (Taylor–Hood).
6. Prioridades 5 y 6 según necesidad.

## Verificación de cada bloque

- `python run_tests.py` en verde (hoy 21 suites) y la documentación actualizada en el mismo commit.
- Prioridad 1: casos hoy frágiles que converjan sin continuación manual: cavidad Re = 1000 desde el reposo, Bingham con
  `m` alto, Cross–WLF en frío y SA arrancando desde un estado uniforme.
- Prioridad 2: canal SST a Re_τ = 180/395/590 con U_c⁺ a ≤ 3 % de la DNS y ley de pared ≤ 7 %, igual que SA.
- Prioridad 3: mismos resultados a 1e-10 y tiempos medidos (cilindro de Schäfer–Turek, reómetro).
- Prioridad 4: Poiseuille, Kovasznay y cavidad con presión de orden 2; corrientes espurias de la estratificación en
  T3 a nivel de redondeo; presión en el eje axisimétrico de orden 2.
