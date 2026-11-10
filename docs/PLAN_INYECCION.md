# Plan: simulación de inyección tipo Cadmould

Documento de traspaso para ampliar el llenado Hele-Shaw de FEM-sim hacia un simulador de inyección completo: llenado, compactación,
enfriamiento del molde, orientación de fibras y alabeo sobre piezas reales importadas de CAD. Se leen primero este archivo,
`docs/ESTADO.md` (estado y validación de cada física), `docs/PLAN_3D.md` (fases 0–6 del 3D, en particular la 6) y `docs/PLAN_ETAPA17.md`.
Escrito el 2026-11-09, sobre `test-branch` en el commit `8457e37`; las 46 suites de `run_tests.py` pasaban.

## Veredicto: viable por fases, con un módulo grande nuevo (alabeo)

Cadmould resuelve el llenado sobre una malla de superficie de la pieza con el espesor asignado localmente: una formulación tipo
Hele-Shaw, no Navier–Stokes 3D. Es la misma familia que `Source/Physics/HeleShaw.py` sobre la superficie media (fase 6 del 3D), así que el
núcleo físico del llenado ya está. Lo que falta es: (1) llegar a la malla con espesor desde un sólido CAD, (2) el sistema de alimentación,
(3) la orientación de fibras, (4) el molde con sus canales y el ciclo, (5) la mecánica de sólidos para el alabeo, y (6) velocidad y
validación. No se ha comparado con Cadmould en detalle: lo dicho de él es lo públicamente conocido, y los resultados de FEM-sim deben
validarse con soluciones exactas y piezas medidas, no contra otro programa.

## Lo que ya existe

| Capacidad | Dónde | Estado |
|---|---|---|
| Llenado no isotermo, Cross–WLF, frente, isócronas | `HeleShaw.fill`, `setThermal` | validado (ver `ESTADO.md`) |
| T(z) por celdas en el espesor, capa congelada, calentamiento viscoso | `setThermal(cells=...)` | validado |
| Llenado compresible y compactación con Tait, contracción volumétrica | `setDensity`, `pack`, `shrinkage`, `switchover` | validado |
| Cristalización | `setCrystallization`, `Source/Physics/Crystallization.py` | validado (parámetros de otro iPP) |
| Piezas plegadas y uniones en T sobre la superficie media | `Mesh.from_gmsh_model(surface=True)`, fase 6 | validado con la división exacta x_i = h_i s(t) |
| Líneas de soldadura (enrollamiento), aire atrapado, fuerza de cierre, esfuerzo en la pared | `Ejemplo_Llenado_Pieza_3D.ipynb` | posproceso en el cuaderno, no en la librería |
| Conducción 3D con CG+AMG | `HeatTransfer` en 3D, fase 1 | validada, no acoplada al llenado |
| Ajuste de materiales (Cross–WLF, Tait) | `Source/Physics/Fitting.py` | POLYFORT FPP 30 GFC K1079 ajustado |

## Lo que falta (inventario)

| Bloque | Qué hace falta | Dificultad |
|---|---|---|
| Rendimiento | ~~el llenado avanza un nodo por paso: la caja de 1850 nodos tarda 29 min~~ hecho en la fase 0 (83 s con `batch=0.1`); falta el llenado compresible y el empaque por lotes, y bajar de unos minutos en 10⁴ nodos | hecho en parte |
| Geometría desde CAD | importar STEP/STL (gmsh OCC lo lee), malla de superficie y espesor local medido en el sólido | alta |
| Sistema de alimentación | elementos 1D (bebedero, canales, canal caliente, entradas) acoplados a la superficie | media |
| Orientación de fibras | Folgar–Tucker con RSC o MRD y un cierre (híbrido, IBOF), por capa del espesor | media–alta |
| Molde y ciclo | conducción 3D del molde con canales de refrigeración, ciclo repetido hasta régimen, tiempo de enfriamiento | media–alta |
| Alabeo | física estructural nueva (no hay ninguna en `Source/Physics`): termoelasticidad con deformación inicial de la contracción, anisótropa por orientación | alta |
| Materiales | base de datos (reología, PVT, cinética, térmicas, mecánicas, fibra) en un formato propio | baja–media |
| Validación | piezas medidas: presión en cavidad, tiros cortos, peso, alabeo | depende de datos del usuario |
| Interfaz y optimización | estudios de entrada y ventana de proceso, tipo Varimos | opcional, al final |

## Plan por fases (cada una con pruebas, documentación y petición de commit)

| Fase | Qué | Criterio para seguir |
|---|---|---|
| 0. Rendimiento del llenado | perfilar (`cProfile`) `fill` y `_pack_step`; llenar varios nodos por paso (todos los que se completan dentro de una tolerancia de tiempo), reutilizar la estructura de la matriz de presión, vectorizar el perfil T(z) por nodos | las suites de Hele-Shaw dan los mismos resultados dentro de su tolerancia; la caja de la fase 6 al menos 10 veces más rápida; medir hasta 2·10⁴ nodos |
| 1. Geometría desde CAD | **1a:** sólido STEP → malla de superficie y espesor por trazado de rayos opuesto a la normal (con filtro de ángulo y de esquinas); **1b:** flujo de doble superficie (*dual domain*): las dos pieles emparejadas, con conectores en los bordes para que el frente avance igual en ambas; alternativa más simple: superficie media dada por el usuario en CAD | una placa sólida da el mismo llenado que la superficie media de la fase 6; la caja como STEP frente a la caja de superficie media; espesor exacto en placas, cuñas y nervios |
| 2. Sistema de alimentación | elementos `line` con sección circular, trapezoidal o semicircular; canal frío (capa congelada como en la cavidad) y caliente (pared a T fija, sin capa congelada); acople al nodo de entrada; molde de varias cavidades | ley de potencia en tubo frente a la solución exacta; dos cavidades simétricas llenan a la vez; molde de familia: el desequilibrio cambia con el diámetro del canal como predice el balance de caudales |
| 3. Orientación de fibras | tensor de orientación a₂ por capa del espesor (las celdas de `setThermal`); Folgar–Tucker con RSC o MRD; cierre híbrido e IBOF; transporte con la velocidad media de la capa y gradiente en el espesor | estado estacionario de Folgar–Tucker en corte simple frente a la integración con `solve_ivp`; periodo de Jeffery sin interacción; estructura piel–núcleo en una placa; comparar con datos publicados (tomar los valores de las fuentes, no de memoria) |
| 4. Molde y ciclo | malla 3D del bloque del molde con canales (gmsh), conducción con `HeatTransfer`, convección en canales (Dittus–Boelter o Gnielinski en 1D), acople con la pieza por el HTC y el flujo de calor de las capas; ciclo repetido hasta régimen; criterio de expulsión | placa en molde frente a la conducción 1D exacta; balance de energía del ciclo; convergencia del ciclo; la temperatura de la pared cae hacia la constante actual cuando el molde conduce mucho |
| 5. Alabeo | termoelasticidad lineal nueva: sólido (tet10 sobre la pieza extruida desde la superficie) o lámina; deformación inicial por la contracción (isótropa primero, luego anisótropa por orientación con Mori–Tanaka y promedio de orientación); tensiones residuales por capa | tira bimetálica (Timoshenko) y placa con gradiente térmico frente a las soluciones exactas; placa con contracción en el espesor asimétrica; rigidez de un compuesto unidireccional frente a Mori–Tanaka |
| 6. Materiales y validación | formato de material en archivo (JSON) con todas las propiedades; validación con piezas del usuario: presión en cavidad, tiros cortos, peso, alabeo | errores medidos y documentados en `ESTADO.md`; sin datos medidos esta fase no se cierra |
| 7. Estudios de proceso (opcional) | barridos de posición de entrada, tiempo de llenado y presión de compactación; optimización simple (curva U ya hecha en el cuaderno) | después de 0–6 |

Orden recomendado: 0 → 1 → 2 → 3 → 4 → 5. La fase 0 va primero porque todas las demás multiplican el coste del llenado; con 0–2 ya hay un
simulador de llenado y compactación útil sobre piezas reales. La 3 es esencial para materiales con fibra (el POLYFORT tiene 30 % de vidrio)
y la 5 depende de 3 y 4.

Fuera de alcance salvo decisión: flujo 3D de volumen en piezas gruesas (Navier–Stokes de dos fases: en 2D el conjunto de nivel con un salto
de viscosidad de 10⁴ no fue fiable), inyección asistida por gas o agua, sobreinyección y bicomponente, espumado, paralelismo MPI.

## Cómo implementarlo con eficacia

- **Medir antes de optimizar** (fase 0) y conservar las suites actuales de Hele-Shaw como regresión: cualquier aceleración debe dar los
  mismos resultados.
- **Reutilizar lo hecho:** las celdas del espesor de `setThermal` sirven para la orientación y las tensiones residuales; `HeatTransfer` 3D
  para el molde; los campos vectoriales y la tangente automática de jax para la elasticidad.
- **Soluciones exactas primero**, piezas reales después; mallas pequeñas en la suite y estudios de malla aparte (`Test/manual/`).
- Los posprocesos del cuaderno (soldaduras por enrollamiento, aire atrapado, fuerza de cierre) pasan a la librería cuando se usen en más
  de un ejemplo, con su prueba.
- Documentar en `docs/ESTADO.md`, `docs/PLAN_ETAPA17.md`, este archivo (apartado Avance) y `README.md` en el mismo commit de cada fase.

## Convenciones de trabajo (las de este proyecto)

- Hablar en español. Implementar, validar y documentar; **pedir aprobación antes de cada commit; nunca push**. Informar con honestidad lo
  que no se logró y lo que no se verificó.
- Mensajes de commit en español sin acentos, terminando en `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Pruebas con `.venv/Scripts/python.exe run_tests.py`; `docs/` está en `.gitignore`: añadir con `git add -f docs/...`.
- Salidas generadas (`resultados_vtk`, `Logs`, `.vtu`) no se suben. Archivos descargados (CAD de terceros incluido): carpeta propia y
  `python -I`.
- gmsh 4.13.1 instalado (lee STEP con OCC).

## Avance

- **Fase 0 hecha (2026-11-09): rendimiento del llenado.** Perfil: un paso por nodo y coste N² (kernel de la fluidez 33 %, temperatura
  27 %, SuperLU 22 %). `fill(batch=f)` avanza hasta que se llena la fracción f del frente; el exceso de los nodos que se llenan antes
  pasa a sus vecinos con su calor (masa y energía exactas). Convección de la temperatura con BiCGSTAB sobre filas escaladas,
  ensamblaje de patrón fijo, kernel solo en los elementos con fundido. `batch=0` (por omisión) reproduce el código anterior a 2e-10 K.
  Caja de `Example_Filling_Box_3D.py` (1850 nodos): 1296 s → 318 s con lote 0 y 83 s con lote 0.1 (15.6 veces; presión +0.07 %,
  tiempos de llenado a 0.9 ms rms). Escalado de la pieza del cuaderno con lote 0.1: de 562 a 16450 nodos en 11 a 947 s (pasos ~√N, tiempo ~N^1.35, 0.61 GB), y a 3.5 mm
  7.3 veces más rápido que nodo a nodo. `Test/test_hele_shaw_batch.py`
  (5 pruebas), `Test/manual/hele_shaw_scaling.py`. **Sin hacer:** lotes en el llenado compresible y en el empaque (con lotes la presión
  divergía con un fundido casi incompresible: ahora los rechaza), y el kernel y Newton (unas 6 iteraciones por paso) siguen dominando.

## Primeros pasos de la nueva sesión

1. Leer este archivo, `docs/ESTADO.md` y la fase 6 de `docs/PLAN_3D.md`; correr la suite completa para la línea base.
2. Fase 1: geometría desde CAD (la fase 0 está hecha; usar `fill(batch=0.1)` en piezas grandes).
