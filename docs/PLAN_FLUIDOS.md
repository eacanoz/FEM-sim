# Plan de trabajo: transporte de fluidos

Acordado el 2026-10-28. Rama de trabajo: `test-branch`. Foco de aplicación: **procesamiento de polímeros**
(flujos viscosos de Re bajo, viscosidad dependiente de la tasa de corte y de la temperatura, disipación viscosa,
geometrías axisimétricas como tubos y boquillas).

## Decisiones

| Tema | Decisión | Motivo |
|---|---|---|
| Velocidad–presión | Igual orden (Q1/P1) estabilizado con PSPG; Taylor–Hood en la etapa de escalado | Encaja con el `DofLayout` intercalado (todos los campos en los mismos nodos) y reutiliza `supg_tau` |
| Turbulencia | Spalart–Allmaras y luego k-ω SST, con pared resuelta | Sin funciones de pared; SA es una sola ecuación y prepara la infraestructura (distancia a la pared, positividad) |
| No newtonianos | Prioridad alta: van justo después de Navier–Stokes estacionario, con acoplamiento térmico | Foco en polímeros: Cross–WLF, Carreau, ley de potencia, disipación viscosa |
| Axisimétrico (r-z) | Prioridad alta, después de no newtonianos | Tubos, boquillas, contracciones |

## Diagnóstico de partida

Ya disponible: campos con varias componentes en `DofLayout`, bloques entre campos, Newton, θ-método, Dirichlet por
DOF (`System.set_dirichlet`), facetas con normales salientes, leyes puntuales con jacobiano automático, SUPG escalar.

Faltaba:
1. Leyes que dependan de **gradientes** de los campos (`(u·∇)u`, `μ(γ̇)`, turbulencia): `VolumeLawTerm` y
   `DiffusionLawTerm` solo reciben valores en los puntos de Gauss.
2. Operadores vectoriales (viscoso, divergencia–presión) y salida VTK de vectores.
3. Transitorio con iteraciones de Newton dentro del paso y BDF2 (hoy es linealmente implícito).
4. Elementos cuadráticos y `DofLayout` con nodos distintos por campo (solo para Taylor–Hood).

## Etapas

| Etapa | Contenido | Validación |
|---|---|---|
| **9. Infraestructura + Darcy** | Operadores vectoriales (viscoso `ε(u):ε(v)`, divergencia–presión, masa y fuente vectoriales); `GradientLawTerm` (ley `f(u, ∇u)` con jacobiano automático respecto a valores y gradientes); Dirichlet por componente; presión de referencia; VTK vectorial. Darcy como primer flujo y velocidad de Darcy hacia la convección de `ht`/`mt` | MMS de elasticidad/Laplaciano vectorial, jacobiano frente a diferencias finitas, Darcy exacto, acoplamiento Darcy → especie |
| **10. Stokes estacionario** | Física de flujo (`u`, `p`), Q1/P1 con PSPG; BC: no deslizamiento, perfil de entrada, salida libre (*do-nothing*), simetría, deslizamiento | Poiseuille y Couette exactos, MMS (orden 2 en `u`), cavidad con Re→0 |
| **11. Navier–Stokes estacionario** | Convección con Newton completo (Picard opcional), SUPG/PSPG/LSIC, continuación en Re | Kovasznay, cavidad Re 100–1000 frente a Ghia et al., escalón hacia atrás |
| **12. No newtonianos + acoplamiento térmico** | Velocidad calculada → convección de calor (bloques cruzados); `μ(γ̇, T)`: ley de potencia, Carreau–Yasuda, Cross–WLF, Bingham/Herschel–Bulkley regularizados (Papanastasiou); disipación viscosa `μγ̇²`; `μ(T)` Arrhenius/WLF | Ley de potencia y Bingham en canal (analíticos), MMS con Carreau, balance de energía con disipación (Brinkman) |
| **13. Axisimétrico r-z** | Operadores con peso `r`, término `u_r/r`, eje de simetría | Hagen–Poiseuille, ley de potencia en tubo, contracción 4:1 |
| **14. Transitorio** | Newton dentro del paso, BDF2, presión como restricción algebraica | Taylor–Green, cilindro Re=100 (Schäfer–Turek) |
| **15. Boussinesq y especies** | Flotación, convección natural, especies arrastradas por la velocidad calculada | Cavidad de de Vahl Davis (Ra 1e3–1e6, Nusselt) |
| **16. Turbulencia RANS** | Distancia a la pared, longitud de mezcla (van Driest), Spalart–Allmaras, k-ω SST con limitadores | Canal Re_τ=180/395 frente a DNS, placa plana (Cf), escalón de Driver–Seegmiller |
| **17. Escalado y extensiones** | Taylor–Hood (T6/Q9, `DofLayout` por campo), precondicionadores de bloques para punto de silla, 3D; viscoelástico (Oldroyd-B, Giesekus) opcional | Mismos casos frente a igual orden; iteraciones según el tamaño |

Fuera de alcance: compresible, superficie libre, multifásico, LES.

## Criterio de cierre de cada etapa

- `python run_tests.py` en verde.
- Al menos un caso de referencia (analítico, manufacturado o de la literatura) con su error u orden medido.
- `README.md`, `docs/ARQUITECTURA.md`, `docs/ESTADO.md` y esta página actualizados en el mismo commit.

## Estado

| Etapa | Estado |
|---|---|
| 9 | hecha: `Test/test_fluids_foundation.py` (11 pruebas), `Example_Darcy_Transport.py` |
| 10 | hecha: `Test/test_stokes.py` (8 pruebas), `Example_Stokes_Contraction.py`. PSPG con esfuerzo viscoso recuperado por parches |
| 11 | hecha: `Test/test_navier_stokes.py` (7 pruebas), `Example_NavierStokes_Cavity.py`. SUPG/PSPG/LSIC con Newton (τ congelado) o Picard |
| 12 | hecha: `Test/test_non_newtonian.py` (8 pruebas), `Example_Polymer_Slit_Die.py`. `Rheology.py`, `FlowTerm`, `FieldConvectionTerm` |
| 13 | siguiente |
| 14–17 | pendiente |
