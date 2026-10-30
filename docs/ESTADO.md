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
| Stokes: parche | `u = (y, x)`, `p = 2x + 3y` con fuerza de cuerpo, mallas distorsionadas Q4 y T3 | `Test/test_stokes.py` | exacto (1e-10) |
| Stokes: Poiseuille | entrada parabólica, salida desarrollada; caudal entrante = saliente | `Test/test_stokes.py` | Q4 nodalmente exacto (1e-11); T3 orden 2 en u y en la caída de presión; caudal 1e-12 |
| Stokes: PSPG sin esfuerzo recuperado | misma prueba | `Test/test_stokes.py` | caída de presión orden 1.5 (error 1.1 % con 16 elementos en la altura) |
| Stokes: solución exacta | `ψ = x eˣ sin y`, tracción exacta en un lado | `Test/test_stokes.py` | u orden 2; p orden 1 (máximo) y 1.5 (cuadrático), Q4 y T3 |
| Stokes: cavidad | `ψ_min = -0.1001` en `y ≈ 0.764` (Shankar y Deshpande 2000), simetría especular, caudal neto nulo en `x = 0.5` | `Test/test_stokes.py` | -0.10011 en y = 0.766 (64×64); simetría 1e-12 |
| Stokes: plano de simetría | media canal con simetría = canal completo en los nodos comunes | `Test/test_stokes.py` | 1e-12 |
| Stokes: arranque entre placas | serie de Fourier con Crank-Nicolson; límite estacionario con Euler | `Test/test_stokes.py` | u 1.2e-3 relativo; estacionario exacto, p → 0 |
| Navier–Stokes: tangente | convección + SUPG + PSPG + LSIC + esfuerzo recuperado frente a diferencias finitas (τ congelado), Q4 y T3 | `Test/test_navier_stokes.py` | 1e-10 relativo |
| Navier–Stokes: Kovasznay (Re = 40) | solución exacta, tracción exacta en la salida, mallas 8 a 32 | `Test/test_navier_stokes.py` | u orden 1.8–2.1, p orden 1.9–2.0; Newton 5–7 iteraciones |
| Navier–Stokes: cavidad Re = 100 | u en x = 0.5 frente a Ghia et al. (1982) | `Test/test_navier_stokes.py`, `Example_NavierStokes_Cavity.py` | 0.006 (32×32 y 64×64); u_min = -0.2143 |
| Navier–Stokes: cavidad Re = 400 y 1000 (ejemplo) | u_min en x = 0.5 frente a Ghia (-0.3273 en y = 0.281; -0.3829 en y = 0.172) | `Example_NavierStokes_Cavity.py` | -0.3305 en y = 0.281; -0.3925 en y = 0.172 (64×64); vórtice principal en (0.53, 0.56) |
| Navier–Stokes: canal desarrollado | Poiseuille con inercia (Re = 100) nodalmente exacto en Q4 | `Test/test_navier_stokes.py` | 1e-10 |
| Navier–Stokes: límites | ρ → 0 igual a Stokes; Picard converge a la misma solución (17 frente a 6 iteraciones) | `Test/test_navier_stokes.py` | pasa |
| Leyes de viscosidad | límites newtoniano y de potencia de Carreau y Cross, Cross–WLF en `T = D2`, esfuerzo de Bingham y Herschel–Bulkley más allá de la fluencia, Arrhenius | `Test/test_non_newtonian.py` | pasa |
| Ley de potencia en canal (n = 0.5) | perfil analítico, presión lineal | `Test/test_non_newtonian.py` | orden 2.00 (error 1e-3 con 32 elementos) |
| Bingham (Papanastasiou, m = 2000) en canal | perfil con tapón analítico | `Test/test_non_newtonian.py` | error 2e-4; tasa de corte en el tapón < 0.1 % de la de pared |
| Calentamiento viscoso (Brinkman) | Couette con paredes isotermas: T parabólica exacta | `Test/test_non_newtonian.py` | exacto en los nodos (1e-9), Q4 y T3 |
| μ(T) en Couette | exponencial: nodalmente exacto (la media de Gauss es proporcional a la exacta); Arrhenius frente a cuadratura | `Test/test_non_newtonian.py` | 1e-12; orden 2.00 |
| Convección por la velocidad calculada | flujo tapón de `ff` = velocidad constante en `ht`, con y sin SUPG | `Test/test_non_newtonian.py` | 1e-9 |
| Jacobiano acoplado u–p–T | Carreau + Arrhenius, inercia, PSPG con esfuerzo recuperado, calentamiento, convección del calor con SUPG, frente a diferencias finitas (τ congelado) | `Test/test_non_newtonian.py` | 1e-7 relativo |
| Dado de rendija (ejemplo) | caudal conservado; aumento de la temperatura de mezcla (0.29 K) menor que el adiabático Δp/ρCp (0.48 K) | `Example_Polymer_Slit_Die.py` | coherente |
| Axisimétrico: medidas y operadores | volumen `∫r dr dz`, superficies lateral y de tapa, `div(r, -2z) = 0` con el término de aro, energía viscosa | `Test/test_axisymmetric.py` | exacto (1e-12) |
| Axisimétrico: conducción radial | cilindro hueco (perfil logarítmico); cilindro macizo con fuente (eje natural) | `Test/test_axisymmetric.py` | orden 1.98; error < 2 % con 8 elementos |
| Hagen–Poiseuille | caudal conservado; perfil y gradiente de presión con el caudal discreto | `Test/test_axisymmetric.py` | u orden 1.8; p orden 1.95 (Q4) y 1.5 (T3) fuera del eje, orden 1 en el eje (0.4 %) |
| Extensión uniaxial | `u = (-a r/2, a z)`, presión constante, malla distorsionada | `Test/test_axisymmetric.py` | exacto (1e-10), también la presión |
| Ley de potencia en tubo (n = 0.5) | perfil analítico y caudal | `Test/test_axisymmetric.py` | orden 2.03 |
| Jacobiano axisimétrico acoplado | Carreau + Arrhenius, inercia, PSPG con términos de aro, calentamiento, convección del calor | `Test/test_axisymmetric.py` | 1e-7 relativo |
| Reómetro capilar (ejemplo) | viscosidad "medida" (Rabinowitsch) frente a la ley Cross del modelo | `Example_Capillary_Rheometer.py` | 0.3–1.1 % (2.2 % en el extremo, n' unilateral) |
| Taylor–Green (Re = 10 en [0, π]², paredes deslizantes) | solución exacta; error temporal frente a una referencia de paso fino en la misma malla | `Test/test_transient_flow.py` | u 3e-4, p 1e-3 (24×24); Euler orden 1.0, BDF2 orden 2.0; Crank–Nicolson 3 a 12 veces más preciso que BDF2 pero de orden 1 asintótico |
| Arranque entre placas con inercia | serie de Fourier; presión monótona (sin oscilación), igual en CN y BDF2 | `Test/test_transient_flow.py` | u 1e-3 relativo |
| Límite estacionario | cavidad Re = 10 con Carreau: final del transitorio BDF2 = solución estacionaria de Newton | `Test/test_transient_flow.py` | 1e-10 |
| Tangente transitoria | flujo + calor con el contexto de paso frente a diferencias finitas | `Test/test_transient_flow.py` | 1e-7 relativo |
| Calor transitorio convectado | flujo tapón de `ff` = velocidad prescrita en `ht`, con SUPG | `Test/test_transient_flow.py` | 1e-8 |
| Fuerzas por reacciones | canal: paredes = Δp H; Couette: −μU L/H; tubo: Δp πR² | `Test/test_transient_flow.py` | 1e-10 |
| Cilindro de Schäfer–Turek 2D-2 (Re = 100, ejemplo) | C_D,max 3.22–3.24; C_L,max 0.99–1.01; St 0.295–0.305 | `Example_Cylinder_Vortex_Shedding.py` | 2926 nodos, BDF2 dt = 0.01: C_D,max 3.231 (dentro); C_L,max 0.966 (−2.4 %); St 0.2905 (−1.5 %); 15 min |
| Convección natural (de Vahl Davis 1983) | Nusselt de la pared caliente por reacciones, u_max y v_max en los planos medios; Ra = 1e3–1e5 (32×32) y 1e3–1e6 (48×48, ejemplo) | `Test/test_buoyancy.py`, `Example_Natural_Convection.py` | 32×32: Nu < 0.4 %, velocidades < 1.7 %; 48×48: Nu < 0.2 %, u_max < 0.9 %, v_max < 1.9 %; calor entrante = saliente (1e-10) |
| Estratificación estable | calentada por arriba: reposo y presión cuadrática | `Test/test_buoyancy.py` | Q4 exacto (1e-10); T3 corrientes espurias de orden 3 (no robusto en presión) |
| Especies en el flujo calculado | Le = 1: especie = temperatura; flotación solutal = térmica | `Test/test_buoyancy.py` | 1e-8 |
| Tangente con flotación | térmica y solutal, Carreau, inercia, calor y especie con SUPG, estacionario y paso de Crank–Nicolson | `Test/test_buoyancy.py` | 1e-7 relativo |
| Flujo de calor por reacciones | pared plana: k ΔT / L | `Test/test_buoyancy.py` | 1e-12 |
| Distancia a la pared | placas paralelas; cilindro (convexo) | `Test/test_turbulence.py` | Q4 exacta (1e-12), T3 O(h) solo donde ∇φ = 0; cilindro: sobreestima ≈ d/(2R), independiente de la malla |
| Canal turbulento (Spalart–Allmaras) | Re_τ = 180, 395 (prueba) y 590 (ejemplo): u⁺ = y⁺, ley de pared de Reichardt, U_c⁺ de la DNS de Moser, Kim y Mansour (1999), C_f de Dean | `Test/test_turbulence.py`, `Example_Turbulent_Channel.py` | subcapa < 2 %; ley de pared < 7 %; U_c⁺ +1.0 %, −0.4 %, −2.3 %; C_f −4 a −6 %; independiente de la malla; Newton 12–17 |
| Límite laminar | sin turbulencia de entrada SA no produce ν̃: Poiseuille | `Test/test_turbulence.py` | ν̃ = 0 (1e-12) |
| Tangente con SA | u, p, ν̃ de ambos signos y la viscosidad turbulenta, frente a diferencias finitas | `Test/test_turbulence.py` | 1e-6 relativo |
| Solver no lineal: misma solución por cualquier camino | cavidad Re = 100: Newton, pseudo-transitorio y subrelajación por campo | `Test/test_nonlinear_globalization.py` | u 1e-7, p 1e-6; 5, 7 y 20 iteraciones |
| Pseudo-transitorio en casos frágiles (sin continuación) | cavidad Re = 5000 desde el reposo (Ghia); Bingham `m` = 2000 sin arranque newtoniano; Cross–WLF desde la temperatura fría | `Test/test_nonlinear_globalization.py` | 24 iter., error 0.037; 128 iter. (Newton 172), error 2e-4; 8 iter., T 1e-6 |
| Escalado por campo | SA en canal Re_τ = 395 desde un estado uniforme: antes Newton se estancaba (200 iter., U_c⁺ = 8) | `Test/test_nonlinear_globalization.py` | 12 iter., U_c⁺ = 20.05 (DNS 20.1) |
| Canal k-ω SST frente a un resolvedor 1D independiente | Re_τ = 180 y 395 (prueba) y 590 (a mano): diferencias finitas con ω como incógnita, sin código del proyecto | `Test/test_sst.py` | U_c⁺, U_b⁺ y k máx. a < 1 %; 9–10 iteraciones |
| Canal k-ω SST frente a la DNS | U_c⁺ de Moser, Kim y Mansour (1999), subcapa y ley de pared de Reichardt | `Test/test_sst.py` | U_c⁺ −3.6 %, −2.5 % y −4.3 % a Re_τ = 180, 395 y 590, igual en el resolvedor 1D (el déficit es del modelo); subcapa < 2 %; ley de pared < 7 % |
| k-ω SST: independencia del arranque y la malla; tangente; transitorio | estado uniforme, malla doble, diferencias finitas en Q4 y T3, BDF2 desde el estacionario | `Test/test_sst.py` | u 1e-5 (126 iter.); U_c⁺ < 0.5 %; tangente 1e-6; u 1e-5 |
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
  unidades de su incógnita) y con el tamaño de su campo (`System.field_scales`, Etapa 17). La tolerancia relativa pasó de
  1e-8 a 1e-10 para conservar la exactitud de los casos 1D.
- `heat.Convection = True` etc. deben asignarse en la parte (`model.physics.heat`), no en la combinación.

### Fluidos (Etapa 9)
- Solo Darcy está operativo como física de flujo; la velocidad pasa a otra física en los puntos de Gauss (acoplamiento
  en un sentido, misma malla). El acoplamiento monolítico con un campo de velocidad calculado llega con Navier–Stokes.
- `GradientLawTerm` evalúa un arreglo `W` de tamaño `nel·ngp·nne·nc·nc(1+dim)` por campo; es genérico pero más lento que
  un operador dedicado. Para Navier–Stokes en mallas grandes conviene un operador de convección vectorizado propio.
- `derived_fields` solo se añade en estudios estacionarios.
- La ley de potencia con `n < 1` necesita regularización (`(γ̇² + ε²)^((n-1)/2)`) y un punto inicial no nulo.

### Fluidos (Etapa 10, Stokes)
- **Presión de igual orden:** la teoría de P1/Q1 estabilizados da presión O(h) en norma máxima y O(h^1.5) en L2, y es
  lo que se mide. El esfuerzo recuperado vuelve exacta la caída de presión en canales (Q4) pero no mejora el orden
  general. Para presión de orden 2 hacen falta Taylor–Hood (Etapa 17).
- **Contorno totalmente Dirichlet:** el caudal discreto de la velocidad interpolada `∮u_h·n` no es cero exactamente;
  la incompatibilidad cae en el nodo donde se fija la presión (pico local que no converge) y contamina la velocidad.
  Con al menos una salida o tracción no ocurre. Si todo el contorno tiene velocidad, use datos con caudal nulo exacto
  (paredes, tapa deslizante).
- **Esfuerzo recuperado cerca de paredes:** con fuerzas viscosas normales grandes en la pared equilibradas por una
  fuerza de cuerpo (casos manufacturados artificiales) la presión en la pared empeora respecto a PSPG simple.
  Se puede desactivar con `physics.viscous_projection = False`.
- **Transitorio:** la presión es algebraica; con Crank-Nicolson oscila de paso a paso (la velocidad sigue de orden 2).
  Con Euler implícito se amortigua.
- Simetría y deslizamiento solo en aristas alineadas con los ejes (se fija una componente).
- `rho` debe ser un número (masa y `tau`); `mu` puede ser número o función de las coordenadas.

### Fluidos (Etapa 11, Navier–Stokes)
- Solo estacionario validado. En transitorio con inercia, las masas estabilizadas (SUPG y PSPG de `ρ u_t`) usan `τ`
  y la velocidad del estado inicial del solve, y el θ-método linealiza una sola vez por paso: el transitorio de
  Navier–Stokes se completa en la Etapa 14 (Newton dentro del paso, BDF2).
- A Re alto Newton con búsqueda de línea se estanca (cavidad Re = 5000 desde el reposo: más de 2 minutos sin converger).
  Usar `globalization='pseudo-transient'` (24 iteraciones, sin continuación; a Re = 10000 converge en 48, sin validar). Picard es más robusto lejos de la solución
  pero converge linealmente.
- El tangente omite `dτ/du`: la convergencia no es estrictamente cuadrática, pero 5–7 iteraciones bastan en los casos
  probados.
- Sin estabilización de captura de discontinuidades (no hace falta en flujo incompresible laminar).

### Fluidos (Etapa 12, no newtonianos y acoplamiento térmico)
- Las leyes con fluencia (Bingham, Herschel–Bulkley) son regularizadas: el tapón es casi rígido, no rígido. Valores de
  `m·τ_y/μ_p` muy altos endurecen Newton (cientos de iteraciones). `globalization='pseudo-transient'` es unas 10 veces más
  rápido en `m` = 2000 (128 iteraciones frente a 172, sin búsquedas de línea), pero con `m` = 2e5 necesita 396 iteraciones
  frente a 176 y queda menos exacto (1.7e-5 frente a 2.1e-6).
- Las leyes de adelgazamiento convergen mejor partiendo de una solución newtoniana (resolver con `setViscosity` y luego
  activar la ley); los ejemplos lo hacen así.
- El residuo SUPG del calor incluye convección y la fuente constante, no la disipación viscosa ni reacciones: pequeña
  inconsistencia O(τ Φ).
- Transitorio: las masas estabilizadas (PSPG, SUPG del flujo y del calor) se congelan con el estado inicial del solve.
- Cross–WLF sin dependencia de la presión (`D3 = 0`).
- Con una ley de viscosidad, cada iteración multiplica `Q @ dσ/dx` (dispersa); es más caro que el caso newtoniano.
  `viscous_projection = False` lo evita a costa de la consistencia de la PSPG.

### Fluidos (Etapa 13, axisimétrico)
- Coordenadas fijas: `x = r`, `y = z`, eje en `x = 0` (radios negativos se rechazan). Sin giro (`u_θ = 0`).
- La presión de los nodos del eje converge con orden 1 (las funciones de prueba del eje llevan el peso `r → 0`).
- `GradientLawTerm` con campos vectoriales no incluye los términos de aro (los escalares sí son correctos).
- Fluidos inelásticos: la pérdida de entrada de una contracción (corrección de Bagley) es pequeña; en polímeros reales
  domina la elasticidad del fundido (viscoelástico, Etapa 17 opcional).

### Fluidos (Etapa 14, transitorio)
- **Esquema recomendado para flujo: BDF2** (orden 2 limpio, L-estable). Crank–Nicolson es estable (presión implícita,
  sin oscilaciones) y muy preciso a pasos prácticos, pero conserva un término de primer orden de coeficiente pequeño
  cuyo origen no se ha aislado. Euler implícito: orden 1.
- La presión del arranque desde el reposo tiene un pequeño error espacial de la PSPG (curvatura grande cerca de las
  paredes) que decae con el tiempo.
- Paso fijo (sin control adaptativo). Condiciones de contorno independientes del tiempo.
- Coste: el término de flujo se reensambla en cada iteración de Newton (en Python vectorizado); el ejemplo del cilindro
  con unos miles de nodos y 800 pasos tarda del orden de veinte minutos.

### Fluidos (Etapa 15, convección natural y especies)
- Boussinesq con β constante; sin dependencia de la densidad en la inercia. La presión es la desviación de la
  hidrostática (`ρ₀ g` no se incluye).
- Triángulos: una fuerza de cuerpo que es un gradiente (estratificación) se equilibra solo de forma aproximada y
  aparecen corrientes espurias pequeñas que convergen con orden 3; Q4 estructurado lo equilibra exactamente.
  Una discretización robusta en presión (Taylor–Hood con proyección, H(div)) lo evitaría (Etapa 17).
- A Ra alto conviene la continuación en Ra (como en los ejemplos).

### Fluidos (Etapa 16a, turbulencia)
- Spalart–Allmaras sin el término de transición `f_t2` y sin funciones de pared: el primer nodo debe quedar en y⁺ ≲ 1.
- La distancia de Spalding sobreestima la distancia a paredes convexas en ≈ d/(2R) (eikonal o p-Poisson en la Etapa 17).
- Solo 2D plano (`GradientLawTerm` no tiene los términos de aro del axisimétrico).
- Newton directo converge en el canal, incluso desde un estado uniforme (escalado por campo). Falta probar SA en flujos con
  separación; para ellos está `globalization='pseudo-transient'`.
- k-ω SST (`sst`): incógnitas k y `ln_omega` (ω queda en `model.sol['omega']`). **Validado solo en el canal**, y solo con
  `globalization='pseudo-transient'`: Newton puro diverge. U_c⁺ queda 2.5–4.3 % bajo la DNS en cualquier malla, también
  en el resolvedor 1D independiente: es del modelo, no de la discretización.
- **k-ω SST en placa plana: sin validar.** A Re_L = 1e6 el residuo se estanca en ~2e-4 (500 iteraciones en 40×48 y 400 en
  80×64), con k recortado a 0 en decenas de nodos junto al borde de ataque. Sin recorte es peor (ln ω oscila entre −23 y
  +24). El C_f ya es razonable y casi no cambia con la malla (−7 a −11 % frente a Schultz–Grunow; SA en 40×48: −6 a −15 %),
  pero no hay convergencia. SA sí converge en la misma malla (38 iteraciones).
- k-ω SST: arrancar con k muy pequeño puede tardar (canal desde estado uniforme: 126 iteraciones frente a 9) y el flujo
  laminar es solución del modelo. Solo 2D plano. El escalón de Driver–Seegmiller no se ha probado.

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
7. **Fluidos.** Ver [PLAN_FLUIDOS.md](PLAN_FLUIDOS.md): Etapas 9 a 15 y 16a (Spalart–Allmaras) hechas; en curso, la Etapa 17 ([PLAN_ETAPA17.md](PLAN_ETAPA17.md)): Prioridades 1 y 2 hechas, salvo k-ω SST en placa plana.

## 5. Criterio para cerrar una etapa

- Todas las suites de `Test/` pasan (`python run_tests.py`).
- Un caso de referencia reproduce el resultado anterior o explica la diferencia.
- La documentación de `README.md`, `docs/ARQUITECTURA.md` y esta página se actualiza en el mismo commit.
