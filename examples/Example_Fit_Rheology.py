## Ejemplo: ajuste de parametros reologicos a datos de reometro (Source/Physics/Fitting.py)
#
# Dos archivos CSV con encabezado, uno por ensayo:
#   viscosidad:   T [C], gamma_dot [1/s], eta [Pa s]          (capilar o rotacional, varias temperaturas)
#   oscilatorio:  T [C], omega [rad/s], Gp [Pa], Gpp [Pa]      (barridos de frecuencia a varias temperaturas)
#
#   python Example_Fit_Rheology.py viscosidad.csv oscilatorio.csv [T_ref]
#
# Sin archivos usa datos SINTETICOS (un PP ilustrativo con 3 % de dispersion), para mostrar el flujo de trabajo:
#   1. curva maestra: factores a_T de los barridos y su ajuste WLF (C1, C2)
#   2. espectro de Maxwell en T_ref: modos (mu_i, lam_i), un tiempo por decada
#   3. parametro no lineal de Giesekus (alpha) desde la viscosidad en T_ref
#   4. Cross-WLF desde la viscosidad a todas las temperaturas
# y escribe las lineas para usarlos: flow.setViscosityLaw(CrossWLF(...)) y polymer.setModes([...]).

import sys

import numpy as np
import matplotlib.pyplot as plt

from Source.Physics.Fitting import (fit_cross_wlf, shift_factors, fit_wlf, fit_spectrum, maxwell_moduli, fit_nonlinear_parameter,
                                    cross_wlf_from_wlf)
from Source.Physics.Rheology import CrossWLF
from Source.Physics.Viscoelastic import SpectrumViscosity


def synthetic():

    """Datos de un material conocido, con 3 % de dispersion: un espectro ancho (13 modos Giesekus de 1e-5 a 10 s), como el de un fundido,
    con desplazamiento WLF. Que la curva de viscosidad sea casi una ley de potencia depende de que haya modos cortos: con un espectro
    que acaba en 1e-3 s, la viscosidad cae cada vez mas rapido por encima de 1e3 1/s y un Cross de un solo n deja un 13 %."""

    rng = np.random.default_rng(7)
    noise = lambda a: a * np.exp(0.03 * rng.standard_normal(np.shape(a)))
    lam = np.logspace(-5, 1, 13)
    modes = [(g * l, l, 0.3) for g, l in zip(2e5 * (lam / 1e-3) ** -0.7 * np.exp(-lam / 8.0), lam)]
    T_ref, C1, C2 = 220.0, 3.5, 160.0
    shift = lambda T: 10.0 ** (-C1 * (T - T_ref) / (C2 + T - T_ref))
    temps = [200.0, 220.0, 240.0, 260.0]
    osc = []
    for T in temps:
        w = np.logspace(-1, 2, 16)
        Gp, Gpp = maxwell_moduli(modes, w * shift(T))
        osc += [(T, wi, a, b) for wi, a, b in zip(w, noise(Gp), noise(Gpp))]
    visc = []
    for T in temps:
        g = np.logspace(0, 4, 10)
        eta = shift(T) * np.asarray(SpectrumViscosity(modes)((g * shift(T)) ** 2, 0.0))    # tiempo-temperatura en la viscosidad
        visc += [(T, gi, e) for gi, e in zip(g, noise(eta))]
    return np.array(visc), np.array(osc), T_ref


if len(sys.argv) >= 3:
    visc = np.genfromtxt(sys.argv[1], delimiter=',', names=True)
    osc = np.genfromtxt(sys.argv[2], delimiter=',', names=True)
    visc = np.c_[visc['T'], visc['gamma_dot'], visc['eta']]
    osc = np.c_[osc['T'], osc['omega'], osc['Gp'], osc['Gpp']]
    temps = np.unique(osc[:, 0])
    T_ref = float(sys.argv[3]) if len(sys.argv) > 3 else float(temps[len(temps) // 2])
    print(f"Datos: {sys.argv[1]} ({len(visc)} puntos), {sys.argv[2]} ({len(osc)} puntos); T_ref = {T_ref} C")
else:
    visc, osc, T_ref = synthetic()
    print(f"Datos SINTETICOS (material conocido, 3 % de dispersion); T_ref = {T_ref} C")

# 1. curva maestra
temps = list(np.unique(osc[:, 0]))
by_T = [osc[osc[:, 0] == T] for T in temps]
aT = shift_factors(temps, [d[:, 1] for d in by_T], [d[:, 2] for d in by_T], [d[:, 3] for d in by_T], T_ref)
wlf = fit_wlf(temps, aT, T_ref)
print("\n1. Factores de desplazamiento:", ", ".join(f"{T:.0f} C: {a:.3g}" for T, a in zip(temps, aT)))
print(wlf)

# 2. espectro sobre la curva maestra
w_master = np.concatenate([d[:, 1] * a for d, a in zip(by_T, aT)])
Gp_master, Gpp_master = np.concatenate([d[:, 2] for d in by_T]), np.concatenate([d[:, 3] for d in by_T])
spectrum = fit_spectrum(w_master, Gp_master, Gpp_master)
print(f"\n2. Espectro en {T_ref} C: {len(spectrum.model)} modos, error relativo rms {spectrum.rms:.2%}, eta0 = {spectrum.params['eta0']:.4g} Pa s")
for mu, lam, _ in spectrum.model:
    print(f"   G = {mu / lam:10.4g} Pa   lam = {lam:10.4g} s   mu = {mu:10.4g} Pa s")

# 3. parametro no lineal desde la viscosidad en T_ref, dentro de la ventana del espectro: fuera de 1 / lam_min la viscosidad de los
#    modos ya no la describen los datos oscilatorios
at_ref = visc[np.isclose(visc[:, 0], T_ref) & (visc[:, 1] <= w_master.max())]
giesekus = fit_nonlinear_parameter(spectrum.model, at_ref[:, 1], at_ref[:, 2], 'giesekus')
print(f"\n3. Giesekus (viscosidad a {T_ref} C hasta {w_master.max():.3g} 1/s, {len(at_ref)} puntos): alpha = {giesekus.params['alpha']:.4f} "
      f"+- {giesekus.std['alpha']:.4f}, error rms {giesekus.rms:.2%}")
print("   (solo con la viscosidad de corte: datos extensionales lo fijarian mejor, y las perdidas de Bagley dependen de el)")

# 4. Cross-WLF a todas las temperaturas
cross = fit_cross_wlf(visc[:, 1], visc[:, 0], visc[:, 2])
print(f"\n4. Cross-WLF (A2 fijo en 51.6 K):\n{cross}")

p = cross.params
print("\nPara usarlos:")
print(f"   flow.setViscosityLaw(CrossWLF(n={p['n']:.4g}, tau_star={p['tau_star']:.4g}, D1={p['D1']:.4g}, D2={p['D2']:.5g}, "
      f"A1={p['A1']:.4g}, A2={p['A2']:.4g}))")
print("   polymer.setModes([" + ", ".join(f"({mu:.4g}, {lam:.4g}, {a:.3g})" for mu, lam, a in giesekus.model) + "])")
print(f"   el desplazamiento WLF de los tiempos de relajacion: C1 = {wlf.params['C1']:.4g}, C2 = {wlf.params['C2']:.4g} K, "
      f"T_ref = {T_ref} C")

# figuras
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
w = np.logspace(np.log10(w_master.min()), np.log10(w_master.max()), 200)
G1, G2 = maxwell_moduli(spectrum.model, w)
axes[0].loglog(w_master, Gp_master, 'o', mfc='none', label="G' (curva maestra)")
axes[0].loglog(w_master, Gpp_master, 's', mfc='none', label="G''")
axes[0].loglog(w, G1, '-', w, G2, '--')
axes[0].set_xlabel('omega a_T [rad/s]'); axes[0].set_ylabel('G [Pa]'); axes[0].legend(); axes[0].set_title(f'Espectro en {T_ref} C')

g = np.logspace(np.log10(visc[:, 1].min()), np.log10(visc[:, 1].max()), 100)
for T in np.unique(visc[:, 0]):
    m = np.isclose(visc[:, 0], T)
    line, = axes[1].loglog(visc[m, 1], visc[m, 2], 'o', mfc='none', label=f'{T:.0f} C')
    axes[1].loglog(g, np.asarray(cross.model(g ** 2, np.full_like(g, T))), '-', color=line.get_color())
axes[1].set_xlabel('gamma_dot [1/s]'); axes[1].set_ylabel('eta [Pa s]'); axes[1].legend(); axes[1].set_title('Cross-WLF')

axes[2].loglog(at_ref[:, 1], at_ref[:, 2], 'o', mfc='none', label='datos')
axes[2].loglog(g, np.asarray(SpectrumViscosity(giesekus.model)(g ** 2, 0.0)), '-', label=f"Giesekus, alpha {giesekus.params['alpha']:.3f}")
axes[2].set_xlabel('gamma_dot [1/s]'); axes[2].set_ylabel('eta [Pa s]'); axes[2].legend(); axes[2].set_title(f'Viscosidad del espectro, {T_ref} C')
for ax in axes:
    ax.grid(True, which='both', alpha=0.3)
plt.show()
