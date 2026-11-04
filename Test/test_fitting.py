"""Fitting of rheological parameters: synthetic data of known parameters, exact and with noise.

Run from the project root:  python Test/test_fitting.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Physics.Fitting import (fit_cross_wlf, shift_factors, fit_wlf, fit_arrhenius, cross_wlf_from_wlf, fit_spectrum,
                                    maxwell_moduli, fit_nonlinear_parameter)
from Source.Physics.Rheology import CrossWLF
from Source.Physics.Viscoelastic import SpectrumViscosity

# A polypropylene of the kind injection-molding databases list (parameters illustrative)
TRUE = dict(n=0.2758, tau_star=2.62e4, D1=4.42e14, D2=263.15, A1=32.0, A2=51.6)
MODES = [(1e5 * 0.01, 0.01, 0.0), (3e4 * 0.1, 0.1, 0.0), (1e4 * 1.0, 1.0, 0.0), (1e3 * 10.0, 10.0, 0.0)]


def cross_data(noise=0.0, seed=1):
    T = np.repeat([200.0, 220.0, 240.0, 260.0], 12)
    g = np.tile(np.logspace(0, 4, 12), 4)
    law = CrossWLF(**TRUE)
    eta = np.asarray(law(g ** 2, T), float)
    return g, T, eta * np.exp(noise * np.random.default_rng(seed).standard_normal(len(eta)))


def test_cross_wlf_from_exact_data():

    """Exact data: the parameters come back to round-off, and the law returned gives the data."""

    g, T, eta = cross_data()
    fit = fit_cross_wlf(g, T, eta)
    print(fit)
    for name in ('n', 'tau_star', 'D1', 'D2', 'A1'):
        assert abs(fit.params[name] / TRUE[name] - 1) < 1e-6, name
    assert np.max(np.abs(np.asarray(fit.model(g ** 2, T)) / eta - 1)) < 1e-8


def test_cross_wlf_with_noise():

    """3 % scatter: rms of about 3 %, the law within 5 % of the exact one, the parameters within four standard deviations (D1 and
    tau_star on their logarithm, which is how they are fitted). D1, D2 and A1 one by one are poorly determined by four temperatures
    over 60 K (D1 moves by orders of magnitude from one noise sample to another), but the zero-shear viscosity they give over the
    measured temperatures is not."""

    g, T, eta = cross_data(0.03)
    fit = fit_cross_wlf(g, T, eta)
    exact = np.asarray(CrossWLF(**TRUE)(g ** 2, T))
    print(f"   rms {fit.rms:.3f}; n {fit.params['n']:.4f} +- {fit.std['n']:.4f} (true {TRUE['n']}); "
          f"largest error of the law {np.max(np.abs(np.asarray(fit.model(g ** 2, T)) / exact - 1)):.3f}")
    assert 0.02 < fit.rms < 0.04
    for name in ('n', 'D2', 'A1'):
        assert abs(fit.params[name] - TRUE[name]) < 4 * fit.std[name], name
    for name in ('tau_star', 'D1'):
        assert abs(np.log(fit.params[name] / TRUE[name])) < 4 * fit.std[name] / fit.params[name], name
    assert np.max(np.abs(np.asarray(fit.model(g ** 2, T)) / exact - 1)) < 0.05
    measured = np.unique(T)
    ratio = np.asarray(fit.model.zero_shear(measured)) / np.asarray(CrossWLF(**TRUE).zero_shear(measured))
    print(f"   D1 {fit.params['D1'] / TRUE['D1']:.3g} times the true one; mu0 at the measured temperatures {np.round(ratio, 4)}")
    assert np.max(np.abs(ratio - 1)) < 0.04


def tts_data(log_aT, T, T_ref=200.0, points=30):
    omega = [np.logspace(-1, 2, points) for _ in T]
    Gp, Gpp = zip(*[maxwell_moduli(MODES, w * 10.0 ** a) for w, a in zip(omega, log_aT)])
    return omega, Gp, Gpp


def test_time_temperature_superposition():

    """Shift factors of a WLF material from curves over three decades each, and C1, C2 from them; Arrhenius alike."""

    T, T_ref, C1, C2 = [180.0, 200.0, 220.0, 240.0, 260.0], 200.0, 4.0, 150.0
    d = np.array(T) - T_ref
    log_aT = -C1 * d / (C2 + d)
    aT = shift_factors(T, *tts_data(log_aT, T), T_ref)
    print(f"   log a_T {np.round(np.log10(aT), 4)} (exact {np.round(log_aT, 4)})")
    assert np.max(np.abs(np.log10(aT) - log_aT)) < 2e-3

    wlf = fit_wlf(T, aT, T_ref)
    print(f"   C1 {wlf.params['C1']:.4f}, C2 {wlf.params['C2']:.3f} (exact {C1}, {C2})")
    assert abs(wlf.params['C1'] / C1 - 1) < 0.01 and abs(wlf.params['C2'] / C2 - 1) < 0.02
    exact = fit_wlf(T, 10.0 ** log_aT, T_ref)
    assert abs(exact.params['C1'] / C1 - 1) < 1e-8 and abs(exact.params['C2'] / C2 - 1) < 1e-8

    E_R = 5000.0
    a = np.exp(E_R * (1 / (np.array(T) + 273.15) - 1 / (T_ref + 273.15)))
    assert abs(fit_arrhenius(T, a, T_ref).params['E_R'] / E_R - 1) < 1e-12


def test_cross_wlf_from_a_wlf_shift():

    """The zero-shear viscosity of the law built from C1, C2 follows the WLF shift exactly."""

    T_ref, C1, C2, mu0 = 200.0, 4.0, 150.0, 2.0e4
    law = cross_wlf_from_wlf(0.3, 3e4, mu0, T_ref, C1, C2)
    T = np.array([180.0, 200.0, 250.0])
    d = T - T_ref
    assert np.allclose(np.asarray(law.zero_shear(T)), mu0 * 10.0 ** (-C1 * d / (C2 + d)), rtol=1e-12)


def spectrum_data(noise=0.0, seed=2):
    w = np.logspace(-2, 3, 51)
    Gp, Gpp = maxwell_moduli(MODES, w)
    r = np.random.default_rng(seed)
    return w, Gp * np.exp(noise * r.standard_normal(len(w))), Gpp * np.exp(noise * r.standard_normal(len(w)))


def test_spectrum():

    """With the true times the moduli come back exactly; with one time per decade, G' and G'' and the zero-shear viscosity are
    reproduced; with 2 % noise, within it."""

    w, Gp, Gpp = spectrum_data()
    fit = fit_spectrum(w, Gp, Gpp, lam=[m[1] for m in MODES])
    assert np.allclose([m[0] for m in fit.model], [m[0] for m in MODES], rtol=1e-8)

    eta0 = sum(m[0] for m in MODES)
    for noise, rms, viscosity in ((0.0, 0.01, 0.01), (0.02, 0.03, 0.05)):
        w, Gp, Gpp = spectrum_data(noise)
        for refine in (False, True):
            fit = fit_spectrum(w, Gp, Gpp, refine=refine)
            print(f"   noise {noise:.0%}, refine {refine}: {len(fit.model)} modes, rms {fit.rms:.4f}, "
                  f"eta0 {fit.params['eta0']:.1f} (exact {eta0:.1f})")
            assert fit.rms < rms and abs(fit.params['eta0'] / eta0 - 1) < viscosity


def test_nonlinear_parameter():

    """alpha of Giesekus and epsilon of PTT from the shear viscosity of the spectrum, exact and with 2 % noise."""

    g = np.logspace(-2, 3, 30)
    for constitutive, a in (('giesekus', 0.25), ('ptt-linear', 0.1), ('ptt-exp', 0.2)):
        law = SpectrumViscosity([(mu, lam, a) for mu, lam, _ in MODES], 0.0, constitutive)
        eta = np.asarray(law(g ** 2, 0.0), float)
        fit = fit_nonlinear_parameter(MODES, g, eta, constitutive)
        name = next(iter(fit.params))
        noisy = fit_nonlinear_parameter(MODES, g, eta * np.exp(0.02 * np.random.default_rng(3).standard_normal(len(g))), constitutive)
        print(f"   {constitutive}: {name} {fit.params[name]:.6f} (exact {a}); with 2 % noise {noisy.params[name]:.4f} "
              f"+- {noisy.std[name]:.4f}")
        assert abs(fit.params[name] / a - 1) < 1e-5
        assert abs(noisy.params[name] - a) < 4 * noisy.std[name]
        assert all(m[2] == fit.params[name] for m in fit.model)


def test_whole_chain_on_a_broad_spectrum():

    """The chain of Example_Fit_Rheology.py on a broad spectrum (13 Giesekus modes from 1e-5 to 10 s, alpha = 0.3, WLF shift) with
    3 % scatter: shift factors and C1, C2 from four sweeps over three decades, the spectrum of the master curve, alpha from the
    viscosity inside its window. C1 and C2 one by one move by some 8 % with the noise of four shift factors (they are correlated); the
    shift they give, between the measured temperatures, is what is asked of them."""

    rng = np.random.default_rng(7)
    noise = lambda a: a * np.exp(0.03 * rng.standard_normal(np.shape(a)))
    lam = np.logspace(-5, 1, 13)
    modes = [(g * l, l, 0.3) for g, l in zip(2e5 * (lam / 1e-3) ** -0.7 * np.exp(-lam / 8.0), lam)]
    T, T_ref, C1, C2 = [200.0, 220.0, 240.0, 260.0], 220.0, 3.5, 160.0
    shift = lambda t: 10.0 ** (-C1 * (t - T_ref) / (C2 + t - T_ref))
    omega = [np.logspace(-1, 2, 16) for _ in T]
    moduli = [maxwell_moduli(modes, w * shift(t)) for w, t in zip(omega, T)]
    Gp, Gpp = [noise(m[0]) for m in moduli], [noise(m[1]) for m in moduli]

    aT = shift_factors(T, omega, Gp, Gpp, T_ref)
    wlf = fit_wlf(T, aT, T_ref)
    w = np.concatenate([o * a for o, a in zip(omega, aT)])
    spectrum = fit_spectrum(w, np.concatenate(Gp), np.concatenate(Gpp))
    g = np.logspace(0, np.log10(w.max()), 8)
    eta = noise(np.asarray(SpectrumViscosity(modes)(g ** 2, 0.0)))
    giesekus = fit_nonlinear_parameter(spectrum.model, g, eta)
    eta0 = sum(m[0] for m in modes)
    print(f"   C1 {wlf.params['C1']:.3f}, C2 {wlf.params['C2']:.1f} (exact {C1}, {C2}); spectrum rms {spectrum.rms:.3f}, eta0 "
          f"{spectrum.params['eta0']:.0f} (exact {eta0:.0f}); alpha {giesekus.params['alpha']:.3f} +- {giesekus.std['alpha']:.3f}")
    span = np.linspace(T[0], T[-1], 50)
    fitted = -wlf.params['C1'] * (span - T_ref) / (wlf.params['C2'] + span - T_ref)
    error = np.max(np.abs(fitted - np.log10(shift(span))))
    print(f"   largest error of log10 a_T between {T[0]} and {T[-1]} C: {error:.4f}")
    assert error < 0.02
    assert spectrum.rms < 0.035 and abs(spectrum.params['eta0'] / eta0 - 1) < 0.03
    assert abs(giesekus.params['alpha'] - 0.3) < 3 * giesekus.std['alpha']


if __name__ == "__main__":
    tests = [test_cross_wlf_from_exact_data, test_cross_wlf_with_noise, test_time_temperature_superposition,
             test_cross_wlf_from_a_wlf_shift, test_spectrum, test_nonlinear_parameter, test_whole_chain_on_a_broad_spectrum]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__} {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
