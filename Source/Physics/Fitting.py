# Fitting of rheological parameters to measured data, so that the laws of Source.Physics.Rheology and the polymer modes of
# Source.Physics.Viscoelastic describe a real material instead of an illustrative one.
#
#     fit_cross_wlf(gamma_dot, T, eta)              Cross-WLF from steady shear viscosity at several temperatures
#     shift_factors(T, omega, Gp, Gpp, T_ref)       time-temperature superposition: a_T of each temperature
#     fit_wlf(T, a_T, T_ref), fit_arrhenius(...)    C1, C2 (or E/R) of the shift factors
#     cross_wlf_from_wlf(...)                       the Cross-WLF law of a WLF shift and a Cross curve at T_ref
#     fit_spectrum(omega, Gp, Gpp)                  discrete Maxwell spectrum [(mu_i, lam_i, 0)], for ve.setModes
#     fit_nonlinear_parameter(modes, gamma_dot, eta) alpha of Giesekus or epsilon of PTT from the shear viscosity
#
# Every fit works on relative errors (logarithms of the data): rheological data span decades and their scatter is a
# percentage. Each returns a Fit with the parameters, their standard deviations from the linearized problem (an estimate:
# the data errors are taken as independent and of equal size) and the rms of the relative error.
#
# Temperatures in °C, as in the rest of the code.

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares, nnls, minimize_scalar

from Source.Physics.Physics import KELVIN_OFFSET
from Source.Physics.Rheology import CrossWLF


@dataclass
class Fit:

    params: dict
    std: dict = field(default_factory=dict)     # standard deviation of each parameter (linearized)
    rms: float = np.nan                         # rms of ln(model / data): about the relative error
    model: object = None                        # the law, or the modes

    def __str__(self):
        rows = [f"  {k:10s} {v: .6g}" + (f"  +- {self.std[k]:.2g}" if k in self.std and np.isfinite(self.std[k]) else "")
                for k, v in self.params.items()]
        return "\n".join(rows + [f"  rms relative error {self.rms:.2%}"])


def _standard_deviations(result, scale):

    """sigma of each variable of a least_squares result: s^2 (J^T J)^-1, times d(parameter)/d(variable)."""

    J, r = result.jac, result.fun
    dof = max(len(r) - J.shape[1], 1)
    try:
        cov = np.linalg.inv(J.T @ J) * (r @ r) / dof
        return np.sqrt(np.maximum(np.diag(cov), 0.0)) * np.abs(scale)
    except np.linalg.LinAlgError:
        return np.full(J.shape[1], np.inf)


# ----- Cross-WLF -----

CROSS_WLF_LOG = ('tau_star', 'D1')          # fitted through their logarithm (positive, over decades)


def _cross_wlf_eta(gamma_dot, T, p, n, tau_star, D1, D2, A1, A2, D3):
    dT = np.asarray(T, dtype=float) + KELVIN_OFFSET - D2
    mu0 = D1 * np.exp(-A1 * (dT - D3 * p) / (A2 + dT))
    return mu0 / (1.0 + (mu0 * gamma_dot / tau_star) ** (1.0 - n))


def fit_cross_wlf(gamma_dot, T, eta, p=None, free=('n', 'tau_star', 'D1', 'D2', 'A1'), fixed=None, guess=None):

    """Cross-WLF (Rheology.CrossWLF) from the shear viscosity eta [Pa s] at the shear rates gamma_dot [1/s] and temperatures T
    [°C], and the pressures p [Pa] if D3 is fitted. free: the fitted parameters; the others take `fixed`, by default A2 = 51.6 K
    (the value the injection-molding databases use: D2 and A2 are strongly correlated, and data over a usual window of
    temperature cannot separate them) and D3 = 0. guess: starting values of some parameters. Returns a Fit whose model is the
    CrossWLF law.

    D1, D2 and A1 one by one are poorly determined unless the temperatures span well over 100 K: with four of them over 60 K and
    3 % scatter, D1 changes by orders of magnitude between noise samples, while mu0(T) over the measured range stays within the
    scatter. Do not extrapolate far in temperature; fixing D2 (near the glass transition, if known) makes the others well posed.
    The std of tau_star and D1 is that of their logarithm times the value: read it as a relative error."""

    g, T, eta = (np.ravel(np.asarray(a, dtype=float)) for a in (gamma_dot, T, eta))
    p = np.zeros_like(g) if p is None else np.ravel(np.asarray(p, dtype=float))
    if not (len(g) == len(T) == len(eta) == len(p)):
        raise ValueError("gamma_dot, T, eta (y p) deben tener la misma longitud.")
    if np.any(g <= 0) or np.any(eta <= 0):
        raise ValueError("Las tasas y las viscosidades deben ser positivas.")

    values = {'A2': 51.6, 'D3': 0.0}
    values.update(fixed or {})
    free = tuple(free)
    unknown = set(free) - {'n', 'tau_star', 'D1', 'D2', 'A1', 'A2', 'D3'}
    if unknown:
        raise ValueError(f"Parametros desconocidos: {sorted(unknown)}.")

    TK = T + KELVIN_OFFSET
    stress = eta * g
    start = {'n': 0.3, 'tau_star': float(np.median(stress)), 'A1': 25.0, 'D2': float(TK.min() - 100.0)}
    # power-law index from the highest rates of each temperature
    slopes = []
    for t in np.unique(T):
        m = T == t
        if m.sum() >= 3:
            top = np.argsort(g[m])[-3:]
            slopes.append(np.polyfit(np.log(g[m][top]), np.log(eta[m][top]), 1)[0])
    if slopes:
        start['n'] = float(np.clip(1.0 + np.median(slopes), 0.05, 0.9))
    start.update(guess or {})

    def complete(x):
        out = dict(values)
        for name, v in zip(free, x):
            out[name] = float(np.exp(v)) if name in CROSS_WLF_LOG else float(v)
        return out

    def residual(x):
        q = complete(x)
        model = _cross_wlf_eta(g, T, p, q['n'], q['tau_star'], q['D1'], q['D2'], q['A1'], q['A2'], q['D3'])
        return np.log(model) - np.log(eta)

    lower = {'n': 0.0, 'tau_star': -np.inf, 'D1': -np.inf, 'D2': 0.0, 'A1': 0.1, 'A2': 1.0, 'D3': -np.inf}
    upper = {'n': 0.999, 'tau_star': np.inf, 'D1': np.inf, 'D2': float(TK.min() - 1.0), 'A1': 500.0, 'A2': 1e3, 'D3': np.inf}

    best = None
    for shift in ((0.0,) if 'D2' in (guess or {}) else (0.0, -50.0, 40.0)):         # D2 has several local minima: a few starts
        s = dict(start)
        s['D2'] = min(start['D2'] + shift, TK.min() - 5.0)
        # D1 so that the plateau of the start passes through the most viscous point
        k = np.argmax(eta)
        dT = TK[k] - s['D2']
        q = dict(values, **s)
        if 'D1' not in (guess or {}):
            s['D1'] = float(eta[k] * np.exp(q['A1'] * dT / (q['A2'] + dT)))
        x0 = [np.log(s[name]) if name in CROSS_WLF_LOG else s.get(name, values.get(name, 0.0)) for name in free]
        lo = [lower[name] for name in free]
        hi = [upper[name] for name in free]
        x0 = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        result = least_squares(residual, x0, bounds=(lo, hi), x_scale='jac', method='trf')
        if best is None or result.cost < best.cost:
            best = result

    params = complete(best.x)
    scale = [params[name] if name in CROSS_WLF_LOG else 1.0 for name in free]
    std = dict(zip(free, _standard_deviations(best, scale)))
    law = CrossWLF(n=params['n'], tau_star=params['tau_star'], D1=params['D1'], D2=params['D2'], A1=params['A1'], A2=params['A2'],
                   D3=params['D3'])

    return Fit(params, std, float(np.sqrt(np.mean(best.fun ** 2))), law)


# ----- time-temperature superposition -----

def _misfit(log_w, log_moduli, master_w, master_moduli):

    """Mean squared difference of shifted curves against the master curve, where they overlap; inf without overlap."""

    inside = (log_w >= master_w[0]) & (log_w <= master_w[-1])
    if inside.sum() < 2:
        return np.inf
    return float(np.mean([(np.interp(log_w[inside], master_w, m) - c[inside]) ** 2 for c, m in zip(log_moduli, master_moduli)]))


def shift_factors(T, omega, Gp, Gpp, T_ref):

    """Horizontal shift factors a_T of oscillatory data at several temperatures: G'(omega a_T, T_ref) = G'(omega, T).

    T: list of temperatures [°C]; omega, Gp, Gpp: lists of arrays, one per temperature. T_ref must be one of them. Each curve is
    shifted onto the master curve built so far (from T_ref outwards, nearest temperatures first), minimizing the difference of
    log G' and log G'' where they overlap. No vertical shift (the factor T rho / T_ref rho_ref is close to 1). Returns a_T for
    each T, in the order given."""

    T = [float(t) for t in T]
    if T_ref not in T:
        raise ValueError("T_ref debe ser una de las temperaturas medidas.")
    data = [(np.log10(np.asarray(w, float)), np.log10(np.asarray(a, float)), np.log10(np.asarray(b, float)))
            for w, a, b in zip(omega, Gp, Gpp)]
    log_a = {T.index(T_ref): 0.0}

    def master():
        w = np.concatenate([data[i][0] + log_a[i] for i in log_a])
        order = np.argsort(w)
        return w[order], [np.concatenate([data[i][1] for i in log_a])[order], np.concatenate([data[i][2] for i in log_a])[order]]

    for i in sorted(range(len(T)), key=lambda k: abs(T[k] - T_ref)):
        if i in log_a:
            continue
        mw, mm = master()
        lw, g1, g2 = data[i]
        cost = lambda s: _misfit(lw + s, (g1, g2), mw, mm)
        grid = np.linspace(-12.0, 12.0, 4801)                        # coarse search: the misfit has flat parts without overlap
        values = np.array([cost(s) for s in grid])
        if not np.isfinite(values).any():
            raise ValueError(f"La curva a {T[i]} C no se superpone con la curva maestra.")
        s0 = grid[np.nanargmin(np.where(np.isfinite(values), values, np.nan))]
        log_a[i] = float(minimize_scalar(cost, bounds=(s0 - 0.01, s0 + 0.01), method='bounded', options={'xatol': 1e-8}).x)

    return np.array([10.0 ** log_a[i] for i in range(len(T))])


def fit_wlf(T, a_T, T_ref):

    """log10 a_T = -C1 (T - T_ref) / (C2 + T - T_ref): C1 and C2 [K] by least squares on log10 a_T."""

    T, y = np.asarray(T, float), np.log10(np.asarray(a_T, float))
    d = T - T_ref

    def residual(x):
        return -x[0] * d / (x[1] + d) - y

    lo = max(1e-3, float(-d.min()) + 1e-3)                            # C2 + T - T_ref > 0 at every temperature
    result = least_squares(residual, [5.0, max(100.0, 2 * lo)], bounds=([0.0, lo], [np.inf, np.inf]))
    std = _standard_deviations(result, [1.0, 1.0])
    return Fit({'C1': float(result.x[0]), 'C2': float(result.x[1]), 'T_ref': float(T_ref)}, {'C1': std[0], 'C2': std[1]},
               float(np.sqrt(np.mean(result.fun ** 2)) * np.log(10.0)))


def fit_arrhenius(T, a_T, T_ref):

    """ln a_T = E_R (1/T - 1/T_ref), T in K: the activation temperature E_R = E_a / R [K] of Rheology.Arrhenius."""

    x = 1.0 / (np.asarray(T, float) + KELVIN_OFFSET) - 1.0 / (T_ref + KELVIN_OFFSET)
    y = np.log(np.asarray(a_T, float))
    E_R = float(x @ y / (x @ x))
    r = E_R * x - y
    std = float(np.sqrt((r @ r) / max(len(r) - 1, 1) / (x @ x)))
    return Fit({'E_R': E_R, 'T_ref': float(T_ref)}, {'E_R': std}, float(np.sqrt(np.mean(r ** 2))))


def cross_wlf_from_wlf(n, tau_star, mu0_ref, T_ref, C1, C2):

    """The Cross-WLF law whose zero-shear viscosity follows a WLF shift: D1 = mu0(T_ref), D2 = T_ref (K), A1 = C1 ln 10, A2 = C2."""

    return CrossWLF(n=n, tau_star=tau_star, D1=mu0_ref, D2=T_ref + KELVIN_OFFSET, A1=C1 * np.log(10.0), A2=C2)


# ----- linear viscoelastic spectrum -----

def maxwell_moduli(modes, omega):

    """G'(omega), G''(omega) of modes [(mu_i, lam_i, ...)], G_i = mu_i / lam_i."""

    w = np.asarray(omega, float)[:, None]
    mu, lam = np.array([m[0] for m in modes], float), np.array([m[1] for m in modes], float)
    x = w * lam
    G = mu / lam
    return (G * x ** 2 / (1 + x ** 2)).sum(axis=1), (G * x / (1 + x ** 2)).sum(axis=1)


def fit_spectrum(omega, Gp, Gpp, lam=None, per_decade=2.0, extend=1.0, refine=False):

    """Discrete Maxwell spectrum of G'(omega), G''(omega) [Pa, omega in rad/s]: modes [(mu_i, lam_i, 0.0)] for ve.setModes, with
    mu_i = G_i lam_i (so sum mu_i is the zero-shear viscosity of the data). The relaxation times are fixed, `per_decade` per decade
    between 1 / omega_max and 1 / omega_min widened by `extend` decades on each side (or `lam`), and the moduli G_i >= 0 come from
    nonnegative least squares on the relative error of G' and G''. Modes with G_i = 0 are dropped.

    The problem is ill-posed: more than about two modes per decade fits the noise, one per decade leaves 3-10 % when the modes of the
    material fall between the times, and without the extra decade the modes shorter than 1 / omega_max pile up in the first one
    (12 % against 2.4 % on a broad spectrum with 3 % noise, Example_Fit_Rheology.py). refine=True then moves the times and moduli of
    the remaining modes (least squares on their logarithms): a smaller rms, but it can move the longest modes and with them the
    zero-shear viscosity (+9 % in that case). The spectrum describes the window of the data only: beyond it, it is an extrapolation."""

    w, Gp, Gpp = (np.asarray(a, float) for a in (omega, Gp, Gpp))
    if lam is None:
        lo, hi = np.log10(1.0 / w.max()) - extend, np.log10(1.0 / w.min()) + extend
        count = max(int(np.ceil((hi - lo) * per_decade)) + 1, 1)
        lam = np.logspace(lo, hi, count)
    lam = np.asarray(lam, float)

    x = w[:, None] * lam[None, :]
    A = np.vstack([x ** 2 / (1 + x ** 2) / Gp[:, None], x / (1 + x ** 2) / Gpp[:, None]])
    scale = np.linalg.norm(A, axis=0)                                  # columns of very different size: nnls stalls without it
    G, _ = nnls(A / scale, np.ones(A.shape[0]), maxiter=100 * A.shape[1])
    G = G / scale
    keep = G > 0
    G, lam = G[keep], lam[keep]

    def residual(z):
        modes = [(np.exp(a) * np.exp(b), np.exp(b)) for a, b in zip(z[:len(G)], z[len(G):])]
        p, q = maxwell_moduli(modes, w)
        return np.concatenate([np.log(p / Gp), np.log(q / Gpp)])

    z = np.concatenate([np.log(G), np.log(lam)])
    result = least_squares(residual, z) if refine else None
    if refine:
        z = result.x
    G, lam = np.exp(z[:len(G)]), np.exp(z[len(G):])
    order = np.argsort(lam)
    G, lam = G[order], lam[order]
    modes = [(float(g * l), float(l), 0.0) for g, l in zip(G, lam)]
    r = residual(np.concatenate([np.log(G), np.log(lam)]))
    params = {f'G{i}': float(g) for i, g in enumerate(G)}
    params.update({f'lam{i}': float(l) for i, l in enumerate(lam)})
    params['eta0'] = float(np.sum(G * lam))
    return Fit(params, {}, float(np.sqrt(np.mean(r ** 2))), modes)


# ----- nonlinear parameter of the modes -----

def fit_nonlinear_parameter(modes, gamma_dot, eta, constitutive='giesekus', solvent=0.0, bounds=None):

    """The parameter alpha of Giesekus, or epsilon of PTT ('ptt-linear', 'ptt-exp'), common to every mode, from the steady shear
    viscosity eta(gamma_dot) of the material, with the linear spectrum `modes` already fitted (fit_spectrum). Uses the closed form
    of Viscoelastic.SpectrumViscosity. Only the shear viscosity enters: elongational data would pin the parameter better, and
    the Bagley losses depend on it. Returns a Fit whose model is the list of modes with that parameter."""

    from Source.Physics.Viscoelastic import SpectrumViscosity           # here: Viscoelastic imports jax-heavy modules

    g, eta = np.asarray(gamma_dot, float), np.asarray(eta, float)
    if bounds is None:
        bounds = (1e-4, 0.5) if constitutive == 'giesekus' else (1e-4, 1.0)

    def residual(a):
        law = SpectrumViscosity([(mu, lam, a) for mu, lam, *_ in modes], solvent, constitutive)
        return np.log(np.asarray(law(g ** 2, 0.0), float)) - np.log(eta)

    result = minimize_scalar(lambda a: float(np.sum(residual(a) ** 2)), bounds=bounds, method='bounded', options={'xatol': 1e-10})
    a = float(result.x)
    r = residual(a)
    h = 1e-6 * max(a, 1e-3)
    J = (residual(a + h) - residual(a - h)) / (2 * h)
    std = float(np.sqrt((r @ r) / max(len(r) - 1, 1) / max(J @ J, 1e-300)))
    name = 'alpha' if constitutive == 'giesekus' else 'epsilon'
    return Fit({name: a}, {name: std}, float(np.sqrt(np.mean(r ** 2))), [(mu, lam, a) for mu, lam, *_ in modes])
