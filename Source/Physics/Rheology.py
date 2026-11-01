# +++++++++++++++++++++++++++++++++++++++++++++++
# Author: Edgar Alejandro Cano Zapata
# E-mail: edgara.cano@outlook.com
# Blog: ---
# +++++++++++++++++++++++++++++++++++++++++++++++

# Viscosity laws of generalized Newtonian fluids: mu(gamma_dot, T).
#
# The shear rate is gamma_dot = sqrt(2 eps:eps), eps = (grad u + grad u^T) / 2 (gamma_dot = |du/dy| in simple shear).
# A law is called with gamma2 = gamma_dot^2 (smooth in the velocity gradient) and the temperature T in the units of
# the model (°C); laws that need absolute temperatures convert internally. They are written with jax.numpy, so the
# flow physics differentiates them automatically (Newton tangent with dmu/dgamma and dmu/dT):
#
#     flow.setViscosityLaw(CarreauYasuda(mu0=3000, mu_inf=0, lam=0.5, n=0.35))
#     flow.setViscosityLaw(CrossWLF(n=0.27, tau_star=3e4, D1=1e13, D2=263.15, A1=28.0, A2=51.6))   # needs a 'T' field
#
# Shear-thinning laws with a zero-shear plateau (Carreau, Cross) behave well at gamma_dot = 0. The power law and the
# yield-stress laws are singular there and are regularized: a minimum shear rate (power law) and the exponential
# regularization of Papanastasiou (Bingham, Herschel-Bulkley).

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from Source.Physics.Physics import KELVIN_OFFSET

_TINY = 1e-30   # keeps sqrt(gamma2) differentiable at gamma2 = 0


def _gamma(gamma2, gamma_min=0.0):
    return jnp.sqrt(gamma2 + gamma_min ** 2 + _TINY)


class ViscosityLaw:

    """Base class: mu(gamma2, T) with jax.numpy. uses_temperature tells the flow physics to read a temperature field."""

    uses_temperature = False
    uses_pressure = False      # the law reads the pressure too: mu(gamma2, T, p)

    def __call__(self, gamma2, T):
        raise NotImplementedError

    def viscosity(self, gamma_dot, T=0.0):

        """mu at shear rates (and temperatures) given as numbers or arrays, for plots and tables."""

        g, t = np.broadcast_arrays(np.asarray(gamma_dot, dtype=float), np.asarray(T, dtype=float))
        return np.asarray(jax.vmap(self.__call__)(jnp.asarray(g.ravel() ** 2), jnp.asarray(t.ravel()))).reshape(g.shape)


class SlipLaw:

    """Wall shear stress as a function of the slip velocity, tau(u_s), with jax.numpy; odd in u_s, so that it opposes the slip."""

    def __call__(self, slip):
        raise NotImplementedError


class NavierSlip(SlipLaw):

    """tau = beta u_s: beta [Pa s/m] is the friction coefficient, mu / beta the extrapolation length. Linear."""

    def __init__(self, beta):
        self.beta = float(beta)

    def __call__(self, slip):
        return self.beta * slip


class PowerSlip(SlipLaw):

    """Power law of polymer melts (Hatzikiriakos), tau = k |u_s|^m sign(u_s), with an optional stress tau_c below which the wall
    sticks (tau_c (1 - exp(-|u_s| / u_c)) sign(u_s) is added, smooth for Newton: smaller u_c, sharper threshold). u_min regularizes the
    power at zero slip, singular for m < 1: below u_min the law is linear in u_s."""

    def __init__(self, k, m, tau_c=0.0, u_c=1e-6, u_min=1e-9):
        self.k, self.m, self.tau_c, self.u_c, self.u_min = map(float, (k, m, tau_c, u_c, u_min))

    def __call__(self, slip):
        magnitude = jnp.sqrt(slip * slip + self.u_min ** 2)
        tau = self.k * magnitude ** (self.m - 1.0) * slip
        if self.tau_c:
            tau = tau + self.tau_c * (-jnp.expm1(-magnitude / self.u_c)) * slip / magnitude
        return tau


class Newtonian(ViscosityLaw):

    def __init__(self, mu):
        self.mu = float(mu)

    def __call__(self, gamma2, T):
        return self.mu + 0.0 * gamma2


class PowerLaw(ViscosityLaw):

    """mu = K gamma_dot^(n - 1) (Ostwald-de Waele), with gamma_dot >= gamma_min so that mu stays finite for n < 1."""

    def __init__(self, K, n, gamma_min=1e-3):
        self.K, self.n, self.gamma_min = float(K), float(n), float(gamma_min)

    def __call__(self, gamma2, T):
        return self.K * (gamma2 + self.gamma_min ** 2) ** ((self.n - 1.0) / 2.0)


class CarreauYasuda(ViscosityLaw):

    """mu = mu_inf + (mu0 - mu_inf) (1 + (lam gamma_dot)^a)^((n - 1) / a). a = 2 is the Carreau law."""

    def __init__(self, mu0, lam, n, a=2.0, mu_inf=0.0):
        self.mu0, self.lam, self.n, self.a, self.mu_inf = float(mu0), float(lam), float(n), float(a), float(mu_inf)

    def __call__(self, gamma2, T):
        lg_a = (self.lam ** 2 * gamma2 + _TINY) ** (self.a / 2.0)
        return self.mu_inf + (self.mu0 - self.mu_inf) * (1.0 + lg_a) ** ((self.n - 1.0) / self.a)


class Cross(ViscosityLaw):

    """mu = mu0 / (1 + (mu0 gamma_dot / tau_star)^(1 - n)): zero-shear plateau mu0, power law of index n beyond
    the shear stress tau_star [Pa]."""

    def __init__(self, mu0, tau_star, n):
        self.mu0, self.tau_star, self.n = float(mu0), float(tau_star), float(n)

    def __call__(self, gamma2, T):
        x2 = (self.mu0 / self.tau_star) ** 2 * gamma2 + _TINY
        return self.mu0 / (1.0 + x2 ** ((1.0 - self.n) / 2.0))


class CrossWLF(ViscosityLaw):

    """Cross-WLF model of polymer melts (the form used by injection-molding software):

        mu = mu0(T) / (1 + (mu0(T) gamma_dot / tau_star)^(1 - n)),   mu0(T) = D1 exp(-A1 (T - D2) / (A2 + T - D2))

    with T and D2 in Kelvin (the law converts the model temperature from °C). The pressure enters through D3 [K/Pa]:
    T* = D2 + D3 p and A2 = A2~ + D3 p, so that mu0 = D1 exp(-A1 (T - D2 - D3 p) / (A2~ + T - D2)). With D3 > 0 the
    pressure is read as a field of the flow itself, with its absolute level: use the true pressure at the outlet.
    Valid above T* (roughly the glass transition)."""

    uses_temperature = True

    def __init__(self, n, tau_star, D1, D2, A1, A2, D3=0.0):
        self.n, self.tau_star, self.D1, self.D2, self.A1, self.A2, self.D3 = map(float, (n, tau_star, D1, D2, A1, A2, D3))
        self.uses_pressure = self.D3 != 0.0

    def zero_shear(self, T, p=0.0):
        dT = T + KELVIN_OFFSET - self.D2
        return self.D1 * jnp.exp(-self.A1 * (dT - self.D3 * p) / (self.A2 + dT))

    def __call__(self, gamma2, T, p=0.0):
        mu0 = self.zero_shear(T, p)
        x2 = (mu0 / self.tau_star) ** 2 * gamma2 + _TINY
        return mu0 / (1.0 + x2 ** ((1.0 - self.n) / 2.0))


class Bingham(ViscosityLaw):

    """Bingham plastic with the Papanastasiou regularization: mu = mu_p + tau_y (1 - exp(-m gamma_dot)) / gamma_dot.

    tau_y [Pa] yield stress, mu_p plastic viscosity, m [s] regularization (larger m: closer to the ideal Bingham
    fluid, harder for Newton; m tau_y / mu_p of 100 to 1000 is usual)."""

    def __init__(self, mu_p, tau_y, m):
        self.mu_p, self.tau_y, self.m = float(mu_p), float(tau_y), float(m)

    def __call__(self, gamma2, T):
        g = _gamma(gamma2)
        mg = self.m * g
        # (1 - exp(-m g)) / g, written to stay accurate (-> m) at small shear rates
        ratio = jnp.where(mg < 1e-6, self.m * (1.0 - 0.5 * mg), -jnp.expm1(-mg) / g)
        return self.mu_p + self.tau_y * ratio


class HerschelBulkley(ViscosityLaw):

    """mu = K gamma_dot^(n - 1) + tau_y (1 - exp(-m gamma_dot)) / gamma_dot (Papanastasiou), gamma_dot >= gamma_min."""

    def __init__(self, K, n, tau_y, m, gamma_min=1e-3):
        self.K, self.n, self.tau_y, self.m, self.gamma_min = map(float, (K, n, tau_y, m, gamma_min))

    def __call__(self, gamma2, T):
        g = _gamma(gamma2)
        mg = self.m * g
        ratio = jnp.where(mg < 1e-6, self.m * (1.0 - 0.5 * mg), -jnp.expm1(-mg) / g)
        return self.K * (gamma2 + self.gamma_min ** 2) ** ((self.n - 1.0) / 2.0) + self.tau_y * ratio


class Arrhenius(ViscosityLaw):

    """Temperature dependence of any law by time-temperature superposition:

        mu(gamma_dot, T) = a_T mu_ref(a_T gamma_dot),   a_T = exp(E_R (1/T - 1/T_ref))   (T, T_ref in K from °C)

    E_R = E_a / R [K]. For a Newtonian law it reduces to mu_ref * a_T."""

    uses_temperature = True

    def __init__(self, law, E_R, T_ref):
        self.law, self.E_R, self.T_ref = law, float(E_R), float(T_ref)

    def shift(self, T):
        return jnp.exp(self.E_R * (1.0 / (T + KELVIN_OFFSET) - 1.0 / (self.T_ref + KELVIN_OFFSET)))

    def __call__(self, gamma2, T):
        aT = self.shift(T)
        return aT * self.law(aT ** 2 * gamma2, T)


class Exponential(ViscosityLaw):

    """mu(gamma_dot, T) = mu_ref(gamma_dot) exp(-b (T - T_ref)): a simple temperature factor (b [1/K])."""

    uses_temperature = True

    def __init__(self, law, b, T_ref):
        self.law, self.b, self.T_ref = law, float(b), float(T_ref)

    def __call__(self, gamma2, T):
        return self.law(gamma2, T) * jnp.exp(-self.b * (T - self.T_ref))
