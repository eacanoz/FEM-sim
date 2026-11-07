# Quiescent crystallization kinetics of semicrystalline polymers, non-isothermal Nakamura with the Hoffman-Lauritzen rate written on
# the half-time of crystallization (the form of Chan & Isayev, Polym. Eng. Sci. 34, 461 (1994), used in injection-molding simulation):
#
#     X = 1 - exp(-tau^n),   tau = integral of K(T) dt   (the Nakamura time: it is transported with the melt, and X = 0 is not singular)
#     K(T) = ln(2)^(1/n) (1/t_1/2)_0 exp(-U* / (R (T - T_inf))) exp(-Kg (Tm0 + T) / (2 T^2 (Tm0 - T))),   T_inf < T < Tm0, else 0
#
# X is the relative crystallinity (0 to 1); the absolute one is X times the maximum the polymer reaches. Temperatures of the kinetic
# parameters in Kelvin, as published; the model calls rate() with the temperature in °C.

import numpy as np

from Source.Physics.Physics import KELVIN_OFFSET

R_GAS = 8.314462618


class NakamuraHoffman:

    """Nakamura kinetics with the Hoffman-Lauritzen rate. ln_inv_t_half0: ln((1/t_1/2)_0 [1/s]); Kg [K^2]; U_star [J/mol]; T_inf, Tm0 [K]
    (T_inf = Tg - 30 to 50 K, Tm0 the equilibrium melting temperature); n the Avrami index."""

    def __init__(self, ln_inv_t_half0, Kg, U_star, T_inf, Tm0, n=3.0):
        self.ln_inv_t_half0, self.Kg, self.U_star = float(ln_inv_t_half0), float(Kg), float(U_star)
        self.T_inf, self.Tm0, self.n = float(T_inf), float(Tm0), float(n)

    def rate(self, T):

        """K(T) [1/s] at T [°C]; zero above Tm0 and below T_inf."""

        TK = np.asarray(T, dtype=float) + KELVIN_OFFSET
        inside = (TK > self.T_inf) & (TK < self.Tm0)
        TKs = np.where(inside, TK, 0.5 * (self.T_inf + self.Tm0))
        log_K = (np.log(np.log(2.0)) / self.n + self.ln_inv_t_half0 - self.U_star / (R_GAS * (TKs - self.T_inf))
                 - self.Kg * (self.Tm0 + TKs) / (2.0 * TKs ** 2 * (self.Tm0 - TKs)))
        return np.where(inside, np.exp(log_K), 0.0)

    def half_time(self, T):

        """Isothermal half-time of crystallization [s] at T [°C] (infinite where there is none)."""

        with np.errstate(divide='ignore'):
            return np.log(2.0) ** (1.0 / self.n) / self.rate(T)

    def crystallinity(self, tau):

        """Relative crystallinity X of the Nakamura time tau."""

        return 1.0 - np.exp(-np.maximum(tau, 0.0) ** self.n)
