# The feed system of a mold for the Hele-Shaw filling: sprue, runners and gates as a network of circular tubes (1D elements) that carry
# the melt from the machine nozzle to the cavities. The flow rate of a tube of radius R at the pressure gradient G is q = S(R, G) G with
#
#     S = (pi / 2) integral_0^R r^3 / eta(gamma_dot(r)) dr,     gamma_dot(r) eta(gamma_dot(r)) = r G / 2
#
# (the shear stress grows linearly with r; Newtonian S = pi R^4 / (8 mu), power law q = pi n / (3n + 1) R^3 (R G / (2K))^(1/n)).
# HeleShawFilling.addRunners solves the pressure of the network and of the cavity together: each tube element is a conductance S / L
# between its two nodes, with the exact tangent of q(G) (tabulated in log q, log G as the gap fluidity), and the nodes of the network
# fill by the same control volumes as the cavity (V = pi R^2 L / 2 to each end), so the front runs along the runners first and enters
# the cavities through the gates. A gate is a short tube (its own diameter and land length) from a node of the network to the cavity node
# nearest to a point (a pin gate), without volume. The inlet node (the nozzle) starts filled and takes the flow rate of the machine.
#
# The melt in the network is at the injection temperature (the law at T_melt, or at the fixed T of an isothermal filling): a hot runner,
# or a runner whose cooling is neglected. A cold runner (frozen layer across the radius) and the packing through the network are not
# done here. SI units.

import numpy as np
import scipy.sparse as sp


class TubeFluidity:

    """S(R, G) of a circular tube for a viscosity law, with `layers` Gauss-Legendre points across the radius; the local shear rate from
    gamma_dot eta(gamma_dot) = r G / 2 by Newton (the solver of GapFluidity)."""

    def __init__(self, viscosity, T=None, layers=24):
        from Source.Physics.HeleShaw import GapFluidity
        self._gap = GapFluidity(viscosity, T, layers)
        self.s, self.ws = self._gap.s, self._gap.ws

    def __call__(self, R, G):
        R, G = np.broadcast_arrays(np.asarray(R, dtype=float), np.asarray(G, dtype=float))
        r = R[..., None] * self.s
        gamma = self._gap.shear_rate(0.5 * r * G[..., None])
        return 0.5 * np.pi * R * np.sum(self.ws * r ** 3 / self._gap.eta(gamma ** 2), axis=-1)

    def table(self, R, G_min=1e-3, G_max=1e12, points=1201):

        """log G and log q_m(G) = log(S(R, G) G) on a log grid: q_m grows monotonically with G."""

        logG = np.linspace(np.log(G_min), np.log(G_max), points)
        G = np.exp(logG)
        return logG, np.log(self(np.full_like(G, R), G) * G)


class RunnerSystem:

    """The network: nodes in space, tubes between them, the inlet and the gates. Build it, then HeleShawFilling.addRunners(system).

        system = RunnerSystem()
        a = system.addNode((x, y, z))
        b = system.addSegment(a, (x2, y2, z2), diameter, divisions=4)      # returns the end node
        system.setInlet(a, Q)
        system.addGate(b, (xg, yg[, zg]), diameter, length)               # to the cavity node nearest to the point"""

    def __init__(self):
        self.xyz = []
        self.elements = []                                   # (node a, node b, diameter)
        self.gates = []                                      # (node, cavity point, diameter, length)
        self.inlet = None

    def addNode(self, xyz):
        self.xyz.append(tuple(float(c) for c in xyz) + (0.0,) * (3 - len(xyz)))
        return len(self.xyz) - 1

    def addSegment(self, start, end, diameter, divisions=1):

        """A straight tube from node `start` to the point (or node) `end`, divided in `divisions` elements; returns the end node."""

        a = np.array(self.xyz[start])
        if np.ndim(end) == 0:
            stop = int(end)
            b = np.array(self.xyz[stop])
        else:
            b = np.array(tuple(float(c) for c in end) + (0.0,) * (3 - len(end)))
            stop = None
        previous = start
        for k in range(1, divisions + 1):
            node = stop if (k == divisions and stop is not None) else self.addNode(a + (b - a) * k / divisions)
            self.elements.append((previous, node, float(diameter)))
            previous = node
        return previous

    def setInlet(self, node, Q):
        self.inlet = (int(node), float(Q))

    def addGate(self, node, point, diameter, length):
        self.gates.append((int(node), tuple(float(c) for c in point), float(diameter), float(length)))

    # ----- bound to a cavity (HeleShawFilling.addRunners) -----

    def bind(self, cavity_xy, fluidity):

        """Arrays for the solver: the cavity node of each gate, the volumes of the nodes, the tables of q(G) per diameter."""

        if self.inlet is None:
            raise ValueError("La red de alimentacion necesita su entrada: setInlet(nodo, Q).")
        self.n = len(self.xyz)
        X = np.array(self.xyz)
        a, b, d = (np.array([e[k] for e in self.elements]) for k in range(3))
        self.length_run = np.linalg.norm(X[b] - X[a], axis=1)
        self.volume = np.zeros(self.n)
        np.add.at(self.volume, a, 0.5 * np.pi * (d / 2) ** 2 * self.length_run)
        np.add.at(self.volume, b, 0.5 * np.pi * (d / 2) ** 2 * self.length_run)
        cav = np.asarray(cavity_xy, dtype=float)
        gate_cavity = []
        for node, point, _, _ in self.gates:
            pt = np.array(point[:cav.shape[1]])
            gate_cavity.append(int(np.argmin(np.linalg.norm(cav - pt, axis=1))))
        self.gate_cavity = np.array(gate_cavity, dtype=int)
        # all the tubes: runner elements (network a, b) and gates (network node, cavity node); diameters and lengths
        self.tube_a = np.concatenate([a, [g[0] for g in self.gates]]).astype(int)            # in the network
        self.tube_b = np.concatenate([b, self.gate_cavity]).astype(int)                      # network node, or cavity node for a gate
        self.tube_gate = np.concatenate([np.zeros(len(a), bool), np.ones(len(self.gates), bool)])
        self.tube_d = np.concatenate([d, [g[2] for g in self.gates]])
        self.tube_L = np.concatenate([self.length_run, [g[3] for g in self.gates]])
        keys = np.round(self.tube_d, 12)
        unique, self._table_index = np.unique(keys, return_inverse=True)
        self._tables = [fluidity.table(0.5 * float(D)) for D in unique]
        self.fill_factor = np.zeros(self.n)
        self.fill_factor[self.inlet[0]] = 1.0
        self.fill_time = np.full(self.n, np.nan)
        self.fill_time[self.inlet[0]] = 0.0
        self.p = np.zeros(self.n)

    def flow(self, G):

        """q_m(G) [m3/s] and dq_m/dG of each tube at the gradients G > 0, interpolated in log q, log G; beyond the table, its end slopes
        (a power law: a Picard iterate from rest can land far outside, and Newton needs a derivative there)."""

        q, dq = np.empty_like(G), np.empty_like(G)
        logG = np.log(np.maximum(G, 1e-300))
        for k, (lg, lq) in enumerate(self._tables):
            m = self._table_index == k
            x = logG[m]
            i = np.clip(np.searchsorted(lg, x) - 1, 0, len(lg) - 2)
            slope = (lq[i + 1] - lq[i]) / (lg[i + 1] - lg[i])
            log_q = lq[i] + slope * (x - lg[i])                     # linear in log-log, also past both ends
            q[m] = np.exp(log_q)
            dq[m] = q[m] / np.exp(x) * slope
        return q, dq

    def assemble(self, p_cavity, p_network, n_cavity, filled_cavity, filled_network):

        """Stiffness and tangent of the tubes in the joint numbering (cavity nodes first, then the network): the flow of a tube from a to b
        is q_m(|dp| / L) sign(dp); K holds q / dp (the secant), J dq / ddp. Only tubes with a filled end carry melt. Returns K, J (csr)
        and the flow of each tube (a to b)."""

        pa = p_network[self.tube_a]
        pb = np.where(self.tube_gate, p_cavity[self.tube_b], p_network[np.where(self.tube_gate, 0, self.tube_b)])
        fa = filled_network[self.tube_a]
        fb = np.where(self.tube_gate, filled_cavity[self.tube_b], filled_network[np.where(self.tube_gate, 0, self.tube_b)])
        carry = fa | fb
        dp = pa - pb
        G = np.abs(dp) / self.tube_L
        # a tube at rest (both ends at one pressure: a node just filled) of a shear-thinning melt has S -> 0: it is evaluated at a
        # millionth of the largest gradient of the network, so that it conducts and the tangent is not singular
        floor = 1e-6 * np.max(G[carry]) if np.any(carry & (G > 0)) else np.exp(self._tables[0][0][0])
        Ge = np.maximum(G, floor)
        q, dq = self.flow(Ge)
        k = q / (Ge * self.tube_L)                                                            # secant conductance q / dp
        q = np.where(G >= floor, q, k * G * self.tube_L)
        k = np.where(carry, k, 0.0)
        kt = np.where(carry, dq / self.tube_L, 0.0)                                           # tangent dq / ddp
        ia = n_cavity + self.tube_a
        ib = np.where(self.tube_gate, self.tube_b, n_cavity + np.where(self.tube_gate, 0, self.tube_b))
        N = n_cavity + self.n
        rows = np.concatenate([ia, ia, ib, ib])
        cols = np.concatenate([ia, ib, ia, ib])
        K = sp.csr_matrix((np.concatenate([k, -k, -k, k]), (rows, cols)), shape=(N, N))
        J = sp.csr_matrix((np.concatenate([kt, -kt, -kt, kt]), (rows, cols)), shape=(N, N))
        return K, J, np.where(carry, np.sign(dp) * q, 0.0)
