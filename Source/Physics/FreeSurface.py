# Steady free surface of an extrudate (die swell), by the streamline method of Tanner: the free surface is a streamline with no
# traction on it. Each iteration
#
#   1. solves the flow on the current domain (the free surface is a natural boundary: zero traction),
#   2. integrates the kinematic condition dh/ds = u_t / u_s along the surface from the lip of the die, h(s0) = H, so that the
#      surface becomes the streamline that leaves the lip (s: coordinate along the flow, t: across it, from the symmetry line),
#   3. moves every node downstream of the lip across the flow, t -> t h_new(s) / h_old(s), which keeps the mesh valid for
#      moderate swell, and the discretization is built again (Model.moveNodes),
#
# with relaxation, until h stops changing. The flow physics needs nothing else: any fluid of the code (Newtonian, generalized
# Newtonian, viscoelastic, non-isothermal) and plane or axisymmetric flow. No surface tension or gravity.
#
# Plane flow along x: s = x, t = y (along=0). Axisymmetric (x = r, y = z) flow along z: s = z, t = r (along=1).

import time

import numpy as np

from Source.logger_config import sim_logger


class FreeSurface:

    def __init__(self, model, edges, lip, along=0, relaxation=1.0):

        """model: solved by model.solve() with its own configuration. edges: the edges of the free surface, straight at
        t = H at the start (its nodes, midsides too, define h). lip: the coordinate s0 of the lip of the die, where the surface
        starts. along: 0 if the flow goes along x (plane), 1 along y (axisymmetric, z). relaxation: fraction of the update of h."""

        self.model = model
        self.along, self.across = along, 1 - along
        self.s0 = float(lip)
        self.relaxation = float(relaxation)
        xy = self._xy()
        nodes = np.unique(np.ravel(np.asarray(edges, dtype=int)))
        self.nodes = nodes[np.argsort(xy[nodes, self.along])]
        self.H = float(xy[self.nodes[0], self.across])
        if np.ptp(xy[self.nodes, self.across]) > 1e-12 * max(1.0, abs(self.H)):
            raise ValueError("La superficie libre debe empezar recta (t = H constante).")
        if abs(xy[self.nodes[0], self.along] - self.s0) > 1e-12 * max(1.0, abs(self.s0)):
            raise ValueError("La superficie libre debe empezar en el borde del dado (s = lip).")
        self.history = []

    def _xy(self):
        return np.array([node.coor[:2] for node in self.model.mesh.NL], dtype=float)

    def height(self):

        """s and h of the surface nodes, ordered along the flow."""

        xy = self._xy()
        return xy[self.nodes, self.along], xy[self.nodes, self.across]

    def swell(self):

        """Swell ratio h(end) / H."""

        return self.height()[1][-1] / self.H

    def streamline(self):

        """The surface that leaves the lip with the current velocity, h(s0) = H, dh/ds = u_t / u_s (trapezoidal rule on the
        nodes); and the largest |u . n| / |u| on the current surface, which is zero once it is a streamline."""

        s, h = self.height()
        u = np.asarray(self.model.sol['u'])[self.nodes]
        speed = np.where(np.abs(u[:, self.along]) > 0, u[:, self.along], 1.0)
        slope = np.where(np.abs(u[:, self.along]) > 0, u[:, self.across] / speed, 0.0)
        slope[0] = slope[1]                                       # the lip is a no-slip node: u = 0, its slope that of the next node
        new =self.H + np.concatenate([[0.0], np.cumsum(0.5 * (slope[1:] + slope[:-1]) * np.diff(s))])
        tangent = np.gradient(h, s)[1:]                           # without the lip, where u = 0
        flux = np.abs(u[1:, self.across] - tangent * u[1:, self.along]) / np.linalg.norm(u[1:], axis=1) / np.sqrt(1 + tangent ** 2)
        return new, float(flux.max())

    def move(self, new):

        """Moves the surface to h = new and the nodes downstream of the lip across the flow in proportion."""

        s, h = self.height()
        xy = self._xy()
        down = xy[:, self.along] > self.s0
        factor = np.interp(xy[down, self.along], s, new) / np.interp(xy[down, self.along], s, h)
        xy[down, self.across] *= factor
        xy[self.nodes, self.across] = new                         # exactly, also where interp would round
        self.model.moveNodes(xy)

    def solve(self, max_iterations=50, tol=1e-6, progress=True):

        """Iterates until the largest change of h is below tol * H. Returns True if it converged; self.history keeps, for each
        iteration, (change of h / H, swell, largest |u . n| / |u|, nonlinear iterations of the flow, seconds)."""

        start = time.perf_counter()
        for k in range(1, max_iterations + 1):
            t = time.perf_counter()
            self.model.solve()
            if getattr(self.model, 'nonlinear_converged', True) is False:
                sim_logger.warning(f"Superficie libre: el flujo no converge en la iteracion {k}.")
                return False
            new, flux = self.streamline()
            _, h = self.height()
            change = float(np.max(np.abs(new - h))) / abs(self.H)
            self.move(h + self.relaxation * (new - h))
            self.history.append((change, self.swell(), flux, getattr(self.model, 'newton_iterations', None) or 0,
                                 time.perf_counter() - t))
            if progress:
                print(f"   superficie libre {k:3d}: cambio de h {change:.2e} H, hinchamiento {self.swell():.5f}, |u.n|/|u| max "
                      f"{flux:.1e}, {self.history[-1][3]} it. del flujo, {self.history[-1][4]:.1f} s "
                      f"(total {time.perf_counter() - start:.0f} s)", flush=True)
            if change < tol:
                self.model.solve()                               # the flow on the final surface
                return True
        return False
