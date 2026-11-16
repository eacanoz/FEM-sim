"""Thin classes per application (phase E.7 of docs/PLAN_RENDIMIENTO_GPU_ML.md): the model, the boundary conditions, the mesh and the
typical outputs of a case, written once over Model and HeleShawFilling instead of in every script. Each one is what a case file
(Source/Cases.py) of its type runs, and can be used directly:

    SlitDie(material, half_aperture, length)              .solve(q), .curve(flow_rates)      flat slit die, melt with viscous heating
    CapillaryRheometer(material, T, ...)                  .measure(apparent_rates)           Bagley and Rabinowitsch corrections
    InjectionCase.plate(material, ...)                    .run()                              fill: isochrones, pressure, clamp force

`material` is a Source.Material.material with a viscosity law (material.from_file) or the name of one of the library. The models
they build are in .model (SlitDie, CapillaryRheometer) or .filling (InjectionCase), for anything these classes do not cover.
"""

import os

import numpy as np

from Source.Material import material as Material


def _as_material(mat):
    return Material.from_file(mat) if isinstance(mat, (str, os.PathLike)) else mat


def _property(mat, name, override=None):

    value = override if override is not None else getattr(mat, name, None)
    if value is None:
        raise ValueError(f"material {mat.name!r}: falta la propiedad {name!r} (en el archivo o como argumento).")
    return float(value)


def _viscosity_law(mat):

    if getattr(mat, 'viscosity_law', None) is None:
        raise ValueError(f"material {mat.name!r}: sin seccion 'viscosity'; hace falta una ley de viscosidad.")
    return mat.viscosity_law


def rest_viscosity(law, T):

    """Viscosity of a law at rest (a start for the Newtonian solution): its zero-shear value, or at 1 1/s if it has none."""

    if hasattr(law, 'zero_shear'):
        return float(law.zero_shear(T))
    if hasattr(law, 'mu0'):
        return float(law.mu0)
    return float(law(1.0, T))


# ----- flat slit die -----

class SlitDie:

    """Melt in a flat slit die, the half of the slit by symmetry (examples/Example_Polymer_Slit_Die.py): a parabolic profile of the given
    flow rate at the inlet, no slip at the wall, free outlet, velocity, pressure and temperature solved together (viscous heating on
    request). flow_rate is per unit depth in the half slit [m2/s]; T in °C; rho, Cp, k default to the material's."""

    def __init__(self, material, half_aperture=1.0e-3, length=20.0e-3, nx=80, ny=16, T_inlet=230.0, T_wall=None,
                 viscous_heating=True, rho=None, Cp=None, k=None):

        from Source.Pre_processing.Mesh import Mesh
        from Source.core import Model
        from Source.Physics.FluidFlow import ff
        from Source.Physics.HeatTransfer import ht
        from Source.enums import ElementType

        self.material = _as_material(material)
        self.law = _viscosity_law(self.material)
        self.H, self.L = float(half_aperture), float(length)
        self.T_in = float(T_inlet)
        self.T_wall = self.T_in if T_wall is None else float(T_wall)
        self.viscous_heating = bool(viscous_heating)

        mesh = self.mesh = Mesh()
        mesh.generate2DMesh((self.L, self.H), (nx, ny), ElementType.quadrilateral)
        self.inlet = mesh.boundary_edges(lambda x, y: x < 1e-12 * self.L)
        self.outlet = mesh.boundary_edges(lambda x, y: x > self.L * (1 - 1e-12))
        axis = mesh.boundary_edges(lambda x, y: y < 1e-12 * self.H)
        wall = mesh.boundary_edges(lambda x, y: y > self.H * (1 - 1e-12))
        self.xy = mesh.coords

        thermal = {'rho': _property(self.material, 'rho', rho), 'Cp': _property(self.material, 'Cp', Cp),
                   'k': _property(self.material, 'k', k)}
        mu0 = rest_viscosity(self.law, self.T_in)
        self.model = Model(name='slit_die', dim=2, mesh=mesh, mat=Material('melt', miu=mu0, **thermal), psc=[ff, ht])
        self.fluid, heat = self.model.physics.fluid, self.model.physics.heat

        self.fluid.addBC_SymmetryEdges(axis)
        self.fluid.addBC_NoSlipEdges(wall)
        self.fluid.addBC_OutletEdges(self.outlet, 0.0)
        heat.addBC_TemperatureEdges(self.inlet, self.T_in)
        heat.addBC_TemperatureEdges(wall, self.T_wall)
        self.model.physics.setVelocity('u')
        self.model.physics.setStabilization('SUPG')
        self.model.physics.initField('T', self.T_in)
        self.model.solverConfiguration(Study='steady', Type='nonlinear', Method='direct')

        self._law_on = False

    def _set_flow_rate(self, q):
        u_max = 1.5 * q / self.H
        self.fluid.addBC_VelocityEdges(self.inlet, lambda x, y: (u_max * (1 - (y / self.H) ** 2), 0.0))

    def solve(self, flow_rate):

        """Solves at a flow rate (starting from the previous solution) and returns the results: pressure_drop [Pa], T_max and
        T_outlet_mixed [°C], flow_in, flow_out [m2/s], wall_shear_rate [1/s] (apparent, 3 q / H^2), converged, newton_iterations."""

        q, model = float(flow_rate), self.model
        self._set_flow_rate(q)

        if not self._law_on:
            model.solve()                                              # Newtonian (rest viscosity): the starting point
            self.fluid.setViscosityLaw(self.law)
            self.fluid.setViscousHeating(self.viscous_heating)
            self._law_on = True

        model.solve()

        nodes = np.unique(np.ravel(self.inlet))
        column = np.abs(self.xy[:, 0] - self.L) < 1e-9 * self.L
        order = np.argsort(self.xy[column, 1])
        y = self.xy[column, 1][order]
        u_out, T_out = model.sol['u'][column, 0][order], model.sol['T'][column][order]

        return {
            'pressure_drop': float(model.sol['p'][nodes].mean()),
            'T_max': float(model.sol['T'].max()),
            'T_outlet_mixed': float(np.trapezoid(u_out * T_out, y) / np.trapezoid(u_out, y)),
            'flow_in': float(-self.fluid.flowRate(self.inlet)),
            'flow_out': float(self.fluid.flowRate(self.outlet)),
            'wall_shear_rate': float(3.0 * q / self.H ** 2),
            'converged': bool(model.nonlinear_converged),
            'newton_iterations': int(model.newton_iterations),
        }

    def curve(self, flow_rates):

        """The die curve: a list of results, one per flow rate, each starting from the previous solution (continuation)."""

        return [dict(flow_rate=float(q), **self.solve(q)) for q in flow_rates]


# ----- capillary rheometer -----

class CapillaryRheometer:

    """Simulated capillary rheometer (examples/Example_Capillary_Rheometer.py): an axisymmetric barrel and capillary with a flat (180°)
    entry, isothermal at T [°C], the viscosity the material's law at that temperature. measure() processes the runs as in the
    laboratory:

        apparent shear rate   gamma_a = 4 Q / (pi R^3)
        wall stress           tau_w = (R / 2) |dp/dz|, the gradient of the developed stretch of the capillary
        entrance loss         dp_e = total drop - (developed stretches of barrel and capillary)   (Bagley)
        Rabinowitsch          gamma_w = gamma_a (3 n' + 1) / (4 n'),  n' = d ln tau_w / d ln gamma_a
        true viscosity        eta = tau_w / gamma_w, compared with the law given to the model

    mesh_scale multiplies the element sizes (1: those of the example, about 0.12 R in the capillary and finer at the corner)."""

    def __init__(self, material, T=230.0, barrel_radius=4.75e-3, barrel_length=8.0e-3, capillary_radius=0.5e-3,
                 capillary_length=16.0e-3, mesh_scale=1.0):

        from Source.core import Model
        from Source.Physics.FluidFlow import ff
        from Source.Physics.Rheology import AtTemperature

        self.material = _as_material(material)
        self.T = float(T)
        self.law = AtTemperature(_viscosity_law(self.material), self.T)
        self.RB, self.LB, self.RC, self.LC = (float(v) for v in (barrel_radius, barrel_length, capillary_radius, capillary_length))

        self.mesh = self._mesh(float(mesh_scale))
        self.xy = self.mesh.coords

        self.model = Model(name='capillary', dim=2, mesh=self.mesh, mat=Material('melt', miu=rest_viscosity(self.law.law, self.T)),
                           psc=ff, axisymmetric=True)
        self.fluid = self.model.physics
        self.inlet, self.outlet = self.mesh.boundary_edges('inlet'), self.mesh.boundary_edges('outlet')
        self.fluid.addBC_SymmetryEdges(self.mesh.boundary_edges('axis'))
        self.fluid.addBC_NoSlipEdges(self.mesh.boundary_edges('wall'))
        self.fluid.addBC_OutletEdges(self.outlet, 0.0)
        self._law_on = False

    def _mesh(self, scale):

        import gmsh
        from Source.Pre_processing.Mesh import Mesh

        lc_barrel, lc_cap, lc_corner = 0.7 * self.RC * scale, 0.12 * self.RC * scale, 0.03 * self.RC * scale
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            gmsh.model.add('capillary')
            geo = gmsh.model.geo
            points = [(0, 0, lc_barrel), (self.RB, 0, lc_barrel), (self.RB, self.LB, lc_barrel), (self.RC, self.LB, lc_corner),
                      (self.RC, self.LB + self.LC, lc_cap), (0, self.LB + self.LC, lc_cap), (0, self.LB, lc_cap)]
            p = [geo.addPoint(x, y, 0, s) for x, y, s in points]
            lines = [geo.addLine(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]
            geo.addPlaneSurface([geo.addCurveLoop(lines)])
            geo.synchronize()
            gmsh.model.addPhysicalGroup(1, [lines[0]], name='inlet')
            gmsh.model.addPhysicalGroup(1, [lines[1], lines[2], lines[3]], name='wall')
            gmsh.model.addPhysicalGroup(1, [lines[4]], name='outlet')
            gmsh.model.addPhysicalGroup(1, [lines[5], lines[6]], name='axis')
            gmsh.model.mesh.generate(2)
            return Mesh.from_gmsh_model()
        finally:
            gmsh.finalize()

    def _set_flow_rate(self, q):
        u_mean = q / (np.pi * self.RB ** 2)                  # the piston: developed Newtonian profile in the barrel
        self.fluid.addBC_VelocityEdges(self.inlet, lambda r, z: (0.0, 2 * u_mean * (1 - (r / self.RB) ** 2)))

    def _axial_gradient(self, z0, z1):
        sel = (self.xy[:, 0] < 1e-12) & (self.xy[:, 1] > z0) & (self.xy[:, 1] < z1)
        return -np.polyfit(self.xy[sel, 1], self.model.sol['p'][sel], 1)[0]

    def measure(self, apparent_shear_rates=(20.0, 60.0, 200.0, 600.0, 2000.0)):

        """Runs the shear rates in increasing order (each from the previous solution) and returns a dictionary of arrays: gamma_apparent,
        tau_wall, dp_total, dp_entrance, n_prime, gamma_wall, eta_measured, eta_law, error (eta_measured / eta_law - 1), newton_iterations.
        At least three rates are needed (n' is a derivative of the curve)."""

        rates = [float(g) for g in apparent_shear_rates]
        if len(rates) < 3 or rates != sorted(rates):
            raise ValueError("measure: al menos tres tasas de corte aparentes, en orden creciente.")

        model, rows = self.model, []

        self._set_flow_rate(rates[0] * np.pi * self.RC ** 3 / 4)
        model.solverConfiguration(Study='steady', Type='linear', Method='direct')
        model.solve()                                               # Newtonian start
        self.fluid.setViscosityLaw(self.law)
        model.solverConfiguration(Study='steady', Type='nonlinear', Method='direct')

        inlet_nodes = np.unique(np.ravel(self.inlet))
        for gamma_a in rates:
            self._set_flow_rate(gamma_a * np.pi * self.RC ** 3 / 4)
            model.solve()
            q = self.fluid.flowRate(self.outlet)
            dp_total = model.sol['p'][inlet_nodes].mean()
            grad_cap = self._axial_gradient(self.LB + 0.4 * self.LC, self.LB + 0.9 * self.LC)
            grad_bar = self._axial_gradient(0.1 * self.LB, 0.4 * self.LB)
            rows.append((4 * q / (np.pi * self.RC ** 3), self.RC / 2 * grad_cap, dp_total,
                         dp_total - grad_cap * self.LC - grad_bar * self.LB, model.newton_iterations))

        rows = np.array(rows)
        gamma_app, tau_w = rows[:, 0], rows[:, 1]
        n_prime = np.gradient(np.log(tau_w), np.log(gamma_app))
        gamma_w = gamma_app * (3 * n_prime + 1) / (4 * n_prime)
        eta = tau_w / gamma_w
        eta_law = self.law.law.viscosity(gamma_w, self.T)

        return {'gamma_apparent': gamma_app, 'tau_wall': tau_w, 'dp_total': rows[:, 2], 'dp_entrance': rows[:, 3],
                'n_prime': n_prime, 'gamma_wall': gamma_w, 'eta_measured': eta, 'eta_law': eta_law,
                'error': eta / eta_law - 1.0, 'newton_iterations': rows[:, 4].astype(int)}


# ----- injection molding: filling of a cavity -----

class InjectionCase:

    """Filling of a thin cavity with the Hele-Shaw model (Source/Physics/HeleShaw.py): a mesh of its planform, the gate (boundary
    edges), the gap, and the filling at a constant flow rate (volume / fill_time). With thermal=True the temperature across the gap
    (viscous heating, frozen layer, short shot) with the density, heat capacity and conductivity of the material, a melt at T_melt, a
    mold at T_mold and a contact conductance htc [W/(m2 K)]. The material's viscosity law may be of temperature (Cross-WLF).

    run() fills and returns: fill_time [s], gate_pressure_peak and gate_pressure_end [Pa], clamp_force_peak [N] (the integral of the
    pressure over the projected area, at each step), short_shot (True if the flow froze before filling), filled_fraction (volume),
    frozen_fraction_mean (thermal), steps. Fields afterwards: filling.fill_time (isochrones: when each node filled), filling.p,
    clamp_history [(t, F)]. isochrones(n) gives the levels; plot_isochrones draws them."""

    def __init__(self, material, mesh, gate_edges, gap, fill_time, T_melt=230.0, T_mold=50.0, thermal=True, htc=5000.0, T_noflow=None,
                 batch=0.1, rho=None, Cp=None, k=None):

        from Source.Physics.HeleShaw import HeleShawFilling

        self.material = _as_material(material)
        self.law = _viscosity_law(self.material)
        self.mesh = mesh
        self.fill_duration = float(fill_time)
        self.T_melt, self.T_mold = float(T_melt), float(T_mold)
        self.batch = float(batch)

        self.filling = HeleShawFilling(mesh, gap, self.law, T=self.T_melt)
        self.volume = float(np.sum(self.filling.volume))
        self.filling.addGate(gate_edges, self.volume / self.fill_duration)

        if thermal:
            if T_noflow is None and getattr(self.material, 'pvt', None) is not None:
                T_noflow = self.material.pvt.b['b5'] - 273.15           # the transition temperature of its Tait fit
            self.filling.setThermal(self.T_melt, self.T_mold, _property(self.material, 'rho', rho), _property(self.material, 'Cp', Cp),
                                    _property(self.material, 'k', k), htc=htc, T_noflow=T_noflow, cells=20)
        self.thermal = bool(thermal)

        disc = self.filling.disc
        from Source.Physics.HeleShaw import _scatter_vector
        self.nodal_area = _scatter_vector(disc, np.einsum('eg,gi->ei', disc.detJw, disc.N), disc.conn)
        self.clamp_history = []

    @classmethod
    def plate(cls, material, length=0.10, width=0.06, thickness=2e-3, gate_width=0.01, fill_time=1.0, element_size=2.5e-3, **options):

        """A rectangular plate length x width x thickness with a gate of gate_width at the middle of the edge x = 0."""

        from Source.Pre_processing.Mesh import Mesh
        from Source.enums import ElementType

        mesh = Mesh()
        mesh.generate2DMesh((length, width), (max(2, round(length / element_size)), max(2, round(width / element_size))),
                            ElementType.quadrilateral)
        gate = mesh.boundary_edges(lambda x, y: x < 1e-9 * length and abs(y - width / 2) <= gate_width / 2 + 1e-9 * width)
        return cls(material, mesh, gate, thickness, fill_time, **options)

    def _record(self, filling):
        self.clamp_history.append((filling.time, float(self.nodal_area @ filling.p)))

    def run(self, progress=False):

        filling = self.filling
        self.clamp_history = []
        filling.fill(progress=progress, on_step=self._record, batch=self.batch)
        self._record(filling)

        gate_nodes = filling._gate_nodes()
        pressures = [row[2] for row in filling.history]
        results = {
            'fill_time': float(filling.time),
            'gate_pressure_peak': float(max(pressures)) if pressures else 0.0,
            'gate_pressure_end': float(filling.p[gate_nodes].mean()),
            'clamp_force_peak': float(max(force for _, force in self.clamp_history)),
            'short_shot': filling.short_shot is not None,
            'filled_fraction': float(filling.melt_volume() / self.volume),
            'steps': len(filling.history),
        }
        if self.thermal:
            results['frozen_fraction_mean'] = float(filling.frozen_fraction().mean())
        return results

    def isochrones(self, n=10):

        """n equally spaced fill times between 0 and the last node filled [s]."""

        last = np.nanmax(self.filling.fill_time)
        return np.linspace(0.0, last, n + 1)[1:]

    def plot_isochrones(self, ax=None, n=10):

        """Contours of the fill time on the planform (matplotlib); returns the axes."""

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        triang = self.mesh.triangulation(scale=1e3)
        t = np.where(np.isnan(self.filling.fill_time), np.nanmax(self.filling.fill_time), self.filling.fill_time)
        cs = ax.tricontourf(triang, t, levels=n, cmap='viridis')
        ax.figure.colorbar(cs, ax=ax, label='tiempo de llenado [s]')
        ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
        return ax
