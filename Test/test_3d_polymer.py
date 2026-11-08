"""Checks for the polymer-melt flows in 3D (phase 3 of docs/PLAN_3D.md): viscosity laws, thermal coupling with viscous heating, the
pressure dependence of Cross-WLF, compressible flow (Tait) and wall slip.

Two kinds of checks. (1) The same plane channel solved in 2D (validated by the 2D suites) and as a 3D slab extruded in z with symmetry
on the z faces must give the same fields at the same (x, y): this exercises every 3D code path of the physics against the 2D one.
(2) 3D cases with their own references: the full tangent against finite differences, slip with its exact profile, and a wide
rectangular die against the parallel plates.

Run from the project root:  python Test/test_3d_polymer.py
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff, _isotropic_slip
from Source.Physics.HeatTransfer import ht
from Source.Physics.Rheology import PowerLaw, CarreauYasuda, CrossWLF, NavierSlip, PowerSlip, TaitPVT
from Source.Simulation.Discretization import Discretization
from Source.enums import ElementType, StudyType, ProblemType

# Two-domain Tait of a glass-filled PP (Test/test_compressible.py)
PP_GF = dict(b1m=1.006e-3, b2m=4.936e-7, b3m=7.989e7, b4m=3.53e-3, b1s=9.073e-4, b2s=1.49e-7, b3s=2.601e8, b4s=2.353e-3,
             b5=438.15, b6=1.1e-7, b7=1.006e-4, b8=0.05, b9=1.1e-8)


def slab(dim, element, L, H, nx, ny, order=2, W=0.5):

    """Plane channel 0 < x < L, 0 < y < H: a 2D mesh, or in 3D a slab of thickness W with one cell in z."""

    mesh = Mesh()
    if dim == 2:
        mesh.generate2DMesh((L, H), (nx, ny), element, order=order)
    else:
        kind = ElementType.hexahedron if element == ElementType.quadrilateral else ElementType.tetrahedron
        mesh.generate3DMesh((L, H, W), (nx, ny, 1), kind, order=order)
    return mesh


def faces(mesh, test):

    """Boundary facets whose midpoint passes test(x, y) (also for 3D meshes, where z is ignored)."""

    return mesh.boundary_facets(lambda *c: test(c[0], c[1]))


def common_nodes(mesh2, mesh3):

    """Index of the 2D node at the (x, y) of each 3D node."""

    xy = np.round(np.array([n.coor[:2] for n in mesh2.NL]), 9)
    lookup = {tuple(c): k for k, c in enumerate(xy)}
    return np.array([lookup[tuple(np.round(n.coor[:2], 9))] for n in mesh3.NL])


def compare(build, element=ElementType.quadrilateral, fields=('u', 'p'), **kwargs):

    """Solves build(dim, mesh, ...) on the 2D channel and on the 3D slab; returns {field: max difference / max value} and both models."""

    models = {}
    for dim in (2, 3):
        mesh = slab(dim, element, **kwargs)
        model = build(dim, mesh)
        model.solve()
        assert model.nonlinear_converged in (None, True), dim
        models[dim] = (model, mesh)
    (m2, mesh2), (m3, mesh3) = models[2], models[3]
    index = common_nodes(mesh2, mesh3)
    out = {}
    for name in fields:
        a, b = np.asarray(m2.sol[name], dtype=float), np.asarray(m3.sol[name], dtype=float)
        if a.ndim > 1 and a.shape[-1] == m2.timeVector.size if m2.timeVector is not None else False:
            a, b = a[..., -1], b[..., -1]
        if name == 'u':
            a3 = a[index]
            diff = max(np.max(np.abs(b[:, :2] - a3)), np.max(np.abs(b[:, 2])))
            scale = np.max(np.abs(a))
        else:
            diff, scale = np.max(np.abs(b - a[index])), np.max(np.abs(a))
        out[name] = diff / max(scale, 1e-300)
    return out, models


def symmetric_z(flow, mesh, dim, W=0.5):
    if dim == 3:
        flow.addBC_SymmetryEdges(mesh.boundary_faces(lambda x, y, z: z < 1e-9 or z > W - 1e-9))


# ----- 3D against 2D -----

def test_power_law_and_carreau_match_2d():
    """Shear-thinning laws (power law n = 0.5, Carreau-Yasuda) in a pressure-driven channel: the 3D slab equals the 2D channel, and the power
    law matches its exact profile u = n / (n + 1) (G / K)^(1/n) [(H/2)^(1 + 1/n) - |y - H/2|^(1 + 1/n)]."""

    L, H, G = 2.0, 1.0, 2.0
    for law in (PowerLaw(1.0, 0.5), CarreauYasuda(mu0=1.0, lam=2.0, n=0.4)):
        def build(dim, mesh):
            model = Model(name='law', dim=dim, mesh=mesh, mat=material('X', miu=1.0), psc=ff)
            flow = model.physics
            flow.setViscosityLaw(law)
            flow.addBC_NoSlipEdges(faces(mesh, lambda x, y: y < 1e-9 or y > H - 1e-9))
            flow.addBC_OutletEdges(faces(mesh, lambda x, y: x < 1e-9), G * L)
            flow.addBC_OutletEdges(faces(mesh, lambda x, y: x > L - 1e-9), 0.0)
            symmetric_z(flow, mesh, dim)
            model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
            return model

        diff, models = compare(build, L=L, H=H, nx=2, ny=8)
        print(f"   {type(law).__name__}: 3D - 2D, u {diff['u']:.1e}, p {diff['p']:.1e}")
        assert diff['u'] < 1e-8 and diff['p'] < 1e-8, diff
        if isinstance(law, PowerLaw):
            model, mesh = models[3]
            y = np.array([n.coor[1] for n in mesh.NL])
            n = 0.5
            exact = n / (n + 1) * G ** (1 / n) * ((H / 2) ** (1 + 1 / n) - np.abs(y - H / 2) ** (1 + 1 / n))
            err = np.max(np.abs(np.asarray(model.sol['u'])[:, 0] - exact)) / exact.max()
            print(f"   power law 3D against the exact profile: {err:.1e}")
            assert err < 2e-3


def test_cross_wlf_with_pressure_and_heating_match_2d():
    """A melt (Cross-WLF with pressure dependence D3) entering at 230 into walls at 50 with viscous heating, heat convected by the flow:
    velocity, pressure and temperature of the 3D slab equal those of the 2D channel."""

    L, H = 4.0, 1.0
    law = CrossWLF(n=0.3, tau_star=3e4, D1=1e13, D2=263.15, A1=30.0, A2=51.6, D3=1e-7)

    def build(dim, mesh):
        model = Model(name='melt', dim=dim, mesh=mesh, mat=material('PP', miu=1e3, rho=900.0, Cp=2000.0, k=0.2), psc=[ff, ht])
        flow, heat = model.physics.fluid, model.physics.heat
        flow.setViscosityLaw(law)
        flow.setViscousHeating(True)
        model.physics.setVelocity('u')
        flow.addBC_NoSlipEdges(faces(mesh, lambda x, y: y < 1e-9 or y > H - 1e-9))
        inlet = faces(mesh, lambda x, y: x < 1e-9)
        flow.addBC_VelocityEdges(inlet, lambda *c: (6 * 0.01 * c[1] * (H - c[1]) / H ** 2,) + (0.0,) * (dim - 1))
        flow.addBC_OutletEdges(faces(mesh, lambda x, y: x > L - 1e-9), 0.0)
        symmetric_z(flow, mesh, dim)
        heat.addBC_TemperatureEdges(inlet, 230.0)
        heat.addBC_TemperatureEdges(faces(mesh, lambda x, y: y < 1e-9 or y > H - 1e-9), 200.0)
        model.physics.initField('T', 230.0)
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
        return model

    diff, _ = compare(build, fields=('u', 'p', 'T'), L=L, H=H, nx=4, ny=4)
    print(f"   Cross-WLF + D3 + heating: 3D - 2D, u {diff['u']:.1e}, p {diff['p']:.1e}, T {diff['T']:.1e}")
    assert max(diff.values()) < 1e-8, diff


def test_compressible_match_2d():
    """Compressible channel, rho = rho0 exp(kappa p) (27 % density change, as Test/test_compressible.py), and a thermal Tait melt with
    compression heating in a short transient: 3D slab equal to the 2D channel."""

    L, H, kappa, Q = 20.0, 1.0, 1e-3, 1.0

    def build(dim, mesh):
        model = Model(name='comp', dim=dim, mesh=mesh, mat=material('X', miu=1.0, rho=1.0), psc=ff)
        flow = model.physics
        flow.setDensityLaw(lambda P, T: 1.0 * jnp.exp(kappa * (P - 1.0e5)), T=0.0)
        flow.addBC_NoSlipEdges(faces(mesh, lambda x, y: y < 1e-9 or y > H - 1e-9))
        flow.addBC_VelocityEdges(faces(mesh, lambda x, y: x < 1e-9),
                                 lambda *c: (6 * Q * c[1] * (H - c[1]) / H ** 3,) + (0.0,) * (dim - 1))
        flow.addBC_OutletEdges(faces(mesh, lambda x, y: x > L - 1e-9), 0.0)
        symmetric_z(flow, mesh, dim)
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
        return model

    diff, _ = compare(build, L=L, H=H, nx=10, ny=2)
    print(f"   exponential density, steady: 3D - 2D, u {diff['u']:.1e}, p {diff['p']:.1e}")
    assert max(diff.values()) < 1e-8, diff

    Lp, T0 = 2.0, 230.0

    def packing(dim, mesh):
        model = Model(name='pack', dim=dim, mesh=mesh, mat=material('PP', miu=1e3, rho=900.0, Cp=2000.0, k=0.2), psc=[ff, ht])
        flow, heat = model.physics.fluid, model.physics.heat
        flow.setDensityLaw(TaitPVT(**PP_GF, width=3.0), compression_heating=True)
        heat.setVelocity('u')
        heat.var['T'].updateField(np.full(mesh.getNoN(), T0))
        flow.addBC_NoSlipEdges(faces(mesh, lambda x, y: y < 1e-9 or y > H - 1e-9 or x > Lp - 1e-9))
        left = faces(mesh, lambda x, y: x < 1e-9)
        flow.addBC_OutletEdges(left, 1e6)
        heat.addBC_TemperatureEdges(left, T0)
        symmetric_z(flow, mesh, dim)
        model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, timeStep=0.002, totalTime=0.01,
                                  timeScheme='BDF2')
        return model

    models = {}
    for dim in (2, 3):
        mesh = slab(dim, ElementType.quadrilateral, Lp, H, 4, 2)
        model = packing(dim, mesh)
        model.solve()
        models[dim] = (model, mesh)
    index = common_nodes(models[2][1], models[3][1])
    diffs = {}
    for name in ('p', 'T'):
        a, b = np.asarray(models[2][0].sol[name])[..., -1], np.asarray(models[3][0].sol[name])[..., -1]
        diffs[name] = np.max(np.abs(b - a[index])) / np.max(np.abs(a))
    rise = np.max(np.asarray(models[3][0].sol['T'])[:, -1]) - T0
    print(f"   Tait + compression heating, transient: 3D - 2D, p {diffs['p']:.1e}, T {diffs['T']:.1e} (heating {rise:.3f} K)")
    assert max(diffs.values()) < 1e-8 and rise > 0, (diffs, rise)


def test_slip_matches_2d_and_its_exact_profile():
    """Navier and power-law slip on both walls of a pressure-driven channel: 3D slab equal to the 2D channel, and the exact profile
    u = G y (H - y) / (2 mu) + u_s with tau(u_s) = G H / 2 (Navier: u_s = G H / (2 beta); power: u_s = (G H / (2 k))^(1/m))."""

    L, H, G, mu = 2.0, 1.0, 2.0, 0.5
    for law, u_s in ((NavierSlip(4.0), G * H / (2 * 4.0)), (PowerSlip(3.0, 0.5), (G * H / (2 * 3.0)) ** 2)):
        def build(dim, mesh):
            model = Model(name='slip', dim=dim, mesh=mesh, mat=material('X', miu=mu), psc=ff)
            flow = model.physics
            flow.addBC_SlipLawEdges(faces(mesh, lambda x, y: y < 1e-9 or y > H - 1e-9), law)
            flow.addBC_OutletEdges(faces(mesh, lambda x, y: x < 1e-9), G * L)
            flow.addBC_OutletEdges(faces(mesh, lambda x, y: x > L - 1e-9), 0.0)
            symmetric_z(flow, mesh, dim)
            model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
            return model

        diff, models = compare(build, L=L, H=H, nx=2, ny=4)
        model, mesh = models[3]
        y = np.array([n.coor[1] for n in mesh.NL])
        exact = G * y * (H - y) / (2 * mu) + u_s
        err = np.max(np.abs(np.asarray(model.sol['u'])[:, 0] - exact))
        print(f"   {type(law).__name__}: 3D - 2D u {diff['u']:.1e}; 3D against exact {err:.1e} (u_s = {u_s:.4f})")
        assert diff['u'] < 1e-8 and err < 1e-8, (diff, err)


def test_isotropic_slip_law():
    """The 3D slip law acts along the tangential velocity with the magnitude of the scalar law, tau = tau(|u_s|) u_s / |u_s|, and is
    smooth at u_s = 0."""

    import jax

    law = PowerSlip(3.0, 0.5, tau_c=0.2, u_c=1e-3)
    vector = _isotropic_slip(law)
    for u in (np.array([0.3, -0.4]), np.array([1e-4, 2e-4]), np.array([0.0, 0.7])):
        tau = np.asarray(vector(jnp.asarray(u)))
        s = np.linalg.norm(u)
        assert np.allclose(tau, float(law(s)) * u / s, rtol=1e-12)
    J = np.asarray(jax.jacfwd(vector)(jnp.zeros(2)))
    assert np.all(np.isfinite(J)) and np.allclose(J, J[0, 0] * np.eye(2))
    assert np.allclose(np.asarray(_isotropic_slip(NavierSlip(4.0))(jnp.asarray([0.3, 0.1]))), [1.2, 0.4])


# ----- 3D references -----

def test_tangent_in_3d_with_every_polymer_term():
    """The Newton tangent of a 3D model with Cross-WLF (D3), viscous heating, a thermal Tait melt with compression heating (BDF2 step) and
    power-law slip on a wall, against central finite differences, on hex27 and tet10 (tau frozen at the state)."""

    for kind in (ElementType.hexahedron, ElementType.tetrahedron):
        mesh = Mesh()
        mesh.generate3DMesh((1.0, 1.0, 1.0), (1, 1, 1), kind, order=2)
        model = Model(name='tangent', dim=3, mesh=mesh, mat=material('PP', miu=1.0, rho=900.0, Cp=2000.0, k=0.2), psc=[ff, ht])
        flow = model.physics.fluid
        # D3 of a melt (1e-7 K/Pa): with the pressures of the Tait law (1e6 Pa) a larger one would push T below the WLF singularity
        flow.setViscosityLaw(CrossWLF(n=0.5, tau_star=1.0, D1=1e6, D2=263.15, A1=20.0, A2=51.6, D3=1e-7))
        flow.setViscousHeating(True)
        flow.setDensityLaw(TaitPVT(**PP_GF, width=3.0), compression_heating=True)
        model.physics.setVelocity('u')
        flow.addBC_SlipLawEdges(mesh.boundary_faces(lambda x, y, z: y < 1e-9), PowerSlip(3.0, 0.5))
        model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, timeStep=0.01, totalTime=0.01,
                                  timeScheme='BDF2')
        system = model._system()
        layout = system.layout
        rng = np.random.default_rng(3)
        x = rng.normal(scale=0.3, size=layout.ndof)
        x[layout.dofs('T')[:, 0]] = 230.0 + rng.normal(size=layout.n_nodes)
        x[layout.dofs('p')[:, 0]] = 1e6 * (1 + 0.1 * rng.normal(size=layout.n_nodes))
        x_old = x + 0.01 * rng.normal(size=layout.ndof)
        system.time = {'alpha': 1.5 / 0.01, 'beta': -1.5 / 0.01 * x_old, 'theta': 1.0}
        for term in system._terms:
            if hasattr(term, 'tau_state'):
                term.tau_state = x.copy()
        J, _ = system._assemble(x)
        J = J.toarray()
        scale = np.abs(x) * 1e-6 + 1e-6
        FD = np.zeros_like(J)
        for k in range(layout.ndof):
            e = np.zeros(layout.ndof)
            e[k] = scale[k]
            FD[:, k] = (system._assemble(x + e)[1] - system._assemble(x - e)[1]) / (2 * scale[k])
        column = np.max(np.abs(FD), axis=0) + 1e-300
        err = np.max(np.abs(J - FD) / column)
        print(f"   {kind.name}: {layout.ndof} unknowns, largest tangent error {err:.1e} (relative to each column)")
        assert err < 1e-5, err


def plates_and_duct(n_index, W_over_H, ny=4, H=1.0, G=2.0, L=1.0):

    """Fully developed power-law flow (K = 1, n = n_index) under the gradient G in a duct of section W x H (hex27, the half 0 < z < W/2
    with symmetry at z = W/2): flow rate per unit width and centreline velocity at mid-width, against the parallel plates."""

    W = W_over_H * H
    nz = max(1, int(round(W_over_H * ny / 2 / 2)))
    mesh = Mesh()
    mesh.generate3DMesh((L, H, W / 2), (1, ny, nz), ElementType.hexahedron, order=2)
    model = Model(name='die', dim=3, mesh=mesh, mat=material('X', miu=1.0), psc=ff)
    flow = model.physics
    if n_index != 1.0:
        flow.setViscosityLaw(PowerLaw(1.0, n_index))
    flow.addBC_NoSlipEdges(mesh.boundary_faces(lambda x, y, z: y < 1e-9 or y > H - 1e-9 or z < 1e-9))
    flow.addBC_SymmetryEdges(mesh.boundary_faces(lambda x, y, z: z > W / 2 - 1e-9))
    inlet = mesh.boundary_faces(lambda x, y, z: x < 1e-9)
    flow.addBC_OutletEdges(inlet, G * L)
    flow.addBC_OutletEdges(mesh.boundary_faces(lambda x, y, z: x > L - 1e-9), 0.0)
    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear if n_index != 1.0 else ProblemType.linear)
    model.solve()

    n = n_index
    q_plates = 2 * n / (2 * n + 1) * G ** (1 / n) * (H / 2) ** (2 + 1 / n)          # per unit width
    u_plates = n / (n + 1) * G ** (1 / n) * (H / 2) ** (1 + 1 / n)
    q = -flow.flowRate(inlet) / (W / 2)
    xyz = np.array([node.coor for node in mesh.NL])
    centre = int(np.argmin(np.linalg.norm(xyz[:, 1:] - [H / 2, W / 2], axis=1) + np.abs(xyz[:, 0] - L / 2)))
    return q / q_plates, np.asarray(model.sol['u'])[centre, 0] / u_plates


def test_wide_die_tends_to_the_parallel_plates():
    """Rectangular die of increasing width W/H = 4, 8, 16: the flow rate per unit width and the centre velocity tend to those of the parallel
    plates, with a flow-rate deficit proportional to H/W (the side walls). Newtonian: the deficit matches the duct series,
    192 / pi^5 (H/W) sum tanh(n pi W / 2H) / n^5 to the mesh error; power law n = 0.4: deficit about halved each time W doubles and centre
    velocity within 1e-3 of the plates at W/H = 16."""

    def series_ratio(r):
        k = np.arange(1, 200, 2)
        return 1 - 192 / (np.pi ** 5 * r) * np.sum(np.tanh(k * np.pi * r / 2) / k ** 5)

    for n_index in (1.0, 0.4):
        results = [(r,) + plates_and_duct(n_index, r) for r in (4, 8, 16)]
        deficits = [1 - q for _, q, _ in results]
        for r, q, u in results:
            note = f", series {series_ratio(r):.5f}" if n_index == 1.0 else ""
            print(f"   n = {n_index}, W/H = {r}: q / q_plates {q:.5f}{note}, u_centre / u_plates {u:.5f}")
        if n_index == 1.0:
            for r, q, _ in results:
                assert abs(q - series_ratio(r)) < 2e-3, (r, q, series_ratio(r))
        assert 1.7 < deficits[0] / deficits[1] < 2.3 and 1.7 < deficits[1] / deficits[2] < 2.3, deficits
        assert abs(results[-1][2] - 1) < 2e-3, results[-1]


def main():
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    failures = 0

    for name, fn in tests:
        start = time.perf_counter()
        try:
            fn()
            print(f"PASS  {name} ({time.perf_counter() - start:.1f} s)")
        except Exception as exc:
            failures += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}: {exc!r}")

    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
