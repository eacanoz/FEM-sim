"""Stage 13 of the fluids plan: axisymmetric (r, z) formulation for heat and flow.

The 2D mesh is the meridian plane: x = r, y = z, the axis at x = 0. Run from the project root:
    python Test/test_axisymmetric.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.HeatTransfer import ht
from Source.Physics.Rheology import PowerLaw, CarreauYasuda, Arrhenius
from Source.Simulation.Discretization import Discretization
import Source.Simulation.Operators as ops
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers


def configure(model, kind=ProblemType.linear):
    model.solverConfiguration(Study=StudyType.steady_state, Type=kind, Method=SolverType.direct,
                              Solver=DirectSolvers.PARDISO)


def rz_mesh(size, div, element=ElementType.quadrilateral, r0=0.0):
    mesh = Mesh()
    mesh.generate2DMesh((size[0] - r0, size[1]), div, element)
    for node in mesh.NL:
        node.coor = (node.coor[0] + r0, node.coor[1])
    return mesh


def xy_of(mesh):
    return np.array([n.coor[:2] for n in mesh.NL], dtype=float)


def edges(mesh, selector):
    return mesh.boundary_edges(selector)


# ----- geometry and operators -----

def test_measures_and_hoop_operators():

    R, L = 0.7, 1.3
    for element in (ElementType.quadrilateral, ElementType.triangle):
        mesh = rz_mesh((R, L), (4, 6), element)
        disc = Discretization(mesh, axisymmetric=True)

        assert np.isclose(disc.detJw.sum(), R ** 2 * L / 2)               # integral r dr dz (volume / 2 pi)
        assert np.isclose(disc.area.sum(), R * L)                         # plane area kept for element sizes
        assert np.isclose(disc.facets(edges(mesh, lambda x, y: x > R - 1e-9)).detJw.sum(), R * L)        # lateral
        assert np.isclose(disc.facets(edges(mesh, lambda x, y: y < 1e-9)).detJw.sum(), R ** 2 / 2)      # end disc

        # u = (r, -2 z): div u = 1 + 1 - 2 = 0 (the hoop term u_r / r is needed); eps:eps = 1 + 1 + 4
        xy = disc.node_xy
        u = np.c_[xy[:, 0], -2.0 * xy[:, 1]]
        ue = u[disc.conn].reshape(disc.n_elements, -1)
        assert np.max(np.abs(np.einsum('eij,ej->ei', ops.divergence(disc), ue))) < 1e-12
        energy = np.einsum('ei,eij,ej->', ue, ops.viscous(disc, 1.5), ue)
        assert np.isclose(energy, 2 * 1.5 * 6 * R ** 2 * L / 2)

        # an axial translation has no strain; a radial one does (hoop strain u_r / r)
        uz = np.c_[np.zeros(len(xy)), np.ones(len(xy))][disc.conn].reshape(disc.n_elements, -1)
        assert abs(np.einsum('ei,eij,ej->', uz, ops.viscous(disc, 1.0), uz)) < 1e-12

    try:
        Discretization(rz_mesh((1.0, 1.0), (2, 2), r0=-0.5), axisymmetric=True)
    except ValueError as error:
        assert 'eje' in str(error)
    else:
        raise AssertionError("expected an error for negative radii")


# ----- heat -----

def test_radial_conduction():

    """Hollow cylinder R1 < r < R2 with fixed temperatures: T = T1 + (T2 - T1) ln(r/R1) / ln(R2/R1). Solid cylinder with
    a uniform source q: T = Tw + q (R^2 - r^2) / (4 k), with nothing imposed on the axis (natural condition)."""

    R1, R2, T1, T2, k = 0.2, 1.0, 100.0, 20.0, 2.0
    errors = []
    for n in (4, 8, 16, 32):
        mesh = rz_mesh((R2, 0.5), (n, 2), r0=R1)
        model = Model(name='hollow', dim=2, mesh=mesh, mat=material('X', k=k, rho=1.0, Cp=1.0), psc=ht, axisymmetric=True)
        model.physics.addBC_TemperatureEdges(edges(mesh, lambda x, y: x < R1 + 1e-9), T1)
        model.physics.addBC_TemperatureEdges(edges(mesh, lambda x, y: x > R2 - 1e-9), T2)
        configure(model)
        model.solve()
        r = xy_of(mesh)[:, 0]
        errors.append(np.max(np.abs(model.sol['T'] - (T1 + (T2 - T1) * np.log(r / R1) / np.log(R2 / R1)))))
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    print(f"   hollow cylinder: errors {['%.1e' % e for e in errors]} rates {['%.2f' % r for r in rates]}")
    assert rates[-1] > 1.8

    R, q, Tw = 1.0, 50.0, 10.0
    for element in (ElementType.quadrilateral, ElementType.triangle):
        mesh = rz_mesh((R, 0.5), (8, 2), element)
        model = Model(name='solid', dim=2, mesh=mesh, mat=material('X', k=k, rho=1.0, Cp=1.0), psc=ht, axisymmetric=True)
        model.physics.F_const = q
        model.physics.addBC_TemperatureEdges(edges(mesh, lambda x, y: x > R - 1e-9), Tw)
        configure(model)
        model.solve()
        r = xy_of(mesh)[:, 0]
        err = np.max(np.abs(model.sol['T'] - (Tw + q * (R ** 2 - r ** 2) / (4 * k))))
        assert err < 0.02 * q * R ** 2 / (4 * k), (element, err)

        # energy balance: heat generated = heat through the wall, 2 pi integral of q r dr dz
        lateral = edges(mesh, lambda x, y: x > R - 1e-9)
        assert np.isclose(2 * np.pi * model.geometry.detJw.sum() * q, 2 * np.pi * R * 0.5 * q * R / 2)
        assert lateral


# ----- flow -----

def pipe(n, element=ElementType.quadrilateral, R=0.5, L=2.0, mu=2.0, U=1.0, projection=True):

    """Pipe of radius R: parabolic inlet at z = 0, fully developed outlet at z = L (p = 0), axis and wall."""

    mesh = rz_mesh((R, L), (n, 4 * n), element)
    model = Model(name='pipe', dim=2, mesh=mesh, mat=material('X', miu=mu), psc=ff, axisymmetric=True)
    model.physics.viscous_projection = projection
    inlet, outlet = edges(mesh, lambda x, y: y < 1e-9), edges(mesh, lambda x, y: y > L - 1e-9)
    model.physics.addBC_VelocityEdges(inlet, lambda r, z: (0.0, 2 * U * (1 - (r / R) ** 2)))
    model.physics.addBC_OutletEdges(outlet, 0.0)
    model.physics.addBC_SymmetryEdges(edges(mesh, lambda x, y: x < 1e-9))
    model.physics.addBC_NoSlipEdges(edges(mesh, lambda x, y: x > R - 1e-9))
    configure(model)
    model.solve()

    return model, mesh, inlet, outlet


def test_hagen_poiseuille():

    """Flow rate conserved exactly; downstream of the inlet the flow is Hagen-Poiseuille with the discrete flow rate
    Q_h: u_z = 2 Q_h / (pi R^2) (1 - r^2/R^2), dp/dz = -8 mu Q_h / (pi R^4). The pressure converges at second order off
    the axis and at first order on it (the test functions of the axis nodes carry the weight r -> 0)."""

    R, L, mu = 0.5, 2.0, 2.0
    for element in (ElementType.quadrilateral, ElementType.triangle):
        eu, ep, ea = [], [], []
        for n in (4, 8, 16):
            model, mesh, inlet, outlet = pipe(n, element, R, L, mu)
            q_in, q_out = -model.physics.flowRate(inlet), model.physics.flowRate(outlet)
            assert abs(q_in - q_out) < 1e-12 * q_in

            xy = xy_of(mesh)
            far = xy[:, 1] > 0.5 * L                                         # away from the inlet
            exact = 2 * q_in / (np.pi * R ** 2) * (1 - (xy[:, 0] / R) ** 2)
            eu.append(np.max(np.abs(model.sol['u'][far, 1] - exact[far])) / exact.max())
            grad = 8 * mu * q_in / (np.pi * R ** 4)
            dp = np.abs(model.sol['p'] - grad * (L - xy[:, 1])) / (grad * L)
            axis = xy[:, 0] < 1e-9
            ep.append(dp[far & ~axis].max())
            ea.append(dp[far & axis].max())
            assert np.max(np.abs(model.sol['u'][axis, 0])) == 0.0                    # u_r = 0 on the axis

        ru, rp = np.log2(eu[-2] / eu[-1]), np.log2(ep[-2] / ep[-1])
        print(f"   {element.name}: velocity {['%.1e' % e for e in eu]} rate {ru:.2f}; pressure off the axis "
              f"{['%.1e' % e for e in ep]} rate {rp:.2f}, on the axis {['%.1e' % e for e in ea]}")
        p_order = 1.8 if element == ElementType.quadrilateral else 1.4       # T3 pressure: O(h^1.5), as in plane flows
        assert ru > 1.7 and rp > p_order and eu[-1] < 5e-3 and ea[-1] < 1e-2


def test_uniaxial_extension_is_exact():

    """u = (-a r / 2, a z) is divergence free with a constant stress and no viscous force (the hoop terms of div(sigma)
    cancel): exact for linear elements, also the pressure, with the recovered stress in PSPG."""

    a, mu, R, L = 0.8, 1.7, 1.0, 2.0
    for element in (ElementType.quadrilateral, ElementType.triangle):
        mesh = rz_mesh((R, L), (5, 7), element)
        rng = np.random.default_rng(2)
        for node in mesh.NL:
            r, z = node.coor
            if 1e-9 < r < R - 1e-9 and 1e-9 < z < L - 1e-9:
                node.coor = (r + 0.04 * rng.uniform(-1, 1), z + 0.04 * rng.uniform(-1, 1))
        model = Model(name='extension', dim=2, mesh=mesh, mat=material('X', miu=mu), psc=ff, axisymmetric=True)
        model.physics.addBC_SymmetryEdges(edges(mesh, lambda x, y: x < 1e-9))
        model.physics.addBC_VelocityEdges(edges(mesh, lambda x, y: x > 1e-9), lambda r, z: (-a * r / 2, a * z))
        model.physics.setPressureReference((R, 0.0), 0.0)
        configure(model)
        model.solve()

        xy = xy_of(mesh)
        assert np.max(np.abs(model.sol['u'] - np.c_[-a * xy[:, 0] / 2, a * xy[:, 1]])) < 1e-10
        assert np.max(np.abs(model.sol['p'])) < 1e-9
        assert np.allclose(model.sol['shear_rate'], np.sqrt(3) * a)          # gamma_dot^2 = 2 (a^2/4 + a^2/4 + a^2)


def test_power_law_pipe_flow():

    """Pressure-driven power-law fluid: u_z = n/(n+1) (G/(2K))^(1/n) (R^((n+1)/n) - r^((n+1)/n))."""

    K, n, G, R, L = 2.0, 0.5, 8.0, 0.5, 0.5
    errors = []
    for nr in (8, 16, 32):
        mesh = rz_mesh((R, L), (nr, 2))
        model = Model(name='power', dim=2, mesh=mesh, mat=material('X', miu=1.0), psc=ff, axisymmetric=True)
        model.physics.addBC_OutletEdges(edges(mesh, lambda x, y: y < 1e-9), G * L)
        model.physics.addBC_OutletEdges(edges(mesh, lambda x, y: y > L - 1e-9), 0.0)
        model.physics.addBC_SymmetryEdges(edges(mesh, lambda x, y: x < 1e-9))
        model.physics.addBC_NoSlipEdges(edges(mesh, lambda x, y: x > R - 1e-9))
        configure(model)
        model.solve()                                                         # Newtonian start
        model.physics.setViscosityLaw(PowerLaw(K, n, gamma_min=1e-4))
        configure(model, ProblemType.nonlinear)
        model.solve()

        r = xy_of(mesh)[:, 0]
        m = (n + 1) / n
        exact = n / (n + 1) * (G / (2 * K)) ** (1 / n) * (R ** m - r ** m)
        errors.append(np.max(np.abs(model.sol['u'][:, 1] - exact)) / exact.max())
        q_exact = np.pi * n / (3 * n + 1) * (G / (2 * K)) ** (1 / n) * R ** (3 + 1 / n)
        q = model.physics.flowRate(edges(mesh, lambda x, y: y > L - 1e-9))
        assert abs(q / q_exact - 1) < 4 * errors[-1] + 1e-3, (q, q_exact)

    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    print(f"   power-law pipe n = {n}: errors {['%.1e' % e for e in errors]} rates {['%.2f' % r for r in rates]}")
    assert rates[-1] > 1.8 and errors[-1] < 2e-3


def test_axisymmetric_coupled_tangent():

    """u, p, T in (r, z): Carreau + Arrhenius, inertia, PSPG with the recovered stress (hoop terms), viscous heating
    (hoop strain) and heat convected by u with SUPG, against finite differences with tau frozen."""

    for element in (ElementType.quadrilateral, ElementType.triangle):
        mesh = rz_mesh((1.0, 1.0), (2, 2), element, r0=0.0)
        model = Model(name='tangent', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.2, Cp=1.5, k=0.3),
                      psc=[ff, ht], axisymmetric=True)
        fluid = model.physics.fluid
        fluid.setViscosityLaw(Arrhenius(CarreauYasuda(mu0=2.0, lam=0.7, n=0.4), E_R=800.0, T_ref=150.0))
        fluid.setInertia(True)
        fluid.setViscousHeating(True)
        model.physics.setVelocity('u')
        model.physics.setStabilization('SUPG')
        configure(model, ProblemType.nonlinear)

        layout = model.layout
        rng = np.random.default_rng(4)
        x = rng.normal(scale=0.5, size=layout.ndof)
        x[layout.dofs('T')[:, 0]] = 150.0 + 20.0 * rng.normal(size=layout.n_nodes)

        system = model._system()
        for term in system._terms:
            term.tau_state = x.copy()

        J, _ = system._assemble(x)
        FD = np.zeros((layout.ndof, layout.ndof))
        for kdof in range(layout.ndof):
            h = 1e-6 * max(1.0, abs(x[kdof]))
            e = np.zeros(layout.ndof)
            e[kdof] = h
            FD[:, kdof] = (system._assemble(x + e)[1] - system._assemble(x - e)[1]) / (2 * h)

        err = np.max(np.abs(J.toarray() - FD)) / np.max(np.abs(FD))
        assert err < 1e-7, (element, err)

    assert 'axisymmetric' in str(model)


if __name__ == "__main__":
    tests = [test_measures_and_hoop_operators, test_radial_conduction, test_hagen_poiseuille,
             test_uniaxial_extension_is_exact, test_power_law_pipe_flow, test_axisymmetric_coupled_tangent]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
