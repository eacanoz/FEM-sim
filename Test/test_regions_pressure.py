"""Stage 17, priority 6: Cross-WLF with pressure dependence (D3) and materials by region.

Run from the project root:  python Test/test_regions_pressure.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import gmsh
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.HeatTransfer import ht
from Source.Physics.Rheology import CrossWLF
from Source.enums import ElementType, StudyType, ProblemType


def xy_of(mesh):
    return np.array([n.coor[:2] for n in mesh.NL], dtype=float)


def steady(model, kind=ProblemType.nonlinear):
    model.solverConfiguration(Study=StudyType.steady_state, Type=kind)
    model.solve()


def layered_mesh(order=1, interface=0.5):

    """Unit square split at y = interface into the regions 'lower' and 'upper', with triangles."""

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        occ = gmsh.model.occ
        lower = occ.addRectangle(0, 0, 0, 1, interface)
        upper = occ.addRectangle(0, interface, 0, 1, 1 - interface)
        occ.fragment([(2, lower)], [(2, upper)])
        occ.synchronize()
        for _, tag in gmsh.model.getEntities(2):
            _, y, _ = occ.getCenterOfMass(2, tag)
            gmsh.model.addPhysicalGroup(2, [tag], name='lower' if y < interface else 'upper')
        gmsh.option.setNumber('Mesh.MeshSizeMax', 0.1)
        gmsh.model.mesh.generate(2)
        if order == 2:
            gmsh.model.mesh.setOrder(2)
        return Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()


# ----- Cross-WLF with pressure -----

def law(D3=1e-7):
    return CrossWLF(n=0.3, tau_star=3e4, D1=1e13, D2=263.15, A1=30.0, A2=51.6, D3=D3)


def test_law_values():

    """T* = D2 + D3 p, A2 = A2~ + D3 p: the pressure acts as a shift of the temperature, T -> T - D3 p."""

    melt = law()
    assert melt.uses_pressure and not law(0.0).uses_pressure
    T, p = 230.0, 5e7
    assert np.isclose(float(melt.zero_shear(T, 0.0)), float(law(0.0).zero_shear(T)))

    d = T + 273.15 - 263.15
    expected = 1e13 * np.exp(-30.0 * (d - 1e-7 * p) / (51.6 + d))
    assert np.isclose(float(melt.zero_shear(T, p)), expected)
    assert float(melt.zero_shear(T, p)) > float(melt.zero_shear(T, 0.0))         # pressure raises the viscosity


def test_flow_reads_the_pressure():

    mesh = Mesh()
    mesh.generate2DMesh((1.0, 1.0), (2, 2), ElementType.quadrilateral, order=2)
    model = Model(name='m', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.0, k=1.0, Cp=1.0), psc=[ff, ht])
    model.physics.fluid.setViscosityLaw(law())
    assert model.physics.fluid.viscosity_fields == ['T', 'p']
    model.physics.fluid.setViscosityLaw(law(0.0))
    assert model.physics.fluid.viscosity_fields == ['T']


def test_tangent_with_pressure():

    """Both element families and both velocity-pressure spaces, against finite differences with tau frozen."""

    for element, order in ((ElementType.quadrilateral, 1), (ElementType.triangle, 1), (ElementType.quadrilateral, 2), (ElementType.triangle, 2)):
        mesh = Mesh()
        mesh.generate2DMesh((1.0, 1.0), (2, 2), element, order=order)
        model = Model(name='tangent', dim=2, mesh=mesh, mat=material('X', miu=1.0, rho=1.0, k=1.0, Cp=1.0), psc=[ff, ht])
        model.physics.fluid.setViscosityLaw(CrossWLF(n=0.5, tau_star=1.0, D1=1e6, D2=263.15, A1=20.0, A2=51.6, D3=0.2))
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)
        system = model._system()
        layout = system.layout
        rng = np.random.default_rng(6)
        x = rng.normal(scale=0.5, size=layout.ndof)
        x[layout.dofs('T')[:, 0]] = 230.0 + rng.normal(size=layout.n_nodes)
        for term in system._terms:
            if hasattr(term, 'tau_state'):
                term.tau_state = x.copy()
        J, _ = system._assemble(x)
        FD = np.zeros((layout.ndof, layout.ndof))
        for k in range(layout.ndof):
            e = np.zeros(layout.ndof)
            e[k] = 1e-6
            FD[:, k] = (system._assemble(x + e)[1] - system._assemble(x - e)[1]) / 2e-6
        err = np.max(np.abs(J.toarray() - FD)) / np.max(np.abs(FD))
        assert err < 1e-6, (element, order, err)


def test_long_slit_against_lubrication():

    """A Newtonian Cross-WLF is mu0 exp(b p), b = A1 D3 / (A2 + T - D2). A long slit with the flow rate q (half gap H) has
    dp/dx = -3 q mu / H^3, so exp(-b p(0)) = 1 - b 3 q mu0 L / H^3: well above the constant viscosity value."""

    H, L, q, T = 1.0, 60.0, 1.0, 230.0
    melt = CrossWLF(n=0.5, tau_star=1e12, D1=1e6, D2=263.15, A1=20.0, A2=51.6, D3=0.6)
    mu0 = float(melt.zero_shear(T, 0.0))
    b = melt.A1 * melt.D3 / (melt.A2 + T + 273.15 - melt.D2)
    constant = 3 * q * mu0 * L / H ** 3
    assert 0.3 < b * constant < 0.8

    errors = {}
    for element, order, label in ((ElementType.quadrilateral, 2, 'Q2/Q1'), (ElementType.quadrilateral, 1, 'Q4')):
        mesh = Mesh()
        mesh.generate2DMesh((L, H), (120, 12), element, order=order)
        model = Model(name='slit', dim=2, mesh=mesh, mat=material('X', miu=mu0, rho=1.0, k=1.0, Cp=1.0), psc=[ff, ht])
        fluid, heat = model.physics.fluid, model.physics.heat
        inlet = mesh.boundary_edges(lambda x, y: x < 1e-9)
        fluid.addBC_VelocityEdges(inlet, lambda x, y: (1.5 * q / H * (1 - (y / H) ** 2), 0.0))
        fluid.addBC_SymmetryEdges(mesh.boundary_edges(lambda x, y: y < 1e-9))
        fluid.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y > H - 1e-9))
        fluid.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x > L - 1e-9), 0.0)
        heat.addBC_TemperatureEdges(mesh.boundary_edges(None), T)
        model.physics.initField('T', T)
        steady(model)                                                            # constant viscosity
        fluid.setViscosityLaw(melt)
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, globalization='pseudo-transient')
        model.solve()

        xy = xy_of(mesh)
        axis = xy[:, 1] < 1e-9
        order_x = np.argsort(xy[axis, 0])
        x_axis, p_axis = xy[axis, 0][order_x], model.sol['p'][axis][order_x]
        exact = -np.log(1 - b * 3 * q * mu0 * (L - x_axis) / H ** 3) / b
        middle = (x_axis > 0.2 * L) & (x_axis < 0.9 * L)
        errors[label] = np.max(np.abs(p_axis[middle] / exact[middle] - 1))
        assert model.nonlinear_converged
    print(f"   exponent b dp {b * constant:.2f}; largest relative error of p against lubrication {errors}")
    assert max(errors.values()) < 0.01


# ----- regions -----

def test_regions_are_element_indices():

    mesh = layered_mesh()
    lower, upper = np.asarray(mesh.regions['lower']), np.asarray(mesh.regions['upper'])
    assert len(lower) + len(upper) == mesh.getNoE() and len(np.intersect1d(lower, upper)) == 0

    centres = np.array([np.mean([n.coor[1] for n in mesh.EL[e].nodes]) for e in range(mesh.getNoE())])
    assert np.all(centres[lower] < 0.5) and np.all(centres[upper] > 0.5)

    model = Model(name='m', dim=2, mesh=mesh, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
    field = model.region_field({'lower': 2.0, 'upper': 5.0})
    assert np.all(field[lower] == 2.0) and np.all(field[upper] == 5.0)

    for bad in ({'lower': 2.0}, {'lower': 2.0, 'upper': 5.0, 'other': 1.0}):
        try:
            model.region_field(bad)
        except (KeyError, ValueError):
            pass
        else:
            raise AssertionError("regions must be known and cover the mesh")
    assert np.all(model.region_field({'lower': 2.0}, default=7.0)[upper] == 7.0)


def test_composite_wall_is_exact():

    """Two conductivities in series: T is linear by pieces, with a kink exactly at the interface."""

    k1, k2 = 1.0, 4.0
    for order in (1, 2):
        mesh = layered_mesh(order)
        model = Model(name='wall', dim=2, mesh=mesh, mat=material('X', k=1.0, rho=1.0, Cp=1.0), psc=ht)
        heat = model.physics
        heat.K_const = model.region_field({'lower': k1, 'upper': k2})
        heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: y < 1e-9), 0.0)
        heat.addBC_TemperatureEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), 1.0)
        steady(model, ProblemType.linear)
        y = xy_of(mesh)[:, 1]
        flux = 1.0 / (0.5 / k1 + 0.5 / k2)
        exact = np.where(y < 0.5, flux * y / k1, flux * 0.5 / k1 + flux * (y - 0.5) / k2)
        assert np.max(np.abs(model.sol['T'] - exact)) < 1e-11, order


def test_two_layer_couette_is_exact():

    """Two viscosities in a shear flow: the same shear stress, u linear by pieces. Taylor-Hood needs no stabilization that
    could feel the jump, so the solution and a zero pressure are exact."""

    mu1, mu2 = 1.0, 9.0
    mesh = layered_mesh(order=2)
    model = Model(name='couette', dim=2, mesh=mesh, mat=material('X', miu=1.0), psc=ff)
    flow = model.physics
    flow.setViscosity(model.region_field({'lower': mu1, 'upper': mu2}))
    flow.addBC_NoSlipEdges(mesh.boundary_edges(lambda x, y: y < 1e-9))
    flow.addBC_VelocityEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9), (1.0, 0.0))
    flow.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x < 1e-9))
    flow.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x > 1 - 1e-9))
    steady(model, ProblemType.linear)
    y = xy_of(mesh)[:, 1]
    tau = 1.0 / (0.5 / mu1 + 0.5 / mu2)
    exact = np.where(y < 0.5, tau * y / mu1, tau * 0.5 / mu1 + tau * (y - 0.5) / mu2)
    assert np.max(np.abs(model.sol['u'][:, 0] - exact)) < 1e-11
    assert np.max(np.abs(model.sol['p'])) < 1e-10


if __name__ == "__main__":
    tests = [test_law_values, test_flow_reads_the_pressure, test_tangent_with_pressure, test_long_slit_against_lubrication,
             test_regions_are_element_indices, test_composite_wall_is_exact, test_two_layer_couette_is_exact]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
