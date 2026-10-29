"""Stage 16 of the fluids plan: RANS turbulence with Spalart-Allmaras (SA-neg) and the wall distance.

Run from the project root:  python Test/test_turbulence.py
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
from Source.Physics.Turbulence import sa, wall_distance, SAViscosity
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers

# Direct numerical simulation of the channel (Moser, Kim & Mansour 1999): centreline velocity in wall units
DNS_CENTRELINE = {180.0: 18.3, 395.0: 20.1}


def configure(model, study=StudyType.steady_state, **time):
    model.solverConfiguration(Study=study, Type=ProblemType.nonlinear, Method=SolverType.direct,
                              Solver=DirectSolvers.PARDISO, **time)


def reichardt(y_plus):

    """Reichardt's law of the wall (smooth fit of the viscous sublayer, buffer and log layers)."""

    return np.log(1 + 0.41 * y_plus) / 0.41 + 7.8 * (1 - np.exp(-y_plus / 11) - y_plus / 11 * np.exp(-y_plus / 3))


def channel(re_tau, ny, beta, element=ElementType.quadrilateral):

    """Half channel 0 < y < 1 (wall at y = 0, symmetry at y = 1), x-independent, driven by a unit body force:
    tau_w = 1, u_tau = 1, nu = 1 / Re_tau, so u is u+ and y Re_tau is y+."""

    mesh = Mesh()
    mesh.generate2DMesh((0.1, 1.0), (2, ny), element)
    for node in mesh.NL:                                          # tanh clustering at the wall
        x, e = node.coor
        node.coor = (x, 1.0 + np.tanh(beta * (e - 1.0)) / np.tanh(beta))
    xy = np.array([n.coor for n in mesh.NL])

    model = Model(name='channel', dim=2, mesh=mesh, mat=material('X', miu=1.0 / re_tau, rho=1.0), psc=[ff, sa])
    fluid, turbulence = model.physics.fluid, model.physics.turbulence
    fluid.setInertia(True)
    fluid.setBodyForce((1.0, 0.0))
    wall = mesh.boundary_edges(lambda x, y: y < 1e-9)
    fluid.addBC_NoSlipEdges(wall)
    fluid.addBC_SymmetryEdges(mesh.boundary_edges(lambda x, y: y > 1 - 1e-9))
    fluid.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x < 1e-9))
    fluid.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x > 0.1 - 1e-9))
    turbulence.setWalls(wall)
    turbulence.coupleFlow()

    y_plus = xy[:, 1] * re_tau                                    # rough initial guess
    fluid.var['u'].updateField(np.c_[reichardt(y_plus), np.zeros(len(xy))])
    turbulence.var['nut'].updateField(0.41 * xy[:, 1] * (1 - xy[:, 1]) + 1e-3)
    configure(model)
    model.solve()

    return model, xy


# ----- tests -----

def test_wall_distance():

    """Spalding's approximation is exact between parallel plates; around a cylinder it is close to the true distance
    near the wall, where the turbulence model needs it."""

    for element, tol in ((ElementType.quadrilateral, 1e-12), (ElementType.triangle, 1e-2)):
        mesh = Mesh()
        mesh.generate2DMesh((2.0, 1.0), (8, 10), element)
        xy = np.array([n.coor for n in mesh.NL])
        d = wall_distance(mesh, mesh.boundary_edges(lambda x, y: y < 1e-9 or y > 1 - 1e-9))
        exact = np.minimum(xy[:, 1], 1 - xy[:, 1])
        assert np.max(np.abs(d - exact)) < tol, (element, np.max(np.abs(d - exact)))
        assert np.max(np.abs(d - exact)[exact < 0.2]) < 1e-12 if element == ElementType.quadrilateral else True

    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        occ = gmsh.model.occ
        surface, _ = occ.cut([(2, occ.addDisk(0, 0, 0, 1.0, 1.0))], [(2, occ.addDisk(0, 0, 0, 0.2, 0.2))])
        occ.synchronize()
        inner = [t for _, t in gmsh.model.getBoundary(surface, oriented=False)
                 if np.hypot(*occ.getCenterOfMass(1, t)[:2]) < 1e-6 and occ.getMass(1, t) < 2]
        gmsh.model.addPhysicalGroup(1, inner, name='inner')
        gmsh.option.setNumber('Mesh.MeshSizeMax', 0.04)
        gmsh.model.mesh.generate(2)
        disk = Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()
    xy = np.array([n.coor for n in disk.NL])
    d = wall_distance(disk, disk.boundary_edges('inner'))
    true = np.hypot(xy[:, 0], xy[:, 1]) - 0.2
    # Spalding's approximation overestimates the distance from a convex wall by about d / (2 R): small next to the
    # wall, where the turbulence model uses it, and independent of the mesh
    for a, b in ((0.02, 0.05), (0.05, 0.1)):
        band = (true > a) & (true <= b)
        assert np.max(np.abs(d[band] / true[band] - 1)) < b / (2 * 0.2) + 0.02


def test_turbulent_channel():

    """Re_tau = 180 and 395: viscous sublayer u+ = y+, law of the wall (Reichardt) in the buffer and log layers, and
    the centreline velocity of the DNS of Moser, Kim & Mansour (1999)."""

    for re_tau, ny, beta in ((180.0, 48, 2.5), (395.0, 64, 3.0)):
        model, xy = channel(re_tau, ny, beta)
        column = np.abs(xy[:, 0]) < 1e-12
        order = np.argsort(xy[column, 1])
        y_plus, u_plus = xy[column, 1][order] * re_tau, model.sol['u'][column, 0][order]

        sublayer = (y_plus > 0) & (y_plus < 2)
        outer = y_plus > 5
        dev = np.max(np.abs(u_plus[outer] / reichardt(y_plus[outer]) - 1))
        centre = u_plus.max()
        print(f"   Re_tau = {re_tau:.0f}: U_c+ {centre:.2f} (DNS {DNS_CENTRELINE[re_tau]}), max deviation from the law "
              f"of the wall {dev:.1%}, Newton {model.newton_iterations}")
        assert np.max(np.abs(u_plus[sublayer] / y_plus[sublayer] - 1)) < 0.02
        assert dev < 0.07 and abs(centre / DNS_CENTRELINE[re_tau] - 1) < 0.02
        assert np.all(model.sol['eddy_viscosity'] >= 0.0) and model.newton_iterations <= 25


def test_laminar_limit():

    """Without free-stream turbulence (nu~ = 0 everywhere) SA produces none: the flow stays laminar Poiseuille."""

    mesh = Mesh()
    mesh.generate2DMesh((1.0, 1.0), (4, 8), ElementType.triangle)
    xy = np.array([n.coor for n in mesh.NL])
    model = Model(name='laminar', dim=2, mesh=mesh, mat=material('X', miu=0.1, rho=1.0), psc=[ff, sa])
    fluid, turbulence = model.physics.fluid, model.physics.turbulence
    fluid.setInertia(True)
    walls = mesh.boundary_edges(lambda x, y: y < 1e-9 or y > 1 - 1e-9)
    inlet = mesh.boundary_edges(lambda x, y: x < 1e-9)
    fluid.addBC_VelocityEdges(inlet, lambda x, y: (4 * y * (1 - y), 0.0))
    fluid.addBC_NoSlipEdges(walls)
    fluid.addBC_OutletEdges(mesh.boundary_edges(lambda x, y: x > 1 - 1e-9))
    turbulence.setWalls(walls)
    turbulence.addBC_TurbulenceEdges(inlet, 0.0)
    turbulence.coupleFlow()
    turbulence.initField('nut', 0.0)
    configure(model)
    model.solve()

    assert np.max(np.abs(model.sol['nut'])) < 1e-12
    assert np.max(np.abs(model.sol['u'][:, 0] - 4 * xy[:, 1] * (1 - xy[:, 1]))) < 2e-2


def test_tangent_of_the_coupled_model():

    """u, p, nu~ (with nu~ of both signs: SA-neg branches) and the eddy viscosity in the flow, against finite
    differences with tau frozen."""

    for element in (ElementType.quadrilateral, ElementType.triangle):
        mesh = Mesh()
        mesh.generate2DMesh((1.0, 1.0), (2, 2), element)
        model = Model(name='tangent', dim=2, mesh=mesh, mat=material('X', miu=0.05, rho=1.0), psc=[ff, sa])
        model.physics.fluid.setInertia(True)
        model.physics.turbulence.setWalls(mesh.boundary_edges(lambda x, y: y < 1e-9))
        model.physics.turbulence.coupleFlow()
        configure(model)

        layout = model.layout
        system = model._system()
        rng = np.random.default_rng(9)
        x = rng.normal(scale=0.5, size=layout.ndof)
        x[layout.dofs('nut')[:, 0]] = 0.05 * rng.normal(size=layout.n_nodes)       # both signs
        x[layout.dofs('wall_distance')[:, 0]] = model.physics.turbulence.var['wall_distance'].values
        free = ~system.is_dirichlet
        for term in system._terms:
            if hasattr(term, 'tau_state'):
                term.tau_state = x.copy()

        J, _ = system._assemble(x)
        FD = np.zeros((layout.ndof, layout.ndof))
        for k in np.nonzero(free)[0]:
            e = np.zeros(layout.ndof)
            e[k] = 1e-7
            FD[:, k] = (system._assemble(x + e)[1] - system._assemble(x - e)[1]) / 2e-7
        err = np.max(np.abs(J.toarray()[:, free] - FD[:, free])) / np.max(np.abs(FD))
        assert err < 1e-6, (element, err)


def test_eddy_viscosity_law():
    law = SAViscosity(mu=2e-3, rho=1.2)
    nut = np.array([-1e-3, 0.0, 1e-5, 1e-3, 1.0])
    mu = law.viscosity(np.ones_like(nut), nut)
    chi = np.maximum(nut, 0) / law.nu
    assert np.allclose(mu, 2e-3 + 1.2 * np.maximum(nut, 0) * chi ** 3 / (chi ** 3 + 7.1 ** 3))
    assert np.isclose(mu[-1], 2e-3 + 1.2 * 1.0, rtol=1e-5)                    # f_v1 -> 1 far from the wall


if __name__ == "__main__":
    tests = [test_wall_distance, test_turbulent_channel, test_laminar_limit, test_tangent_of_the_coupled_model,
             test_eddy_viscosity_law]
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
