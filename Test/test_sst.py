"""Stage 17, priority 2: k-omega SST with ln(omega) as unknown.

The reference for the implementation is an independent 1D solver of the same model, written here with omega as the
unknown of the equation and without any project code. The DNS is the reference for the model, which falls short of it:
both solvers agree on a centreline velocity 3 to 4 % low, on any mesh.

Run from the project root:  python Test/test_sst.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.Turbulence import sst, BETA1
from Source.enums import ElementType, StudyType, ProblemType, SolverType, DirectSolvers

jax.config.update("jax_enable_x64", True)

# Direct numerical simulation of the channel (Moser, Kim & Mansour 1999): centreline velocity in wall units
DNS_CENTRELINE = {180.0: 18.3, 395.0: 20.1, 590.0: 21.4}


def reichardt(y_plus):
    return np.log(1 + 0.41 * y_plus) / 0.41 + 7.8 * (1 - np.exp(-y_plus / 11) - y_plus / 11 * np.exp(-y_plus / 3))


def guess(y, re_tau):

    """Law of the wall for u, and omega and k from its near-wall and log-layer limits."""

    y_safe = np.maximum(y, 1e-6)
    y_plus = y * re_tau
    nu = 1.0 / re_tau
    omega = np.sqrt((6 * nu / (0.075 * y_safe ** 2)) ** 2 + (1 / (0.3 * 0.41 * y_safe)) ** 2)
    nut = 0.41 * y * (1 - 0.5 * y) * (1 - np.exp(-y_plus / 26)) ** 2 + 1e-4
    return reichardt(y_plus), nut * omega * (y > 0), np.log(omega)


def channel_model(re_tau, ny, beta, element=ElementType.quadrilateral):

    """Half channel 0 < y < 1 (wall at y = 0, symmetry at y = 1), x-independent, driven by a unit body force:
    tau_w = 1, u_tau = 1, nu = 1 / Re_tau, so u is u+ and y Re_tau is y+."""

    mesh = Mesh()
    mesh.generate2DMesh((0.1, 1.0), (2, ny), element)
    for node in mesh.NL:
        x, e = node.coor
        node.coor = (x, 1.0 + np.tanh(beta * (e - 1.0)) / np.tanh(beta))
    xy = np.array([n.coor for n in mesh.NL])

    model = Model(name='channel', dim=2, mesh=mesh, mat=material('X', miu=1.0 / re_tau, rho=1.0), psc=[ff, sst])
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

    return model, xy


def solve_channel(re_tau, ny, beta, uniform=False):

    model, xy = channel_model(re_tau, ny, beta)
    fluid, turbulence = model.physics.fluid, model.physics.turbulence

    if uniform:
        n = len(xy)
        fluid.var['u'].updateField(np.c_[10.0 * np.ones(n), np.zeros(n)])
        turbulence.var['k'].updateField(np.full(n, 1e-2))
        turbulence.var['ln_omega'].updateField(np.full(n, np.log(10.0)))
    else:
        u, k, g = guess(xy[:, 1], re_tau)
        fluid.var['u'].updateField(np.c_[u, np.zeros(len(u))])
        turbulence.var['k'].updateField(k)
        turbulence.var['ln_omega'].updateField(g)

    model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, Method=SolverType.direct,
                              Solver=DirectSolvers.PARDISO, globalization='pseudo-transient')
    model.solve()

    return model, xy


def profiles(model, xy, re_tau):
    column = np.abs(xy[:, 0]) < 1e-12
    order = np.argsort(xy[column, 1])
    return xy[column, 1][order], model.sol['u'][column, 0][order], model.sol['k'][column][order]


def reference_1d(re_tau, ny, beta):

    """SST in the fully developed half channel by finite differences on a tanh mesh, with the omega equation in its
    original form (the logarithm only keeps the iterates positive): (U_c+, U_b+, k_max)."""

    bs, a1 = 0.09, 0.31
    sk1, sw1, b1, al1 = 0.85, 0.5, 0.075, 5.0 / 9.0
    sk2, sw2, b2, al2 = 1.0, 0.856, 0.0828, 0.44

    nu = 1.0 / re_tau
    y = 1.0 + np.tanh(beta * (np.linspace(0, 1, ny + 1) - 1.0)) / np.tanh(beta)
    n = ny + 1
    dy, yj = jnp.asarray(np.diff(y)), jnp.asarray(y)
    wall_omega = 60.0 * nu / (b1 * y[1] ** 2)
    h = jnp.concatenate([dy[:1], 0.5 * (dy[:-1] + dy[1:]), dy[-1:] * 0.5])

    def grad(f):
        d = jnp.diff(f) / dy
        inner = (d[:-1] * dy[1:] + d[1:] * dy[:-1]) / (dy[:-1] + dy[1:])
        return jnp.concatenate([d[:1], inner, jnp.zeros(1)])                 # symmetry at y = 1

    def div(mu, f):
        flux = 0.5 * (mu[:-1] + mu[1:]) * jnp.diff(f) / dy
        return jnp.concatenate([jnp.zeros(1), (flux[1:] - flux[:-1]) / h[1:-1], -flux[-1:] / h[-1:]])

    def residual(z):
        u, k, w = z[:n], z[n:2 * n], jnp.exp(z[2 * n:])
        S = jnp.abs(grad(u)) + 1e-30
        gk, gw = grad(k), grad(w)
        kp = jnp.maximum(k, 0.0)
        d = jnp.maximum(yj, 0.5 * yj[1])
        sqrt_k = jnp.sqrt(kp + 1e-30)
        cross = gk * gw / w
        arg1 = jnp.minimum(jnp.maximum(sqrt_k / (bs * w * d), 500 * nu / (d ** 2 * w)),
                           4 * sw2 * kp / (jnp.maximum(2 * sw2 * cross, 1e-20) * d ** 2))
        f1 = jnp.tanh(arg1 ** 4)
        f2 = jnp.tanh(jnp.maximum(2 * sqrt_k / (bs * w * d), 500 * nu / (d ** 2 * w)) ** 2)
        nut = a1 * kp / jnp.maximum(a1 * w, S * f2)
        blend = lambda a, b: f1 * a + (1 - f1) * b
        Ru = div(nu + nut, u) + 1.0
        Rk = div(nu + blend(sk1, sk2) * nut, k) + jnp.minimum(nut * S ** 2, 10 * bs * kp * w) - bs * k * w
        Rw = (div(nu + blend(sw1, sw2) * nut, w) + blend(al1, al2) * S ** 2 - blend(b1, b2) * w ** 2
              + 2 * (1 - f1) * sw2 * cross) / w
        return jnp.concatenate([Ru.at[0].set(u[0]), Rk.at[0].set(k[0]), Rw.at[0].set(z[2 * n] - np.log(wall_omega))])

    u0, k0, g0 = guess(y, re_tau)
    g0[0] = np.log(wall_omega)
    z = np.concatenate([u0, k0, g0])
    F, tangent = jax.jit(residual), jax.jit(jax.jacfwd(residual))

    cfl, norm = 1.0, np.inf
    for _ in range(200):
        R, J = np.asarray(F(z)), np.asarray(tangent(z))
        new_norm = np.linalg.norm(R)
        cfl = min(2.0 * cfl, 1e12) if new_norm < norm else max(0.5 * cfl, 1e-3)
        norm = new_norm
        if norm < 1e-8:
            break
        D = np.abs(np.diag(J)) / cfl
        D[[0, n, 2 * n]] = 0.0                                               # wall rows: no pseudo-time
        z = z + np.linalg.solve(J - np.diag(D), -R)                          # R has the sign of the time derivative

    assert norm < 1e-8, norm
    return z[:n].max(), np.trapezoid(z[:n], y), z[n:2 * n].max()


# ----- tests -----

def test_tangent():

    """The tangent of u, p, k and ln(omega) coupled, with k of both signs, against finite differences."""

    for element in (ElementType.quadrilateral, ElementType.triangle):
        model, xy = channel_model(180.0, 3, 1.0, element)
        model.physics.fluid.setInertia(True)
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear)

        layout = model.layout
        system = model._system()
        rng = np.random.default_rng(9)
        x = rng.normal(scale=0.5, size=layout.ndof)
        x[layout.dofs('k')[:, 0]] = 0.2 * rng.normal(size=layout.n_nodes)
        x[layout.dofs('ln_omega')[:, 0]] = 2.0 + rng.normal(size=layout.n_nodes)
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


def test_channel_against_an_independent_solver_and_the_dns():

    for re_tau, ny, beta, ny_1d in ((180.0, 48, 2.5, 192), (395.0, 64, 3.0, 256)):
        model, xy = solve_channel(re_tau, ny, beta)
        y, u, k = profiles(model, xy, re_tau)
        y_plus = y * re_tau
        centre, bulk = u.max(), np.trapezoid(u, y)
        ref_centre, ref_bulk, ref_k = reference_1d(re_tau, ny_1d, beta)

        sublayer = (y_plus > 0) & (y_plus < 2)
        outer = y_plus > 5
        dev = np.max(np.abs(u[outer] / reichardt(y_plus[outer]) - 1))
        print(f"   Re_tau = {re_tau:.0f}: {model.newton_iterations} iterations; U_c+ {centre:.2f} (1D {ref_centre:.2f}, "
              f"DNS {DNS_CENTRELINE[re_tau]}), U_b+ {bulk:.2f} (1D {ref_bulk:.2f}), k max {k.max():.3f} (1D {ref_k:.3f}), "
              f"law of the wall {dev:.1%}")

        assert model.nonlinear_converged and model.newton_iterations <= 25
        assert abs(centre / ref_centre - 1) < 0.01 and abs(bulk / ref_bulk - 1) < 0.01 and abs(k.max() / ref_k - 1) < 0.01
        assert np.max(np.abs(u[sublayer] / y_plus[sublayer] - 1)) < 0.02
        assert dev < 0.07
        assert abs(centre / DNS_CENTRELINE[re_tau] - 1) < 0.05


def test_channel_does_not_depend_on_the_mesh_or_on_the_start():

    model, xy = solve_channel(180.0, 48, 2.5)
    centre = profiles(model, xy, 180.0)[1].max()

    fine, xy_fine = solve_channel(180.0, 96, 2.5)
    assert abs(profiles(fine, xy_fine, 180.0)[1].max() / centre - 1) < 0.005

    uniform, _ = solve_channel(180.0, 48, 2.5, uniform=True)
    print(f"   from a uniform state: {uniform.newton_iterations} iterations, against {model.newton_iterations}")
    assert uniform.nonlinear_converged
    assert np.max(np.abs(uniform.sol['u'] - model.sol['u'])) < 1e-5


def test_fields_and_wall_values():

    model, xy = solve_channel(180.0, 24, 2.5)
    wall = xy[:, 1] < 1e-12
    first = np.unique(xy[:, 1])[1]
    nu = 1.0 / 180.0

    assert np.allclose(model.sol['omega'][wall], 60.0 * nu / (BETA1 * first ** 2))
    assert np.allclose(model.sol['omega'], np.exp(model.sol['ln_omega']))
    assert np.all(model.sol['k'][wall] == 0.0)                                   # the positivity clip leaves walls alone
    assert np.all(model.sol['k'] >= 0.0) and np.all(model.sol['eddy_viscosity'] >= -1e-12)


def test_steady_state_is_kept_by_the_transient_solver():

    """From the steady solution, BDF2 stays there: the time discretization and the ln(omega) mass are consistent."""

    model, xy = solve_channel(180.0, 24, 2.5)
    u = model.sol['u'].copy()

    model.solverConfiguration(Study=StudyType.transient, Type=ProblemType.nonlinear, timeStep=0.5, totalTime=1.5,
                              timeScheme='BDF2')
    model.solve()
    assert np.max(np.abs(model.sol['u'][..., -1] - u)) < 1e-5


if __name__ == "__main__":
    tests = [test_tangent, test_channel_against_an_independent_solver_and_the_dns,
             test_channel_does_not_depend_on_the_mesh_or_on_the_start, test_fields_and_wall_values,
             test_steady_state_is_kept_by_the_transient_solver]
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
