"""Planar 4:1 contraction of an Oldroyd-B fluid (beta = 1/9), as in Test/test_viscoelastic.py, solved on the same quadrilateral mesh in
2D and as a 3D slab of one layer of hex27 (symmetry on the z faces): the entrance correction C and the fields, and the cost of each.
With sizes (far, corner) the mesh of gmsh; the slab thickness W. Also a square-square 4:1 contraction (genuinely 3D, quarter domain).

    python Test/manual/contraction_3d.py slab [far corner W]
    python Test/manual/contraction_3d.py square [far corner]
"""

import os
import sys
import time

import numpy as np
import gmsh

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from heat_3d_scaling import PeakMemory
from Source.Pre_processing.Mesh import Mesh
from Source.core import Model
from Source.Material import material
from Source.Physics.FluidFlow import ff
from Source.Physics.Viscoelastic import ve
from Source.enums import StudyType, ProblemType

H, h, L = 4.0, 1.0, 20.0


def plane_contraction_mesh(dim, far, corner, W=0.5):

    """The 4:1 half contraction (axis y = 0, corner at (0, h)) meshed by gmsh in quadrilaterals; dim = 3 extrudes it by W in one layer
    of hexahedra. Groups: axis, outlet, walls, inlet (and front/back in 3D)."""

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    # every element split into quadrilaterals (2D) or hexahedra (3D: the layer becomes two, the faces are split as in 2D)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2 if dim == 3 else 1)
    try:
        geo = gmsh.model.geo
        points = [(-L, 0, far), (L, 0, far), (L, h, far), (0, h, corner), (0, H, 0.25 * far), (-L, H, far)]
        tags = [geo.addPoint(x, y, 0, size) for x, y, size in points]
        lines = [geo.addLine(tags[i], tags[(i + 1) % 6]) for i in range(6)]
        surface = geo.addPlaneSurface([geo.addCurveLoop(lines)])
        geo.mesh.setRecombine(2, surface)
        groups = {'axis': [0], 'outlet': [1], 'walls': [2, 3, 4], 'inlet': [5]}
        if dim == 3:
            out = geo.extrude([(2, surface)], 0, 0, W, numElements=[1], recombine=True)
            geo.synchronize()
            top, volume, laterals = out[0][1], out[1][1], [tag for _, tag in out[2:]]
            for name, which in groups.items():
                gmsh.model.addPhysicalGroup(2, [laterals[i] for i in which], name=name)
            gmsh.model.addPhysicalGroup(2, [surface, top], name='sides')
            gmsh.model.addPhysicalGroup(3, [volume], name='fluid')
            gmsh.model.mesh.generate(3)
        else:
            geo.synchronize()
            for name, which in groups.items():
                gmsh.model.addPhysicalGroup(1, [lines[i] for i in which], name=name)
            gmsh.model.addPhysicalGroup(2, [surface], name='fluid')
            gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
        gmsh.model.mesh.setOrder(2)
        return Mesh.from_gmsh_model()
    finally:
        gmsh.finalize()


def solve_contraction(mesh, dim, targets=(0.1, 0.5), solver_options=None):
    mu_s, mu_p = 1.0 / 9.0, 8.0 / 9.0
    model = Model(name='contraction', dim=dim, mesh=mesh, mat=material('X', miu=mu_s), psc=[ff, ve])
    flow, polymer = model.physics.fluid, model.physics.viscoelastic
    polymer.setModes([(mu_p, 0.1, 0.0)])
    polymer.setDEVSS(mu_p)
    group = mesh.boundary_faces if dim == 3 else mesh.boundary_edges
    inlet, outlet = group('inlet'), group('outlet')
    top = 1.5 * h / H
    zero = [0.0] * (dim - 1)
    flow.addBC_VelocityEdges(inlet, lambda *c: [top * (1 - (c[1] / H) ** 2)] + zero)
    flow.addBC_NoSlipEdges(group('walls'))
    flow.addBC_SymmetryEdges(group('axis'))
    if dim == 3:
        flow.addBC_SymmetryEdges(group('sides'))
    flow.addBC_OutletEdges(outlet, 0.0)
    polymer.addBC_OutletEdges(outlet)

    def gradient(*c):
        g = np.zeros((dim, dim))
        g[0, 1] = -2.0 * top * c[1] / H ** 2
        return g

    polymer.addBC_InflowEdges(inlet, gradient)
    W = (np.max([n.coor[2] for n in mesh.NL]) if dim == 3 else 1.0)
    results = []
    de = 0.1
    for target in targets:
        polymer.setWeissenberg(target / de)
        de = target
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear, **(solver_options or {}))
        start = time.perf_counter()
        model.solve()
        q = flow.flowRate(outlet) / W
        dp = np.asarray(model.sol['p'])[np.unique(np.ravel(inlet))].mean()
        C = (dp - 3 * q / H ** 3 * L - 3 * q / h ** 3 * L) / (6 * q / h ** 2)
        results.append((target, C, q, model.nonlinear_converged, time.perf_counter() - start))
    return model, results


def slab(far, corner, W):
    out = {}
    for dim in (2, 3):
        mesh = plane_contraction_mesh(dim, far, corner, W)
        with PeakMemory() as memory:
            model, results = solve_contraction(mesh, dim)
        out[dim] = results
        print(f"{dim}D: {mesh.getNoE()} elements, {model.layout.ndof} unknowns, peak +{(memory.peak - memory.base) / 2 ** 30:.2f} GB")
        for target, C, q, converged, seconds in results:
            print(f"   De = {target}: C = {C:.5f}, q = {q:.10f}, converged {converged}, {seconds:.1f} s")
    for (t2, C2, *_), (t3, C3, *_) in zip(out[2], out[3]):
        print(f"De = {t2}: C 3D - 2D = {C3 - C2:.2e}")


def square_contraction_mesh(cell=0.5, L1=6.0, L2=6.0, A=4.0, a=1.0):

    """A quarter of the square-square 4:1 contraction (upstream duct of half side A, downstream of half side a, symmetry planes y = 0 and
    z = 0, the contraction plane at x = 0): a box of hex27 without the cells outside the narrow duct downstream."""

    from Source.Pre_processing.Mesh import Node, Element
    from Source.enums import ElementType

    box = Mesh()
    box.generate3DMesh((L1 + L2, A, A), (int(round((L1 + L2) / cell)), int(round(A / cell)), int(round(A / cell))),
                       ElementType.hexahedron, order=2, origin=(-L1, 0.0, 0.0))
    kept = []
    for element in box.EL:
        c = np.mean([n.coor for n in element.nodes[:8]], axis=0)
        if c[0] < 0 or (c[1] < a and c[2] < a):
            kept.append(element)
    used = sorted({n.id for e in kept for n in e.nodes})
    index = {old: new for new, old in enumerate(used)}
    mesh = Mesh()
    mesh.PD, mesh.element_type, mesh.meshType = 3, 'hex27', ElementType.hexahedron
    mesh.NL = [Node(id=index[k], coor=box.NL[k].coor) for k in used]
    mesh.EL = [Element(id=i, nodes=[mesh.NL[index[n.id]] for n in e.nodes]) for i, e in enumerate(kept)]
    mesh.NoN, mesh.NoE = len(mesh.NL), len(mesh.EL)
    return mesh


def square(cell=0.5, L1=6.0, L2=6.0, A=4.0, a=1.0, De=0.1):

    """Newtonian (mu = 1) and Oldroyd-B (beta = 1/9, lam from De = lam U / a with U the mean downstream velocity) in the quarter square
    contraction, with Method auto (the block solver above 120 k unknowns): the excess pressure drop over the fully developed one of
    both ducts (the duct series), Newtonian against Oldroyd-B, and the cost. No reference values: a cost study and a consistency check."""

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from test_3d_flow import duct_series

    mesh = square_contraction_mesh(cell, L1, L2, A, a)
    Q = 1.0                                       # flow rate of the quarter
    q_unit_up = duct_series(0.0, 0.0, 2 * A, 2 * A, 1.0, 1.0)[1] / 4      # quarter flow rate for G = 1, mu = 1
    q_unit_down = duct_series(0.0, 0.0, 2 * a, 2 * a, 1.0, 1.0)[1] / 4
    G_up = Q / q_unit_up
    profile = lambda y, z: G_up * duct_series(y + A, z + A, 2 * A, 2 * A, 1.0, 1.0)[0]
    U = Q / (a * a)

    def inlet_gradient(x, y, z, h=1e-6):
        g = np.zeros((3, 3))
        g[0, 1] = (profile(min(y + h, A), z) - profile(max(y - h, 0.0), z)) / (min(y + h, A) - max(y - h, 0.0))
        g[0, 2] = (profile(y, min(z + h, A)) - profile(y, max(z - h, 0.0))) / (min(z + h, A) - max(z - h, 0.0))
        return g

    for viscoelastic in (False, True):
        psc = [ff, ve] if viscoelastic else ff
        mu_s = 1.0 / 9.0 if viscoelastic else 1.0
        model = Model(name='square', dim=3, mesh=mesh, mat=material('X', miu=mu_s), psc=psc)
        flow = model.physics.fluid if viscoelastic else model.physics
        inlet = mesh.boundary_faces(lambda x, y, z: x < -L1 + 1e-9)
        outlet = mesh.boundary_faces(lambda x, y, z: x > L2 - 1e-9)
        flow.addBC_VelocityEdges(inlet, lambda x, y, z: (float(profile(y, z)), 0.0, 0.0))
        flow.addBC_SymmetryEdges(mesh.boundary_faces(lambda x, y, z: y < 1e-9 and x > -L1 + 1e-9 and x < L2 - 1e-9))
        flow.addBC_SymmetryEdges(mesh.boundary_faces(lambda x, y, z: z < 1e-9 and x > -L1 + 1e-9 and x < L2 - 1e-9))
        flow.addBC_NoSlipEdges(mesh.boundary_faces(lambda x, y, z: (y > 1e-9 and z > 1e-9 and -L1 + 1e-9 < x < L2 - 1e-9)))
        flow.addBC_OutletEdges(outlet, 0.0)
        if viscoelastic:
            polymer = model.physics.viscoelastic
            polymer.setModes([(8.0 / 9.0, De * a / U, 0.0)])
            polymer.setDEVSS(8.0 / 9.0)
            polymer.addBC_OutletEdges(outlet)
            polymer.addBC_InflowEdges(inlet, inlet_gradient)
        model.solverConfiguration(Study=StudyType.steady_state, Type=ProblemType.nonlinear if viscoelastic else ProblemType.linear,
                                  Method='auto')
        with PeakMemory() as memory:
            start = time.perf_counter()
            model.solve()
            seconds = time.perf_counter() - start
        p_in = np.asarray(model.sol['p'])[np.unique(np.ravel(inlet))].mean()
        developed = Q / q_unit_up * L1 + Q / q_unit_down * L2                # mu_0 = 1 in both fluids
        excess = p_in - developed
        print(f"{'Oldroyd-B De = %.2f' % De if viscoelastic else 'Newtonian':20s}: {mesh.getNoE()} hex27, {model.layout.ndof} unknowns, "
              f"converged {model.nonlinear_converged}, Krylov {model.solver_iterations}, {seconds:.0f} s, peak "
              f"+{(memory.peak - memory.base) / 2 ** 30:.2f} GB; outlet flow {flow.flowRate(outlet):.6f}, excess pressure drop "
              f"{excess:.4f} ({excess / (G_up * 0 + Q / q_unit_down * a):.4f} in units of the downstream gradient times a)",
              flush=True)


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else 'slab'
    args = [float(a) for a in sys.argv[2:]]
    if kind == 'slab':
        far, corner, W = (args + [1.5, 0.1, 0.3][len(args):])[:3]
        slab(far, corner, W)
    elif kind == 'square':
        cell = args[0] if args else 0.5
        square(cell)
