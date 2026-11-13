# A thin-walled solid (a CAD part) as the two skins of its walls, for the Hele-Shaw filling (dual domain): the surface of the solid is
# meshed, its faces are classified into skins (the two sides of each wall) and edges (the narrow faces across the thickness: the rim of
# a plate, the top of a rib), and the thickness is measured by rays through the solid. Each skin carries half of its wall (from its mold
# wall to the midplane), so the flow solves on the skins with half the measure (HeleShawFilling.from_dual_domain), and connectors join
# every skin node to the point of the opposite skin across the wall, so that the front advances on both together.
#
#   - Orientation: the triangles of each CAD face are turned to the outward normal (parity of the crossings of a ray from just off the
#     face, voted over a few triangles of the face).
#   - Thickness: from the centroid of every triangle a ray inward (-n), the nearest crossing of another triangle. A face is a skin if
#     most of its rays end on a face facing it (normals antiparallel within `angle`) and the median length is below its width 2 A / P
#     (A its area, P its perimeter: about the short side of a strip); the rim of a plate (width = the thickness, its rays cross the whole
#     plate) and the top of a rib are edges. A skin triangle keeps its ray if it ends on a skin facing it and is not longer than twice the
#     median of its face (the outer wall of a box next to the base would cross the whole box); the others (the base under a rib, where the
#     ray runs up the rib) take the mean thickness of their neighbours.
#   - Connectors: from every skin node a ray inward along its normal (mean of its skin triangles); where it ends on a skin facing it, at
#     most 1.5 times the local thickness away, the node is linked to that point of the opposite triangle (its three nodes with the
#     barycentric weights); nodes where it misses (the rim, corners) are linked to the nearest point of a facing skin triangle within that
#     distance. A link is a row v = e_i - sum w_k e_k: HeleShawFilling penalizes v . p with a conductance (the flow crosses the wall in
#     proportion to p_i minus the pressure of the opposite point). Not a graph Laplacian i-k: that would conduct between the three nodes
#     of the triangle through i, a short cut along the wall that made the gate pressure of a strip 85 % low.
#
# Exact for plates of constant thickness; a wedge is measured along the normal of each skin, h cos(angle) on the inclined side; a wall
# thicker than it is wide is not thin and gets no sensible thickness. Linear triangles only.

import numpy as np
import scipy.sparse as sp

from Source.Pre_processing.Mesh import Mesh


def _ray_hits(origins, directions, A, B, C, t_min, eps=1e-9, block=None):

    """Nearest crossing t > t_min of each ray (origin, unit direction) with the triangles A B C (Moller-Trumbore, edges included to eps
    in barycentric coordinates). Returns t (inf if none), the triangle (-1) and its barycentric (u, v) of B and C."""

    n, m = len(origins), len(A)
    E1, E2 = B - A, C - A
    t_best = np.full(n, np.inf)
    tri = np.full(n, -1)
    uv = np.zeros((n, 2))
    block = block or max(1, int(4e6 // max(m, 1)))
    t_min = np.broadcast_to(np.asarray(t_min, dtype=float), (n,))
    for s in range(0, n, block):
        O, D = origins[s:s + block], directions[s:s + block]
        P = np.cross(D[:, None, :], E2[None])                          # (b, m, 3)
        det = np.einsum('bmk,mk->bm', P, E1)
        ok = np.abs(det) > 1e-300
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        T = O[:, None, :] - A[None]
        u = np.einsum('bmk,bmk->bm', T, P) * inv
        Q = np.cross(T, E1[None])
        v = np.einsum('bk,bmk->bm', D, Q) * inv
        t = np.einsum('mk,bmk->bm', E2, Q) * inv
        hit = ok & (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps) & (t > t_min[s:s + block, None])
        t = np.where(hit, t, np.inf)
        k = np.argmin(t, axis=1)
        rows = np.arange(len(O))
        t_best[s:s + block] = t[rows, k]
        found = np.isfinite(t[rows, k])
        tri[s:s + block] = np.where(found, k, -1)
        uv[s:s + block] = np.where(found[:, None], np.stack([u[rows, k], v[rows, k]], axis=1), 0.0)
    return t_best, tri, uv


def _closest_on_triangle(p, A, B, C):

    """Closest point of each triangle A B C (m, 3) to the point p (3,) (Ericson, Real-Time Collision Detection 5.1.5): distance and the
    barycentric weights (m, 3) of A, B, C."""

    ab, ac, ap = B - A, C - A, p - A
    d1, d2 = np.einsum('mk,mk->m', ab, ap), np.einsum('mk,mk->m', ac, ap)
    bp, cp = p - B, p - C
    d3, d4 = np.einsum('mk,mk->m', ab, bp), np.einsum('mk,mk->m', ac, bp)
    d5, d6 = np.einsum('mk,mk->m', ab, cp), np.einsum('mk,mk->m', ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    w = np.zeros((len(A), 3))
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = va + vb + vc
        v, ww = vb / denom, vc / denom
        w[:] = np.stack([1 - v - ww, v, ww], axis=1)                       # inside the face
        edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        t = d1 / (d1 - d3)
        w[edge_ab] = np.stack([1 - t, t, 0 * t], axis=1)[edge_ab]
        edge_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        t = d2 / (d2 - d6)
        w[edge_ac] = np.stack([1 - t, 0 * t, t], axis=1)[edge_ac]
        edge_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        w[edge_bc] = np.stack([0 * t, 1 - t, t], axis=1)[edge_bc]
    for mask, col in (((d1 <= 0) & (d2 <= 0), 0), ((d3 >= 0) & (d4 <= d3), 1), ((d6 >= 0) & (d5 <= d6), 2)):
        w[mask] = np.eye(3)[col]
    q = w[:, :1] * A + w[:, 1:2] * B + w[:, 2:] * C
    return np.linalg.norm(q - p, axis=1), w


class DualDomain:

    """The skins of a thin-walled solid with their thickness and connectors. Build it from the gmsh model (from_gmsh_model) or a STEP
    file (from_step). Attributes:
        mesh        the skin triangles (Mesh, surface in space, outward orientation), regions 'face_<tag>' per CAD face
        thickness   (nel,) the wall thickness of each skin triangle [model units]
        measured    (nel,) True where the thickness is its own ray, False where it was taken from the neighbours
        face        (nel,) the CAD face of each skin triangle;  skins, edges: the CAD faces of each kind
        connectors  sparse (m, n): one row per link, e_i - sum w_k e_k (node i of a skin, the point of the opposite skin)
        link_source (m,) the node i of each link;  link_nodes, link_weights (m, 3): the opposite triangle and its weights
        unjoined    the skin nodes left without a link"""

    def __init__(self, xyz, triangles, faces, angle=30.0):

        xyz = np.asarray(xyz, dtype=float)
        triangles = np.asarray(triangles, dtype=int).copy()
        faces = np.asarray(faces, dtype=int)
        self.angle = float(angle)
        cos_tol = np.cos(np.radians(angle))
        size = np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))

        A, B, C = (xyz[triangles[:, k]] for k in range(3))
        normal = np.cross(B - A, C - A)
        area = 0.5 * np.linalg.norm(normal, axis=1)
        unit = normal / (2 * area[:, None])
        centre = (A + B + C) / 3

        # outward orientation, per CAD face: an odd number of crossings from just outside means the normal points in
        rng = np.random.default_rng(0)
        tilt = np.array([0.1234, -0.0567, 0.0891])
        for f in np.unique(faces):
            idx = np.flatnonzero(faces == f)
            pick = idx[rng.choice(len(idx), size=min(5, len(idx)), replace=False)]
            d = unit[pick] + tilt
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            crossings = []
            for i, di in zip(pick, d):
                origin = centre[i] + 1e-6 * size * unit[i]
                count = self._count(origin, di, A, B, C)
                crossings.append(count % 2)
            if np.mean(crossings) > 0.5:
                triangles[idx] = triangles[idx][:, [0, 2, 1]]
                unit[idx] = -unit[idx]
        A, B, C = (xyz[triangles[:, k]] for k in range(3))

        # rays inward from every centroid
        t_ray, hit, _ = _ray_hits(centre, -unit, A, B, C, t_min=1e-9 * size)
        facing = (hit >= 0) & (np.einsum('ij,ij->i', unit, unit[np.maximum(hit, 0)]) < -cos_tol)

        # skins and edges, per CAD face
        self.skins, self.edges = [], []
        skin_face = {}
        median = {}
        for f in np.unique(faces):
            idx = np.flatnonzero(faces == f)
            width = 2 * area[idx].sum() / self._perimeter(triangles[idx], xyz)
            good = facing[idx]
            med = np.median(t_ray[idx][good]) if np.any(good) else np.inf
            median[f] = med
            skin = np.mean(good) >= 0.5 and med < width
            skin_face[f] = skin
            (self.skins if skin else self.edges).append(int(f))
        is_skin = np.array([skin_face[f] for f in faces])

        # thickness of the skin triangles: their own ray, or the mean of their neighbours
        keep = np.flatnonzero(is_skin)
        own = facing & is_skin & is_skin[np.maximum(hit, 0)] & (t_ray <= 2 * np.array([median[f] for f in faces]))
        thickness = np.where(own, t_ray, np.nan)[keep]
        measured = own[keep]
        tris = triangles[keep]
        edge_neighbours = self._edge_neighbours(tris)
        for _ in range(200):
            missing = np.flatnonzero(np.isnan(thickness))
            if not len(missing):
                break
            for e in missing:
                values = thickness[edge_neighbours[e]]
                values = values[~np.isnan(values)]
                if len(values):
                    thickness[e] = values.mean()
        if np.any(np.isnan(thickness)):
            thickness[np.isnan(thickness)] = np.nanmedian(thickness)

        # the skin mesh, renumbered
        used = np.unique(tris)
        renumber = np.full(len(xyz), -1)
        renumber[used] = np.arange(len(used))
        conn = renumber[tris]
        self.xyz = xyz[used]
        self.thickness = thickness
        self.measured = measured
        self.face = faces[keep]
        regions = {f'face_{f}': np.flatnonzero(self.face == f) for f in self.skins}
        self.mesh = Mesh.from_arrays(self.xyz, conn, 'tri3', regions)
        self.conn = conn
        self.link_source, self.link_nodes, self.link_weights, self.unjoined = self._join(cos_tol)
        m = len(self.link_source)
        rows = np.repeat(np.arange(m), 4)
        cols = np.column_stack([self.link_source, self.link_nodes]).ravel()
        vals = np.column_stack([np.ones(m), -self.link_weights]).ravel()
        self.connectors = sp.csr_matrix((vals, (rows, cols)), shape=(m, len(self.xyz)))

    @staticmethod
    def _count(origin, direction, A, B, C):
        E1, E2 = B - A, C - A
        P = np.cross(direction, E2)
        det = np.einsum('mk,mk->m', P, E1)
        ok = np.abs(det) > 1e-300
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        T = origin - A
        u = np.einsum('mk,mk->m', T, P) * inv
        Q = np.cross(T, E1)
        v = (Q @ direction) * inv
        t = np.einsum('mk,mk->m', E2, Q) * inv
        return int(np.sum(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)))

    @staticmethod
    def _perimeter(tris, xyz):
        edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        outer = unique[counts == 1]
        return float(np.linalg.norm(xyz[outer[:, 0]] - xyz[outer[:, 1]], axis=1).sum())

    @staticmethod
    def _edge_neighbours(tris):
        edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
        owner = np.tile(np.arange(len(tris)), 3)
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        edges, owner = edges[order], owner[order]
        same = np.all(edges[1:] == edges[:-1], axis=1)
        neighbours = [[] for _ in range(len(tris))]
        for k in np.flatnonzero(same):
            a, b = owner[k], owner[k + 1]
            neighbours[a].append(b)
            neighbours[b].append(a)
        return [np.array(n, dtype=int) for n in neighbours]

    def _join(self, cos_tol):

        """The links: source nodes (m,), opposite triangles (m, 3) and weights (m, 3), and the nodes left without one."""

        xyz, conn = self.xyz, self.conn
        n = len(xyz)
        A, B, C = (xyz[conn[:, k]] for k in range(3))
        normal = np.cross(B - A, C - A)
        area = 0.5 * np.linalg.norm(normal, axis=1)
        unit = normal / (2 * area[:, None])
        node_normal = np.zeros((n, 3))
        node_h = np.zeros(n)
        weight = np.zeros(n)
        for k in range(3):
            np.add.at(node_normal, conn[:, k], area[:, None] * unit)
            np.add.at(node_h, conn[:, k], area * self.thickness)
            np.add.at(weight, conn[:, k], area)
        node_normal /= np.maximum(np.linalg.norm(node_normal, axis=1, keepdims=True), 1e-300)
        node_h /= weight
        reach = 1.5 * node_h
        t, tri, uv = _ray_hits(xyz, -node_normal, A, B, C, t_min=1e-6 * node_h)
        ok = (tri >= 0) & (t <= reach)
        ok &= np.einsum('ij,ij->i', node_normal, unit[np.maximum(tri, 0)]) < -cos_tol
        source, nodes, weights = [], [], []
        for i in np.flatnonzero(ok):
            source.append(i); nodes.append(conn[tri[i]]); weights.append((1 - uv[i, 0] - uv[i, 1], uv[i, 0], uv[i, 1]))
        unjoined = []
        for i in np.flatnonzero(~ok):
            facing = np.einsum('k,mk->m', node_normal[i], unit) < -cos_tol
            facing &= ~np.any(conn == i, axis=1)
            cand = np.flatnonzero(facing & (np.linalg.norm((A + B + C) / 3 - xyz[i], axis=1) < reach[i] + np.sqrt(2 * area.max() * 2)))
            if not len(cand):
                unjoined.append(i)
                continue
            d, w = _closest_on_triangle(xyz[i], A[cand], B[cand], C[cand])
            k = int(np.argmin(d))
            if d[k] > reach[i]:
                unjoined.append(i)
                continue
            source.append(i); nodes.append(conn[cand[k]]); weights.append(w[k])
        return (np.array(source, dtype=int), np.array(nodes, dtype=int).reshape(-1, 3), np.array(weights, dtype=float).reshape(-1, 3),
                np.array(unjoined, dtype=int))

    # ----- builders -----

    @classmethod
    def from_gmsh_model(cls, volumes=None, angle=30.0, scale=1.0):

        """From the gmsh model that is open, with the surface of its volumes meshed in triangles (gmsh.model.mesh.generate(2)).
        volumes: the tags of the volumes (default: all). scale multiplies the coordinates (1e-3: a model in mm to metres)."""

        import gmsh

        if volumes is None:
            volumes = [tag for _, tag in gmsh.model.getEntities(3)]
        surfaces = sorted({abs(tag) for _, tag in gmsh.model.getBoundary([(3, v) for v in volumes], oriented=False)})
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        index = {int(t): i for i, t in enumerate(tags)}
        xyz = np.asarray(xyz).reshape(-1, 3) * float(scale)
        triangles, faces = [], []
        for s in surfaces:
            types, _, nodes = gmsh.model.mesh.getElements(2, s)
            for et, nl in zip(types, nodes):
                if et != 2:
                    raise ValueError("La doble superficie usa triangulos lineales (gmsh tipo 2).")
                tri = np.array([index[int(t)] for t in nl], dtype=int).reshape(-1, 3)
                triangles.append(tri)
                faces.append(np.full(len(tri), s))
        return cls(xyz, np.concatenate(triangles), np.concatenate(faces), angle)

    @classmethod
    def from_step(cls, path, size, size_min=None, angle=30.0, curvature=16, scale=1.0):

        """Reads a STEP (or BREP/IGES) solid with gmsh (OpenCASCADE), meshes its surface with triangles of `size` (and down to size_min,
        refined by curvature: `curvature` elements per 2 pi) and builds the skins. gmsh reads STEP in mm: the sizes are in mm, and
        scale=1e-3 gives the skins in metres (for HeleShawFilling, in SI)."""

        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            gmsh.model.occ.importShapes(str(path))
            gmsh.model.occ.synchronize()
            gmsh.option.setNumber('Mesh.MeshSizeMax', size)
            gmsh.option.setNumber('Mesh.MeshSizeMin', size_min if size_min is not None else size / 3)
            if curvature:
                gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', curvature)
            gmsh.model.mesh.generate(2)
            return cls.from_gmsh_model(angle=angle, scale=scale)
        finally:
            gmsh.finalize()
