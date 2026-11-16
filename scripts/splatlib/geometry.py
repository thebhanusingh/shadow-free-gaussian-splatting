"""Sun and shadow geometry: ground frame, sun direction, look-at camera, shadow prediction, overlap score."""
import numpy as np
from scipy.ndimage import maximum_filter


def plane_frame(up):
    """(up, e1, e2): orthonormal frame with the ground normal first."""
    up = up / np.linalg.norm(up)
    e1 = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return up, e1, np.cross(up, e1)


def sun_vec(elev, azim, frame):
    """Unit vector towards the sun from elevation and azimuth in degrees, in the ground frame."""
    up, e1, e2 = frame
    el, az = np.radians(elev), np.radians(azim)
    return np.cos(el) * (np.cos(az) * e1 + np.sin(az) * e2) + np.sin(el) * up


def look_at(pos, target, up):
    """4x4 world-to-camera matrix for a camera at pos looking at target (z forward, y down)."""
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(-up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    w2c = np.eye(4)
    w2c[:3, :3] = np.stack([x, y, z])
    w2c[:3, 3] = -w2c[:3, :3] @ pos
    return w2c


def predict(occ, rec, s, cell, dilate, eps):
    """Boolean shadow prediction for receivers rec (N,3) given occluders occ (M,3) and sun direction s."""
    a = np.cross(s, [1.0, 0, 0]) if abs(s[0]) < 0.9 else np.cross(s, [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(s, a)
    po, pr = np.c_[occ @ a, occ @ b], np.c_[rec @ a, rec @ b]
    lo = np.minimum(po.min(0), pr.min(0)) - cell * (dilate + 1)
    io, ir = ((po - lo) / cell).astype(int), ((pr - lo) / cell).astype(int)
    shape = np.maximum(io.max(0), ir.max(0)) + dilate + 2
    depth = np.full(shape, -np.inf)
    np.maximum.at(depth, (io[:, 0], io[:, 1]), occ @ s)
    if dilate:
        depth = maximum_filter(depth, size=2 * dilate + 1, mode="constant", cval=-np.inf)
    return depth[ir[:, 0], ir[:, 1]] > rec @ s + eps


def ground_footprint(points, up, sun, plane_off, cell, margin):
    """Where on the ground plane the shadow of `points` (M,3) can fall, for a sun at unit direction `sun`.

    The plane is {x : x.up + plane_off = 0}. Every point is slid along the light to the plane, the landing spots are
    drawn on a grid of `cell` units and grown by `margin` units (soft edges, blobs are not a closed surface).
    Returns dict(grid, lo, cell, up, e1, e2); look cells up with in_footprint."""
    up = up / np.linalg.norm(up)
    sun = sun / np.linalg.norm(sun)
    _, e1, e2 = plane_frame(up)
    h = points @ up + plane_off
    land = points - (h / (sun @ up))[:, None] * sun
    uv = np.c_[land @ e1, land @ e2]
    pad = margin + 2 * cell
    lo = uv.min(0) - pad
    ij = np.floor((uv - lo) / cell).astype(int)
    shape = np.floor((uv.max(0) + pad - lo) / cell).astype(int) + 2
    grid = np.zeros(shape, bool)
    grid[ij[:, 0], ij[:, 1]] = True
    r = max(1, int(round(margin / cell)))
    grid = maximum_filter(grid, size=2 * r + 1, mode="constant", cval=False)
    return dict(grid=grid, lo=lo, cell=cell, up=up, e1=e1, e2=e2)


def in_footprint(fp, X):
    """Boolean per point of X (...,3) on the ground plane: inside the footprint grid (False outside the grid)."""
    X = np.asarray(X, float)
    uv = np.stack([X @ fp["e1"], X @ fp["e2"]], -1)
    ij = np.floor((uv - fp["lo"]) / fp["cell"]).astype(int)
    ok = (ij[..., 0] >= 0) & (ij[..., 1] >= 0) & (ij[..., 0] < fp["grid"].shape[0]) & (ij[..., 1] < fp["grid"].shape[1])
    out = np.zeros(ok.shape, bool)
    out[ok] = fp["grid"][ij[..., 0][ok], ij[..., 1][ok]]
    return out


def iou(pred, obs):
    """Intersection over union of two boolean arrays (0 when both are empty)."""
    u = (pred | obs).sum()
    return float((pred & obs).sum() / u) if u else 0.0


def angle_between(u, v):
    """Angle in degrees between two UNIT vectors."""
    return float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))
