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


def iou(pred, obs):
    """Intersection over union of two boolean arrays (0 when both are empty)."""
    u = (pred | obs).sum()
    return float((pred & obs).sum() / u) if u else 0.0


def angle_between(u, v):
    """Angle in degrees between two UNIT vectors."""
    return float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))
