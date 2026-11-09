"""COLMAP binary model readers (the camera models we use) and the pinhole/distortion projection."""
import struct

import numpy as np

CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5), 4: ("OPENCV", 8)}


def read_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            if model_id not in CAMERA_MODELS:
                raise SystemExit(f"ERROR: unsupported COLMAP camera model id {model_id}")
            name, nparams = CAMERA_MODELS[model_id]
            params = np.array(struct.unpack(f"<{nparams}d", f.read(8 * nparams)))
            cams[cam_id] = dict(model=name, width=w, height=h, params=params)
    return cams


def read_images_bin(path):
    """Registered images sorted by file name: id, quaternion q (w,x,y,z), translation t, camera id, name."""
    imgs = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            img_id, qw, qx, qy, qz, tx, ty, tz, cam_id = struct.unpack("<idddddddi", f.read(64))
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (npts,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * npts, 1)
            imgs.append(dict(id=img_id, q=np.array([qw, qx, qy, qz]), t=np.array([tx, ty, tz]), cam=cam_id, name=name.decode()))
    return sorted(imgs, key=lambda i: i["name"])


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def project(cam, xc):
    """Camera-space points (N,3) with z>0 -> pixel coords (N,2), applying lens distortion."""
    x, y = xc[:, 0] / xc[:, 2], xc[:, 1] / xc[:, 2]
    p, m = cam["params"], cam["model"]
    if m == "SIMPLE_PINHOLE":
        fx = fy = p[0]; cx, cy = p[1], p[2]
    elif m == "PINHOLE":
        fx, fy, cx, cy = p
    elif m in ("SIMPLE_RADIAL", "RADIAL"):
        fx = fy = p[0]; cx, cy = p[1], p[2]
        k1, k2 = p[3], (p[4] if m == "RADIAL" else 0.0)
        r2 = x * x + y * y
        s = 1 + k1 * r2 + k2 * r2 * r2
        x, y = x * s, y * s
    else:  # OPENCV
        fx, fy, cx, cy, k1, k2, p1, p2 = p
        r2 = x * x + y * y
        s = 1 + k1 * r2 + k2 * r2 * r2
        x, y = (x * s + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * s + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
    return np.stack([fx * x + cx, fy * y + cy], 1)
