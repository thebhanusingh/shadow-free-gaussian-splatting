#!/usr/bin/env python
"""Masks for the R3DG test: keep the object and the ground around it, drop the far background.

For every image of an undistorted COLMAP model (PINHOLE), renders the alpha of the splat's blobs within --radius x
object height of the object's footprint (gsplat) and writes <data>/masks/<image name> (255 = keep), dilated by --grow
pixels. The far background (car, lawn) is then left out of R3DG, which keeps it inside 8 GB and on the object.

Usage: make_r3dg_masks.py --data data/IMG_6705_r3dg --ply exports/IMG_6705/splat.ply --object exports/IMG_6705_object/object.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization
from PIL import Image
from scipy import ndimage
from splatlib import qvec2rotmat, read_cameras_bin, read_images_bin, read_ply

sys.path.insert(0, str(Path(__file__).parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--object", type=Path, required=True)
    ap.add_argument("--radius", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--grow", type=int, default=6)
    a = ap.parse_args()
    obj = json.loads(a.object.read_text())
    _, g = read_ply(a.ply)
    xyz = np.c_[g["x"], g["y"], g["z"]].astype(np.float32)
    keep = np.linalg.norm(xyz - np.array(obj["footprint_centre"], np.float32), axis=1) < a.radius * obj["height"]
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x, np.float32)).cuda()
    col = lambda *k: np.stack([g[x][keep] for x in k], 1)
    means = t(xyz[keep])
    quats = torch.nn.functional.normalize(t(col("rot_0", "rot_1", "rot_2", "rot_3")), dim=1)
    scales = torch.exp(t(col("scale_0", "scale_1", "scale_2")))
    opac = torch.sigmoid(t(g["opacity"][keep]))
    white = torch.ones(len(means), 3, device="cuda")
    cams = read_cameras_bin(a.data / "sparse/0/cameras.bin")
    ims = read_images_bin(a.data / "sparse/0/images.bin")
    ims = list(ims.values()) if isinstance(ims, dict) else ims
    out = a.data / "masks"
    out.mkdir(exist_ok=True)
    cover = []
    for im in ims:
        c = cams[im["cam"]]
        W, H = int(c["width"]), int(c["height"])
        fx, fy, cx, cy = c["params"][:4] if len(c["params"]) >= 4 else (c["params"][0], c["params"][0], *c["params"][1:3])
        w2c = np.eye(4)
        w2c[:3, :3] = qvec2rotmat(im["q"])
        w2c[:3, 3] = im["t"]
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device="cuda")
        with torch.no_grad():
            _, alpha, _ = rasterization(means, quats, scales, opac, white, t(w2c)[None], K[None], W, H)
        m = alpha[0, ..., 0].cpu().numpy() > a.alpha
        m = ndimage.binary_dilation(m, iterations=a.grow)
        Image.fromarray((m * 255).astype(np.uint8)).save(out / Path(im["name"]).name)
        cover.append(m.mean())
    print(f"{len(ims)} masks ({keep.sum():,} blobs within {a.radius} x height); kept area per image median "
          f"{np.median(cover):.1%} (min {min(cover):.1%}, max {max(cover):.1%}) -> {out}")


if __name__ == "__main__":
    main()
