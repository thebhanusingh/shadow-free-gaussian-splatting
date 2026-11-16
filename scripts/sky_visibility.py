#!/usr/bin/env python
"""How much of the sky can each bear blob see? The sun-map method repeated for many directions over the sky dome.

For each direction a near-orthographic depth map is rendered from that direction through the bear's 2DGS surfels,
exactly as sun_shadow_map.py does for the sun (only the object's own surfels cast, blob position tested against the map
with a margin --eps). A blob is blocked from a direction when more than half of its 3 x 3 pixel window is closer to the
camera by more than --eps. Directions are spread evenly over the sky above --min-elev degrees (Fibonacci spiral in the
ground frame). Check: the same code for the sun direction must reproduce the accepted sun labels.

`build(a)` returns the visibility function for any direction, so other scripts can ask "does this blob see a sun at
direction d" (used by relight_photos.py).

Writes <out>/sky_visibility.npz: index (blobs, same order as the labels), sky_vis (share of directions that see the
blob), sky_vis_cos (weighted by sin(elevation)), n_dirs, sun_check.

Usage: sky_visibility.py --surfels exports/IMG_6705_2dgs_fromsplat/surfels_params.pt --ply exports/IMG_6705/splat.ply
           --labels exports/IMG_6705_sunmap3/sun_visibility.npz --sun exports/IMG_6705_sun_refit_obj/sun_refit.npz
           --object exports/IMG_6705_object/object.json --plane-from exports/IMG_6705_shadowfix_o/stats.json
           --out exports/IMG_6705_skyvis
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from splatlib import look_at, plane_frame, read_ply, sun_vec

sys.path.insert(0, str(Path(__file__).parent))
from train_2dgs_ref import render


def fibonacci_cap(n, min_elev_deg):
    """n directions (elevation, azimuth in degrees) spread evenly over the sky above min_elev."""
    z = np.linspace(np.sin(np.radians(min_elev_deg)), 1.0, n, endpoint=False) + (1 - np.sin(np.radians(min_elev_deg))) / (2 * n)
    az = np.degrees(np.arange(n) * np.pi * (3 - np.sqrt(5)))
    return np.degrees(np.arcsin(z)), az % 360


def add_arguments(ap):
    for k in ("surfels", "ply", "labels", "sun", "object", "plane-from"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--res", type=int, default=2048)
    ap.add_argument("--cover", type=float, default=1.6)
    ap.add_argument("--far", type=float, default=200.0)
    ap.add_argument("--eps", type=float, default=0.093)
    ap.add_argument("--opacity", type=float, default=0.1)
    ap.add_argument("--radius", type=float, default=2.0)


def build(a):
    """Load the bear's surfels and blobs; return dict(visible_from, idx, q, up, sun, frame).
    visible_from(d) -> bool array over the labelled blobs: True where a light at unit direction d reaches the blob."""
    dev = "cuda"
    obj = json.loads(a.object.read_text())
    gp = json.loads(a.plane_from.read_text())["ground_plane"]
    up = np.array(gp["normal"], float)
    up /= np.linalg.norm(up)
    sun = np.load(a.sun)["sun"].astype(float)
    sun /= np.linalg.norm(sun)
    target = np.array(obj["crop"][:3], float)
    p = {k: v.to(dev) for k, v in torch.load(a.surfels).items()}
    tgt = torch.tensor(target, dtype=torch.float32, device=dev)
    keep = (torch.sigmoid(p["opacities"]) >= a.opacity) & ((p["means"] - tgt).norm(dim=1) < a.radius * obj["height"])
    h = p["means"] @ torch.tensor(up, dtype=torch.float32, device=dev) + float(gp["offset"])
    keep &= (h > 2 * float(gp["tol"])) & ((p["means"] - tgt).norm(dim=1) < float(obj["crop"][3]))
    p = {k: v[keep] for k, v in p.items()}
    idx = np.load(a.labels)["index"]
    _, g = read_ply(a.ply)
    q = np.stack([g[k][idx] for k in "xyz"], 1).astype(float)
    N = a.res
    f = N * a.far / (a.cover * obj["height"])
    K = torch.tensor([[f, 0, N / 2], [0, f, N / 2], [0, 0, 1]], dtype=torch.float32, device=dev)

    def visible_from(d):
        pos = target + a.far * d
        cam_up = up - (up @ d) * d
        if np.linalg.norm(cam_up) < 1e-3:
            cam_up = np.cross(d, [1.0, 0, 0])
        cam_up /= np.linalg.norm(cam_up)
        w2c = look_at(pos, target, cam_up)
        with torch.no_grad():
            out = render(p, torch.tensor(w2c, dtype=torch.float32, device=dev), K, N, N, 0)
        alpha = out[1][0, ..., 0]
        depth = torch.where(alpha > 0.5, out[5][0, ..., 0], torch.full_like(alpha, np.inf))
        pc = q @ w2c[:3, :3].T + w2c[:3, 3]
        ui = torch.tensor(np.round(f * pc[:, 0] / pc[:, 2] + N / 2), device=dev).long()
        vi = torch.tensor(np.round(f * pc[:, 1] / pc[:, 2] + N / 2), device=dev).long()
        z = torch.tensor(pc[:, 2], device=dev, dtype=torch.float32)
        blocked = torch.zeros(len(q), device=dev)
        cnt = torch.zeros(len(q), device=dev)
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                uu, vv = ui + du, vi + dv
                ok = (uu >= 0) & (uu < N) & (vv >= 0) & (vv < N)
                dd = torch.full_like(z, np.inf)
                dd[ok] = depth[vv[ok], uu[ok]]
                blocked += ok * (dd < z - a.eps)
                cnt += ok
        return (blocked / cnt.clamp_min(1) <= 0.5).cpu().numpy()

    return dict(visible_from=visible_from, idx=idx, q=q, up=up, sun=sun, frame=plane_frame(up))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dirs", type=int, default=48)
    ap.add_argument("--min-elev", type=float, default=8.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    b = build(a)
    lab = np.load(a.labels)["visible"] >= 0.5
    chk = b["visible_from"](b["sun"])
    agree = float((chk == lab).mean())
    print(f"sun check: {chk.mean():.3f} lit here vs {lab.mean():.3f} in the accepted labels; agreement {agree:.3f}")
    el, az = fibonacci_cap(a.dirs, a.min_elev)
    vis = np.zeros((a.dirs, len(b["idx"])), bool)
    for i, (e, z_) in enumerate(zip(el, az)):
        vis[i] = b["visible_from"](sun_vec(e, z_, b["frame"]))
    w = np.sin(np.radians(el))
    sky = vis.mean(0)
    sky_cos = (vis * w[:, None]).sum(0) / w.sum()
    np.savez_compressed(a.out / "sky_visibility.npz", index=b["idx"], sky_vis=sky.astype(np.float32),
                        sky_vis_cos=sky_cos.astype(np.float32), n_dirs=a.dirs, sun_check=agree)
    print(f"{a.dirs} directions over the sky above {a.min_elev} deg; sky visibility over {len(b['idx']):,} blobs: "
          f"percentiles 5/25/50/75/95 = {np.percentile(sky, [5, 25, 50, 75, 95]).round(2)}; "
          f"cast-shadow blobs median {np.median(sky[~lab]):.2f}, lit blobs median {np.median(sky[lab]):.2f}")


if __name__ == "__main__":
    main()
