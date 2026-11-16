#!/usr/bin/env python
"""Physical-gain photo correction: shadow pixels get gain = 1 + cos / k, with no fitting on the bear.

A cast-shadow pixel misses the sun but keeps the sky. Its lit version is brighter by
    gain_c = (k_c + cos) / k_c = 1 + cos / k_c,    cos = max(0, n.s), sun strength = 1,
n from the 2DGS surfel normals rendered into the photo (smoothed a little), s = the refitted sun, k_c = sky light over sun
light for an open upward surface, per colour channel, from the concrete next to the bear (PPISP run: sun/shade 8.5, 6.6,
4.2 in linear light, sun elevation 45 deg; k_c = cos(45) / (ratio_c - 1)). --k-scale multiplies k (the bear's own edges
fit about 0.5-1 x the concrete's). Mask = the sun-shadow labels rendered into the photo; corrected = photo * gain ** mask,
so pixels outside the mask are untouched. Gain is capped by one common factor (--max-gain, hue-safe) and a corrected
pixel may not exceed --lit-cap x the local mean of the lit pixels (cross-check with sunlit parts).
--lum-only uses one k for the luminance (no colour shift).
Reports the step left across the shadow edge (median |log(lit band / shadow band)|, same bands as photo_correct_edge.py)
before and after, over all edge windows (nothing is fitted, so no hold-out is needed).

Usage: photo_correct_physics.py --colmap ... --images ... --ply ... --labels ... --surfels ... --sun ... --plane-from ...
           --out exports/IMG_6705_photofix_physics --frames frame_00551,...
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from lighting_pixel_test import render_normals
from photo_correct import load_splat, nblur, render_labels, to_photo_space
from photo_correct_edge import step_metric
from PIL import Image
from splatlib import LUM, linear_to_srgb, read_cameras_bin, read_images_bin, srgb_to_linear
from train_2dgs_ref import surfel_normals

RATIO = np.array([8.5, 6.6, 4.2])                                   # concrete sun / shade, linear light
K_RGB = np.cos(np.radians(45)) / (RATIO - 1)
K_LUM = float(np.cos(np.radians(45)) / (RATIO @ LUM - 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("colmap", "images", "ply", "labels", "surfels", "sun", "plane-from", "out"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--k-scale", type=float, default=1.0)
    ap.add_argument("--lum-only", action="store_true")
    ap.add_argument("--max-gain", type=float, default=12.0)
    ap.add_argument("--lit-cap", type=float, default=1.3)
    ap.add_argument("--smooth", type=float, default=2.0, help="px, blur of the rendered normals before n.s")
    ap.add_argument("--gap", type=float, default=4.0)
    ap.add_argument("--band", type=float, default=14.0)
    ap.add_argument("--edge-sigma", type=float, default=18.0)
    ap.add_argument("--min-px", type=float, default=150.0)
    ap.add_argument("--crop-pad", type=float, default=0.25)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cams = read_cameras_bin(a.colmap / "cameras.bin")
    by = {i["name"].rsplit(".", 1)[0]: i for i in read_images_bin(a.colmap / "images.bin")}
    sp = load_splat(a.ply)
    lab = np.load(a.labels)
    sh = lab["visible"] < 0.5
    lr = np.zeros((len(sp["means"]), 3), np.float32)
    lr[:, 2] = 1.0
    lr[lab["index"], 0] = sh
    lr[lab["index"], 1] = ~sh
    lr[lab["index"], 2] = 0.0
    lr = torch.from_numpy(lr).cuda()
    surf = {k: v.cuda() for k, v in torch.load(a.surfels).items()}
    s_n = surfel_normals(surf["quats"])
    s = np.load(a.sun)["sun"].astype(float)
    s /= np.linalg.norm(s)
    k = (np.full(3, K_LUM) if a.lum_only else K_RGB) * a.k_scale
    stats = {}
    for name in a.frames.split(","):
        im = by[name]
        cam = cams[im["cam"]]
        r = to_photo_space(render_labels(sp, lr, im, cam), cam)
        bear = (r[..., 0] + r[..., 1]) / np.maximum(r.sum(2), 1e-6) > 0.5
        m = np.where(bear, r[..., 0] / np.maximum(r[..., 0] + r[..., 1], 1e-6), 0.0).astype(np.float32)
        ncol, alpha = render_normals(surf, s_n, im, cam)
        ncol, alpha = to_photo_space(ncol, cam), to_photo_space(alpha, cam)
        if a.smooth > 0:
            kk = int(a.smooth * 6) | 1
            ncol = cv2.GaussianBlur(ncol, (kk, kk), a.smooth)
        nrm = ncol / np.maximum(np.linalg.norm(ncol, axis=2, keepdims=True), 1e-6)
        cosv = np.maximum(nrm @ s, 0).astype(np.float32)
        orig8 = np.asarray(Image.open(a.images / im["name"]).convert("RGB"))
        P = srgb_to_linear(orig8.astype(np.float32) / 255)
        L = P @ LUM
        shade, lit = bear & (m > 0.5) & (alpha > 0.5), bear & (m < 0.05) & (alpha > 0.5)
        g = 1.0 + cosv[..., None] / k[None, None, :]
        g = g * np.minimum(1.0, a.max_gain / g.max(2, keepdims=True))
        ys, xs = np.nonzero(bear)
        height = ys.max() - ys.min() + 1
        if a.lit_cap > 0:
            ll, dl = nblur(L, lit.astype(np.float32), max(0.12 * height, 8.0))
            allow = np.where(dl > 1e-3, ll * a.lit_cap, np.inf)
            lum_g = g @ LUM
            f = np.minimum(1.0, allow / np.maximum(L * lum_g, 1e-6))
            f = np.maximum(f, 1.0 / np.maximum(lum_g, 1e-6))
            g = g * f[..., None]
        g[~(bear & (alpha > 0.5))] = 1.0
        w = (np.clip(m, 0, 1) * bear)[..., None]
        gw = np.power(np.maximum(g, 1e-6), w)
        res8 = np.round(linear_to_srgb(P * gw, clip=True) * 255).astype(np.uint8)
        res8[m <= 0] = orig8[m <= 0]
        Image.fromarray(res8).save(a.out / f"{name}_corrected.png")
        L1 = srgb_to_linear(res8 / 255.0) @ LUM
        sm = m > 0.5
        info = dict(shadow_px=int(sm.sum()), shadow_lum_before=round(float(L[sm].mean()), 4), shadow_lum_after=round(float(L1[sm].mean()), 4),
                    lit_lum=round(float(L[lit].mean()), 4), median_lum_gain_in_shadow=round(float(np.median((gw @ LUM)[sm])), 2),
                    share_above_lit_median=round(float((L1[shade] > np.median(L[lit])).mean()), 3),
                    cos_median_in_shadow=round(float(np.median(cosv[shade])), 3))
        info["edge_step"] = step_metric(P, gw, shade, lit, a, np.ones(bear.shape, bool))
        # edges where the surface still faces the sun on the shadow side too (a real cast shadow, not a surface turning away)
        info["cast_edge_step"] = step_metric(P, gw, shade, lit, a, cosv > 0.15)
        info["share_of_shadow_px_facing_sun"] = round(float((cosv[shade] > 0.15).mean()), 3)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        ph, pw = int((y1 - y0) * a.crop_pad), int((x1 - x0) * a.crop_pad)
        y0, y1, x0, x1 = max(0, y0 - ph), min(P.shape[0], y1 + ph), max(0, x0 - pw), min(P.shape[1], x1 + pw)
        gv = np.clip((gw.mean(2) - 1) / (a.max_gain - 1), 0, 1)
        gimg = (np.stack([gv, gv * 0.8, np.zeros_like(gv)], 2) * 255).astype(np.uint8)
        Image.fromarray(np.concatenate([x[y0:y1, x0:x1] for x in (orig8, res8, gimg)], 1)).save(a.out / f"{name}_compare.jpg", quality=92)
        stats[name] = info
        print(name, json.dumps(info))
    json.dump(dict(args={q: str(v) for q, v in vars(a).items()}, k=[round(float(v), 3) for v in k], frames=stats),
              open(a.out / "stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
