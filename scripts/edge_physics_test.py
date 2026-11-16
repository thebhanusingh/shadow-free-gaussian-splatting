#!/usr/bin/env python
"""Does physics predict the brightness step at the shadow edge? No fitting of paint, no learned sky.

At a cast-shadow edge both sides are the same paint and the same surface direction, so
    lit / shadow = (k * V_lit * shape(n) + cos) / (k * V_shadow * shape(n))
with cos = max(0, n.s) (sun strength = 1), V = sky visibility of the spot (ray cast on the surfels, sky_visibility.py),
k = sky light over sun light for an open, upward-facing surface. k is NOT fitted from the bear: it comes from the concrete
next to it (PPISP run: sun/shade 8.5, 6.6, 4.2 in linear light; sun elevation 45 deg -> k = cos(45)/(ratio - 1)).
The measured step is the one photo_correct_edge.py uses (normalised blur of lit-band and shadow-band luminance, bands
--gap..--gap+--band px from the mask edge, bear pixels only). Compared over every window with enough pixels on both sides:
median |log(predicted / measured)|, correlation of the logs, and the median ratio predicted / measured.
Variants: sky shape 1 or (1 + n.up)/2; visibility from the ray cast, none (V = 1) or shuffled; and the k that fits best (one
number per frame) next to the concrete's.

Usage: edge_physics_test.py --colmap ... --images ... --ply ... --labels ... --surfels ... --sun ... --plane-from ...
           --skyvis exports/IMG_6705_skyvis/sky_visibility.npz --out exports/IMG_6705_edgephysics --frames frame_00481,...
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lighting_pixel_test import render_normals
from photo_correct import load_splat, nblur, render_labels, to_photo_space
from photo_correct_edge import bands
from PIL import Image
from splatlib import LUM, read_cameras_bin, read_images_bin, srgb_to_linear
from train_2dgs_ref import surfel_normals

K_CONCRETE = [float(np.cos(np.radians(45)) / (r - 1)) for r in (8.5, 6.6, 4.2)]
K_LUM = float(np.cos(np.radians(45)) / (np.array([8.5, 6.6, 4.2]) @ LUM - 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("colmap", "images", "ply", "labels", "surfels", "sun", "plane-from", "skyvis", "out"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--skyvis-key", default="sky_vis_cos")
    ap.add_argument("--gap", type=float, default=4.0)
    ap.add_argument("--band", type=float, default=14.0)
    ap.add_argument("--sigma", type=float, default=18.0)
    ap.add_argument("--min-px", type=float, default=150.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
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
    sv = np.load(a.skyvis)
    ar = np.zeros((len(sp["means"]), 3), np.float32)
    ar[sv["index"], 0] = sv[a.skyvis_key]
    ar[sv["index"], 1] = 1.0
    ar = torch.from_numpy(ar).cuda()
    surf = {k: v.cuda() for k, v in torch.load(a.surfels).items()}
    s_n = surfel_normals(surf["quats"])
    s = np.load(a.sun)["sun"].astype(float)
    s /= np.linalg.norm(s)
    up = np.array(json.load(open(a.plane_from))["ground_plane"]["normal"], float)
    up /= np.linalg.norm(up)
    k_grid = np.geomspace(0.02, 1.0, 40)
    allres = {}
    for name in a.frames.split(","):
        im = by[name]
        cam = cams[im["cam"]]
        r = to_photo_space(render_labels(sp, lr, im, cam), cam)
        bear = (r[..., 0] + r[..., 1]) / np.maximum(r.sum(2), 1e-6) > 0.5
        m = np.where(bear, r[..., 0] / np.maximum(r[..., 0] + r[..., 1], 1e-6), 0.0)
        ra = to_photo_space(render_labels(sp, ar, im, cam), cam)
        V = np.where(ra[..., 1] > 0.5, ra[..., 0] / np.maximum(ra[..., 1], 1e-6), 1.0).astype(np.float32)
        ncol, alpha = render_normals(surf, s_n, im, cam)
        ncol, alpha = to_photo_space(ncol, cam), to_photo_space(alpha, cam)
        nrm = ncol / np.maximum(np.linalg.norm(ncol, axis=2, keepdims=True), 1e-6)
        cosv = np.maximum(nrm @ s, 0).astype(np.float32)
        shape_h = ((1 + nrm @ up) / 2).astype(np.float32)
        L = srgb_to_linear(np.asarray(Image.open(a.images / im["name"]).convert("RGB"), np.float32) / 255) @ LUM
        shade, lit = bear & (m > 0.5) & (alpha > 0.8), bear & (m < 0.05) & (alpha > 0.8)
        bs, bl = bands(shade, lit, a.gap, a.band)
        need = a.min_px / (2 * np.pi * a.sigma ** 2)
        wl, ws = bl.astype(np.float32), bs.astype(np.float32)
        Ll, dl = nblur(L, wl, a.sigma)
        Ls, ds = nblur(L, ws, a.sigma)
        valid = (dl > need) & (ds > need) & bs & (Ls > 1e-5)
        if valid.sum() < 200:
            allres[name] = dict(skipped="too few edge windows")
            print(name, allres[name])
            continue
        meas = np.log(Ll[valid] / Ls[valid])

        def predicted(k, Vp, shape):
            num = nblur(k * Vp * shape + cosv, wl, a.sigma)[0]       # lit side: sky + sun
            den = nblur(k * Vp * shape, ws, a.sigma)[0]              # shadow side: sky only
            return np.log(np.maximum(num[valid], 1e-6) / np.maximum(den[valid], 1e-6))

        Vs = V.copy()
        idx = np.flatnonzero((bs | bl).ravel())
        Vs.ravel()[idx] = V.ravel()[rng.permutation(idx)]
        ones = np.ones_like(V)
        res = dict(windows=int(valid.sum()), measured_median_log_step=round(float(np.median(meas)), 3),
                   measured_median_ratio=round(float(np.exp(np.median(meas))), 2),
                   median_visibility_shadow_band=round(float(np.median(V[bs])), 3),
                   median_visibility_lit_band=round(float(np.median(V[bl])), 3))
        for label, Vp, shape in (("visibility, flat sky", V, ones), ("visibility, hemisphere sky", V, shape_h),
                                 ("no visibility", ones, ones), ("shuffled visibility", Vs, ones)):
            p = predicted(K_LUM, Vp, shape)
            res[f"concrete k | {label}"] = dict(median_abs_log_err=round(float(np.median(np.abs(p - meas))), 3),
                                                  median_ratio_pred_over_meas=round(float(np.exp(np.median(p - meas))), 2),
                                                  corr=round(float(np.corrcoef(p, meas)[0, 1]), 3))
            errs = [np.median(np.abs(predicted(k, Vp, shape) - meas)) for k in k_grid]
            res[f"best k | {label}"] = dict(k=round(float(k_grid[int(np.argmin(errs))]), 3), median_abs_log_err=round(float(min(errs)), 3))
        res["k_from_concrete"] = round(K_LUM, 3)
        allres[name] = res
        print(name, json.dumps(res, indent=1))
    json.dump(allres, open(a.out / "edgephysics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
