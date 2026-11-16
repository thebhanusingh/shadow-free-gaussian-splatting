#!/usr/bin/env python
"""Lighting-model test on PHOTO PIXELS (no blob colours, no blob direction signs).

Same model as lighting_model_test.py: brightness = paint x (sky(n) + sun x max(0, n.s) x [sun reaches it]), per channel,
linear light, sky = 9 smooth coefficients, sun strength fixed to 1. Here every quantity is per photo pixel:
- colour: the real photo (linear light);
- surface direction n: the 2DGS surfel normals rendered into the photo (each surfel turned to face the camera, so the
  sign is right by construction; blended by alpha and renormalised);
- sun reaches it: the sun-shadow labels (sun_visibility.npz) rendered into the photo (mask < 0.05 lit, > 0.5 shadow);
- paint groups: k-means on the chroma of sunlit pixels, spread to the nearest sunlit pixel within --reach x bear height.
Fit on lit pixels + shaded pixels turned away from the sun (sky only); hold out the cast-shadow pixels (shadow, n.s above
--ns-test); predict them with sky light only and compare with the photo. Controls: no directions, shuffled directions.
Also reports how many sunlit pixels face away from the sun (should be about none if normals and labels agree).

Usage: lighting_pixel_test.py --colmap data/IMG_6705/colmap/sparse/0_refined --images data/IMG_6705/images
    --ply exports/IMG_6705/splat.ply --labels exports/IMG_6705_sunmap3/sun_visibility.npz
    --surfels exports/IMG_6705_2dgs_fromsplat/surfels_params.pt --sun exports/IMG_6705_sun_refit_obj/sun_refit.npz
    --plane-from exports/IMG_6705_shadowfix_o/stats.json --out exports/IMG_6705_lightpixel --frames frame_00481,...
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization_2dgs
from lighting_model_test import basis, fit_channel, predict, score
from photo_correct import load_splat, render_labels, to_photo_space
from PIL import Image
from scipy.cluster.vq import kmeans2
from scipy.ndimage import distance_transform_edt
from splatlib import LUM, qvec2rotmat, read_cameras_bin, read_images_bin, srgb_to_linear
from train_2dgs_ref import surfel_normals


def render_normals(s, s_n, im, cam):
    """World-space surface directions (H,W,3), unit length, facing the camera, in the photo's distorted pixels."""
    W, H = int(cam["width"]), int(cam["height"])
    fx, fy, cx, cy = cam["params"][:4]
    w2c = np.eye(4)
    w2c[:3, :3] = qvec2rotmat(im["q"])
    w2c[:3, 3] = im["t"]
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32).cuda()[None]
    vm = torch.tensor(w2c, dtype=torch.float32).cuda()[None]
    camc = torch.linalg.inv(vm[0])[:3, 3]
    n = s_n * torch.where(((camc - s["means"]) * s_n).sum(1, keepdim=True) < 0, -1.0, 1.0)
    with torch.no_grad():
        out = rasterization_2dgs(means=s["means"], quats=s["quats"], scales=torch.exp(s["scales"]),
                                 opacities=torch.sigmoid(s["opacities"]), colors=n, viewmats=vm, Ks=K, width=W, height=H,
                                 backgrounds=torch.zeros(1, 3, device="cuda"))
    col, alpha = out[0][0, ..., :3].cpu().numpy(), out[1][0, ..., 0].cpu().numpy()
    return col, alpha


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("colmap", "images", "ply", "labels", "surfels", "sun", "plane-from", "out"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--skyvis", type=Path, help="sky_visibility.npz from sky_visibility.py: multiply the sky term by it")
    ap.add_argument("--skyvis-key", default="sky_vis_cos", choices=["sky_vis", "sky_vis_cos"])
    ap.add_argument("--groups", type=int, default=6)
    ap.add_argument("--reach", type=float, default=0.08, help="paint spread, share of the bear's height in pixels")
    ap.add_argument("--ns-lit", type=float, default=0.3)
    ap.add_argument("--ns-test", type=float, default=0.15)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--max-px", type=int, default=60000, help="pixels per set drawn at random for the fit")
    ap.add_argument("--min-alpha", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    cams = read_cameras_bin(a.colmap / "cameras.bin")
    by = {i["name"].rsplit(".", 1)[0]: i for i in read_images_bin(a.colmap / "images.bin")}
    sp = load_splat(a.ply)
    lab = np.load(a.labels)
    lab_rgb = np.zeros((len(sp["means"]), 3), np.float32)
    lab_rgb[:, 2] = 1.0
    sh = lab["visible"] < 0.5
    lab_rgb[lab["index"], 0] = sh
    lab_rgb[lab["index"], 1] = ~sh
    lab_rgb[lab["index"], 2] = 0.0
    lab_rgb = torch.from_numpy(lab_rgb).cuda()
    skyvis_rgb = None
    if a.skyvis:
        sv = np.load(a.skyvis)
        arr = np.zeros((len(sp["means"]), 3), np.float32)
        arr[sv["index"], 0] = sv[a.skyvis_key]
        arr[sv["index"], 1] = 1.0
        skyvis_rgb = torch.from_numpy(arr).cuda()
    surf = {k: v.cuda() for k, v in torch.load(a.surfels).items()}
    s_n = surfel_normals(surf["quats"])
    s = np.load(a.sun)["sun"].astype(float)
    s /= np.linalg.norm(s)
    up = np.array(json.load(open(a.plane_from))["ground_plane"]["normal"], float)
    up /= np.linalg.norm(up)
    allres = {}
    for name in a.frames.split(","):
        im = by[name]
        cam = cams[im["cam"]]
        r = to_photo_space(render_labels(sp, lab_rgb, im, cam), cam)
        bear = (r[..., 0] + r[..., 1]) / np.maximum(r.sum(2), 1e-6) > 0.5
        m = np.where(bear, r[..., 0] / np.maximum(r[..., 0] + r[..., 1], 1e-6), 0.0)
        ao = None
        if skyvis_rgb is not None:
            ra = to_photo_space(render_labels(sp, skyvis_rgb, im, cam), cam)
            ao = np.where(ra[..., 1] > 0.5, ra[..., 0] / np.maximum(ra[..., 1], 1e-6), 1.0).astype(np.float32)
        ncol, alpha = render_normals(surf, s_n, im, cam)
        ncol, alpha = to_photo_space(ncol, cam), to_photo_space(alpha, cam)
        nrm = ncol / np.maximum(np.linalg.norm(ncol, axis=2, keepdims=True), 1e-6)
        photo = np.asarray(Image.open(a.images / im["name"]).convert("RGB"), np.float32) / 255
        rgb = srgb_to_linear(photo)
        L = rgb @ LUM
        ns = nrm @ s
        good = bear & (alpha > a.min_alpha) & (L > 1e-4) & (np.abs(np.linalg.norm(ncol, axis=2) - 1) < 0.5)
        lit = good & (m < 0.05)
        shade = good & (m > 0.5)
        info = dict(bear_px=int(bear.sum()), usable_px=int(good.sum()), lit_px=int(lit.sum()), shadow_px=int(shade.sum()))
        info["lit_pixels_facing_away_from_sun"] = round(float((ns[lit] < -0.05).mean()), 3)
        info["lit_pixels_ns_below_0.15"] = round(float((ns[lit] < 0.15).mean()), 3)
        # brightness against facing angle among sunlit pixels (the physics, no model)
        bins = [-1, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.01]
        info["lit_brightness_by_ns"] = {f"{lo:+.1f}": (round(float(np.median(L[lit & (ns >= lo) & (ns < hi)])), 3) if (lit & (ns >= lo) & (ns < hi)).sum() > 100 else None)
                                         for lo, hi in zip(bins[:-1], bins[1:])}
        info["shadow_brightness_by_ns"] = {f"{lo:+.1f}": (round(float(np.median(L[shade & (ns >= lo) & (ns < hi)])), 3) if (shade & (ns >= lo) & (ns < hi)).sum() > 100 else None)
                                            for lo, hi in zip(bins[:-1], bins[1:])}
        # paint groups from sunlit pixels with a clearly sun-facing direction
        core = lit & (ns > a.ns_lit)
        ch = (rgb / np.maximum(rgb.sum(2, keepdims=True), 1e-9))[..., :2]
        if core.sum() < 500:
            info["skipped"] = "too few clearly sunlit pixels"
            allres[name] = info
            print(name, info)
            continue
        pts = ch[core]
        sub = pts[rng.choice(len(pts), min(40000, len(pts)), replace=False)]
        cen, _ = kmeans2(sub, a.groups, minit="++", seed=a.seed)
        lab_img = np.full(bear.shape, -1, int)
        d2 = ((pts[:, None, :] - cen[None]) ** 2).sum(2)
        lab_img[core] = d2.argmin(1)
        dist, (iy, ix) = distance_transform_edt(lab_img < 0, return_indices=True)
        ys, xs = np.nonzero(bear)
        height = ys.max() - ys.min() + 1
        grp = np.where(dist <= a.reach * height, lab_img[iy, ix], -1)
        okp = good & (grp >= 0)
        test = okp & shade & (ns > a.ns_test)
        amb = okp & shade & (ns >= -0.05) & (ns <= a.ns_test)
        away = okp & shade & (ns < -0.05)
        fit = okp & ~test & ~amb & (lit | away)
        info.update(fit_px=int(fit.sum()), turned_away_px=int((fit & away).sum()), held_out_px=int(test.sum()))
        if test.sum() < 500 or fit.sum() < 2000:
            info["skipped"] = "too few held-out or fit pixels"
            allres[name] = info
            print(name, info)
            continue
        fi = np.flatnonzero(fit.ravel())
        fi = rng.choice(fi, min(a.max_px, len(fi)), replace=False)
        ti = np.flatnonzero(test.ravel())
        ti = rng.choice(ti, min(a.max_px, len(ti)), replace=False)
        nf, nt = nrm.reshape(-1, 3)[fi], nrm.reshape(-1, 3)[ti]
        Bf, Bt = basis(nf.astype(float), up), basis(nt.astype(float), up)
        gf, gt = grp.ravel()[fi], grp.ravel()[ti]
        sunf = (np.maximum(nf @ s, 0) * (m.ravel()[fi] < 0.5)).astype(float)
        Bs = Bt.copy()
        for k in range(a.groups):
            q = np.flatnonzero(gt == k)
            if len(q):
                Bs[q] = Bt[rng.permutation(q)]
        obs = rgb.reshape(-1, 3)[ti]
        z0 = np.zeros(len(ti))

        def variant(Bfit, Btest, const_sky=False):
            out, sky_up = np.zeros((len(ti), 3)), {}
            for c in range(3):
                Lf = np.log(np.maximum(rgb.reshape(-1, 3)[fi][:, c], 1e-6))
                lr, A = fit_channel(Lf, gf, Bfit, sunf, a.groups, a.lam, const_sky=const_sky)
                out[:, c] = predict(lr, A, gt, Btest, z0)
                sky_up["RGB"[c]] = round(float(basis(up[None], up)[0] @ A), 3)
            return out, sky_up

        pred, chan = variant(Bf, Bt)
        pred_c, _ = variant(Bf, Bt, const_sky=True)
        pred_s, _ = variant(Bf, Bs)
        info["sky_at_up_over_sun"] = chan
        info["held_out"] = dict(model=score(obs @ LUM, pred @ LUM), no_directions=score(obs @ LUM, pred_c @ LUM),
                                shuffled_directions=score(obs @ LUM, pred_s @ LUM),
                                per_channel={k: score(obs[:, c], pred[:, c]) for c, k in enumerate("RGB")})
        if ao is not None:                       # sky term multiplied by the sky visibility of each pixel
            af, at = ao.ravel()[fi][:, None], ao.ravel()[ti][:, None]
            at_s = at.copy()
            for k in range(a.groups):
                q = np.flatnonzero(gt == k)
                if len(q):
                    at_s[q] = at[rng.permutation(q)]
            p_ao, chan_ao = variant(Bf * af, Bt * at)
            p_ao_c, _ = variant(Bf * af, Bt * at, const_sky=True)
            p_ao_s, _ = variant(Bf * af, Bt * at_s)
            info["sky_at_up_over_sun_with_visibility"] = chan_ao
            info["held_out_with_visibility"] = dict(
                model=score(obs @ LUM, p_ao @ LUM), visibility_only_no_directions=score(obs @ LUM, p_ao_c @ LUM),
                shuffled_visibility=score(obs @ LUM, p_ao_s @ LUM),
                per_channel={k: score(obs[:, c], p_ao[:, c]) for c, k in enumerate("RGB")})
            lo = np.log(np.maximum(obs @ LUM, 1e-6))
            info["held_out_corr_logbrightness_logvisibility"] = round(float(np.corrcoef(lo, np.log(np.maximum(at[:, 0], 1e-3)))[0, 1]), 3)
            info["held_out_median_visibility"] = round(float(np.median(at)), 3)
            info["fit_median_visibility"] = round(float(np.median(af)), 3)
        allres[name] = info
        print(name, json.dumps({k: v for k, v in info.items() if k not in ("lit_brightness_by_ns", "shadow_brightness_by_ns")}))
    json.dump(allres, open(a.out / "lightpixel.json", "w"), indent=1)


if __name__ == "__main__":
    main()
