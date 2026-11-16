#!/usr/bin/env python
"""How consistent is the colour of the corrected photos? Measured per bear blob across the views that see it.

For every bear blob (labels npz index) the blob centre is projected into every --every-th photo (lens distortion included),
kept when the blob is in front of the 2DGS depth map there (visible), and the photo colour is sampled in a 5 x 5 patch, in
the original photos and in the corrected ones (linear light). Then, per blob over its views:
  flicker   = std over views of the chroma (r, g shares): does the colour change from view to view?
  shift     = |mean corrected chroma - mean original chroma|: is there a systematic colour change?
and, per hue cluster (k-means on the chroma of the sunlit blobs), the saturation (max - min) / max of the mean colour:
sunlit blobs, shadowed blobs in the original, shadowed blobs in the corrected photos. If the corrected shadow blobs are more
saturated than the sunlit blobs of the same hue, the lift over-saturates.

Usage: colour_consistency.py --scene IMG_6704 --fixed data/IMG_6704_fixed --labels exports/IMG_6704_sunmap3/sun_visibility.npz
           --ply exports/IMG_6704/splat.ply --surfels exports/IMG_6704_2dgs_fromsplat/surfels_params.pt [--every 16]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.cluster.vq import kmeans2
from scipy.ndimage import uniform_filter
from splatlib import LUM, SH_C0, project, qvec2rotmat, read_cameras_bin, read_images_bin, read_ply, srgb_to_linear
from train_2dgs_ref import render


def sat(c):
    return (c.max(1) - c.min(1)) / np.maximum(c.max(1), 1e-6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--fixed", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--surfels", type=Path, required=True)
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--min-views", type=int, default=8)
    ap.add_argument("--depth-tol", type=float, default=0.02, help="visible when z < depth x (1 + tol)")
    ap.add_argument("--ratio", type=float, nargs=3, required=True, help="concrete sun/shade ratio per channel (calibration.json)")
    ap.add_argument("--cos-up", type=float, default=0.7071)
    ap.add_argument("--floor", type=float, default=0.3)
    ap.add_argument("--nb-radius", type=float, default=0.12, help="units: neighbouring lit blobs within this distance")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    colmap = Path("data") / a.scene / "colmap/sparse/0_refined"
    cams, imgs = read_cameras_bin(colmap / "cameras.bin"), read_images_bin(colmap / "images.bin")
    lab = np.load(a.labels)
    idx, shadow = lab["index"], lab["visible"] < 0.5
    _, g = read_ply(a.ply)
    q = np.stack([g[k][idx] for k in "xyz"], 1).astype(float)
    surf = {k: v.cuda() for k, v in torch.load(a.surfels).items()}
    names = [i for i in imgs][::a.every]
    n = len(idx)
    so, sf, cnt = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n)
    co, cf = [], []                                    # per view sampled chroma, for the flicker
    samp_o = np.full((len(names), n, 3), np.nan, np.float32)
    samp_f = np.full((len(names), n, 3), np.nan, np.float32)
    for fi, im in enumerate(names):
        cam = cams[im["cam"]]
        W, H = int(cam["width"]), int(cam["height"])
        fx, fy, cx, cy = cam["params"][:4]
        R, t = qvec2rotmat(im["q"]), im["t"]
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = R, t
        with torch.no_grad():
            out = render(surf, torch.tensor(w2c, dtype=torch.float32).cuda(),
                         torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32).cuda(), W, H, 0)
        depth = out[5][0, ..., 0].cpu().numpy()
        xc = q @ R.T + t
        ok = xc[:, 2] > 0.1
        pin = np.stack([fx * xc[:, 0] / np.where(ok, xc[:, 2], 1) + cx, fy * xc[:, 1] / np.where(ok, xc[:, 2], 1) + cy], 1)
        okp = ok & (pin[:, 0] > 2) & (pin[:, 0] < W - 3) & (pin[:, 1] > 2) & (pin[:, 1] < H - 3)
        vis = np.zeros(n, bool)
        ii = np.flatnonzero(okp)
        dz = depth[pin[ii, 1].astype(int), pin[ii, 0].astype(int)]
        vis[ii] = xc[ii, 2] < dz * (1 + a.depth_tol)
        px = project(cam, xc[np.flatnonzero(vis)])           # where the photo shows it (distorted)
        inb = (px[:, 0] > 2) & (px[:, 0] < W - 3) & (px[:, 1] > 2) & (px[:, 1] < H - 3)
        vi = np.flatnonzero(vis)[inb]
        px = px[inb]
        for tag, path, store in (("o", Path("data") / a.scene / "images" / im["name"], samp_o),
                                 ("f", a.fixed / "images" / im["name"], samp_f)):
            p = srgb_to_linear(np.asarray(Image.open(path).convert("RGB"), np.float32) / 255)
            p5 = np.stack([uniform_filter(p[..., c], 5) for c in range(3)], 2)
            store[fi, vi] = p5[px[:, 1].astype(int), px[:, 0].astype(int)]
    v_o, v_f = samp_o, samp_f
    nv = np.sum(~np.isnan(v_o[..., 0]), 0)
    keep = nv >= a.min_views
    ch = lambda x: x[..., :2] / np.maximum(x.sum(-1, keepdims=True), 1e-6)
    flick_o = np.nanmean(np.nanstd(ch(v_o), axis=0), axis=1)
    flick_f = np.nanmean(np.nanstd(ch(v_f), axis=0), axis=1)
    mo, mf = np.nanmean(v_o, axis=0), np.nanmean(v_f, axis=0)
    shift = np.abs(ch(mf) - ch(mo)).sum(1)
    res = dict(photos=len(names), blobs=int(n), blobs_with_enough_views=int(keep.sum()))
    for tag, m in (("lit", keep & ~shadow), ("shadow", keep & shadow)):
        res[f"flicker_{tag}"] = dict(original=round(float(np.median(flick_o[m])), 4), corrected=round(float(np.median(flick_f[m])), 4))
        res[f"chroma_shift_{tag}"] = round(float(np.median(shift[m])), 4)
    lit = keep & ~shadow & (mo @ LUM > 0.02)
    ch_lit = ch(mo)[lit]
    _, grp = kmeans2(ch_lit[:, :2], 5, minit="++", seed=0)
    cen = np.array([ch_lit[grp == k].mean(0) for k in range(5)])
    allc = ch(mo)
    assign = ((allc[:, None, :] - cen[None]) ** 2).sum(2).argmin(1)
    rows = []
    for k in range(5):
        L = keep & ~shadow & (assign == k)
        S0 = keep & shadow & (assign == k)
        if L.sum() > 50 and S0.sum() > 50:
            rows.append(dict(hue_cluster=k, chroma=[round(float(x), 3) for x in cen[k]], lit_blobs=int(L.sum()), shadow_blobs=int(S0.sum()),
                             sat_lit=round(float(np.median(sat(mo[L]))), 3), sat_shadow_original=round(float(np.median(sat(mo[S0]))), 3),
                             sat_shadow_corrected=round(float(np.median(sat(mf[S0]))), 3),
                             lum_lit=round(float(np.median(mo[L] @ LUM)), 3), lum_shadow_original=round(float(np.median(mo[S0] @ LUM)), 3),
                             lum_shadow_corrected=round(float(np.median(mf[S0] @ LUM)), 3)))
    res["by_hue_cluster"] = rows
    if a.out:
        np.savez_compressed(a.out.with_suffix(".npz"), mo=mo, mf=mf, keep=keep, shadow=shadow, q=q, assign=assign, cen=cen)
    # does a colour shift of the lift (the sky tint of the shade) bring the shadowed blobs' chroma closer to the sunlit blobs
    # right next to them (same surface, same paint)? tint t: 0 = brightness only .. 1 = the full sun/shade ratio per channel
    from scipy.spatial import cKDTree
    from splatlib import sun_vec  # noqa: F401  (kept for symmetry with the other scripts)
    ratio = np.array(a.ratio, float)
    cos_up = float(a.cos_up)
    lit_ix, sh_ix = np.flatnonzero(keep & ~shadow), np.flatnonzero(keep & shadow)
    tree = cKDTree(q[lit_ix])
    dist, nb = tree.query(q[sh_ix], k=10, distance_upper_bound=a.nb_radius)
    ok = np.isfinite(dist[:, 3])
    scan = {}
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        k_l = cos_up / (ratio @ LUM - 1)
        k_c = np.power(k_l, 1 - t) * np.power(cos_up / (ratio - 1), t)
        g = (k_c + a.floor) / k_c                      # lift of a turned-away shadow pixel: sun term raised to the floor
        corr = mo[sh_ix] * g[None, :]
        cl = np.array([np.nanmedian(ch(mo[lit_ix[nb[i][np.isfinite(dist[i])]]]), axis=0) for i in np.flatnonzero(ok)])
        d = np.abs(ch(corr[ok]) - cl).sum(1)
        scan[str(t)] = dict(median_chroma_gap=round(float(np.median(d)), 4), mean_gain_ratio_R_over_B=round(float(g[0] / g[2]), 2))
    res["tint_scan_chroma_gap_to_neighbouring_lit_blobs"] = dict(blobs=int(ok.sum()), by_tint=scan)
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
