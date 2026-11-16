#!/usr/bin/env python
"""Shadow-edge gain: correct the bear's sun-shadow pixels with the ratio measured WHERE LIT MEETS SHADOW.

Same mask as photo_correct.py (shadow labels rendered into the photo). The earlier tests matched shadow pixels to lit
paint far away by colour and failed (too weak, or too orange). Here the reference is only the paint right next to the
shadow edge, where both sides are the same surface:
1. Edge bands: lit bear pixels --gap..--gap+--band px from the nearest shadow pixel, and shadow bear pixels the same
   distance from the nearest lit pixel (the gap skips the soft edge and any mask misalignment). Silhouette edges
   (bear against background) are never used: the bands only contain bear pixels of the other class.
2. Edge ratio per channel, in LINEAR light: local mean of the lit band / local mean of the shadow band (normalised
   Gaussian blur, --edge-sigma px), kept only where both bands hold at least --min-px pixels. No white balance is
   assumed: the tint of the shade is measured on the same paint.
3. Spread inward: the gain of a shadow pixel is the normalised blur (--reach x bear height) of the edge ratios sampled in
   the shadow band. Pixels with no edge sample within reach stay unchanged (reported as uncovered). Gain >= 1, <= --max-gain.
4. corrected = photo * gain ** mask. Mask 0 gives exactly the photo.
Held-out test: fit on edge samples from even vertical stripes (--stripe px wide), measure the step left on the odd
stripes (|log(lit band / shadow band)|, before vs after), and the reverse.

Usage: photo_correct_edge.py --colmap ... --images ... --ply ... --labels ... --out exports/IMG_6705_photofix_edge
           --frames frame_00551,frame_00481,...
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from photo_correct import load_splat, nblur, render_labels, to_photo_space
from splatlib import LUM, linear_to_srgb, read_cameras_bin, read_images_bin, srgb_to_linear


def dist_to(mask):
    """Distance (px) from every pixel to the nearest True pixel of mask."""
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)


def bands(shade, lit, gap, width):
    d_lit, d_sh = dist_to(lit), dist_to(shade)
    return shade & (d_lit >= gap) & (d_lit <= gap + width), lit & (d_sh >= gap) & (d_sh <= gap + width)


def edge_ratio(P, band_sh, band_lit, sigma, min_px):
    """Per-channel lit/shadow ratio field and a validity mask, from normalised blurs of the two edge bands."""
    k = 2 * np.pi * sigma ** 2
    need = min_px / k
    out = np.ones(P.shape, np.float32)
    ok = None
    for c in range(3):
        ml, dl = nblur(P[..., c], band_lit.astype(np.float32), sigma)
        ms, ds = nblur(P[..., c], band_sh.astype(np.float32), sigma)
        v = (dl > need) & (ds > need) & (ms > 1e-5)
        out[..., c] = np.where(v, ml / np.maximum(ms, 1e-5), 1.0)
        ok = v if ok is None else ok & v
    return out, ok


def spread(ratio, ok, samples, shade, reach, max_gain, L=None, lit=None, lit_cap=0.0):
    """Gain field for shadow pixels: normalised blur of the edge ratios taken at the sample pixels.

    The cap max_gain scales R, G, B by one common factor (never changes the hue). With lit_cap > 0 a corrected pixel may
    not exceed lit_cap x the local mean luminance of the lit pixels (cross-check with the sunlit parts)."""
    w = (samples & ok).astype(np.float32)
    g = np.ones(ratio.shape, np.float32)
    cover = np.zeros(shade.shape, bool)
    for c in range(3):
        v, d = nblur(ratio[..., c], w, reach)
        have = d > 1e-3
        g[..., c] = np.where(have, v, 1.0)
        cover = have if c == 0 else cover & have
    g = np.maximum(g, 1.0)
    g = g * np.minimum(1.0, max_gain / g.max(2, keepdims=True))
    if lit_cap > 0 and L is not None:
        ll, dl = nblur(L, lit.astype(np.float32), reach)
        allow = np.where(dl > 1e-3, ll * lit_cap, np.inf)
        lum_g = g @ LUM
        k = np.minimum(1.0, allow / np.maximum(L * lum_g, 1e-6))
        k = np.maximum(k, 1.0 / np.maximum(lum_g, 1e-6))        # still never darkens
        g = g * k[..., None]
    g[~(shade & cover)] = 1.0
    return g, shade & cover


def step_metric(P, g, shade, lit, args, region):
    """Leftover lit/shadow step on the edges inside `region`: median |log| of lit band / shadow band luminance."""
    L0 = P @ LUM
    L1 = (P * g) @ LUM
    bs, bl = bands(shade, lit, args.gap, args.band)
    bs, bl = bs & region, bl & region
    k = 2 * np.pi * args.edge_sigma ** 2
    ml, dl = nblur(L0, bl.astype(np.float32), args.edge_sigma)
    m0, ds = nblur(L0, bs.astype(np.float32), args.edge_sigma)
    m1, _ = nblur(L1, bs.astype(np.float32), args.edge_sigma)
    v = (dl > args.min_px / k) & (ds > args.min_px / k) & bs
    if v.sum() < 50:
        return None
    b, a = np.log(ml[v] / np.maximum(m0[v], 1e-6)), np.log(ml[v] / np.maximum(m1[v], 1e-6))
    return dict(samples=int(v.sum()), step_before=round(float(np.median(np.abs(b))), 3),
                step_after=round(float(np.median(np.abs(a))), 3), bias_after=round(float(np.median(a)), 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--colmap", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--gap", type=float, default=4.0, help="px skipped next to the mask edge")
    ap.add_argument("--band", type=float, default=14.0, help="px width of each edge band")
    ap.add_argument("--edge-sigma", type=float, default=18.0)
    ap.add_argument("--min-px", type=float, default=150.0, help="edge pixels needed on EACH side to trust a ratio")
    ap.add_argument("--reach", type=float, default=0.12, help="spread radius as a share of the bear's height in px")
    ap.add_argument("--max-gain", type=float, default=10.0, help="cap on the largest channel gain (common factor)")
    ap.add_argument("--lit-cap", type=float, default=1.3, help="corrected pixel <= this x local mean lit luminance (0 = off)")
    ap.add_argument("--stripe", type=int, default=96, help="px width of the held-out stripes")
    ap.add_argument("--lum-only", action="store_true", help="same as --tint 0: one luminance ratio for R, G, B")
    ap.add_argument("--tint", type=float, default=1.0, help="how much of the measured colour shift to apply, 0..1")
    ap.add_argument("--crop-pad", type=float, default=0.25)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cams = read_cameras_bin(args.colmap / "cameras.bin")
    by = {i["name"].rsplit(".", 1)[0]: i for i in read_images_bin(args.colmap / "images.bin")}
    sp = load_splat(args.ply)
    lab = np.load(args.labels)
    lab_rgb = np.zeros((len(sp["means"]), 3), np.float32)
    lab_rgb[:, 2] = 1.0
    sh = lab["visible"] < 0.5
    lab_rgb[lab["index"], 0] = sh
    lab_rgb[lab["index"], 1] = ~sh
    lab_rgb[lab["index"], 2] = 0.0
    lab_rgb = torch.from_numpy(lab_rgb).cuda()
    stats = {}
    for name in args.frames.split(","):
        im = by[name]
        cam = cams[im["cam"]]
        r = to_photo_space(render_labels(sp, lab_rgb, im, cam), cam)
        bear = (r[..., 0] + r[..., 1]) / np.maximum(r.sum(2), 1e-6) > 0.5
        m = np.where(bear, r[..., 0] / np.maximum(r[..., 0] + r[..., 1], 1e-6), 0.0).astype(np.float32)
        orig8 = np.asarray(Image.open(args.images / im["name"]).convert("RGB"))
        P = srgb_to_linear(orig8.astype(np.float32) / 255)
        shade, lit = bear & (m > 0.5), bear & (m < 0.05)
        ys, xs = np.nonzero(bear)
        height = ys.max() - ys.min() + 1
        args_reach = max(args.reach * height, 8.0)
        bs, bl = bands(shade, lit, args.gap, args.band)
        t = 0.0 if args.lum_only else args.tint
        ratio, ok = edge_ratio(P, bs, bl, args.edge_sigma, args.min_px)
        if t < 1.0:     # blend with the luminance-only ratio: tint strength t (0 = brightness only, 1 = full tint)
            r_l, ok_l = edge_ratio(np.repeat((P @ LUM)[..., None], 3, 2), bs, bl, args.edge_sigma, args.min_px)
            ratio = np.power(np.maximum(ratio, 1e-6), t) * np.power(np.maximum(r_l, 1e-6), 1 - t)
            ok = ok & ok_l
        info = dict(bear_px=int(bear.sum()), shadow_px=int(shade.sum()), edge_samples=int((bs & ok).sum()),
                    reach_px=round(float(args_reach), 1))
        # held-out: even stripes fit, odd stripes judge, and the reverse
        col = (np.arange(P.shape[1]) // args.stripe) % 2
        stripe = np.broadcast_to(col[None, :], bear.shape)
        held = []
        for fit_par in (0, 1):
            g, _ = spread(ratio, ok, bs & (stripe == fit_par), shade, args_reach, args.max_gain, P @ LUM, lit, args.lit_cap)
            mm = step_metric(P, g, shade, lit, args, stripe != fit_par)
            if mm:
                held.append(mm)
        info["held_out"] = held
        g, cover = spread(ratio, ok, bs, shade, args_reach, args.max_gain, P @ LUM, lit, args.lit_cap)
        info["covered_px"] = int(cover.sum())
        info["uncovered_shadow_px"] = int(shade.sum() - cover.sum())
        w = (np.clip(m, 0, 1) * bear)[..., None]
        gw = np.power(np.maximum(g, 1e-6), w)
        res = linear_to_srgb(P * gw, clip=True)
        res8 = np.round(res * 255).astype(np.uint8)
        outside = m <= 0
        res8[outside] = orig8[outside]
        Image.fromarray(res8).save(args.out / f"{name}_corrected.png")
        Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).save(args.out / f"{name}_mask.png")
        L0, L1 = P @ LUM, srgb_to_linear(res8 / 255.0) @ LUM
        sm = m > 0.5
        lit_med = float(np.median(L0[lit])) if lit.any() else None
        info.update(shadow_lum_before=round(float(L0[sm].mean()), 4), shadow_lum_after=round(float(L1[sm].mean()), 4),
                    lit_lum=round(float(L0[lit].mean()), 4) if lit.any() else None,
                    median_gain_lum=round(float(np.median((g @ LUM)[cover])), 2) if cover.any() else None,
                    tint=[round(float(v), 2) for v in np.median(g[cover], 0)] if cover.any() else None,
                    share_above_lit_median=round(float((L1[cover] > lit_med).mean()), 3) if lit_med else None)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        ph, pw = int((y1 - y0) * args.crop_pad), int((x1 - x0) * args.crop_pad)
        y0, y1 = max(0, y0 - ph), min(P.shape[0], y1 + ph)
        x0, x1 = max(0, x0 - pw), min(P.shape[1], x1 + pw)
        ov = orig8.copy()
        ov[bs & ok] = (0.4 * ov[bs & ok] + 0.6 * np.array([255, 255, 0])).astype(np.uint8)
        ov[bl & ok] = (0.4 * ov[bl & ok] + 0.6 * np.array([0, 255, 255])).astype(np.uint8)
        ov[(m > 0.5) & ~(bs & ok)] = (0.7 * ov[(m > 0.5) & ~(bs & ok)] + 0.3 * np.array([255, 40, 40])).astype(np.uint8)
        gv = np.clip((g.mean(2) - 1) / (args.max_gain - 1), 0, 1)
        gimg = (np.stack([gv, gv * 0.8, np.zeros_like(gv)], 2) * 255).astype(np.uint8)
        strip = np.concatenate([a[y0:y1, x0:x1] for a in (orig8, ov, res8, gimg)], 1)
        Image.fromarray(strip).save(args.out / f"{name}_compare.jpg", quality=92)
        stats[name] = info
        print(name, json.dumps({k: v for k, v in info.items()}))
    json.dump(dict(args={k: str(v) for k, v in vars(args).items()}, frames=stats), open(args.out / "stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
