#!/usr/bin/env python
"""Relight the bear in the photos with a moved sun: does the back light up, and does it stay clean?

Per bear pixel, linear light, per channel c:
    old light  E_old = k_c + max(0, n.s_old) * (1 - shadow mask)      (sun strength 1, k_c = sky/sun from the concrete)
    paint      rho   = photo / E_old
    new light  E_new = k_c + max(0, n.s_new) * [sun at s_new reaches the blob]
    relit      = rho * E_new
n = 2DGS normals rendered into the photo, [sun reaches the blob] = the depth-map visibility of sky_visibility.py for the new
direction (rendered into the photo through the blobs), shadow mask = the accepted sun labels. Sky shape is flat, no sky
visibility (it did not help). Pixels outside the bear are the photo.
New suns: back = the real sun turned 180 degrees about the vertical (same elevation); overhead = same azimuth at 80 deg.
Reports, per frame: brightness of the pixels the back sun now lights (before / after), noise (coefficient of variation in
5 x 5 blocks of those pixels against sunlit pixels of the photo), share clipped above 1, and how well the paint matches
across the old shadow edges (median |log(paint lit band / paint shadow band)|), cast edges (shadow side faces the sun)
against attached edges (shadow side faces away: its paint comes from sky-only pixels).

Usage: relight_photos.py --surfels ... --ply ... --labels ... --sun ... --object ... --plane-from ... --colmap ... --images ...
           --out exports/IMG_6705_relight_photos --frames frame_00551,...
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from lighting_pixel_test import render_normals
from photo_correct import load_splat, nblur, render_labels, to_photo_space
from photo_correct_edge import bands
from PIL import Image
from sky_visibility import add_arguments, build
from splatlib import LUM, linear_to_srgb, plane_frame, read_cameras_bin, read_images_bin, srgb_to_linear, sun_vec
from train_2dgs_ref import surfel_normals

RATIO = np.array([8.5, 6.6, 4.2])
K_RGB = np.cos(np.radians(45)) / (RATIO - 1)


def block_cv(L, sel, size=5):
    """Coefficient of variation of luminance in size x size blocks that lie fully inside sel."""
    H, W = L.shape
    h, w = H // size * size, W // size * size
    b = L[:h, :w].reshape(h // size, size, w // size, size).transpose(0, 2, 1, 3).reshape(-1, size * size)
    ok = sel[:h, :w].reshape(h // size, size, w // size, size).transpose(0, 2, 1, 3).reshape(-1, size * size).all(1)
    b = b[ok]
    return float(np.median(b.std(1) / np.maximum(b.mean(1), 1e-6))) if len(b) > 50 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    ap.add_argument("--colmap", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--k-scale", type=float, default=1.0)
    ap.add_argument("--smooth", type=float, default=2.0)
    ap.add_argument("--gap", type=float, default=4.0)
    ap.add_argument("--band", type=float, default=14.0)
    ap.add_argument("--edge-sigma", type=float, default=18.0)
    ap.add_argument("--min-px", type=float, default=150.0)
    ap.add_argument("--crop-pad", type=float, default=0.25)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    b = build(a)
    up, s_old, frame = b["up"], b["sun"], b["frame"]
    hor = s_old - (s_old @ up) * up
    s_back = (s_old @ up) * up - hor
    az = np.degrees(np.arctan2(s_old @ frame[2], s_old @ frame[1]))
    s_top = sun_vec(80, az, frame)
    suns = {"back": s_back / np.linalg.norm(s_back), "overhead": s_top}
    new_vis = {k: b["visible_from"](v) for k, v in suns.items()}
    print({k: f"{v.mean():.2f} of blobs lit" for k, v in new_vis.items()}, "elevation old",
          round(float(np.degrees(np.arcsin(s_old @ up))), 1), "back", round(float(np.degrees(np.arcsin(suns["back"] @ up))), 1))
    cams = read_cameras_bin(a.colmap / "cameras.bin")
    by = {i["name"].rsplit(".", 1)[0]: i for i in read_images_bin(a.colmap / "images.bin")}
    sp = load_splat(a.ply)
    n = len(sp["means"])
    idx = b["idx"]

    def blob_colours(flag):
        arr = np.zeros((n, 3), np.float32)
        arr[:, 2] = 1.0
        arr[idx, 0], arr[idx, 1], arr[idx, 2] = flag, 1.0, 0.0
        return torch.from_numpy(arr).cuda()

    old_lit = np.load(a.labels)["visible"] >= 0.5
    col_old = blob_colours(~old_lit)                      # channel 0 = shadow flag (old sun)
    col_new = {k: blob_colours(v) for k, v in new_vis.items()}
    surf = {k: v.cuda() for k, v in torch.load(a.surfels).items()}
    s_n = surfel_normals(surf["quats"])
    k = K_RGB * a.k_scale
    stats = {}
    for name in a.frames.split(","):
        im = by[name]
        cam = cams[im["cam"]]

        def flag_img(col):
            r = to_photo_space(render_labels(sp, col, im, cam), cam)
            return np.where(r[..., 1] > 0.5, r[..., 0] / np.maximum(r[..., 1], 1e-6), 0.0).astype(np.float32), r[..., 1] > 0.5

        m, bear = flag_img(col_old)
        ncol, alpha = render_normals(surf, s_n, im, cam)
        ncol, alpha = to_photo_space(ncol, cam), to_photo_space(alpha, cam)
        if a.smooth > 0:
            kk = int(a.smooth * 6) | 1
            ncol = cv2.GaussianBlur(ncol, (kk, kk), a.smooth)
        nrm = ncol / np.maximum(np.linalg.norm(ncol, axis=2, keepdims=True), 1e-6)
        orig8 = np.asarray(Image.open(a.images / im["name"]).convert("RGB"))
        P = srgb_to_linear(orig8.astype(np.float32) / 255)
        L = P @ LUM
        use = bear & (alpha > 0.5)
        E_old = k[None, None, :] + (np.maximum(nrm @ s_old, 0) * (1 - m))[..., None]
        rho = P / E_old
        outs, info = {}, {}
        for tag, sv in suns.items():
            fl, _ = flag_img(col_new[tag])
            E_new = k[None, None, :] + (np.maximum(nrm @ sv, 0) * fl)[..., None]
            lin = np.where(use[..., None], rho * E_new, P)
            res = linear_to_srgb(lin, clip=True)
            outs[tag] = np.round(res * 255).astype(np.uint8)
            Image.fromarray(outs[tag]).save(a.out / f"{name}_{tag}.png")
            newly = use & (np.maximum(nrm @ sv, 0) * fl > 0.3) & (m > 0.5)      # lit now, in the old shadow
            Ln = lin @ LUM
            ref = use & (m < 0.05) & (np.maximum(nrm @ s_old, 0) > 0.3)
            info[tag] = dict(newly_lit_px=int(newly.sum()),
                             brightness_before=round(float(L[newly].mean()), 4) if newly.any() else None,
                             brightness_after=round(float(Ln[newly].mean()), 4) if newly.any() else None,
                             noise_cv_newly_lit_after=block_cv(Ln, newly), noise_cv_newly_lit_before=block_cv(L, newly),
                             noise_cv_sunlit_photo=block_cv(L, ref), clipped_share=round(float((lin[use] > 1).any(1).mean()), 3))
        paint = np.where(use[..., None], rho * (k + 0.3)[None, None, :], P)
        paint8 = np.round(linear_to_srgb(paint, clip=True) * 255).astype(np.uint8)
        Image.fromarray(paint8).save(a.out / f"{name}_paint.png")
        # paint match across the old shadow edges
        cosv = np.maximum(nrm @ s_old, 0)
        shade, lit = use & (m > 0.5), use & (m < 0.05)
        bs, bl = bands(shade, lit, a.gap, a.band)
        R = rho @ LUM
        need = a.min_px / (2 * np.pi * a.edge_sigma ** 2)
        paint_step = {}
        for label, reg in (("cast_edges", cosv > 0.15), ("attached_edges", cosv <= 0.15)):
            sb = bs & reg
            ml, dl = nblur(R, bl.astype(np.float32), a.edge_sigma)
            ms, ds = nblur(R, sb.astype(np.float32), a.edge_sigma)
            v = (dl > need) & (ds > need) & sb
            paint_step[label] = (dict(windows=int(v.sum()), median_abs_log=round(float(np.median(np.abs(np.log(ml[v] / np.maximum(ms[v], 1e-9))))), 3),
                                      median_signed_log=round(float(np.median(np.log(ml[v] / np.maximum(ms[v], 1e-9)))), 3)) if v.sum() > 200 else None)
        ys, xs = np.nonzero(use)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        ph, pw = int((y1 - y0) * a.crop_pad), int((x1 - x0) * a.crop_pad)
        y0, y1, x0, x1 = max(0, y0 - ph), min(P.shape[0], y1 + ph), max(0, x0 - pw), min(P.shape[1], x1 + pw)
        Image.fromarray(np.concatenate([x[y0:y1, x0:x1] for x in (orig8, paint8, outs["back"], outs["overhead"])], 1)).save(
            a.out / f"{name}_relight.jpg", quality=92)
        stats[name] = dict(relight=info, paint_step=paint_step)
        print(name, json.dumps(stats[name]))
    json.dump(dict(args={q: str(v) for q, v in vars(a).items()}, k=[round(float(v), 3) for v in k], frames=stats),
              open(a.out / "stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
