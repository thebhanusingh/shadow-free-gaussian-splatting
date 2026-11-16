#!/usr/bin/env python
"""Photo-correction test: brighten the bear's sun-shadow pixels in the real photos, leave every other pixel unchanged.

For each chosen photo:
1. Mask. The shadow labels (sun_visibility.npz, visible < 0.5 = shadow, from the sun refit + bear-only casters) are
   rendered into the photo's camera with gsplat: each blob carries its label as colour (R shadow bear, G lit bear,
   B everything else), so occlusion is handled by the splat itself. The render has no lens distortion, so it is warped
   into the photo's distorted pixel space (OPENCV model from COLMAP). Mask = shadow share among bear pixels.
2. Gain (the user's adjacent brightness-and-colour rule, applied to pixels), in LINEAR light:
   paint groups = k-means on the chroma of bear pixels after removing the shade tint; per group, gain =
   blurred lit luminance / blurred shadow luminance (normalised Gaussian blur, so the strength follows the spot, soft
   edges get less); one luminance factor for R, G, B plus a global shade-to-sun white balance (the concrete's
   8.5 / 6.6 / 4.2 from the PPISP run); never darkens; capped at --max-gain; the corrected pixel may not exceed the
   lit neighbours' mean luminance (cross-check with sunlit parts of the same paint).
3. corrected = photo * gain ** mask. Mask 0 gives exactly the photo (checked, reported as "outside max diff").

Usage: photo_correct.py --colmap data/IMG_6705/colmap/sparse/0_refined --images data/IMG_6705/images
           --ply exports/IMG_6705/splat.ply --labels exports/IMG_6705_sunmap3/sun_visibility.npz
           --out exports/IMG_6705_photofix [--frames frame_00321,... | --auto 5]
Writes <out>/<frame>_corrected.png, <frame>_mask.png, <frame>_compare.jpg (crop: photo | mask | corrected | gain),
stats.json.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization
from PIL import Image
from splatlib import LUM, linear_to_srgb, qvec2rotmat, read_cameras_bin, read_images_bin, read_ply, srgb_to_linear

SHADE_RATIO = np.array([8.5, 6.6, 4.2])          # concrete sun / shade, linear light (PPISP run, 2026-10-28)
WB = SHADE_RATIO / (LUM @ SHADE_RATIO)           # shade -> sun colour shift, luminance 1


def load_splat(path):
    _, d = read_ply(path)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).cuda()
    q = t(np.stack([d[f"rot_{i}"] for i in range(4)], 1))
    return dict(means=t(np.stack([d[k] for k in "xyz"], 1)), quats=q / q.norm(dim=1, keepdim=True),
                scales=torch.exp(t(np.stack([d[f"scale_{i}"] for i in range(3)], 1))),
                opac=torch.sigmoid(t(d["opacity"])))


def render_labels(sp, lab_rgb, im, cam):
    W, H = int(cam["width"]), int(cam["height"])
    fx, fy, cx, cy = cam["params"][:4]
    w2c = np.eye(4)
    w2c[:3, :3] = qvec2rotmat(im["q"])
    w2c[:3, 3] = im["t"]
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32).cuda()[None]
    vm = torch.tensor(w2c, dtype=torch.float32).cuda()[None]
    with torch.no_grad():
        out, alpha, _ = rasterization(sp["means"], sp["quats"], sp["scales"], sp["opac"], lab_rgb, vm, K, W, H,
                                      sh_degree=None, render_mode="RGB")
    return out[0].cpu().numpy()                      # (H, W, 3): R shadow bear, G lit bear, B other (alpha weighted)


def to_photo_space(img, cam):
    """Warp an image rendered WITHOUT lens distortion into the photo's distorted pixels."""
    W, H = int(cam["width"]), int(cam["height"])
    fx, fy, cx, cy = cam["params"][:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    dist = np.array(cam["params"][4:8], float)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pts = np.stack([gx.ravel(), gy.ravel()], 1)[:, None, :].astype(np.float64)
    und = cv2.undistortPoints(pts, K, dist, P=K).reshape(H, W, 2).astype(np.float32)
    return cv2.remap(img, und[..., 0], und[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def nblur(x, w, sigma):
    """Normalised Gaussian blur: weighted local mean of x (H,W) with weights w; second return = weight mass."""
    k = int(sigma * 6) | 1
    num = cv2.GaussianBlur((x * w).astype(np.float32), (k, k), sigma)
    den = cv2.GaussianBlur(w.astype(np.float32), (k, k), sigma)
    return num / np.maximum(den, 1e-6), den


def correct(photo_lin, mask, bear, args):
    """photo_lin (H,W,3) linear; mask (H,W) 0..1 shadow share; bear (H,W) bool. Returns corrected, gain (H,W,3), info."""
    H, W, _ = photo_lin.shape
    ys, xs = np.nonzero(bear)
    height = ys.max() - ys.min() + 1
    sig = max(args.sigma * height, 4.0)
    shade = bear & (mask > 0.5)
    lit = bear & (mask < 0.05)
    L = photo_lin @ LUM
    gain = np.ones((H, W), np.float32)
    info = dict(bear_px=int(bear.sum()), shadow_px=int(shade.sum()), lit_px=int(lit.sum()), sigma_px=round(float(sig), 1))
    if shade.sum() < 50 or lit.sum() < 50:
        return photo_lin, np.ones((H, W, 3), np.float32), info
    # paint groups: chroma after removing the shade tint from shadow pixels
    fix = photo_lin.copy()
    fix[shade] = fix[shade] * WB
    ch = fix / np.maximum(fix.sum(2, keepdims=True), 1e-6)
    sel = np.flatnonzero((shade | lit).ravel())
    feats = ch.reshape(-1, 3)[sel][:, :2].astype(np.float32)
    k = args.groups
    _, lab, cen = cv2.kmeans(feats, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01), 3,
                             cv2.KMEANS_PP_CENTERS)
    group = np.full(H * W, -1, int)
    group[sel] = lab.ravel()
    group = group.reshape(H, W)
    gl = []
    for j in range(k):
        wl, ws = (lit & (group == j)).astype(np.float32), (shade & (group == j)).astype(np.float32)
        if wl.sum() < 30 or ws.sum() < 30:
            continue
        Ll, dl = nblur(L, wl, sig)
        Ls, ds = nblur(L, ws, sig)
        Ll2, dl2 = nblur(L, wl, sig * 3)
        Ls2, ds2 = nblur(L, ws, sig * 3)
        Ll = np.where(dl > 1e-3, Ll, Ll2)
        Ls = np.where(ds > 1e-3, Ls, Ls2)
        g = np.clip(Ll / np.maximum(Ls, 1e-6), 1.0, args.max_gain)
        cap = np.maximum(1.0, Ll / np.maximum(L, 1e-6))           # not brighter than the lit neighbours of this paint
        g = np.minimum(g, cap)
        m = (group == j) & shade
        gain[m] = g[m]
        gl.append(dict(group=j, chroma=[round(float(v), 3) for v in cen[j]], shadow_px=int(ws.sum()), lit_px=int(wl.sum()),
                       median_gain=round(float(np.median(g[m])), 2) if m.any() else None))
    # shadow pixels left without a paint group (no lit reference of that paint): use the median gain of the others
    miss = shade & (gain == 1.0)
    if gl and miss.any():
        med = float(np.median(gain[shade & (gain > 1.0)])) if (shade & (gain > 1.0)).any() else 1.0
        gain[miss] = med
    info["groups"] = gl
    info["unmatched_px"] = int(miss.sum())
    gain3 = gain[..., None] * WB[None, None, :]
    gain3 = np.where(gain[..., None] > 1.0, gain3, 1.0)           # tint fix only where the pixel is brightened
    w = np.clip(mask, 0, 1)[..., None] * bear[..., None]
    out = photo_lin * np.power(np.maximum(gain3, 1e-6), w)
    return out, np.power(np.maximum(gain3, 1e-6), w), info


def pick_frames(imgs, n):
    up = np.mean([-qvec2rotmat(i["q"]).T @ np.array([0, 1.0, 0]) for i in imgs], 0)
    up /= np.linalg.norm(up)
    pos = np.array([-qvec2rotmat(i["q"]).T @ i["t"] for i in imgs])
    c = pos.mean(0)
    a = np.arctan2((pos - c) @ np.cross(up, [1, 0, 0]), (pos - c) @ np.cross(up, np.cross(up, [1, 0, 0])))
    return np.floor((a + np.pi) / (2 * np.pi) * n).astype(int).clip(0, n - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--colmap", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", default="", help="comma list of frame names (without .png)")
    ap.add_argument("--auto", type=int, default=0, help="pick this many frames spread around the bear, most shadow each")
    ap.add_argument("--every", type=int, default=10, help="with --auto: consider every N-th frame")
    ap.add_argument("--sigma", type=float, default=0.06, help="reference blur as a share of the bear's height in pixels")
    ap.add_argument("--groups", type=int, default=6, help="paint groups (k-means on chroma)")
    ap.add_argument("--max-gain", type=float, default=6.0)
    ap.add_argument("--crop-pad", type=float, default=0.25, help="crop margin as a share of the bear box")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cams = read_cameras_bin(args.colmap / "cameras.bin")
    imgs = read_images_bin(args.colmap / "images.bin")
    by = {i["name"].rsplit(".", 1)[0]: i for i in imgs}
    sp = load_splat(args.ply)
    lab = np.load(args.labels)
    n = len(sp["means"])
    lab_rgb = np.zeros((n, 3), np.float32)
    lab_rgb[:, 2] = 1.0
    idx, sh = lab["index"], lab["visible"] < 0.5
    lab_rgb[idx, 0] = sh
    lab_rgb[idx, 1] = ~sh
    lab_rgb[idx, 2] = 0.0
    lab_rgb = torch.from_numpy(lab_rgb).cuda()

    def frame_mask(name):
        im = by[name]
        cam = cams[im["cam"]]
        r = to_photo_space(render_labels(sp, lab_rgb, im, cam), cam)
        tot = np.maximum(r.sum(2), 1e-6)
        bear = (r[..., 0] + r[..., 1]) / tot > 0.5
        m = np.where(bear, r[..., 0] / np.maximum(r[..., 0] + r[..., 1], 1e-6), 0.0)
        return im, cam, bear, m.astype(np.float32)

    names = [f for f in args.frames.split(",") if f]
    if args.auto:
        cand = sorted(by)[::args.every]
        cim = [by[c] for c in cand]
        sector = pick_frames(cim, args.auto)
        scores = {}
        for c in cand:
            _, _, bear, m = frame_mask(c)
            scores[c] = float(((m > 0.5) & bear).sum())
        for s in range(args.auto):
            pool = [c for c, sec in zip(cand, sector) if sec == s]
            if pool:
                names.append(max(pool, key=lambda c: scores[c]))
        print("auto frames:", names, {c: int(scores[c]) for c in names})
    stats = {}
    for name in names:
        im, cam, bear, m = frame_mask(name)
        photo = np.asarray(Image.open(args.images / im["name"]).convert("RGB"), np.float32) / 255
        lin = srgb_to_linear(photo)
        out, gain, info = correct(lin, m, bear, args)
        res = linear_to_srgb(out, clip=True).astype(np.float32)
        res8 = np.round(res * 255).astype(np.uint8)
        orig8 = np.asarray(Image.open(args.images / im["name"]).convert("RGB"))
        outside = m <= 0
        diff_out = int(np.abs(res8.astype(int) - orig8.astype(int))[outside].max()) if outside.any() else 0
        res8[outside] = orig8[outside]                        # exact copy outside the mask (no round-trip noise)
        Image.fromarray(res8).save(args.out / f"{name}_corrected.png")
        Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).save(args.out / f"{name}_mask.png")
        ys, xs = np.nonzero(bear)
        if len(ys):
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            ph, pw = int((y1 - y0) * args.crop_pad), int((x1 - x0) * args.crop_pad)
            y0, y1 = max(0, y0 - ph), min(photo.shape[0], y1 + ph)
            x0, x1 = max(0, x0 - pw), min(photo.shape[1], x1 + pw)
            ov = orig8.copy()
            ov[(m > 0.5)] = (0.5 * ov[(m > 0.5)] + 0.5 * np.array([255, 40, 40])).astype(np.uint8)
            gv = np.clip((gain.mean(2) - 1) / (args.max_gain - 1), 0, 1)
            gimg = (np.stack([gv, gv * 0.8, np.zeros_like(gv)], 2) * 255).astype(np.uint8)
            strip = np.concatenate([a[y0:y1, x0:x1] for a in (orig8, ov, res8, gimg)], 1)
            Image.fromarray(strip).save(args.out / f"{name}_compare.jpg", quality=92)
        L0, L1 = (lin @ LUM), (srgb_to_linear(res8 / 255.0) @ LUM)
        sm = m > 0.5
        info.update(frame=name, outside_max_diff_before_copy=diff_out, outside_px_changed=0,
                    shadow_lum_before=round(float(L0[sm].mean()), 4) if sm.any() else None,
                    shadow_lum_after=round(float(L1[sm].mean()), 4) if sm.any() else None,
                    lit_lum=round(float(L0[bear & (m < 0.05)].mean()), 4) if (bear & (m < 0.05)).any() else None)
        stats[name] = info
        print(name, {k: v for k, v in info.items() if k != "groups"})
    json.dump(dict(args={k: str(v) for k, v in vars(args).items()}, wb=[round(float(v), 3) for v in WB], frames=stats),
              open(args.out / "stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
