#!/usr/bin/env python
"""Correct the sun shadows in the photos with the lighting physics, bear and ground, for training a normal splat.

BEAR (per pixel, linear light, per channel c; sun strength 1, n = 2DGS normals rendered into the photo, s = refitted sun):
    light now      E_old = (1-m)(k_c + cos) + m * k_c * scale_class        m = sun-shadow mask, cos = max(0, n.s)
    paint          rho   = photo / E_old
    light wanted   E_new = k_c + max(cos, --floor)                        sun term raised to at least --floor
    result         rho * E_new, never darker than the photo, at most --lit-cap x the local mean of the lit pixels
A lit pixel with cos >= floor has E_new = E_old, so it stays the photo; shadowed and turned-away pixels are lifted to the
floor (cast shadows on sun-facing surfaces get at least their own cos back). Surfaces turned away from the sun get at most --attached-max (they keep their form shading). The lifted colour may not be more saturated than the sunlit paint of the same hue (--sat-cap: per-hue saturation percentile of the sunlit bear pixels, measured in the calibration). The gain is feathered over the bear's edge.
GROUND: the shadow the bear throws on the concrete (the SDDNet photo mask on ground-plane pixels) is lifted by the concrete's
sun/shade ratio, measured from the photos themselves in the calibration, and takes the colour of the lit concrete around it.
CALIBRATION (no hand values): on every --calib-every-th photo the script measures, across the old shadow edges, how the
paint of the lit side compares with the paint of the shadow side. The colour strength t of the sky (k_c from the concrete
= 0 luminance only .. 1 full colour) is the value that makes the paint chroma match across edges where the shadow side
faces away from the sun; the scale of k for shadow pixels (cast: faces the sun; attached: faces away) is the value that
makes the paint brightness match across the same edges. These are pooled over the photos and fixed for all of them.

Writes <out>/images/<same names>.png (all photos; unselected ones are not written), <out>/calibration.json, compare crops.

Usage: correct_photos.py --surfels ... --ply ... --labels ... --sun ... --object ... --plane-from ... --colmap ... --images ...
           --out data/IMG_6705_fixed [--frames a,b,c | --every N]
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
from scipy.spatial import cKDTree
from splatlib import LUM, SH_C0, linear_to_srgb, qvec2rotmat, read_cameras_bin, read_images_bin, read_ply, srgb_to_linear
from train_2dgs_ref import render, surfel_normals

RATIO = np.array([8.5, 6.6, 4.2])                                # concrete sun / shade, linear light
K_RGB = np.cos(np.radians(45)) / (RATIO - 1)
K_LUM = float(np.cos(np.radians(45)) / (RATIO @ LUM - 1))


def smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


class Scene:
    """Everything that does not change from photo to photo."""

    def __init__(self, a):
        self.a = a
        self.cams = read_cameras_bin(a.colmap / "cameras.bin")
        self.by = {i["name"].rsplit(".", 1)[0]: i for i in read_images_bin(a.colmap / "images.bin")}
        self.names = sorted(self.by)
        self.sp = load_splat(a.ply)
        n = len(self.sp["means"])
        b = build(a)                                              # bear: sun visibility of the labelled blobs
        self.up, self.sun = b["up"], b["sun"]
        obj = json.load(open(a.object))
        gp = json.load(open(a.plane_from))["ground_plane"]
        self.plane_off, self.plane_tol = float(gp["offset"]), float(gp["tol"])
        self.foot, self.height = np.array(obj["footprint_centre"], float), float(obj["height"])
        lab = np.load(a.labels)
        self.bear_idx = lab["index"]
        arr = np.zeros((n, 3), np.float32)
        arr[:, 2] = 1.0
        arr[self.bear_idx, 0] = ~(lab["visible"] >= 0.5)           # shadow flag
        arr[self.bear_idx, 1], arr[self.bear_idx, 2] = 1.0, 0.0
        self.col_bear = torch.from_numpy(arr).cuda()
        del b
        # chroma target for every shadowed bear blob: the median chroma of the sunlit bear blobs next to it (same surface, same
        # paint, from the splat's own colours). One value per blob, so every photo asks for the same colour (consistency).
        _, gg0 = read_ply(a.ply)
        obj0 = json.load(open(a.object))
        rgb0 = srgb_to_linear(np.clip(np.stack([gg0[f"f_dc_{i}"][self.bear_idx] for i in range(3)], 1) * SH_C0 + 0.5, 0, 1))
        pos0 = np.stack([gg0[k][self.bear_idx] for k in "xyz"], 1).astype(float)
        lit0 = lab["visible"] >= 0.5
        chroma0 = rgb0 / np.maximum(rgb0.sum(1, keepdims=True), 1e-6)
        tree = cKDTree(pos0[lit0 & (rgb0 @ LUM > 0.01)])
        lit_chroma = chroma0[lit0 & (rgb0 @ LUM > 0.01)]
        sh_ix = np.flatnonzero(~lit0)
        dist, nb = tree.query(pos0[sh_ix], k=10, distance_upper_bound=a.chroma_radius * float(obj0["height"]))
        tgt = np.zeros((len(self.bear_idx), 3), np.float32)
        okb = np.isfinite(dist[:, 2])
        for j in np.flatnonzero(okb):
            tgt[sh_ix[j], :2] = np.median(lit_chroma[nb[j][np.isfinite(dist[j])]][:, :2], axis=0)
            tgt[sh_ix[j], 2] = 1.0
        arr2 = np.zeros((n, 3), np.float32)
        arr2[self.bear_idx] = tgt
        self.col_target = torch.from_numpy(arr2).cuda()
        print(f"chroma targets for {int(okb.sum()):,} of {len(sh_ix):,} shadowed bear blobs")
        a.out.mkdir(parents=True, exist_ok=True)
        surf = {k: v.cuda() for k, v in torch.load(a.surfels).items()}
        self.surf, self.s_n = surf, surfel_normals(surf["quats"])

    def frame(self, name):
        a, im = self.a, self.by[name]
        cam = self.cams[im["cam"]]

        def flag(col):
            r = to_photo_space(render_labels(self.sp, col, im, cam), cam)
            return np.where(r[..., 1] > 0.5, r[..., 0] / np.maximum(r[..., 1], 1e-6), 0.0).astype(np.float32), r[..., 1] > 0.5

        mb, bear = flag(self.col_bear)
        rt = to_photo_space(render_labels(self.sp, self.col_target, im, cam), cam)
        ct_ok = rt[..., 2] > 0.3
        ct = np.where(ct_ok[..., None], rt[..., :2] / np.maximum(rt[..., 2:3], 1e-6), 0.0).astype(np.float32)
        ncol, alpha = render_normals(self.surf, self.s_n, im, cam)
        ncol, alpha = to_photo_space(ncol, cam), to_photo_space(alpha, cam)
        if a.smooth > 0:
            kk = int(a.smooth * 6) | 1
            ncol = cv2.GaussianBlur(ncol, (kk, kk), a.smooth)
        nrm = ncol / np.maximum(np.linalg.norm(ncol, axis=2, keepdims=True), 1e-6)
        orig8 = np.asarray(Image.open(a.images / im["name"]).convert("RGB"))
        cos = np.maximum(nrm @ self.sun, 0).astype(np.float32)
        use0 = bear & (alpha > 0.3)
        use = cv2.morphologyEx(use0.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)).astype(bool)
        hole = use & ~use0                                           # small gaps inside the bear: take the neighbours' values
        if hole.any():
            mb = np.where(hole, nblur(mb, use0.astype(np.float32), 5.0)[0], mb)
            cos = np.where(hole, nblur(cos, use0.astype(np.float32), 5.0)[0], cos)
        mask = np.asarray(Image.open(a.masks / (name + ".png"))) > 127
        gpx = ground_pixels(self, im, cam, a.ground_radius)
        bear_d = cv2.dilate(use.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        ground = mask & gpx & ~bear_d
        ground = cv2.erode(ground.astype(np.uint8), np.ones((a.ground_erode, a.ground_erode), np.uint8)).astype(bool) if a.ground_erode > 1 else ground
        mg = np.clip(cv2.GaussianBlur(ground.astype(np.float32), (0, 0), a.ground_soft), 0, 1)
        lit_g = gpx & ~cv2.dilate((mask & gpx).astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool) & ~bear_d
        ring = lit_g & cv2.dilate(ground.astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool)
        inner = cv2.erode(ground.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
        gdist = cv2.distanceTransform(ground.astype(np.uint8), cv2.DIST_L2, 3)       # px from the shadow's edge, inside it
        return dict(gdist=gdist, name=name, im=im, orig8=orig8, P=srgb_to_linear(orig8.astype(np.float32) / 255), mb=mb, bear=bear,
                    mg=mg, ground=mg > 0.02, lit_ground=lit_g, cos=cos, use=use, ring=ring, inner=inner, ct=ct, ct_ok=ct_ok)


def ground_pixels(scene, im, cam, radius):
    """Photo pixels whose surfel surface lies on the ground plane within `radius` x object height of the bear's footprint."""
    W, H = int(cam["width"]), int(cam["height"])
    fx, fy, cx, cy = cam["params"][:4]
    w2c = np.eye(4)
    w2c[:3, :3] = qvec2rotmat(im["q"])
    w2c[:3, 3] = im["t"]
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32).cuda()
    vm = torch.tensor(w2c, dtype=torch.float32).cuda()
    with torch.no_grad():
        out = render(scene.surf, vm, K, W, H, 0)                      # same call as the sun map: out[5] = median depth (z)
    alpha, depth = out[1][0, ..., 0].cpu().numpy(), out[5][0, ..., 0].cpu().numpy()
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    xc = np.stack([(gx - cx) / fx * depth, (gy - cy) / fy * depth, depth], 2)
    Xw = (xc - w2c[:3, 3]) @ w2c[:3, :3]                             # R^T (x - t)
    h = Xw @ scene.up + scene.plane_off
    r = np.linalg.norm(Xw - scene.foot, axis=2)
    ok = ((alpha > 0.5) & (np.abs(h) < 4 * scene.plane_tol) & (r < radius * scene.height) & (depth > 0)).astype(np.float32)
    return to_photo_space(ok, cam) > 0.5


HUE_BINS = 24


def hue_sat(c):
    """Hue bin (0..HUE_BINS-1) and saturation (max - min) / max of linear RGB (..., 3)."""
    mx, mn = c.max(-1), c.min(-1)
    h = np.arctan2(np.sqrt(3.0) * (c[..., 1] - c[..., 2]), 2 * c[..., 0] - c[..., 1] - c[..., 2])
    return ((h + np.pi) / (2 * np.pi) * HUE_BINS).astype(int) % HUE_BINS, (mx - mn) / np.maximum(mx, 1e-6)


def saturation_caps(frames, pct):
    """Per hue bin, the pct-th percentile of the saturation of the clearly sunlit bear pixels (pooled over the photos)."""
    hs, ss = [], []
    for f in frames:
        sel = f["use"] & (f["mb"] < 0.05) & (f["cos"] > 0.3) & ((f["P"] @ LUM) > 0.02)
        if sel.sum() > 500:
            ix = np.flatnonzero(sel.ravel())
            ix = ix[::max(1, len(ix) // 4000)]
            h, sat_ = hue_sat(f["P"].reshape(-1, 3)[ix])
            hs.append(h)
            ss.append(sat_)
    h, sat_ = np.concatenate(hs), np.concatenate(ss)
    caps = np.ones(HUE_BINS)
    for b in range(HUE_BINS):
        v = sat_[h == b]
        caps[b] = np.percentile(v, pct) if len(v) > 100 else 1.0
    return caps


def k_of(t, ratio=RATIO, cos_up=float(np.cos(np.radians(45)))):
    """Sky light over sun light for an open upward surface, from the concrete's sun/shade ratio and the sun's height:
    ratio = 1 + cos_up / k.  t = 0 luminance only .. 1 full colour."""
    ratio = np.asarray(ratio, float)
    return np.power(cos_up / (ratio @ LUM - 1), 1 - t) * np.power(cos_up / (ratio - 1), t)


def paint_and_light(f, k, scale_att, scale_cast):
    """rho (H,W,3) and E_old for the bear pixels."""
    cos, m = f["cos"], f["mb"]
    sc = scale_att + (scale_cast - scale_att) * smoothstep(cos, 0.05, 0.25)
    E_old = (1 - m)[..., None] * (k[None, None, :] + cos[..., None]) + m[..., None] * (k[None, None, :] * sc[..., None])
    return f["P"] / np.maximum(E_old, 1e-4), E_old


def edge_stats(f, rho, a):
    """Signed log paint step (lit / shadow) and chroma distance across the old shadow edges, cast and attached."""
    use, m, cos = f["use"], f["mb"], f["cos"]
    shade, lit = use & (m > 0.5), use & (m < 0.05)
    bs, bl = bands(shade, lit, a.gap, a.band)
    need = a.min_px / (2 * np.pi * a.edge_sigma ** 2)
    R = rho @ LUM
    ch = rho / np.maximum(rho.sum(2, keepdims=True), 1e-9)
    out = {}
    for label, reg in (("cast", cos > 0.15), ("attached", cos <= 0.15)):
        sb = bs & reg
        ml, dl = nblur(R, bl.astype(np.float32), a.edge_sigma)
        ms, ds = nblur(R, sb.astype(np.float32), a.edge_sigma)
        v = (dl > need) & (ds > need) & sb
        if v.sum() < 200:
            out[label] = None
            continue
        cd = 0.0
        for c in range(3):
            cl, _ = nblur(ch[..., c], bl.astype(np.float32), a.edge_sigma)
            cs, _ = nblur(ch[..., c], sb.astype(np.float32), a.edge_sigma)
            cd = cd + np.abs(cl[v] - cs[v])
        out[label] = dict(n=int(v.sum()), log_step=float(np.median(np.log(ml[v] / np.maximum(ms[v], 1e-9)))), chroma=float(np.median(cd)))
    return out


def measure_ground_ratio(frames):
    """Concrete sun/shade ratio per channel (linear light) from the photos: lit concrete in a ring around the bear's shadow
    over the shadow's inner part, median over the photos."""
    rows, w = [], []
    for f in frames:
        if f["ring"].sum() > 3000 and f["inner"].sum() > 3000:
            rows.append([f["P"][..., c][f["ring"]].mean() / max(f["P"][..., c][f["inner"]].mean(), 1e-6) for c in range(3)])
            w.append(f["inner"].sum())
    if len(rows) < 3:
        return None, len(rows)
    rows, w = np.array(rows), np.array(w, float)
    out = []
    for c in range(3):
        o = np.argsort(rows[:, c])
        out.append(float(rows[o, c][np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.5)]))
    return np.array(out), len(rows)


def calibrate(scene, names, a):
    frames = [scene.frame(n) for n in names]
    ratio, nr = measure_ground_ratio(frames)
    cos_up = float(scene.sun @ scene.up)
    print(f"  sun {np.degrees(np.arcsin(cos_up)):.1f} deg above the ground; concrete sun/shade ratio from {nr} photos: "
          f"{None if ratio is None else ratio.round(2)} (the PPISP measure on bear 2 was {RATIO})")
    if ratio is None:
        ratio = RATIO
    best_t, rows = None, {}
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        k = k_of(t, ratio, cos_up)
        tot, cnt = 0.0, 0
        for f in frames:
            rho, _ = paint_and_light(f, k, 1.0, 1.0)
            e = edge_stats(f, rho, a)["attached"]
            if e:
                tot, cnt = tot + e["chroma"] * e["n"], cnt + e["n"]
        rows[t] = tot / max(cnt, 1)
        print(f"  colour strength t={t}: paint chroma step across attached edges {rows[t]:.4f}")
    best_t = min(rows, key=rows.get)
    k = k_of(best_t, ratio, cos_up)
    steps = {"cast": [], "attached": []}
    for f in frames:
        rho, _ = paint_and_light(f, k, 1.0, 1.0)
        for lab, e in edge_stats(f, rho, a).items():
            if e:
                steps[lab].append((e["log_step"], e["n"]))
    scales = {}
    for lab, v in steps.items():
        if v:
            vals, w = np.array([x[0] for x in v]), np.array([x[1] for x in v], float)
            o = np.argsort(vals)
            med = vals[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.5)]
            scales[lab] = float(np.clip(np.exp(-med), 0.3, 1.2))
        else:
            scales[lab] = 1.0
    return dict(t=best_t, chroma_by_t=rows, scale_attached=scales["attached"], scale_cast=scales["cast"], frames=names,
                ratio=[float(v) for v in ratio], cos_up=cos_up, sat_caps=[float(v) for v in saturation_caps(frames, a.sat_pct)])


def correct(f, cal, a):
    P, use, m, cos = f["P"], f["use"], f["mb"], f["cos"]
    ratio = np.array(cal.get("ratio", RATIO), float)
    k = k_of(cal["t"], ratio, cal.get("cos_up", float(np.cos(np.radians(45)))))
    rho, _ = paint_and_light(f, k, cal["scale_attached"], cal["scale_cast"])
    E_new = k[None, None, :] + np.maximum(cos, a.floor)[..., None]
    gain = np.maximum(rho * E_new / np.maximum(P, 1e-6), 1.0)
    # surfaces turned away from the sun keep their form shading: their lift is limited (cast shadows get the full gain)
    cap_g = a.attached_max + (1e3 - a.attached_max) * smoothstep(cos, 0.05, 0.25)
    gain = np.minimum(gain, cap_g[..., None])
    L = P @ LUM
    ys, xs = np.nonzero(use)
    height = max(ys.max() - ys.min() + 1, 50) if len(ys) else 100
    lit = use & (m < 0.05) & (cos > 0.3)
    ll, dl = nblur(L, lit.astype(np.float32), max(0.12 * height, 8.0))
    frame_lit = float(L[lit].mean()) if lit.sum() > 200 else np.inf        # no sunlit pixel near: the frame's mean
    allow = np.where(dl > 1e-3, ll, frame_lit) * a.lit_cap
    lum_gain = gain @ LUM
    kfac = np.minimum(1.0, allow / np.maximum(L * lum_gain, 1e-6))
    kfac = np.maximum(kfac, 1.0 / np.maximum(lum_gain, 1e-6))
    gain = gain * kfac[..., None]
    gain = np.where(use[..., None], gain, 1.0)
    if a.chroma_beta > 0:
        # colour consistency: the lifted shadow keeps its brightness but its chroma moves toward the sunlit paint next to it
        out0 = P * gain
        Y0 = np.maximum(out0 @ LUM, 1e-6)
        c0 = out0 / np.maximum(out0.sum(2, keepdims=True), 1e-6)
        ct3 = np.concatenate([f["ct"], 1 - f["ct"].sum(2, keepdims=True)], 2)
        step = np.clip(ct3 - c0, -a.chroma_cap, a.chroma_cap)
        wgt = (np.clip(m, 0, 1) * a.chroma_beta * f["ct_ok"] * use)[..., None]
        c1 = np.clip(c0 + step * wgt, 1e-4, None)
        out1 = c1 * (Y0 / np.maximum(c1 @ LUM, 1e-6))[..., None]
        gain = np.where(wgt > 0, out1 / np.maximum(P, 1e-6), gain)
    if a.sat_cap and "sat_caps" in cal:
        # the lifted shadow may not be more saturated than the sunlit paint of the same hue: desaturate toward grey at the same brightness
        out0 = P * gain
        Y0 = out0 @ LUM
        hb, sa = hue_sat(out0)
        cap = np.asarray(cal["sat_caps"])[hb]
        over = use & (m > 0.05) & (sa > cap)
        fac = np.where(over, cap / np.maximum(sa, 1e-6), 1.0)[..., None]
        out1 = Y0[..., None] + (out0 - Y0[..., None]) * fac
        gain = np.where(over[..., None], np.maximum(out1, 0) / np.maximum(P, 1e-6), gain)
    # feather over the bear's rim: extend the gain a few pixels outward, fade with the distance
    uf = use.astype(np.float32)
    lg = np.log(np.maximum(gain, 1e-6))
    near = np.stack([nblur(lg[..., c], uf, 4.0)[0] for c in range(3)], 2)          # mean log gain of the bear pixels around
    fade = np.clip(cv2.GaussianBlur(uf, (0, 0), a.feather), 0, 1)
    gain = np.exp(np.where(use[..., None], lg, near * fade[..., None]))
    # ground: the concrete's measured ratio where the bear's shadow lies
    # The lifted shadow keeps its own brightness detail but takes the colour of the lit concrete around it (concrete is
    # one neutral paint; the shadow's colour, a few hundred times noisier after the lift, is mostly video chroma blocking).
    mg = np.clip(f["mg"], 0, 1)
    Y = np.maximum(L, 1e-6)
    ref_px = f["ring"] if f["ring"].sum() > 500 else f["lit_ground"]
    if ref_px.sum() > 200:
        rc = np.median(P[ref_px] / Y[ref_px][:, None], axis=0)
        lum_gain = np.power(ratio @ LUM, mg)
        if f["lit_ground"].sum() > 500:                               # not brighter than the lit concrete around
            gl_, gd = nblur(L, f["lit_ground"].astype(np.float32), 100.0)
            allow_g = np.where(gd > 1e-3, gl_, np.inf) * a.ground_cap
            lum_gain = np.maximum(np.minimum(lum_gain, allow_g / np.maximum(L, 1e-6)), 1.0)
        target = (Y * lum_gain)[..., None] * rc[None, None, :]
        gg = (1 - mg)[..., None] + mg[..., None] * target / np.maximum(P, 1e-6)
        gain = np.where((f["ground"] & ~f["use"])[..., None], gg, gain)
    out = P * gain
    res8 = np.round(linear_to_srgb(out, clip=True) * 255).astype(np.uint8)
    changed = (gain > 1.001).any(2)
    res8[~changed] = f["orig8"][~changed]
    return res8, gain


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    ap.add_argument("--colmap", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", default="")
    ap.add_argument("--every", type=int, default=0, help="correct every N-th photo (0 with no --frames: all)")
    ap.add_argument("--calib-every", type=int, default=13)
    ap.add_argument("--calib-from", type=Path, help="calibration.json to reuse instead of measuring")
    ap.add_argument("--floor", type=float, default=0.3)
    ap.add_argument("--chroma-beta", type=float, default=0.0, help="0 = off (default; it smears neighbouring paints); share of the move toward the neighbouring sunlit chroma")
    ap.add_argument("--chroma-cap", type=float, default=0.10, help="largest change of a chroma share (r, g, b) from this step")
    ap.add_argument("--chroma-radius", type=float, default=0.06, help="sunlit blobs within this x object height give the target")
    ap.add_argument("--sat-cap", type=int, default=1, help="1 = limit the lifted shadow's saturation per hue to the sunlit paint's")
    ap.add_argument("--sat-pct", type=float, default=90.0, help="percentile of the sunlit saturation used as the cap")
    ap.add_argument("--lit-cap", type=float, default=1.0)
    ap.add_argument("--attached-max", type=float, default=2.5, help="largest gain on pixels turned away from the sun")
    ap.add_argument("--smooth", type=float, default=2.0)
    ap.add_argument("--feather", type=float, default=2.5)
    ap.add_argument("--masks", type=Path, default=Path("data/IMG_6705/shadow_masks"), help="SDDNet photo shadow masks (ground)")
    ap.add_argument("--ground-soft", type=float, default=2.0, help="px, softening of the ground shadow edge")
    ap.add_argument("--ground-cap", type=float, default=1.5, help="ground gain keeps the pixel at most this x the local lit concrete")
    ap.add_argument("--ground-radius", type=float, default=6.0, help="ground pixels within this x object height of the bear")
    ap.add_argument("--ground-erode", type=int, default=3, help="px, shrink the photo mask so the correction stops inside the shadow")
    ap.add_argument("--gap", type=float, default=4.0)
    ap.add_argument("--band", type=float, default=14.0)
    ap.add_argument("--edge-sigma", type=float, default=18.0)
    ap.add_argument("--min-px", type=float, default=150.0)
    ap.add_argument("--crop-pad", type=float, default=0.6)
    ap.add_argument("--compare-every", type=int, default=100, help="save a comparison crop for every N-th corrected photo")
    a = ap.parse_args()
    scene = Scene(a)
    if a.frames:
        names = a.frames.split(",")
    else:
        names = scene.names[::a.every] if a.every else scene.names
    (a.out / "images").mkdir(parents=True, exist_ok=True)
    if a.calib_from:
        cal = json.load(open(a.calib_from))
    else:
        cn = scene.names[::a.calib_every]
        print(f"calibrating on {len(cn)} photos")
        cal = calibrate(scene, cn, a)
        json.dump(cal, open(a.out / "calibration.json", "w"), indent=1)
    print(f"colour strength t = {cal['t']}, k scale: cast {cal['scale_cast']:.2f}, attached {cal['scale_attached']:.2f}")
    summary = {}
    for i, name in enumerate(names):
        f = scene.frame(name)
        res8, gain = correct(f, cal, a)
        Image.fromarray(res8).save(a.out / "images" / scene.by[name]["name"])
        sm = f["use"] & (f["mb"] > 0.5)
        L0, L1 = f["P"] @ LUM, srgb_to_linear(res8 / 255.0) @ LUM
        summary[name] = dict(changed_share=round(float((gain > 1.001).any(2).mean()), 3),
                             bear_shadow_lum=[round(float(L0[sm].mean()), 4), round(float(L1[sm].mean()), 4)] if sm.any() else None,
                             clipped_bear=round(float((srgb_to_linear(res8 / 255.0)[f["use"]] > 0.98).any(1).mean()), 3) if f["use"].any() else None)
        if a.compare_every and (i % a.compare_every == 0 or a.frames):
            ys, xs = np.nonzero(f["use"])
            if len(ys):
                H, W = f["use"].shape
                cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
                hh = int((max(ys.max() - ys.min(), xs.max() - xs.min()) / 2) * (1 + a.crop_pad))
                y0, y1, x0, x1 = max(0, cy - hh), min(H, cy + hh), max(0, cx - hh), min(W, cx + hh)
                Image.fromarray(np.concatenate([f["orig8"][y0:y1, x0:x1], res8[y0:y1, x0:x1]], 1)).save(a.out / f"compare_{name}.jpg", quality=92)
        if i % 25 == 0:
            print(f"{i + 1}/{len(names)} {name} {summary[name]}", flush=True)
    json.dump(dict(calibration=cal, args={q: str(v) for q, v in vars(a).items()}, frames=summary), open(a.out / "summary.json", "w"), indent=1)
    print("done", len(names), "photos ->", a.out / "images")


if __name__ == "__main__":
    main()
