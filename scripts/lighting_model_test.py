#!/usr/bin/env python
"""Lighting-model test: can the lighting alone explain the cast shadows on the bear?

Model, per colour channel, in LINEAR light, for a blob with paint group g and surface direction n:
    brightness = paint_g * ( sky(n) + sun * max(0, n.s) * [sun reaches the blob] )
sky(n) = 9 smooth coefficients (second-order spherical harmonics of n: sky, ground bounce), sun strength fixed to 1 so
sky is measured in units of the sun. Paint groups = k-means on the chroma of clearly sunlit blobs; every blob takes the
group of the sunlit blobs around it (the user's adjacent rule: paint is spatially continuous).

Test (can fail): fit on all blobs EXCEPT the cast-shadowed ones on sun-facing surfaces (lit blobs + blobs turned
away from the sun), then predict the held-out cast-shadowed blobs, which get sky light only. Compared with what the
splat stores (the colour the photos show). Controls: the same fit with sky(n) forced constant (no surface directions),
and the real fit with the held-out blobs' directions shuffled inside their paint group.

Usage: lighting_model_test.py --ply exports/IMG_6705/splat.ply --labels exports/IMG_6705_sunmap3/sun_visibility.npz
           --borrow exports/IMG_6705_2dgs_borrow/borrowed.npz --sun exports/IMG_6705_sun_refit_obj/sun_refit.npz
           --plane-from exports/IMG_6705_shadowfix_o/stats.json --object exports/IMG_6705_object/object.json
           --out exports/IMG_6705_lightmodel
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from splatlib import LUM, SH_C0, read_ply, srgb_to_linear

NB = 9


def basis(n, up):
    """Second-order basis of the surface direction in a frame with 'up' as z."""
    e1 = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    x, y, z = n @ e1, n @ e2, n @ up
    return np.stack([np.ones_like(x), x, y, z, x * y, y * z, 3 * z * z - 1, x * z, x * x - y * y], 1)


def fit_channel(logL, g, B, sun, G, lam, const_sky=False):
    """Least squares in log space for log-paint per group and the sky coefficients (sun strength fixed to 1)."""
    lr0 = np.array([np.median(logL[g == k]) if (g == k).any() else 0.0 for k in range(G)])
    A0 = np.zeros(NB)
    A0[0] = 0.15

    def res(x):
        lr, A = x[:G], x[G:]
        if const_sky:
            A = np.concatenate([A[:1], np.zeros(NB - 1)])
        m = np.maximum(B @ A + sun, 1e-3)
        r = logL - (lr[g] + np.log(m))
        return np.concatenate([r, lam * A[1:]])

    x0 = np.concatenate([lr0 - np.log(0.5), A0])
    sol = least_squares(res, x0, loss="soft_l1", f_scale=0.3, max_nfev=300)
    lr, A = sol.x[:G], sol.x[G:]
    if const_sky:
        A = np.concatenate([A[:1], np.zeros(NB - 1)])
    return lr, A


def predict_brightness(lr, A, g, B, sun):
    return np.exp(lr[g]) * np.maximum(B @ A + sun, 1e-3)


def score(obs, pred):
    lr = np.log(np.maximum(obs, 1e-6) / np.maximum(pred, 1e-6))
    return dict(n=int(len(obs)), median_ratio_obs_over_pred=round(float(np.exp(np.median(lr))), 3),
                median_abs_log=round(float(np.median(np.abs(lr))), 3),
                within_25pct=round(float((np.abs(lr) < np.log(1.25)).mean()), 3),
                corr_log=round(float(np.corrcoef(np.log(np.maximum(obs, 1e-6)), np.log(np.maximum(pred, 1e-6)))[0, 1]), 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("ply", "labels", "borrow", "sun", "plane-from", "object", "out"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--groups", type=int, default=6)
    ap.add_argument("--radius", type=float, default=0.06, help="paint-group neighbourhood, share of the object height")
    ap.add_argument("--ns-lit", type=float, default=0.3, help="sunlit blobs used for paint groups: n.s above this")
    ap.add_argument("--ns-test", type=float, default=0.15, help="held-out blobs: blocked and n.s above this")
    ap.add_argument("--lam", type=float, default=0.05, help="pull of sky coefficients toward a constant")
    ap.add_argument("--min-opacity", type=float, default=0.3)
    ap.add_argument("--orient", choices=["centre", "consistent"], default="consistent")
    ap.add_argument("--orient-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    _, d = read_ply(a.ply)
    lab, bor = np.load(a.labels), np.load(a.borrow)
    idx = lab["index"]
    assert (bor["index"][bor["on_object"]] == idx).all(), "borrowed normals and labels are not the same blobs"
    n = bor["normal"][bor["on_object"]].astype(float)
    conf = bor["confident"][bor["on_object"]]
    s = np.load(a.sun)["sun"].astype(float)
    s /= np.linalg.norm(s)
    up = np.array(json.load(open(a.plane_from))["ground_plane"]["normal"], float)
    up /= np.linalg.norm(up)
    height = float(json.load(open(a.object))["height"])
    p = np.stack([d[k][idx] for k in "xyz"], 1).astype(float)
    op = 1 / (1 + np.exp(-d["opacity"][idx].astype(float)))
    rgb = srgb_to_linear(np.clip(np.stack([d[f"f_dc_{i}"][idx] for i in range(3)], 1) * SH_C0 + 0.5, 0, 1))
    vis = lab["visible"] >= 0.5

    # surface direction signs are not reliable. orient=centre: away from the middle of the object. orient=consistent:
    # make neighbouring directions agree (tangent-plane propagation in open3d), then point the majority away from the middle.
    centre = np.average(p, axis=0, weights=op)
    ok = conf & (op > a.min_opacity) & (rgb @ LUM > 1e-4)
    n0 = n.copy()
    if a.orient == "consistent":
        import open3d as o3d
        sub = np.flatnonzero(ok)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(p[sub])
        pc.normals = o3d.utility.Vector3dVector(n[sub])
        pc.orient_normals_consistent_tangent_plane(a.orient_k)
        n[sub] = np.asarray(pc.normals)
        if ((p[sub] - centre) * n[sub]).sum(1).mean() < 0:
            n[sub] *= -1
    else:
        n = np.where((((p - centre) * n).sum(1) < 0)[:, None], -n, n)
    ns = n @ s
    toward_middle = (((p - centre) * n).sum(1) < 0)[ok].mean()
    changed = ((n0 * n).sum(1) < 0)[ok].mean()
    print(f"blobs {len(idx):,}; usable (confident direction, opaque) {ok.sum():,}; orient={a.orient}: "
          f"{toward_middle:.1%} of usable point toward the middle, {changed:.1%} changed sign")

    # paint groups from clearly sunlit blobs, spread to neighbours
    lit_core = ok & vis & (ns > a.ns_lit)
    ch = rgb / np.maximum(rgb.sum(1, keepdims=True), 1e-9)
    _, grp_lit = kmeans2(ch[lit_core][:, :2], a.groups, minit="++", seed=a.seed)
    gl = np.full(len(idx), -1)
    gl[np.flatnonzero(lit_core)] = grp_lit
    tree = cKDTree(p[lit_core])
    dist, nb = tree.query(p, k=10, distance_upper_bound=a.radius * height)
    g = np.full(len(idx), -1)
    for i in range(len(idx)):
        v = nb[i][np.isfinite(dist[i])]
        if len(v):
            g[i] = np.bincount(grp_lit[v], minlength=a.groups).argmax()
    ok &= g >= 0
    print(f"blobs with a paint group {ok.sum():,}")

    test = ok & ~vis & (ns > a.ns_test)                      # cast-shadowed, sun-facing surface: held out
    amb = ok & ~vis & (ns >= -0.05) & (ns <= a.ns_test)      # ambiguous: left out of both
    fit = ok & ~test & ~amb
    sunflag = (vis & (ns > 0)).astype(float)
    sun = np.maximum(ns, 0) * sunflag
    B = basis(n, up)
    print(f"fit {fit.sum():,} (lit {(fit & vis).sum():,}, turned away {(fit & ~vis).sum():,}); held out {test.sum():,}; ambiguous {amb.sum():,}")

    out = dict(counts=dict(fit=int(fit.sum()), test=int(test.sum()), ambiguous=int(amb.sum()), lit=int((fit & vis).sum()),
                           turned_away=int((fit & ~vis).sum())), sun_dir=[round(float(v), 3) for v in s], channels={})
    pred = np.zeros((len(idx), 3))
    pred_c = np.zeros((len(idx), 3))
    pred_sh = np.zeros((len(idx), 3))
    sky_up = []
    for c in range(3):
        L = np.log(np.maximum(rgb[:, c], 1e-6))
        lr, A = fit_channel(L[fit], g[fit], B[fit], sun[fit], a.groups, a.lam)
        pred[:, c] = predict_brightness(lr, A, g, B, sun * 0)           # sky only (for held-out cast-shadowed blobs)
        full = predict_brightness(lr, A, g, B, sun)                    # with sun, for the fit set
        pred_fit = np.where(fit, full, 0)
        lrc, Ac = fit_channel(L[fit], g[fit], B[fit], sun[fit], a.groups, a.lam, const_sky=True)
        pred_c[:, c] = predict_brightness(lrc, Ac, g, B, sun * 0)
        # shuffled directions inside each paint group (null)
        Bs = B.copy()
        for k in range(a.groups):
            m = np.flatnonzero(test & (g == k))
            Bs[m] = B[rng.permutation(m)] if len(m) else Bs[m]
        pred_sh[:, c] = predict_brightness(lr, A, g, Bs, sun * 0)
        pred[fit, c] = pred_fit[fit]
        sky_up.append(float(basis(up[None], up)[0] @ A))
        out["channels"][["R", "G", "B"][c]] = dict(sky_at_up_over_sun=round(sky_up[-1], 3), sky_coeffs=[round(float(v), 3) for v in A],
                                                 paint=[round(float(np.exp(v)), 3) for v in lr])
    obs = rgb @ LUM
    lum = lambda P: P @ LUM
    out["fit_set_self_consistency"] = score(obs[fit], lum(pred[fit]))
    out["held_out_cast_shadow"] = {
        "lighting_model": score(obs[test], lum(pred[test])),
        "control_no_directions": score(obs[test], lum(pred_c[test])),
        "control_shuffled_directions": score(obs[test], lum(pred_sh[test])),
        "per_channel_lighting_model": {k: score(rgb[test, c], pred[test, c]) for c, k in enumerate("RGB")}}
    out["per_paint_group"] = []
    for k in range(a.groups):
        m = test & (g == k)
        if m.sum() > 30:
            out["per_paint_group"].append(dict(group=k, held_out=int(m.sum()), lit_chroma=[round(float(v), 3) for v in np.median(ch[lit_core & (g == k)], 0)],
                                               **score(obs[m], lum(pred[m]))))
    # what the sky/sun ratio should be from the concrete next to the bear (PPISP: sun/shade 8.5, 6.6, 4.2; elevation 45 deg)
    out["concrete_expectation_sky_over_sun"] = [round(float(1 / ((r - 1) / np.sin(np.radians(45)))), 3) for r in (8.5, 6.6, 4.2)]
    json.dump(out, open(a.out / "lightmodel.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_paint_group", "channels")}, indent=1))
    for r in out["per_paint_group"]:
        print(r)
    print({k: v["sky_at_up_over_sun"] for k, v in out["channels"].items()}, "expected from the concrete:", out["concrete_expectation_sky_over_sun"])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
        for axi, (title, P, m) in zip(ax, [("fit set: lit + turned away", lum(pred), fit), ("held out: cast shadow, model", lum(pred), test),
                                           ("held out: no directions (control)", lum(pred_c), test)]):
            q = rng.choice(np.flatnonzero(m), min(6000, m.sum()), replace=False)
            axi.loglog(obs[q], P[q], ".", ms=2, alpha=0.35)
            axi.plot([1e-3, 1], [1e-3, 1], "k-", lw=0.8)
            axi.set_xlim(1e-3, 1); axi.set_ylim(1e-3, 1); axi.set_title(title)
            axi.set_xlabel("stored blob brightness (linear)"); axi.set_ylabel("model prediction")
        fig.tight_layout()
        fig.savefig(a.out / "lightmodel_scatter.png", dpi=110)
    except Exception as e:  # plotting is optional
        print("no plot:", e)


if __name__ == "__main__":
    main()
