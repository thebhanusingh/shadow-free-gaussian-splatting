#!/usr/bin/env python
"""Score corrected photos against the original photos, from the single sun capture alone (no second lighting needed).

Every number is measured in LINEAR light on the photos that the splat will be trained on, with the same shadow labels and
ground masks the correction used (the labels come from the geometry, not from the corrected photos). For each photo:

  bear_edge_step_cast / _attached   median |log(lit side / shadow side)| of the brightness across the old shadow edges, bear
                                    pixels only, split by surface facing the sun (cast) or away (attached). 0 = no step left.
                                    The original photos give the starting value (typically 1 to 2.5 = a 3x to 12x step).
  bear_edge_chroma_cast / _attached same, for the colour: summed |difference of the r, g, b shares| across the edge. 0 = same colour.
  lit_changed_median / lit_changed_share  how much the pixels that should NOT change moved (sunlit bear pixels facing the
                                    sun, and the lit concrete): median |log ratio| and share of pixels moved by more than 5%. 0 = untouched.
  ground_ratio                      lit concrete in a ring around the bear's shadow / shadow interior (mean brightness). 1 = the
                                    shadow is gone; the original is typically 6 to 25.
  ground_overshoot_share            share of shadow-interior pixels brighter than 1.3 x the lit ring (over-lifted).
  ground_texture                    coefficient of variation inside the lifted shadow / in the lit ring. About 1 = same texture;
                                    far below 1 = a flat patch.
  other_shadow_changed_share        share of the photo-mask ground pixels OUTSIDE the bear's own shadow area (the photographer's
                                    shadow, a wall's) that moved by more than 5%. 0 = left alone, as it should be.
  saturation_over_share             share of the bear's shadow pixels more saturated than the sunlit paint of the same hue
                                    (90th percentile, pooled over the scored photos). The original is the baseline; lower is better.

Prints a table (median over the photos, original vs corrected) and writes <out>/score.json with the per-photo numbers.

Usage (from scripts/): python score_correction.py --surfels ... --ply ... --labels ... --sun ... --object ... --plane-from ...
           --colmap ... --images data/IMG_6704/images --masks data/IMG_6704/shadow_masks --fixed data/IMG_6704_fixed/images
           --eps 0.122 --out exports/IMG_6704_score_v2 [--frames a,b,c | --every 13]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sky_visibility import add_arguments
from splatlib import LUM, srgb_to_linear

import correct_photos as cp


def load_lin(path):
    return srgb_to_linear(np.asarray(Image.open(path).convert("RGB"), np.float32) / 255)


def edge_numbers(f, P, a):
    e = cp.edge_stats(f, P, a)
    out = {}
    for lab in ("cast", "attached"):
        v = e[lab]
        out[f"bear_edge_step_{lab}"] = None if v is None else abs(v["log_step"])
        out[f"bear_edge_chroma_{lab}"] = None if v is None else v["chroma"]
    return out


def ground_numbers(f, P):
    L = P @ LUM
    ring, inner = f["ring"], f["inner"]
    if ring.sum() < 3000 or inner.sum() < 3000:
        return dict(ground_ratio=None, ground_overshoot_share=None, ground_texture=None)
    lr = float(L[ring].mean())
    cv = lambda x: float(x.std() / max(x.mean(), 1e-9))
    return dict(ground_ratio=lr / max(float(L[inner].mean()), 1e-9),
                ground_overshoot_share=float((L[inner] > 1.3 * lr).mean()),
                ground_texture=cv(L[inner]) / max(cv(L[ring]), 1e-9))


def lit_numbers(f, P0, P1):
    """How far the pixels that must stay as the photo moved."""
    keep = (f["use"] & (f["mb"] < 0.05) & (f["cos"] > 0.3)) | (f["lit_ground"] & ~f["ground"])
    if keep.sum() < 500:
        return dict(lit_changed_median=None, lit_changed_share=None)
    d = np.abs(np.log(np.maximum(P1 @ LUM, 1e-5)[keep] / np.maximum(P0 @ LUM, 1e-5)[keep]))
    return dict(lit_changed_median=float(np.median(d)), lit_changed_share=float((d > np.log(1.05)).mean()))


def other_numbers(f, P0, P1):
    """Shadows on the ground that are not the bear's (the photographer's, a wall's) must stay as they were."""
    o = f["ground_other"]
    if o.sum() < 500:
        return dict(other_shadow_changed_share=None)
    d = np.abs(np.log(np.maximum(P1 @ LUM, 1e-5)[o] / np.maximum(P0 @ LUM, 1e-5)[o]))
    return dict(other_shadow_changed_share=float((d > np.log(1.05)).mean()))


def saturation_over(f, P, caps):
    sel = f["use"] & (f["mb"] > 0.5) & ((P @ LUM) > 0.003)
    if sel.sum() < 500:
        return None
    hb, sa = cp.hue_sat(P[sel])
    return float((sa > caps[hb]).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    ap.add_argument("--colmap", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True, help="the ORIGINAL photos")
    ap.add_argument("--fixed", type=Path, required=True, help="the corrected photos (a folder of images)")
    ap.add_argument("--masks", type=Path, required=True, help="SDDNet photo shadow masks (ground) of THIS scene")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", default="")
    ap.add_argument("--every", type=int, default=13, help="score every N-th photo that exists in --fixed")
    # same defaults as correct_photos.py, so the labels and masks are the ones the correction used
    ap.add_argument("--chroma-beta", type=float, default=0.0)
    ap.add_argument("--chroma-radius", type=float, default=0.06)
    ap.add_argument("--sat-pct", type=float, default=90.0)
    ap.add_argument("--smooth", type=float, default=2.0)
    ap.add_argument("--ground-erode", type=int, default=3)
    ap.add_argument("--ground-soft", type=float, default=2.0)
    ap.add_argument("--ground-radius", type=float, default=6.0)
    ap.add_argument("--footprint-cell", type=float, default=0.01)
    ap.add_argument("--footprint-margin", type=float, default=0.05)
    ap.add_argument("--gap", type=float, default=4.0)
    ap.add_argument("--band", type=float, default=14.0)
    ap.add_argument("--edge-sigma", type=float, default=18.0)
    ap.add_argument("--min-px", type=float, default=150.0)
    a = ap.parse_args()
    scene = cp.Scene(a)
    have = [n for n in scene.names if (a.fixed / scene.by[n]["name"]).exists()]
    names = a.frames.split(",") if a.frames else have[::max(a.every, 1)]
    names = [n for n in names if n in have]
    if not names:
        raise SystemExit(f"no corrected photo from the list found in {a.fixed}")
    print(f"scoring {len(names)} photos from {a.fixed}")
    frames = {n: scene.frame(n) for n in names}
    caps = cp.saturation_caps(list(frames.values()), a.sat_pct)
    rows = {}
    for n, f in frames.items():
        P0, P1 = f["P"], load_lin(a.fixed / scene.by[n]["name"])
        r = {}
        for tag, P in (("original", P0), ("corrected", P1)):
            d = {}
            d.update(edge_numbers(f, P, a))
            d.update(ground_numbers(f, P))
            d["saturation_over_share"] = saturation_over(f, P, caps)
            r[tag] = d
        r["corrected"].update(lit_numbers(f, P0, P1))
        r["corrected"].update(other_numbers(f, P0, P1))
        rows[n] = r
    keys = sorted({k for r in rows.values() for t in r.values() for k in t})
    summary = {}
    for k in keys:
        summary[k] = {}
        for tag in ("original", "corrected"):
            v = [r[tag][k] for r in rows.values() if tag in r and r[tag].get(k) is not None]
            summary[k][tag] = round(float(np.median(v)), 4) if v else None
            summary[k][f"n_{tag}"] = len(v)
    print(f"\n{'measure':32s} {'original':>10s} {'corrected':>10s}   (median over photos)")
    for k in keys:
        o, c = summary[k]["original"], summary[k]["corrected"]
        print(f"{k:32s} {'-' if o is None else format(o, '10.4f'):>10s} {'-' if c is None else format(c, '10.4f'):>10s}")
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(fixed=str(a.fixed), photos=names, summary=summary, per_photo=rows,
                   args={q: str(v) for q, v in vars(a).items()}), open(a.out / "score.json", "w"), indent=1)
    print("\nwrote", a.out / "score.json")


if __name__ == "__main__":
    main()
