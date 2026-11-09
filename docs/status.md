# Current status (read after CLAUDE.md; details in docs/lab-notes.md)

Last updated 2026-11-04. Short hand-off so a new session can pick up without the old conversation.

## Where things stand
- **Bear 2 (IMG_6705, seated cowboy-hat bear, 16 cm, concrete)** ran end to end with ONE command on 2026-10-27:
  `scripts/runs/run_full_6705.sh` (log `logs/IMG_6705_full.log`, load `logs/IMG_6705_load.csv`), 1 h 52 min, no hand values.
  Final `exports/IMG_6705_final/splat_shadowfree.ply`; renders `exports/IMG_6705_final_compare/`.
- Worked: auto object finding + scale (`scripts/find_object.py`, K = 0.767 vs bear 1), sun elevation from video metadata
  (`scripts/sun_from_metadata.py`, 44.47 deg; no compass heading in the file, so azimuth is fitted from the shadow with
  `fit_sun_edges.py --fix-elevation`), ground shadow removal (texture fill U + splat clone stamp Y).
- **Failed: object (bear) fill**: bear turns orange / glassy. Cause 1: `--adj-radius` scaled to 1.72 units (18 cm) is
  bigger than the bear, so the adjacent rule compares each spot with the whole bear; paint is read as shade.
  Cause 2 (root): standard 3DGS blobs have no reliable surface direction (axis vs PCA normals 42 deg apart, ~1 in 4
  confident), so the fill cannot use geometry (sun-blocked + facing) and falls back on brightness.
- Other open issues: predicted ground shadow ~3x too big in stage 4a (IoU 0.34); receiver offset at the scan edge in both
  sun fits; grey smudge at the bear's feet (127 unfilled clone cells); own sun fit +7 deg without metadata.

## Decided next step (user, 2026-10-28): surface test
- Train a **2DGS** (flat surfel) splat of bear 2 on the EXISTING poses (`data/IMG_6705`, no COLMAP rerun); gsplat 1.4.0 in
  env `gsplat` has `rasterization_2dgs`. About 45-60 min.
- User does NOT want lower visual quality: plan is two splats, normal splatfacto splat for looks, 2DGS only as a geometry
  reference (each normal blob borrows the nearest surfel's normal). Measure before deciding: held-out PSNR/SSIM/LPIPS of
  both + crops of the hat; share of blobs with a confident normal.
- After that: geometry-driven fill = brightness x E_lit / E, E = sky + sun * max(0, n.s) * visible, with the sun/sky
  ratio measured on the concrete next to the object (not fitted on the painted bear).

## Surface test done (2026-10-28): 2DGS directions work
- `scripts/train_2dgs_ref.py --init-ply exports/IMG_6705/splat.ply --steps 10000 --dist-start 0 --normal-start 1000
  --refine-stop 0` (50.6 min, 2.2 GB GPU) -> `exports/IMG_6705_2dgs_fromsplat/` (719,172 surfels). Held-out (same 65
  frames as ns-eval): 2DGS PSNR 28.16 / SSIM 0.879 / LPIPS 0.207 vs 3DGS 28.45 / 0.878 / 0.170. Hat crops nearly equal.
- `scripts/borrow_normals.py` -> `exports/IMG_6705_2dgs_borrow/`: bear blobs with a confident direction 69.8% (old rule
  19.7%). `scripts/normals_picture.py`: 2DGS directions clean, 3DGS speckle (`exports/IMG_6705_2dgs_compare/`).
- New `scripts/fill_geometry.py` (user's rules: no relight on the sun side; per-spot shadow strength; cross-check with
  sunlit same-paint neighbours; adjacent rule is the user's, radius 0.06 x height). `--margin 0.4` -> 39% of the bear
  filled. Then clone stamp Y -> **new final `exports/IMG_6705_clone_y_geom/splat_clone.ply`**, renders
  `exports/IMG_6705_final_geom_compare/`. Bear keeps real colours (orange/glassy gone), lift modest; ground shadow gone;
  left: dark patch at the scan edge (frame 407), pale rim at the bear's base, hat brim edge a bit pale.
- Not yet run: full from-scratch 2DGS (not needed so far). Bounce light from the removed ground shadow deliberately not added.
- User asked for splatting that knows each object: proposed lifting SAM 2 masks onto the splat (FlashSplat-style
  voting); SAM 2 not installed (download needs the user's OK).

## Relight work (2026-10-28 evening): paused by the user ("lets wait it out")
- `scripts/relight_sh.py`: bear = paint x light; light (sky + concrete bounce + sun) as 9 SH coefficients, measured
  per channel on the concrete; per-spot sun visibility from same-colour sunlit neighbours (user's adjacent rule);
  relight without a shadow term. Flags: `--cast-from` (sunlit reference only from map-lit blobs), `--lift-averted`
  (light the sun-averted side too), `--linear` + `--crf-json` + `--match-ply` (PPISP splat). Base for runs:
  `exports/IMG_6705_clone_y_base/splat_clone.ply` (ground fixed, bear untouched).
- Results (orbit page https://claude.ai/artifact/SxA6HuVon5NBkQCQ8A84mn, `scripts/render_orbit.py`):
  relight 1 lifted everything evenly (no shadow removed); relight 2 (`exports/IMG_6705_relight_sh2`) removes much
  more shade but adds orange/green; relight 3 (`exports/IMG_6705_relight_sh3`, PPISP linear) right brightness (lit
  98% unchanged, cast 0.68 -> 0.86 of lit, averted to about 0.8 of lit) but worse colour blotches.
- Cause: per-channel gains of 3-6x on dark shade blobs amplify their colour noise. **Proposed next (not run):
  relight 4 = one brightness factor for R, G, B per blob + one global shade-to-sun white balance from the concrete.**
- PPISP (`scripts/train_ppisp_ft.py`, ppisp 1.3.0 installed in env gsplat, built with CC=gcc-13 CXX=g++-13):
  `exports/IMG_6705_ppisp/` held-out PSNR 29.13 / SSIM 0.893 / LPIPS 0.174 vs original 28.45 / 0.878 / 0.170.
  Learned camera curve is mild (not sRGB); exposure nearly constant. In linear colours the concrete sun/shade ratio is
  8.5 / 6.6 / 4.2 (R, G, B), not 27 / 19 / 10: the sRGB assumption had inflated the sun about 3x.
- Uncommitted on the laptop: `fill_geometry.py`, `render_orbit.py`, `relight_sh.py`, `train_ppisp_ft.py`, notes,
  status (commit needs the user's OK).

## Shadow detection fixed (2026-11-02): sun refit + bear-only shadow casters
- Root cause of all shadow-detection trouble: non-bear surfels (ground layer, faint strays) acted as false blockers.
  With only the bear's own 2DGS surfels as casters (`scripts/sun_refit.py --object-casters`), the predicted ground
  shadow matches the photo masks at IoU 0.87 (was 0.14-0.30); refit sun elevation 44.97 (metadata 44.47), azimuth
  -4.5 deg: `exports/IMG_6705_sun_refit_obj/sun_refit.npz`.
- Bear labels: `scripts/sun_shadow_map.py --sun-npz ... --object-casters --test point --eps 0.093` (shadow when part of
  the bear is > 10 mm closer to the sun): 30% shadow, accepted by the user (`exports/IMG_6705_sunmap3`, page
  https://claude.ai/artifact/DQGoj6j8LQPnRg9AvZmU9r).
- Fill `scripts/fill_labels.py` (only labelled blobs; base `exports/IMG_6705_clone_y_base`, ground fixed, bear untouched):
  fill 3 (`--near-mm 2 --propagate 0.1`) cleanest colours but hard shadows stay; fill 5 (`--local-strength
  --local-tint 0.2 --chroma 0.2`) / fill 7 (+ `--brightness-only`) remove most hard shadows but the shade blobs' own
  colours are off (purple-grey brim underside, orange back). Fill 6 (`--tint-colour`) worse (blue). Orbit page:
  https://claude.ai/artifact/SxA6HuVon5NBkQCQ8A84mn. Conclusion: the shaded blobs' stored colour is wrong; brightness
  methods cannot recover it.
- Next (user, 2026-11-02): BRDF decomposition with R3DG using OUR sun direction, to get paint colours and remove
  shadows. Not started.

## R3DG test (2026-11-02/03): ran, shadows went into the paint
- Env `r3dg` (Python 3.9, PyTorch 2.0.1 cu118, conda CUDA 11.8 nvcc + cudart, conda gcc 11; conda-forge CUDA 13.x
  packages removed), code `~/third_party/Relightable3DGaussian` (commit a102f14; Inria GS licence, non-commercial).
  Local changes there (originals `*.orig`): mask fix + mask broadcast in `scene/dataset_readers.py`, float32 images
  in `scene/utils.py`, sun patch (`scripts/patch_r3dg_sun.py`: fixed-direction sun, learnable strength, ray-traced
  visibility, saved with the light checkpoint), `eval_sun.py` (= `scripts/r3dg_eval_sun.py`).
- Data `data/IMG_6705_r3dg_half` (960 x 538 undistorted, masks of bear + ground within 2 x height; full size runs out
  of RAM). Stage 1 (`outputs_r3dg/IMG_6705/3dgs`, 42 min, 636k Gaussians, PSNR 27.4 masked). Stage 2
  (`scripts/runs/r3dg_stage2.sh 40000 8`, `outputs_r3dg/IMG_6705/neilf_sun`, about 1 h; 32 samples spilled the 8 GB card).
- Result: paint x light PSNR 23.8; learned sun strength stayed at its start (3.0); the ground shadow and bear shading
  are baked into the base colour, so unblocking sun / sky changes only the edges. Summary image
  `outputs_r3dg/IMG_6705/neilf_sun/sun_eval_40000/r3dg_summary.jpg`.
- Offered next (user paused, "lets do stuff tomorrow"): retry stage 2 with the sun strength FIXED from the concrete
  measurement and the per-Gaussian local light heavily penalised, so only the traced visibility can darken shadows.
- Uncommitted: scripts/make_r3dg_masks.py, patch_r3dg_sun.py, r3dg_eval_sun.py, runs/setup_r3dg_env.sh,
  runs/build_r3dg_ext.sh, runs/prep_r3dg_data.sh, runs/prep_r3dg_half.sh, runs/r3dg_stage2.sh, notes, status (ask).

## R3DG retry and research (2026-11-04): R3DG dropped, photo correction next
- Retry (`scripts/runs/r3dg_stage2_fix.sh 40000 8`, `outputs_r3dg/IMG_6705/neilf_sun_fix`): sun fixed 3.8/3.5/3.0, no local light, shade-guided paint smoothness. Held-out paint x light PSNR 23.5. Failed the same way: ground paint 0.03 in photo shadow vs 0.27 lit, though traced visibility is right (96.6% / 0.4%); bear paint oversaturated orange with glassy shards. Conclusion: one lighting condition does not constrain R3DG enough.
- Deep research: `reports/Deshadow models for splats.md` in the Windows checkout (notes in `research_notes/`), not on the laptop WSL repo. Ranked: control = user adjacent method per photo inside the geometry mask with per-blob median across views; SGCR-SR gain; Colorful Diffuse gain; mask-fed models; DiffusionRenderer Cosmos / UniRelight on the school machine.
- Next (proposed, not started): per-photo correction before training a normal 3DGS (user idea: work out the light first). Note WSL can restart when idle: tmux, ttyd and tunnel then need restarting.

## Pending decisions / reminders
- Commit the new uncommitted scripts? (`sun_from_metadata.py`, `find_object.py`, `sample_load.py`,
  `runs/run_full_6705.sh`, `fit_sun_edges.py --fix-elevation`; plus older uncommitted bear 1 scripts). Ask first.
- Remind the user: GLOMAP (`colmap4` env, `global_mapper`) vs incremental mapping timing test. Not queued.
- User may film new videos (AE/AF lock, slower, 2-3 loops incl. one high, own shadow off the scene).

## House rules (from the user)
- One command at a time; nothing new without the user's yes; ask before commits; log every step in lab-notes.
- Report each scene on its own, comparisons only at the end. Ground shadows must be removed too.
- The adjacent brightness-and-colour rule is the user's own method: credit it.
- Visible terminal: tmux `splat`, window `bear2`; read-only ttyd (port 7681, tmux session `ttyd`, password in `~/.ttyd_pass`) + cloudflared tunnel (tmux session `tunnel`, link in `~/cloudflared_ttyd.log`). After a laptop restart all three must be started again; the user approves the public link.
