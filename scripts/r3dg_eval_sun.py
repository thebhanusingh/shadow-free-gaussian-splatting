#
# Local addition for the bear 2 test (not part of R3DG). Based on R3DG's eval_nvs.py (Inria / Gaussian-Splatting
# licence: non-commercial, research and evaluation use).
#
# For each held-out view: photo | R3DG paint x light (as learned, with shadows) | the same with the sun unblocked
# everywhere (cast sun shadows removed) | with sun AND sky unblocked (all cast shadows removed) | base colour (paint).
# Needs the sun patch (patch_r3dg_sun.py) and R3DG_SUN set to the direction used in training.
import os
import torch
from tqdm import tqdm
from torchvision.utils import save_image
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render_fn_dict
from scene import Scene, GaussianModel
from scene.direct_light_map import DirectLightMap
from utils.general_utils import safe_state
from utils.image_utils import psnr


def main(dataset, pipeline, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, render_type="neilf")
        scene = Scene(dataset, gaussians, shuffle=False)
        background = torch.zeros(3, device="cuda")
        iteration = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=False)
        gaussians.update_visibility(args.sample_num)
        light = DirectLightMap(dataset.env_resolution)
        env_ckpt = os.path.dirname(args.checkpoint) + "/env_light_" + os.path.basename(args.checkpoint)
        light.create_from_ckpt(env_ckpt, restore_optimizer=False)
        kw = {"sample_num": args.sample_num, "env_light": light}
        out = os.path.join(dataset.model_path, "sun_eval_{}".format(iteration))
        os.makedirs(out, exist_ok=True)
        render = render_fn_dict["neilf"]
        vis_env, vis_sun = gaussians._visibility_tracing, gaussians._sun_visibility
        p_shadow, p_free = 0.0, 0.0
        views = scene.getTestCameras()
        for idx, view in enumerate(tqdm(views, desc="sun eval")):
            gt = view.original_image[0:3].cuda()
            r = render(view, gaussians, pipeline, background, dict_params=kw)
            gaussians._sun_visibility = torch.ones_like(vis_sun)
            r_sun = render(view, gaussians, pipeline, background, dict_params=kw)
            gaussians._visibility_tracing = torch.ones_like(vis_env)
            r_all = render(view, gaussians, pipeline, background, dict_params=kw)
            gaussians._visibility_tracing, gaussians._sun_visibility = vis_env, vis_sun
            row = torch.cat([gt, r["pbr"], r_sun["pbr"], r_all["pbr"], r["base_color"]], dim=2).clamp(0, 1)
            save_image(row, os.path.join(out, "{0:03d}.jpg".format(idx)))
            p_shadow += psnr(r["pbr"], gt).mean().item()
        print("held-out {} views: paint x light PSNR {:.2f}; sun irradiance {}; images -> {}".format(
            len(views), p_shadow / len(views), light.get_sun.tolist(), out))


if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)
    main(model.extract(args), pipeline.extract(args), args)
