"""Add a fixed-direction sun to R3DG's NeILF stage (local patch; originals kept as *.orig).

R3DG_SUN="x,y,z"   sun direction (towards the sun, world = COLMAP coordinates) -> turns the sun term on
R3DG_SUN_INIT=3.0  starting sun irradiance per channel (learned, softplus); the direction stays fixed
The sun's visibility is ray traced once per Gaussian with R3DG's own tracer (like its 32 sampled incident directions),
and the sun adds (diffuse + GGX specular) x sun x visibility x max(0, n.s) to each Gaussian's shaded colour.
"""
import shutil
from pathlib import Path

R = Path.home() / "third_party/Relightable3DGaussian"


def patch(rel, pairs):
    f = R / rel
    o = f.with_suffix(f.suffix + ".orig")
    if not o.exists():
        shutil.copy(f, o)
    s = o.read_text()
    for a, b in pairs:
        assert s.count(a) == 1, (rel, a[:60])
        s = s.replace(a, b)
    f.write_text(s)
    print("patched", rel)


patch("scene/direct_light_map.py", [
    ("class DirectLightMap:\n\n    def __init__(self, H=128, light_init=0.5):\n",
     "class DirectLightMap:\n\n    def __init__(self, H=128, light_init=0.5):\n"
     "        import os, math  # local patch: optional fixed-direction sun (R3DG_SUN)\n"
     "        s = os.environ.get(\"R3DG_SUN\")\n"
     "        self.sun_dir = None\n"
     "        if s:\n"
     "            self.sun_dir = F.normalize(torch.tensor([float(v) for v in s.split(\",\")], device=\"cuda\"), dim=0)\n"
     "            init = float(os.environ.get(\"R3DG_SUN_INIT\", \"3.0\"))\n"
     "            self.sun_raw = nn.Parameter(torch.full((3,), math.log(math.expm1(init)), device=\"cuda\"))\n"
     "            fix = os.environ.get(\"R3DG_SUN_FIX\")  # fixed sun irradiance r,g,b: not learned\n"
     "            self.sun_fixed = None\n"
     "            if fix:\n"
     "                self.sun_fixed = torch.tensor([float(v) for v in fix.split(\",\")], device=\"cuda\")\n"
     "                self.sun_raw.requires_grad_(False)\n"),
    ("            {'params': [self.env], 'lr': training_args.env_lr, \"name\": \"env\"}\n        ]\n",
     "            {'params': [self.env], 'lr': training_args.env_lr, \"name\": \"env\"}\n        ]\n"
     "        if self.sun_dir is not None and self.sun_fixed is None:\n"
     "            l.append({'params': [self.sun_raw], 'lr': training_args.env_lr, \"name\": \"sun\"})\n"),
    ("    def step(self):\n",
     "    @property\n    def get_sun(self):\n"
     "        return self.sun_fixed if self.sun_fixed is not None else F.softplus(self.sun_raw)\n\n    def step(self):\n"),
    # save and load the learned sun with the light checkpoint
    ("            self.optimizer.state_dict(),\n        ]\n",
     "            self.optimizer.state_dict(),\n        ]\n"
     "        if self.sun_dir is not None:\n"
     "            captured_list.append(self.sun_raw)\n"),
    ("         opt_dict) = model_args[:2]\n",
     "         opt_dict) = model_args[:2]\n"
     "        if self.sun_dir is not None and len(model_args) > 2:\n"
     "            self.sun_raw = model_args[2]\n"
     "            print(f\"loaded sun irradiance {F.softplus(self.sun_raw).tolist()}\")\n"),
])

patch("scene/gaussian_model.py", [
    ("        self._visibility_tracing = incident_visibility_result\n",
     "        self._visibility_tracing = incident_visibility_result\n"
     "        import os  # local patch: visibility toward the fixed sun (R3DG_SUN)\n"
     "        s = os.environ.get(\"R3DG_SUN\")\n"
     "        self._sun_visibility = None\n"
     "        if s:\n"
     "            sd = F.normalize(torch.tensor([float(v) for v in s.split(\",\")], device=\"cuda\"), dim=0)\n"
     "            vis = []\n"
     "            for offset in range(0, gaussians_xyz.shape[0], chunk_size):\n"
     "                xyz = gaussians_xyz[offset:offset + chunk_size, None]\n"
     "                tr = raytracer.trace_visibility(xyz, sd.expand(xyz.shape[0], 1, 3).contiguous(), gaussians_xyz,\n"
     "                                                gaussians_inverse_covariance, gaussians_opacity, gaussians_normal)\n"
     "                vis.append(tr[\"visibility\"])\n"
     "            self._sun_visibility = torch.cat(vis, dim=0)\n"
     "            print(f\"sun visibility: mean {self._sun_visibility.mean().item():.3f}, \"\n"
     "                  f\"blocked (< 0.5) {(self._sun_visibility < 0.5).float().mean().item():.1%}\")\n"),
])

patch("gaussian_renderer/neilf.py", [
    ("            incident_dirs_precompute=pc._incident_dirs, incident_areas_precompute=pc._incident_areas)\n",
     "            incident_dirs_precompute=pc._incident_dirs, incident_areas_precompute=pc._incident_areas,\n"
     "            sun_visibility=getattr(pc, \"_sun_visibility\", None))\n"),
    ("                incident_areas_precompute=pc._incident_areas[i:i + chunk_size])\n",
     "                incident_areas_precompute=pc._incident_areas[i:i + chunk_size],\n"
     "                sun_visibility=None if getattr(pc, \"_sun_visibility\", None) is None else pc._sun_visibility[i:i + chunk_size])\n"),
    ("                              visibility_precompute=None, incident_dirs_precompute=None, incident_areas_precompute=None):\n",
     "                              visibility_precompute=None, incident_dirs_precompute=None, incident_areas_precompute=None,\n"
     "                              sun_visibility=None):\n"),
    # R3DG_NO_LOCAL=1: no per-Gaussian local light, so it cannot paint light in or out spot by spot
    ("    incident_lights = local_incident_lights + global_incident_lights\n",
     "    if os.environ.get(\"R3DG_NO_LOCAL\"):\n"
     "        local_incident_lights = local_incident_lights * 0\n"
     "    incident_lights = local_incident_lights + global_incident_lights\n"),
    # R3DG_SHADE_GUIDE=1: paint smoothness may only break at edges of the photo divided by the modelled light, so a
    # shadow edge the traced visibility explains is no excuse for an edge in the paint
    ("        loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color * image_mask, gt_image)\n",
     "        guide = gt_image\n"
     "        if os.environ.get(\"R3DG_SHADE_GUIDE\"):\n"
     "            guide = gt_image / rendered_diffuse.detach().mean(0, keepdim=True).clamp_min(0.05)\n"
     "            guide = (guide / guide.mean().clamp_min(1e-6)).clamp(0, 3) * gt_image.mean()\n"
     "        loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color * image_mask, guide)\n"),
    ("import math\n", "import math\nimport os\n"),
    ("    diffuse_light = transport.mean(dim=-2)\n",
     "    diffuse_light = transport.mean(dim=-2)\n"
     "    if sun_visibility is not None and getattr(direct_light_env_light, \"sun_dir\", None) is not None:\n"
     "        # local patch: fixed-direction sun with ray-traced visibility\n"
     "        sd = direct_light_env_light.sun_dir.view(1, 1, 3).expand(normals.shape[0], 1, 3)\n"
     "        n_d_s = (normals[:, None] * sd).sum(-1, keepdim=True).clamp(min=0)\n"
     "        f_s_sun = GGX_specular(normals, viewdirs, sd, roughness, fresnel=0.04)\n"
     "        sun_in = direct_light_env_light.get_sun.view(1, 1, 3) * sun_visibility * n_d_s\n"
     "        pbr = pbr + ((f_d + f_s_sun) * sun_in)[:, 0]\n"
     "        specular = specular + (f_s_sun * sun_in)[:, 0]\n"
     "        diffuse_light = diffuse_light + sun_in[:, 0]\n"),
])
print("R3DG SUN PATCH DONE")
