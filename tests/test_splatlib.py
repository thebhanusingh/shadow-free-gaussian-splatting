"""Tests for the shared library and the shadow geometry. Run from the repo root, no GPU and no data needed:

    python -m unittest tests.test_splatlib -v        (or: python tests/test_splatlib.py)

What is checked: colour maths, PLY and COLMAP file round trips, camera projection, the sun / shadow geometry on a
synthetic ball on a plane with a known sun, and that no script keeps its own copy of a library function.
"""
import re
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from splatlib import (LUM, SH_C0, angle_between, chroma, iou, linear_to_srgb, look_at, plane_frame, ply_header,  # noqa: E402
                      predict, project, qvec2rotmat, read_cameras_bin, read_images_bin, read_ply, srgb_to_linear,
                      sun_vec, write_ply)


class Colour(unittest.TestCase):
    def test_round_trip(self):
        c = np.linspace(0, 1, 101)
        np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(c)), c, atol=1e-9)

    def test_known_values(self):
        self.assertAlmostEqual(float(srgb_to_linear(np.array(0.5))), 0.2140411, places=6)   # mid grey in sRGB
        self.assertAlmostEqual(float(linear_to_srgb(np.array(0.18))), 0.4614, places=3)     # photographic grey card
        self.assertEqual(float(srgb_to_linear(np.array(0.0))), 0.0)
        self.assertAlmostEqual(float(srgb_to_linear(np.array(1.0))), 1.0, places=9)

    def test_clip_only_when_asked(self):
        self.assertGreater(float(linear_to_srgb(np.array(2.0))), 1.0)
        self.assertAlmostEqual(float(linear_to_srgb(np.array(2.0), clip=True)), 1.0, places=12)
        self.assertEqual(float(linear_to_srgb(np.array(-0.5))), 0.0)

    def test_luminance_weights_sum_to_one(self):
        self.assertAlmostEqual(float(LUM.sum()), 1.0, places=9)

    def test_chroma_ignores_brightness(self):
        rgb = np.array([[0.2, 0.4, 0.4], [0.02, 0.04, 0.04]])
        c = chroma(rgb)
        np.testing.assert_allclose(c[0], c[1])
        np.testing.assert_allclose(c.sum(1), 1.0)

    def test_gain_in_linear_light_keeps_hue(self):
        """The rule behind every shadow fill: a gain applied in linear light must not change chroma."""
        lin = srgb_to_linear(np.array([[0.30, 0.20, 0.10]]))
        np.testing.assert_allclose(chroma(lin), chroma(lin * 3.5))


class PlyFiles(unittest.TestCase):
    def test_round_trip_and_count_rewrite(self):
        dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("f_dc_0", "<f4"), ("opacity", "<f4")]
        data = np.zeros(7, dtype=dtype)
        data["x"] = np.arange(7)
        data["opacity"] = 0.5
        header = ply_header(dtype)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.ply"
            write_ply(p, header, data)
            h2, d2 = read_ply(p)
            self.assertEqual(len(d2), 7)
            np.testing.assert_array_equal(d2["x"], data["x"])
            write_ply(p, h2, d2[:3])                      # the header count must follow the data
            self.assertEqual(len(read_ply(p)[1]), 3)

    def test_rejects_ascii_ply(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ascii.ply"
            p.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
            with self.assertRaises(SystemExit):
                read_ply(p)

    def test_sh_constant(self):
        self.assertAlmostEqual(SH_C0, 1 / (2 * np.sqrt(np.pi)), places=12)


class Colmap(unittest.TestCase):
    def test_read_binary_model(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with open(d / "cameras.bin", "wb") as f:
                f.write(struct.pack("<Q", 1))
                f.write(struct.pack("<iiQQ", 1, 1, 640, 480))                    # PINHOLE
                f.write(struct.pack("<4d", 500.0, 510.0, 320.0, 240.0))
            with open(d / "images.bin", "wb") as f:
                f.write(struct.pack("<Q", 2))
                for i, name in ((2, b"b.jpg"), (1, b"a.jpg")):                    # written out of name order
                    f.write(struct.pack("<idddddddi", i, 1, 0, 0, 0, 0.1, 0.2, 0.3, 1))
                    f.write(name + b"\x00")
                    f.write(struct.pack("<Q", 0))
            cams, imgs = read_cameras_bin(d / "cameras.bin"), read_images_bin(d / "images.bin")
            self.assertEqual(cams[1]["model"], "PINHOLE")
            self.assertEqual([i["name"] for i in imgs], ["a.jpg", "b.jpg"])         # sorted by name
            np.testing.assert_allclose(imgs[0]["t"], [0.1, 0.2, 0.3])

    def test_unknown_camera_model_stops(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cameras.bin"
            with open(p, "wb") as f:
                f.write(struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 99, 10, 10))
            with self.assertRaises(SystemExit):
                read_cameras_bin(p)

    def test_quaternion(self):
        np.testing.assert_allclose(qvec2rotmat([1, 0, 0, 0]), np.eye(3), atol=1e-12)
        r = qvec2rotmat([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])               # 90 degrees about z
        np.testing.assert_allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-12)
        q = np.random.default_rng(0).normal(size=4)
        r = qvec2rotmat(q / np.linalg.norm(q))
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r), 1.0, places=12)

    def test_projection(self):
        cam = dict(model="PINHOLE", params=np.array([500.0, 500.0, 320.0, 240.0]))
        px = project(cam, np.array([[0.0, 0.0, 2.0], [0.2, -0.1, 2.0]]))
        np.testing.assert_allclose(px, [[320, 240], [370, 215]])
        # radial distortion pushes points outward from the centre for k1 > 0
        cam = dict(model="SIMPLE_RADIAL", params=np.array([500.0, 320.0, 240.0, 0.1]))
        a = project(cam, np.array([[1.0, 0.0, 2.0]]))[0, 0] - 320
        self.assertGreater(a, 500 * 0.5)


class SunAndShadow(unittest.TestCase):
    def test_sun_vector_is_unit_and_has_the_right_elevation(self):
        frame = plane_frame(np.array([0.0, 0.0, 1.0]))
        for elev, az in ((45, 0), (30, 120), (80, -200)):
            s = sun_vec(elev, az, frame)
            self.assertAlmostEqual(float(np.linalg.norm(s)), 1.0, places=12)
            self.assertAlmostEqual(float(np.degrees(np.arcsin(s @ frame[0]))), elev, places=9)

    def test_plane_frame_is_orthonormal(self):
        for up in ([0, 0, 1], [0.2, -0.9, 0.3], [1, 0, 0]):
            u, e1, e2 = plane_frame(np.array(up, float))
            np.testing.assert_allclose([u @ e1, u @ e2, e1 @ e2], 0, atol=1e-12)
            np.testing.assert_allclose([np.linalg.norm(v) for v in (u, e1, e2)], 1)

    def test_angle_between(self):
        self.assertAlmostEqual(angle_between(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])), 90.0, places=9)
        self.assertAlmostEqual(angle_between(np.array([1.0, 0, 0]), np.array([1.0, 0, 0])), 0.0, places=6)

    def test_look_at_puts_target_in_front_on_the_axis(self):
        pos, target, up = np.array([3.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        w2c = look_at(pos, target, up)
        t = w2c @ np.append(target, 1)
        np.testing.assert_allclose(t[:2], 0, atol=1e-12)
        self.assertAlmostEqual(t[2], np.linalg.norm(target - pos), places=12)
        np.testing.assert_allclose(w2c[:3, :3] @ w2c[:3, :3].T, np.eye(3), atol=1e-12)

    def test_ball_on_a_plane_casts_the_shadow_a_ruler_predicts(self):
        """Known truth: a ball of radius 1 centred at height 2 over the plane z=0, sun at 45 degrees elevation.
        The shadow is a disc of radius 1 whose centre is 2 / tan(45) = 2 away from the foot, away from the sun."""
        rng = np.random.default_rng(1)
        u = rng.normal(size=(40000, 3))
        ball = 2.0 * np.array([0, 0, 1.0]) + u / np.linalg.norm(u, axis=1, keepdims=True)   # surface points
        gx, gy = np.meshgrid(np.linspace(-6, 6, 241), np.linspace(-6, 6, 241))
        ground = np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]
        frame = plane_frame(np.array([0.0, 0.0, 1.0]))
        s = sun_vec(45, 0, frame)
        shadow = predict(ball, ground, s, cell=0.05, dilate=1, eps=0.02)
        centre = -2.0 * s[:2] / s[2]                                   # where the ball's centre ray meets the ground
        d = np.linalg.norm(ground[:, :2] - centre, axis=1)
        # the shadow of a sphere is an ellipse: semi-axis 1 across the sun, 1 / sin(elevation) = 1.414 along it
        inside = d < 0.95                                              # inside the ellipse whatever its direction
        outside = d > 1.0 / np.sin(np.radians(45)) + 0.15              # beyond the long axis
        self.assertGreater(shadow[inside].mean(), 0.97)
        self.assertLess(shadow[outside].mean(), 0.01)

    def test_no_shadow_with_nothing_above(self):
        occ = np.array([[0.0, 0.0, -1.0], [0.5, 0.5, -1.0]])           # below the receivers
        rec = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
        s = sun_vec(45, 0, plane_frame(np.array([0.0, 0.0, 1.0])))
        self.assertFalse(predict(occ, rec, s, cell=0.05, dilate=0, eps=0.01).any())

    def test_shadow_moves_with_the_sun_azimuth(self):
        ball = np.array([[0.0, 0.0, 2.0]])
        rec = np.array([[-2.0, 0, 0], [2.0, 0, 0], [0, 2.0, 0], [0, -2.0, 0]])
        frame = plane_frame(np.array([0.0, 0.0, 1.0]))
        for az in (0, 90, 180, -90):
            s = sun_vec(45, az, frame)
            expect = int(np.argmin(np.linalg.norm(rec[:, :2] - (-2.0 * s[:2] / s[2]), axis=1)))   # opposite the sun
            hit = predict(ball, rec, s, cell=0.1, dilate=1, eps=0.01)
            self.assertEqual(hit.sum(), 1, f"azimuth {az}: shadow landed on {hit}")
            self.assertEqual(int(np.argmax(hit)), expect, f"azimuth {az}: shadow landed on {hit}")

    def test_iou(self):
        a = np.array([1, 1, 0, 0], bool)
        self.assertEqual(iou(a, a), 1.0)
        self.assertEqual(iou(a, ~a), 0.0)
        self.assertAlmostEqual(iou(a, np.array([1, 0, 1, 0], bool)), 1 / 3)
        self.assertEqual(iou(~a & False, ~a & False), 0.0)             # both empty: defined as 0


class RepoDiscipline(unittest.TestCase):
    """The shared functions live in scripts/splatlib only. A copy in a script is how drift started."""
    SHARED = ("read_ply", "write_ply", "srgb_to_linear", "linear_to_srgb", "read_cameras_bin", "read_images_bin",
              "qvec2rotmat", "project", "look_at", "plane_frame", "sun_vec", "predict", "iou", "angle_between")
    ALLOWED = {("fill_shadows.py", "linear_to_srgb")}      # thin wrapper that fixes clip=True for the fill scripts

    def test_no_script_redefines_a_library_function(self):
        bad = []
        for f in sorted((ROOT / "scripts").glob("*.py")):
            for name in re.findall(r"^def (\w+)\(", f.read_text(), flags=re.M):
                if name in self.SHARED and (f.name, name) not in self.ALLOWED:
                    bad.append(f"{f.name}: {name}")
        self.assertEqual(bad, [], "move these into scripts/splatlib and import them")

    def test_every_script_compiles(self):
        for f in sorted((ROOT / "scripts").glob("*.py")):
            compile(f.read_text(), str(f), "exec")


if __name__ == "__main__":
    unittest.main(verbosity=2)
