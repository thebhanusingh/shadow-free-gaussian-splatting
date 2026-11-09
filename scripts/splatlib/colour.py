"""Colour: sRGB <-> linear light, luminance weights, chroma. Do gain maths in LINEAR light."""
import numpy as np

LUM = np.array([0.2126, 0.7152, 0.0722])  # Rec.709 luminance weights for linear RGB


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c, clip=False):
    """clip=True also clamps to 0..1 (used when writing finished colours); otherwise only negatives are clamped."""
    c = np.clip(c, 0, 1) if clip else np.maximum(c, 0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def chroma(rgb):
    """Colour without brightness: each channel's share of the sum."""
    return rgb / np.maximum(rgb.sum(1, keepdims=True), 1e-6)
