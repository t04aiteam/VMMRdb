# /// script
# dependencies = ["numpy", "pillow"]
# ///
"""Crop-and-match color heuristic: no dataset, no model. Drops the top third of
a vehicle crop (windshield/roof glare), buckets every remaining pixel into one
of 14 named colors by hue/saturation/value, and returns the majority bucket.

Palette (14): white, black, gray, silver, red, orange, yellow, gold, brown,
beige, green, blue, purple, pink -- chosen to match the ~10-12 categories
real-world automotive color-popularity reports (Axalta/PPG/KBB) actually
track, not an arbitrary color-naming list.

ponytail: histogram-snap heuristic, not a trained model -- there is no
per-image color ground truth anywhere in this repo's datasets to train
against. A resnet18 fallback recipe exists at code/color/train_color.py
(trained on CompCars sv_data color labels) if this heuristic misclassifies
metallics/two-tone paint too often in practice.

Usage:
  uv run code/color/color_heuristic.py   # runs the self-check demo
"""
import numpy as np
from PIL import Image


def estimate_color(img: Image.Image) -> dict:
    """PIL crop -> {"color": <name>, "confidence": <fraction of pixels agreeing>}."""
    rgb = np.asarray(img.convert("RGB"))
    rgb = rgb[rgb.shape[0] // 3:]  # drop top third: windshield/roof glare
    if rgb.size == 0:
        return {"color": None, "confidence": 0.0}

    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.float32)
    h = hsv[..., 0].ravel() * (360.0 / 255.0)
    s = hsv[..., 1].ravel() / 255.0
    v = hsv[..., 2].ravel() / 255.0

    color = np.full(h.shape, "", dtype=object)

    achromatic = s < 0.12
    color[achromatic & (v < 0.18)] = "black"
    color[achromatic & (v >= 0.18) & (v < 0.45)] = "gray"
    color[achromatic & (v >= 0.45) & (v < 0.75)] = "silver"
    color[achromatic & (v >= 0.75)] = "white"

    chromatic = ~achromatic
    color[chromatic & ((h >= 345) | (h < 15))] = "red"

    orange_band = chromatic & (h >= 15) & (h < 45)
    color[orange_band & (v < 0.35)] = "brown"
    color[orange_band & (color == "") & (s < 0.3)] = "beige"
    color[orange_band & (color == "")] = "orange"

    yellow_band = chromatic & (h >= 45) & (h < 70)
    color[yellow_band & (s < 0.35) & (v >= 0.55)] = "beige"
    color[yellow_band & (color == "") & (v < 0.6)] = "gold"
    color[yellow_band & (color == "")] = "yellow"

    color[chromatic & (h >= 70) & (h < 170)] = "green"
    color[chromatic & (h >= 170) & (h < 255)] = "blue"
    color[chromatic & (h >= 255) & (h < 290)] = "purple"

    pink_band = chromatic & (h >= 290) & (h < 345)
    color[pink_band & (s > 0.65) & (v < 0.55)] = "red"
    color[pink_band & (color == "")] = "pink"

    color[color == ""] = "gray"  # unreachable given the ranges above cover 0-360; safety net

    names, counts = np.unique(color, return_counts=True)
    winner = counts.argmax()
    return {"color": str(names[winner]), "confidence": round(float(counts[winner] / counts.sum()), 4)}


def demo():
    import colorsys

    cases = [
        (0.5, 0.05, 0.90, "white"),
        (0.5, 0.05, 0.10, "black"),
        (0.5, 0.05, 0.35, "gray"),
        (0.5, 0.05, 0.60, "silver"),
        (0 / 360, 0.80, 0.80, "red"),
        (30 / 360, 0.90, 0.90, "orange"),
        (30 / 360, 0.90, 0.20, "brown"),
        (30 / 360, 0.15, 0.85, "beige"),
        (60 / 360, 0.90, 0.90, "yellow"),
        (60 / 360, 0.90, 0.45, "gold"),
        (60 / 360, 0.20, 0.85, "beige"),
        (120 / 360, 0.70, 0.60, "green"),
        (210 / 360, 0.70, 0.60, "blue"),
        (270 / 360, 0.70, 0.60, "purple"),
        (320 / 360, 0.30, 0.90, "pink"),
        (320 / 360, 0.90, 0.30, "red"),
    ]
    for h, s, v, expected in cases:
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        img = Image.new("RGB", (40, 60), (round(r * 255), round(g * 255), round(b * 255)))
        result = estimate_color(img)
        assert result["color"] == expected, f"h={h} s={s} v={v}: got {result}, expected {expected}"
    print(f"color_heuristic demo: all {len(cases)} cases passed")


if __name__ == "__main__":
    demo()
