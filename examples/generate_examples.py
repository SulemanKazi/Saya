#!/usr/bin/env python
"""Generate synthetic binary shadow images for examples.

Run once after installing the package:
    python examples/generate_examples.py
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE = 256


def save(arr: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)
    print(f"  Saved {path}")


def circle(size: int = SIZE) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    Y, X = np.ogrid[:size, :size]
    cx, cy, r = size // 2, size // 2, size // 3
    img[(X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2] = 255
    return img


def square(size: int = SIZE) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    m = size // 5
    img[m : size - m, m : size - m] = 255
    return img


def triangle(size: int = SIZE) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    m = size // 8
    draw.polygon(
        [(size // 2, m), (size - m, size - m), (m, size - m)],
        fill=255,
    )
    return np.array(img)


def star(size: int = SIZE, points: int = 5) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    cx, cy = size / 2, size / 2
    outer_r = size * 0.38
    inner_r = outer_r * 0.45
    verts = []
    for i in range(points * 2):
        angle = np.pi * i / points - np.pi / 2
        r = outer_r if i % 2 == 0 else inner_r
        verts.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))
    draw.polygon(verts, fill=255)
    return np.array(img)


def cross(size: int = SIZE) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    t = size // 6  # thickness
    c = size // 2
    img[c - t : c + t, :] = 255  # horizontal bar
    img[:, c - t : c + t] = 255  # vertical bar
    return img


def letter_a(size: int = SIZE) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    m = size // 8
    # Outer triangle
    draw.polygon(
        [(size // 2, m), (size - m, size - m), (m, size - m)],
        fill=255,
    )
    # Inner cutout (hollow triangle)
    inner_m = size // 4
    draw.polygon(
        [
            (size // 2, size // 3),
            (size - inner_m, size - inner_m),
            (inner_m, size - inner_m),
        ],
        fill=0,
    )
    # Crossbar
    bar_y = int(size * 0.55)
    bar_h = size // 12
    draw.rectangle(
        [size // 4, bar_y - bar_h, 3 * size // 4, bar_y + bar_h],
        fill=255,
    )
    return np.array(img)


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    print("Generating two_view examples …")
    save(circle(), os.path.join(base, "two_view", "shadow_0.png"))
    save(square(), os.path.join(base, "two_view", "shadow_1.png"))

    print("Generating three_view examples …")
    save(triangle(), os.path.join(base, "three_view", "shadow_0.png"))
    save(star(), os.path.join(base, "three_view", "shadow_1.png"))
    save(cross(), os.path.join(base, "three_view", "shadow_2.png"))

    print("Done.")
