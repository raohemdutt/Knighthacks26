"""Writes the benchmark prompt set's fixture images (images/*.png), deterministically, with the standard library only.

    python fixtures/lab/bench/make_images.py

The images are drawn shapes, never camera frames (kernels rubric K1 d: raw frames never leave the Jetson). Running it
again writes the same bytes, so the prompt-set hash in lab/bench.py stays put; a changed image is a new prompt set.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def png(width: int, height: int, pixel) -> bytes:
    """An 8-bit RGB PNG whose pixel (x, y) is pixel(x, y) -> (r, g, b)."""
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # filter: none
        for x in range(width):
            rows.extend(pixel(x, y))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(rows), 9))
            + chunk(b"IEND", b""))


def bars(x: int, y: int) -> tuple[int, int, int]:
    """1280x720: five vertical bars of different heights and colours on a light background, with a baseline."""
    heights = (420, 260, 560, 180, 340)
    colours = ((200, 60, 60), (60, 140, 200), (80, 170, 90), (230, 170, 40), (140, 90, 180))
    if 640 <= y <= 644 and 120 <= x <= 1160:
        return (30, 30, 30)
    for i, (h, c) in enumerate(zip(heights, colours)):
        left = 160 + i * 200
        if left <= x < left + 120 and 640 - h <= y < 640:
            return c
    return (245, 245, 240)


def shapes(x: int, y: int) -> tuple[int, int, int]:
    """640x480: a red circle, a blue square and a green triangle on white."""
    if (x - 160) ** 2 + (y - 240) ** 2 <= 90 ** 2:
        return (210, 40, 40)
    if 280 <= x < 400 and 180 <= y < 300:
        return (40, 80, 210)
    if 300 <= y < 400 and abs(x - 530) <= (y - 300) * 0.8:
        return (40, 160, 70)
    return (255, 255, 255)


IMAGES = {"bars_1280x720.png": (1280, 720, bars), "shapes_640x480.png": (640, 480, shapes)}


def main() -> None:
    for name, (w, h, fn) in IMAGES.items():
        (HERE / "images" / name).write_bytes(png(w, h, fn))
        print(name)


if __name__ == "__main__":
    main()
