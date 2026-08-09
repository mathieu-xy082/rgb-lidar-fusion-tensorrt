#!/usr/bin/env python3
"""Convert a simple 8-bit PNG RGB/RGBA image to ASCII PPM P3.

This helper intentionally avoids Pillow/ImageMagick so onboarding can render a
KITTI camera image with only Python's standard library.

Usage:
  pdm run python docs/onboarding/scripts/png_to_ppm.py input.png output.ppm
"""
from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def parse_png(path: Path) -> tuple[int, int, int, int, bytes]:
    data = path.read_bytes()
    if not data.startswith(PNG_SIG):
        raise ValueError(f"{path} is not a PNG file")
    offset = len(PNG_SIG)
    width = height = bit_depth = color_type = None
    compressed = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        chunk_type = data[offset : offset + 4]
        offset += 4
        chunk_data = data[offset : offset + length]
        offset += length
        offset += 4  # crc
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            if compression != 0 or filter_method != 0 or interlace != 0:
                raise ValueError("only non-interlaced standard PNG images are supported")
        elif chunk_type == b"IDAT":
            compressed.extend(chunk_data)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None or bit_depth is None or color_type is None:
        raise ValueError("PNG missing IHDR")
    if bit_depth != 8 or color_type not in {2, 6}:
        raise ValueError(
            f"only 8-bit RGB/RGBA PNG is supported, got bit_depth={bit_depth}, color_type={color_type}"
        )
    return width, height, bit_depth, color_type, zlib.decompress(bytes(compressed))


def unfilter(width: int, height: int, color_type: int, raw: bytes) -> bytes:
    channels = 3 if color_type == 2 else 4
    bpp = channels
    row_bytes = width * channels
    rows: list[bytes] = []
    pos = 0
    prev = bytearray(row_bytes)
    for _ in range(height):
        filter_type = raw[pos]
        pos += 1
        src = raw[pos : pos + row_bytes]
        pos += row_bytes
        out = bytearray(row_bytes)
        for i, value in enumerate(src):
            left = out[i - bpp] if i >= bpp else 0
            up = prev[i]
            up_left = prev[i - bpp] if i >= bpp else 0
            if filter_type == 0:
                recon = value
            elif filter_type == 1:
                recon = value + left
            elif filter_type == 2:
                recon = value + up
            elif filter_type == 3:
                recon = value + ((left + up) // 2)
            elif filter_type == 4:
                recon = value + paeth(left, up, up_left)
            else:
                raise ValueError(f"unsupported PNG filter type {filter_type}")
            out[i] = recon & 0xFF
        rows.append(bytes(out))
        prev = out
    return b"".join(rows)


def write_ppm(path: Path, width: int, height: int, pixels: bytes, color_type: int) -> None:
    channels = 3 if color_type == 2 else 4
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write(f"P3\n{width} {height}\n255\n")
        for y in range(height):
            values = []
            start = y * width * channels
            for x in range(width):
                i = start + x * channels
                values.append(f"{pixels[i]} {pixels[i + 1]} {pixels[i + 2]}")
            f.write(" ".join(values) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_png", type=Path)
    parser.add_argument("output_ppm", type=Path)
    args = parser.parse_args()
    width, height, _bit_depth, color_type, raw = parse_png(args.input_png)
    pixels = unfilter(width, height, color_type, raw)
    write_ppm(args.output_ppm, width, height, pixels, color_type)
    print(f"wrote {args.output_ppm} image_shape={height}x{width}")


if __name__ == "__main__":
    main()
