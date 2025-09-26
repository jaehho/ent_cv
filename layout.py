#!/usr/bin/env python3
# Combine 4 JPG images into a single 2x2 collage.

import sys
import argparse
from PIL import Image, ImageOps

def parse_tile_spec(s):
    if not s:
        return None
    s = str(s).lower().strip()
    if "x" in s:
        w, h = s.split("x", 1)
        return int(w), int(h)
    n = int(s)
    return n, n

def prepare_tile(img: Image.Image, tile_w: int, tile_h: int, bg_color: str) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    # Fit the image into the tile while preserving aspect ratio, centered on background
    fitted = img.copy()
    fitted.thumbnail((tile_w, tile_h), Image.LANCZOS)
    tile = Image.new("RGB", (tile_w, tile_h), bg_color)
    ox = (tile_w - fitted.width) // 2
    oy = (tile_h - fitted.height) // 2
    tile.paste(fitted, (ox, oy))
    return tile

def main(argv):
    parser = argparse.ArgumentParser(description="Organize 4 JPG images into 1 (2x2 grid).")
    parser.add_argument("img1")
    parser.add_argument("img2")
    parser.add_argument("img3")
    parser.add_argument("img4")
    parser.add_argument("output", help="Output image path (e.g., collage.jpg)")
    parser.add_argument("--tile", help="Tile size per image. Examples: 512 or 800x600. Default: auto from inputs.", default=None)
    parser.add_argument("--margin", type=int, default=0, help="Margin in pixels around and between tiles. Default: 0")
    parser.add_argument("--bg", default="#ffffff", help="Background color (e.g., #ffffff or black). Default: white")
    args = parser.parse_args(argv)

    paths = [args.img1, args.img2, args.img3, args.img4]
    images = []
    try:
        for p in paths:
            img = Image.open(p)
            img = ImageOps.exif_transpose(img)
            images.append(img)
    except Exception as e:
        print(f"Failed to open images: {e}", file=sys.stderr)
        return 1

    # Determine tile size
    tile_spec = parse_tile_spec(args.tile)
    if tile_spec:
        tile_w, tile_h = tile_spec
    else:
        # Auto: use the min width/height among inputs to avoid upscaling
        tile_w = min(im.width for im in images)
        tile_h = min(im.height for im in images)

    tiles = [prepare_tile(im, tile_w, tile_h, args.bg) for im in images]

    m = max(0, int(args.margin))
    out_w = tile_w * 2 + m * 3
    out_h = tile_h * 2 + m * 3
    out = Image.new("RGB", (out_w, out_h), args.bg)

    positions = [
        (m, m),  # top-left
        (m * 2 + tile_w, m),  # top-right
        (m, m * 2 + tile_h),  # bottom-left
        (m * 2 + tile_w, m * 2 + tile_h),  # bottom-right
    ]

    for tile, pos in zip(tiles, positions):
        out.paste(tile, pos)

    try:
        # If output is JPG, ensure RGB mode
        if args.output.lower().endswith((".jpg", ".jpeg")) and out.mode != "RGB":
            out = out.convert("RGB")
        out.save(args.output, quality=95)
    except Exception as e:
        print(f"Failed to save output: {e}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))