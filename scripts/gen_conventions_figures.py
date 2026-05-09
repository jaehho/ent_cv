"""Generate figures for docs/main.typ.

For each selected frame, emit a side-by-side PNG (OLD boxes | NEW boxes) into
the docs figures directory. Selection is stratified by area-growth ratio.

Output: docs/figures/*.png (in repo)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ent_cv.config import DATASETS_DIR

REPO = Path(__file__).resolve().parents[1]
OLD_LABELS = DATASETS_DIR / "exports" / "dataset_old_style_backup_20260508" / "labels" / "train"
NEW_LABELS = DATASETS_DIR / "exports" / "combined_new" / "labels" / "train"
IMAGES = DATASETS_DIR / "dataset" / "images" / "train"
FIG_DIR = REPO / "docs" / "figures"

CLASS_NAMES = [
    "Forceps", "Non-forcep instruments", "Suction", "Navigation probe",
    "Navigation suction", "Energy-based hemostasis without suction",
    "Saline-enhanced energy delivery", "Suction bovie", "Microdebrider",
    "Drill", "Empty Hand", "Not Sure", "Patient",
]

OLD_COLOR = (255, 80, 80)
NEW_COLOR = (80, 224, 112)
PANEL_W = 720
GAP = 16


@dataclass
class Box:
    cls: int
    cx: float
    cy: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h


def parse_label(path: Path) -> list[Box]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    out: list[Box] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        out.append(Box(int(parts[0]), float(parts[1]), float(parts[2]),
                       float(parts[3]), float(parts[4])))
    return out


def load_font(size: int):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def render_panel(img_path: Path, boxes: list[Box], color: tuple[int, int, int],
                 header: str) -> Image.Image:
    font = load_font(16)
    header_font = load_font(18)
    with Image.open(img_path) as im:
        ow, oh = im.size
        scale = PANEL_W / ow
        ph = int(oh * scale)
        im = im.convert("RGB").resize((PANEL_W, ph), Image.LANCZOS)
        draw = ImageDraw.Draw(im)
        for b in boxes:
            x1 = (b.cx - b.w / 2) * PANEL_W
            y1 = (b.cy - b.h / 2) * ph
            x2 = (b.cx + b.w / 2) * PANEL_W
            y2 = (b.cy + b.h / 2) * ph
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            txt = CLASS_NAMES[b.cls] if b.cls < len(CLASS_NAMES) else f"cls{b.cls}"
            tx, ty = x1 + 3, max(0, y1 - 21)
            tb = draw.textbbox((tx, ty), txt, font=font)
            draw.rectangle(tb, fill=color)
            draw.text((tx, ty), txt, fill=(0, 0, 0), font=font)
        hb = draw.textbbox((8, 8), header, font=header_font)
        draw.rectangle(hb, fill=color)
        draw.text((8, 8), header, fill=(0, 0, 0), font=header_font)
    return im


def make_pair(stem: str, old: list[Box], new: list[Box], out_path: Path) -> None:
    img_path = IMAGES / f"{stem}.png"
    left = render_panel(img_path, old, OLD_COLOR, "OLD (instrument-only)")
    right = render_panel(img_path, new, NEW_COLOR, "NEW (instrument + grasp)")
    h = max(left.height, right.height)
    canvas = Image.new("RGB", (PANEL_W * 2 + GAP, h), (17, 17, 17))
    canvas.paste(left, (0, (h - left.height) // 2))
    canvas.paste(right, (PANEL_W + GAP, (h - right.height) // 2))
    canvas.save(out_path, "PNG", optimize=True)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    old_files = {p.stem for p in OLD_LABELS.glob("*.txt") if p.stat().st_size > 0}
    new_files = {p.stem for p in NEW_LABELS.glob("*.txt") if p.stat().st_size > 0}
    rows = []
    for stem in sorted(old_files & new_files):
        if not (IMAGES / f"{stem}.png").exists():
            continue
        ob = parse_label(OLD_LABELS / f"{stem}.txt")
        nb = parse_label(NEW_LABELS / f"{stem}.txt")
        if not ob or not nb:
            continue
        oa = sum(b.area for b in ob)
        na = sum(b.area for b in nb)
        rows.append((stem, ob, nb, na / oa if oa > 0 else float("inf")))

    rows.sort(key=lambda r: r[3], reverse=True)
    n = len(rows)

    # Pick: 3 largest, 2 median, 1 smallest growth. Skip extreme outliers (top of
    # the distribution can be dominated by edge cases). Use rank-based picks.
    selections = [
        ("01_largest_growth_a", rows[2]),     # rank 3
        ("02_largest_growth_b", rows[8]),     # rank 9
        ("03_largest_growth_c", rows[20]),    # rank 21
        ("04_median_growth_a", rows[n // 2 - 5]),
        ("05_median_growth_b", rows[n // 2 + 5]),
        ("06_smallest_growth", rows[-3]),     # rank n-2 (skip degenerate end)
    ]

    print(f"Total overlap rows: {n}")
    for name, (stem, ob, nb, ratio) in selections:
        out = FIG_DIR / f"{name}.png"
        make_pair(stem, ob, nb, out)
        print(f"  {name}: {stem}  ratio={ratio:.2f}x  -> {out}")


if __name__ == "__main__":
    main()
