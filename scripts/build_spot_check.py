"""Build a self-contained HTML for OLD-vs-NEW ground-truth comparison.

Renders each frame twice (same image, OLD boxes vs NEW boxes), embeds both as
base64 so the file is portable. Sampled and stratified by box-area change so
the most/least-changed frames are visible at a glance.

Inputs (paths fixed for this project):
  OLD labels: /mnt/data/.../exports/tool-only/2026-05-08/labels/train
  NEW labels: /mnt/data/.../exports/grasp-coupled/2026-05-08/labels/train
  Images:    /mnt/data/.../datasets/current/images/train

Output: spot_check.html (repo root)
Run:    uv run python scripts/build_spot_check.py
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from html import escape
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ent_cv.config import DATASETS_DIR

REPO = Path(__file__).resolve().parents[1]
OLD_LABELS = DATASETS_DIR / "exports" / "tool-only" / "2026-05-08" / "labels" / "train"
NEW_LABELS = DATASETS_DIR / "exports" / "grasp-coupled" / "2026-05-08" / "labels" / "train"
IMAGES = DATASETS_DIR / "current" / "images" / "train"
OUT = REPO / "spot_check.html"

CLASS_NAMES = [
    "Forceps",
    "Non-forcep instruments",
    "Suction",
    "Navigation probe",
    "Navigation suction",
    "Energy-based hemostasis without suction",
    "Saline-enhanced energy delivery",
    "Suction bovie",
    "Microdebrider",
    "Drill",
    "Empty Hand",
    "Not Sure",
    "Patient",
]

OLD_COLOR = "#ff5050"
NEW_COLOR = "#50e070"
THUMB_W = 560
JPEG_Q = 78


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
        out.append(
            Box(int(parts[0]), float(parts[1]), float(parts[2]),
                float(parts[3]), float(parts[4]))
        )
    return out


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def render(img_path: Path, boxes: list[Box], color: str, header: str) -> str:
    """Render thumbnail with bbox overlay; return base64 data URI."""
    font = load_font(13)
    header_font = load_font(14)
    with Image.open(img_path) as im:
        ow, oh = im.size
        scale = THUMB_W / ow
        th = int(oh * scale)
        im = im.convert("RGB").resize((THUMB_W, th), Image.LANCZOS)
        draw = ImageDraw.Draw(im)
        for b in boxes:
            x1 = (b.cx - b.w / 2) * THUMB_W
            y1 = (b.cy - b.h / 2) * th
            x2 = (b.cx + b.w / 2) * THUMB_W
            y2 = (b.cy + b.h / 2) * th
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label_text = (
                CLASS_NAMES[b.cls] if b.cls < len(CLASS_NAMES) else f"cls{b.cls}"
            )
            tx, ty = x1 + 2, max(0, y1 - 17)
            tb = draw.textbbox((tx, ty), label_text, font=font)
            draw.rectangle(tb, fill=color)
            draw.text((tx, ty), label_text, fill="black", font=font)
        # corner header (OLD / NEW badge)
        hb = draw.textbbox((6, 6), header, font=header_font)
        draw.rectangle(hb, fill=color)
        draw.text((6, 6), header, fill="black", font=header_font)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=JPEG_Q)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def collect_overlap() -> list[tuple[str, list[Box], list[Box], float]]:
    """Return list of (stem, old_boxes, new_boxes, ratio) for frames in both sets."""
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
        old_a = sum(b.area for b in ob)
        new_a = sum(b.area for b in nb)
        ratio = new_a / old_a if old_a > 0 else float("inf")
        rows.append((stem, ob, nb, ratio))
    return rows


def stratified_sample(
    rows: list[tuple[str, list[Box], list[Box], float]], per_bucket: int
) -> list[tuple[str, list[Box], list[Box], float, str]]:
    """Return sampled rows tagged with bucket label."""
    by_ratio = sorted(rows, key=lambda r: r[3], reverse=True)
    n = len(by_ratio)
    sliced = [
        (by_ratio[:per_bucket], "Largest box growth (NEW >> OLD)"),
        (by_ratio[(n // 2) - per_bucket // 2: (n // 2) + per_bucket - per_bucket // 2],
         "Median box growth"),
        (by_ratio[-per_bucket:], "Smallest box growth (NEW ~ OLD)"),
    ]
    out: list = []
    for chunk, tag in sliced:
        for stem, ob, nb, r in chunk:
            out.append((stem, ob, nb, r, tag))
    return out


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 0; padding: 24px; background: #111; color: #eee; max-width: 1700px; }
h1 { font-size: 18px; margin: 0 0 4px; font-weight: 500; }
.sub { color: #888; font-size: 13px; margin: 0 0 24px; max-width: 90ch; line-height: 1.5; }
h2 { margin: 36px 0 4px; font-size: 14px; color: #ddd;
     border-bottom: 1px solid #2a2a2a; padding-bottom: 4px; font-weight: 600; }
.note { color: #888; font-size: 12px; margin: 4px 0 12px; }
.grid { display: grid; grid-template-columns: 1fr; gap: 18px; }
.row { background: #1a1a1a; border-radius: 6px; padding: 10px; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.pair img { width: 100%; display: block; border-radius: 4px; }
.cap { font-family: 'SF Mono', monospace; font-size: 11.5px; color: #bbb;
       padding: 8px 6px 2px; line-height: 1.5; }
.ratio { color: #ffd54a; font-weight: 600; }
.boxlist { color: #888; font-size: 11px; }
.legend { display: inline-block; padding: 1px 6px; border-radius: 3px;
          font-weight: 600; color: #111; font-family: 'SF Mono', monospace; }
"""


def fmt_boxes(boxes: list[Box]) -> str:
    parts = []
    for b in boxes:
        nm = CLASS_NAMES[b.cls] if b.cls < len(CLASS_NAMES) else f"cls{b.cls}"
        parts.append(f"{nm} (w={b.w:.3f}, h={b.h:.3f})")
    return "; ".join(parts)


def render_section(
    title: str, items: list, note: str
) -> str:
    out = [
        f"<h2>{escape(title)} <span style='color:#888;font-weight:400'>"
        f"({len(items)} frames)</span></h2>",
        f"<p class='note'>{escape(note)}</p>",
        "<div class='grid'>",
    ]
    for stem, ob, nb, ratio, _ in items:
        img = IMAGES / f"{stem}.png"
        old_uri = render(img, ob, OLD_COLOR, "OLD")
        new_uri = render(img, nb, NEW_COLOR, "NEW")
        out.append(
            "<div class='row'>"
            "<div class='pair'>"
            f"<img src='{old_uri}'>"
            f"<img src='{new_uri}'>"
            "</div>"
            "<div class='cap'>"
            f"<b>{escape(stem)}</b> &middot; "
            f"area ratio NEW/OLD = <span class='ratio'>{ratio:.2f}x</span><br>"
            f"<span class='boxlist'><span class='legend' "
            f"style='background:{OLD_COLOR}'>OLD</span> {escape(fmt_boxes(ob))}</span><br>"
            f"<span class='boxlist'><span class='legend' "
            f"style='background:{NEW_COLOR}'>NEW</span> {escape(fmt_boxes(nb))}</span>"
            "</div></div>"
        )
    out.append("</div>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=15,
                    help="Frames per bucket (top/median/bottom by area ratio).")
    args = ap.parse_args()

    rows = collect_overlap()
    print(f"Overlap with both labels and image: {len(rows)} frames")
    sampled = stratified_sample(rows, args.per_bucket)

    sections = []
    for tag in [
        "Largest box growth (NEW >> OLD)",
        "Median box growth",
        "Smallest box growth (NEW ~ OLD)",
    ]:
        items = [r for r in sampled if r[4] == tag]
        notes = {
            "Largest box growth (NEW >> OLD)":
                "Frames where the new box is dramatically larger than the old. "
                "This is where you most clearly see the convention shift "
                "(instrument-only -> instrument+gloved-hand).",
            "Median box growth":
                "Typical mid-distribution frames -- representative of the average "
                "convention difference (~2.3x area).",
            "Smallest box growth (NEW ~ OLD)":
                "Frames where old and new boxes are nearly identical -- usually "
                "instruments captured with very little visible hand, so the two "
                "conventions converge.",
        }
        sections.append(render_section(tag, items, notes[tag]))

    intro = (
        "<p class='sub'>Each row shows the <b>same frame</b> rendered twice: "
        "<span class='legend' style='background:" + OLD_COLOR + "'>OLD</span> on the "
        "left (instrument-only convention) and "
        "<span class='legend' style='background:" + NEW_COLOR + "'>NEW</span> on the "
        "right (instrument + gloved-hand convention). "
        f"There are <b>{len(rows)} overlap frames</b> where both label sets exist; "
        f"this page shows {args.per_bucket * 3} of them, stratified by area ratio.</p>"
    )

    html = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<title>OLD vs NEW GT Comparison</title>",
        f"<style>{CSS}</style></head><body>",
        "<h1>Ground Truth Convention Comparison: OLD vs NEW</h1>",
        intro,
        *sections,
        "</body></html>",
    ]
    OUT.write_text("\n".join(html))
    size_mb = OUT.stat().st_size / 1e6
    print(f"Wrote {OUT} ({size_mb:.1f} MB)")
    print(f"Frames embedded: {len(sampled)} (each rendered twice)")


if __name__ == "__main__":
    main()
