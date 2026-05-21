"""Render every frame as a side-by-side Convention A vs Convention B image and
build an index HTML that lazy-loads them.

  Convention A (tool-only):    red boxes, left panel.
  Convention B (grasp-coupled): green boxes, right panel.

Frames present in only one convention are still rendered; the missing panel is
labeled clearly. Output lives under <repo>/reports/compare_conventions/ (gitignored).

Run: uv run python scripts/render_convention_pairs.py [--workers N] [--limit N]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ent_cv.config import DATASETS_DIR

REPO = Path(__file__).resolve().parents[1]
A_LABELS = DATASETS_DIR / "exports" / "tool-only" / "2026-05-08" / "labels" / "train"
B_LABELS = DATASETS_DIR / "exports" / "grasp-coupled" / "2026-05-08" / "labels" / "train"
IMAGES = DATASETS_DIR / "current" / "images" / "train"
OUT_DIR = REPO / "reports" / "compare_conventions"
THUMBS_DIR = OUT_DIR / "thumbs"

CLASS_NAMES = [
    "Forceps", "Non-forcep instruments", "Suction", "Navigation probe",
    "Navigation suction", "Energy-based hemostasis without suction",
    "Saline-enhanced energy delivery", "Suction bovie", "Microdebrider",
    "Drill", "Empty Hand", "Not Sure", "Patient",
]

A_COLOR = "#ff5050"
B_COLOR = "#50e070"
MUTED = "#666"
PANEL_W = 560
GAP = 6
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
        out.append(Box(int(parts[0]), float(parts[1]), float(parts[2]),
                       float(parts[3]), float(parts[4])))
    return out


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_panel(im: Image.Image, boxes: list[Box], color: str, header: str,
               present: bool) -> Image.Image:
    font = load_font(13)
    header_font = load_font(14)
    ow, oh = im.size
    scale = PANEL_W / ow
    th = int(oh * scale)
    out = im.convert("RGB").resize((PANEL_W, th), Image.LANCZOS)
    draw = ImageDraw.Draw(out)
    if present:
        for b in boxes:
            x1 = (b.cx - b.w / 2) * PANEL_W
            y1 = (b.cy - b.h / 2) * th
            x2 = (b.cx + b.w / 2) * PANEL_W
            y2 = (b.cy + b.h / 2) * th
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label_text = (CLASS_NAMES[b.cls] if b.cls < len(CLASS_NAMES)
                          else f"cls{b.cls}")
            tx, ty = x1 + 2, max(0, y1 - 17)
            tb = draw.textbbox((tx, ty), label_text, font=font)
            draw.rectangle(tb, fill=color)
            draw.text((tx, ty), label_text, fill="black", font=font)
        badge_color = color
    else:
        # dim the panel and stamp "NO LABEL" across center
        overlay = Image.new("RGB", (PANEL_W, th), (0, 0, 0))
        out = Image.blend(out, overlay, 0.55)
        draw = ImageDraw.Draw(out)
        msg = "(no Convention B label)"
        big = load_font(22)
        bb = draw.textbbox((0, 0), msg, font=big)
        mw, mh = bb[2] - bb[0], bb[3] - bb[1]
        draw.text(((PANEL_W - mw) // 2, (th - mh) // 2), msg,
                  fill="#aaa", font=big)
        badge_color = MUTED
    hb = draw.textbbox((6, 6), header, font=header_font)
    draw.rectangle(hb, fill=badge_color)
    draw.text((6, 6), header, fill="black", font=header_font)
    return out


def stats_for(stem: str) -> tuple[int, int, float]:
    """Cheap: read labels and return (n_a_boxes, n_b_boxes, B/A area ratio)."""
    a_boxes = parse_label(A_LABELS / f"{stem}.txt")
    b_boxes = parse_label(B_LABELS / f"{stem}.txt")
    a_area = sum(b.area for b in a_boxes)
    b_area = sum(b.area for b in b_boxes)
    ratio = (b_area / a_area) if a_area > 0 else 0.0
    return len(a_boxes), len(b_boxes), ratio


def render_one(stem: str) -> str:
    """Render the side-by-side composite for one frame to disk; return stem."""
    img_path = IMAGES / f"{stem}.png"
    a_boxes = parse_label(A_LABELS / f"{stem}.txt")
    b_boxes = parse_label(B_LABELS / f"{stem}.txt")
    a_present = (A_LABELS / f"{stem}.txt").exists()
    b_present = (B_LABELS / f"{stem}.txt").exists()

    with Image.open(img_path) as im:
        left = draw_panel(im, a_boxes, A_COLOR, "A (tool-only)", a_present)
        right = draw_panel(im, b_boxes, B_COLOR, "B (grasp-coupled)", b_present)

    h = max(left.height, right.height)
    canvas = Image.new("RGB", (PANEL_W * 2 + GAP, h), (17, 17, 17))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (PANEL_W + GAP, 0))
    canvas.save(THUMBS_DIR / f"{stem}.jpg", "JPEG", quality=JPEG_Q)
    return stem


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 0; padding: 24px; background: #111; color: #eee; max-width: 1300px; }
h1 { font-size: 18px; margin: 0 0 4px; font-weight: 500; }
.sub { color: #888; font-size: 13px; margin: 0 0 20px; max-width: 90ch; line-height: 1.5; }
.controls { position: sticky; top: 0; background: #111; padding: 8px 0;
            border-bottom: 1px solid #2a2a2a; margin-bottom: 16px; z-index: 10;
            font-size: 13px; }
.controls button { background: #222; color: #ddd; border: 1px solid #333;
                   padding: 4px 10px; margin-right: 4px; border-radius: 4px;
                   cursor: pointer; font: inherit; }
.controls button.active { background: #50e070; color: #111; border-color: #50e070; }
.row { background: #1a1a1a; border-radius: 6px; padding: 8px; margin-bottom: 12px; }
.row img { width: 100%; display: block; border-radius: 4px; }
.cap { font-family: 'SF Mono', monospace; font-size: 11.5px; color: #bbb;
       padding: 6px 4px 0; line-height: 1.5; }
.tag { display: inline-block; padding: 1px 6px; border-radius: 3px;
       font-weight: 600; color: #111; font-size: 10.5px; margin-left: 6px; }
.case-header { color: #888; font-size: 13px; margin: 24px 0 6px;
               border-bottom: 1px solid #2a2a2a; padding-bottom: 4px; }
"""

JS = """
function setFilter(name, btn) {
  document.querySelectorAll('.controls button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.row').forEach(r => {
    r.style.display = (name === 'all' || r.dataset.kind === name) ? '' : 'none';
  });
  document.querySelectorAll('.case-header').forEach(h => {
    let n = h.nextElementSibling;
    let any = false;
    while (n && n.classList.contains('row')) {
      if (n.style.display !== 'none') { any = true; break; }
      n = n.nextElementSibling;
    }
    h.style.display = any ? '' : 'none';
  });
}
"""


def case_of(stem: str) -> str:
    return stem.rsplit("_f", 1)[0]


def kind_of(na: int, nb: int) -> str:
    """Classify by annotation presence (boxes), not by file presence."""
    if na > 0 and nb > 0:
        return "both"
    if na > 0:
        return "only-a"
    if nb > 0:
        return "only-b"
    return "background"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only process the first N frames (for quick tests).")
    ap.add_argument("--force", action="store_true",
                    help="Re-render thumbnails even when they already exist.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)

    a = {p.stem for p in A_LABELS.glob("*.txt")}
    b = {p.stem for p in B_LABELS.glob("*.txt")}
    imgs = {p.stem for p in IMAGES.glob("*.png")}
    stems = sorted((a | b) & imgs)
    if args.limit:
        stems = stems[: args.limit]

    # Render only thumbnails that don't already exist (unless --force).
    to_render = [s for s in stems
                 if args.force or not (THUMBS_DIR / f"{s}.jpg").exists()]
    print(f"{len(stems)} frames total; {len(to_render)} need rendering "
          f"({args.workers} workers)")
    if to_render:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(render_one, s): s for s in to_render}
            for i, fut in enumerate(as_completed(futs), 1):
                fut.result()
                if i % 100 == 0 or i == len(to_render):
                    print(f"  {i}/{len(to_render)}")

    # Build HTML using cheap stat reads (kind classified by annotation presence).
    rows_html: list[str] = []
    last_case = None
    counts = {"both": 0, "only-a": 0, "only-b": 0, "background": 0}
    for stem in stems:
        na, nb, ratio = stats_for(stem)
        kind = kind_of(na, nb)
        counts[kind] += 1
        case = case_of(stem)
        if case != last_case:
            rows_html.append(f"<div class='case-header'>{escape(case)}</div>")
            last_case = case
        tag = ""
        if kind == "only-a":
            tag = f"<span class='tag' style='background:{A_COLOR}'>only A</span>"
        elif kind == "only-b":
            tag = f"<span class='tag' style='background:{B_COLOR}'>only B</span>"
        elif kind == "background":
            tag = f"<span class='tag' style='background:{MUTED};color:#eee'>background</span>"
        ratio_txt = f"{ratio:.2f}x" if ratio > 0 else "—"
        rows_html.append(
            f"<div class='row' data-kind='{kind}'>"
            f"<img loading='lazy' src='thumbs/{stem}.jpg'>"
            f"<div class='cap'><b>{escape(stem)}</b>{tag} &middot; "
            f"A boxes: {na} &middot; B boxes: {nb} &middot; B/A area: {ratio_txt}"
            f"</div></div>"
        )

    intro = (
        "<p class='sub'>Every frame rendered side by side: "
        f"<span style='color:{A_COLOR}'>Convention A (tool-only)</span> on the left, "
        f"<span style='color:{B_COLOR}'>Convention B (grasp-coupled)</span> on the right. "
        "<b>Categories use annotation presence</b> (≥1 box), not file presence: "
        f"both: <b>{counts['both']}</b> &middot; "
        f"only A: <b>{counts['only-a']}</b> &middot; "
        f"only B: <b>{counts['only-b']}</b> &middot; "
        f"background (no boxes either side): <b>{counts['background']}</b> &middot; "
        f"total: <b>{len(stems)}</b>.</p>"
    )
    controls = (
        "<div class='controls'>Filter: "
        "<button class='active' onclick=\"setFilter('all', this)\">All</button>"
        "<button onclick=\"setFilter('both', this)\">Both</button>"
        "<button onclick=\"setFilter('only-a', this)\">Only A</button>"
        "<button onclick=\"setFilter('only-b', this)\">Only B</button>"
        "<button onclick=\"setFilter('background', this)\">Background</button>"
        "</div>"
    )
    html = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Convention A vs B — every frame</title>",
        f"<style>{CSS}</style></head><body>",
        "<h1>Convention A (tool-only) vs Convention B (grasp-coupled)</h1>",
        intro,
        controls,
        *rows_html,
        f"<script>{JS}</script>",
        "</body></html>",
    ]
    index = OUT_DIR / "index.html"
    index.write_text("\n".join(html))
    print(f"Wrote {index}")
    print(f"Categories: {counts}")


if __name__ == "__main__":
    main()
