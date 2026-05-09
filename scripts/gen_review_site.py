"""Generate a static HTML site to review all ground-truth annotations.

For each labeled frame, draw the GT bounding boxes onto a thumbnail. Build
per-class and per-case galleries plus a top-level index. Click any thumbnail
to open the original full-resolution PNG.

Output: /mnt/data/ent_cv/review_site/
  index.html
  class/<label>.html
  case/<case>.html
  thumbs/<stem>.jpg
  images -> symlink to original images dir

Run:
    uv run python scripts/gen_review_site.py
Serve:
    python -m http.server 8080 --directory /mnt/data/ent_cv/review_site
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import yaml

from ent_cv.config import DATA_DIR, DATASETS_DIR

DATASET_DIR = DATASETS_DIR / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
IMAGES_DIR = DATASET_DIR / "images" / "train"
LABELS_DIR = DATASET_DIR / "labels" / "train"

OUT_DIR = DATA_DIR / "review_site"
THUMBS_DIR = OUT_DIR / "thumbs"
THUMB_W = 640

PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324",
]


def parse_yolo_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    out = []
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = (float(x) for x in parts[1:5])
        out.append((cls, cx, cy, w, h))
    return out


def parse_case(stem: str) -> str:
    parts = stem.split("_")
    return "_".join(parts[:2])


def make_thumb(args: tuple[Path, list, list[str], Path]) -> str:
    img_path, labels, class_names, out_path = args
    if out_path.exists():
        return out_path.name
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except OSError:
        font = ImageFont.load_default()
    with Image.open(img_path) as im:
        orig_w, orig_h = im.size
        scale = THUMB_W / orig_w
        thumb_h = int(orig_h * scale)
        im = im.convert("RGB").resize((THUMB_W, thumb_h), Image.LANCZOS)
        draw = ImageDraw.Draw(im)
        for cls, cx, cy, w, h in labels:
            x1 = (cx - w / 2) * THUMB_W
            y1 = (cy - h / 2) * thumb_h
            x2 = (cx + w / 2) * THUMB_W
            y2 = (cy + h / 2) * thumb_h
            color = PALETTE[cls % len(PALETTE)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label_text = class_names[cls] if cls < len(class_names) else str(cls)
            tx, ty = x1 + 2, max(0, y1 - 18)
            tb = draw.textbbox((tx, ty), label_text, font=font)
            draw.rectangle(tb, fill=color)
            draw.text((tx, ty), label_text, fill="black", font=font)
        im.save(out_path, "JPEG", quality=85)
    return out_path.name


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 0; padding: 16px; background: #111; color: #eee; }
h1 { margin: 0 0 16px; font-size: 18px; font-weight: 500; }
h2 { margin: 24px 0 8px; font-size: 14px; color: #aaa;
     border-bottom: 1px solid #2a2a2a; padding-bottom: 4px; font-weight: 500; }
.nav { margin-bottom: 12px; font-size: 13px; }
.nav a { color: #6af; text-decoration: none; margin-right: 16px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 6px; }
.cell { background: #1a1a1a; border-radius: 4px; overflow: hidden; }
.cell img { width: 100%; display: block; cursor: pointer; }
.cell .caption { padding: 5px 8px; font-size: 10.5px; font-family: 'SF Mono', monospace;
                 color: #999; line-height: 1.4; word-break: break-all; }
table { border-collapse: collapse; width: 100%; max-width: 720px; }
td, th { padding: 5px 12px; text-align: left; font-size: 13px; }
tr:hover { background: #1a1a1a; }
a { color: #6af; text-decoration: none; }
a:hover { text-decoration: underline; }
.count { color: #888; }
"""


def write_grid_html(path: Path, title: str, items: list[dict]) -> None:
    head = (
        f'<!DOCTYPE html><html><head><meta charset="utf-8">'
        f"<title>{escape(title)}</title><style>{CSS}</style></head><body>"
    )
    body = (
        f'<div class="nav"><a href="../index.html">&larr; index</a></div>'
        f"<h1>{escape(title)} &nbsp;<span class='count'>({len(items)} frames)</span></h1>"
        '<div class="grid">'
    )
    cells = []
    for it in items:
        cells.append(
            f'<div class="cell">'
            f'<a href="{escape(it["src"])}" target="_blank">'
            f'<img src="{escape(it["thumb"])}" loading="lazy"></a>'
            f'<div class="caption">{escape(it["caption"])}</div></div>'
        )
    path.write_text(head + body + "".join(cells) + "</div></body></html>")


def write_index_html(
    path: Path,
    class_stats: dict[str, int],
    case_stats: dict[str, int],
    total_frames: int,
    unlabeled_frames: int,
) -> None:
    head = (
        f'<!DOCTYPE html><html><head><meta charset="utf-8">'
        f"<title>GT Review</title><style>{CSS}</style></head><body>"
    )
    summary = (
        f"<h1>Ground Truth Review</h1>"
        f"<p style='color:#888;font-size:13px'>"
        f"{total_frames} total frames, {total_frames - unlabeled_frames} labeled, "
        f"{unlabeled_frames} unlabeled.</p>"
    )
    cls_rows = "".join(
        f'<tr><td><a href="class/{escape(c.replace(" ", "_").replace("/", "_"))}.html">'
        f"{escape(c)}</a></td><td class='count'>{n} frames</td></tr>"
        for c, n in sorted(class_stats.items(), key=lambda kv: -kv[1])
    )
    case_rows = "".join(
        f'<tr><td><a href="case/{escape(c)}.html">{escape(c)}</a></td>'
        f"<td class='count'>{n} frames</td></tr>"
        for c, n in sorted(case_stats.items())
    )
    path.write_text(
        head + summary
        + "<h2>By class</h2><table>" + cls_rows + "</table>"
        + "<h2>By case</h2><table>" + case_rows + "</table>"
        + "</body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-thumbs", action="store_true")
    args = parser.parse_args()

    yaml_data = yaml.safe_load(DATA_YAML.read_text())
    raw_names = yaml_data["names"]
    if isinstance(raw_names, dict):
        class_names = [raw_names[i] for i in sorted(raw_names)]
    else:
        class_names = list(raw_names)
    print(f"Classes: {class_names}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "class").mkdir(exist_ok=True)
    (OUT_DIR / "case").mkdir(exist_ok=True)

    images_link = OUT_DIR / "images"
    if images_link.exists() and not images_link.is_symlink():
        raise SystemExit(f"{images_link} exists and is not a symlink")
    if not images_link.exists():
        images_link.symlink_to(IMAGES_DIR)

    label_paths = sorted(LABELS_DIR.glob("*.txt"))
    print(f"Walking {len(label_paths)} label files...")

    thumb_jobs: list[tuple[Path, list, list[str], Path]] = []
    class_index: dict[str, list[dict]] = defaultdict(list)
    case_index: dict[str, list[dict]] = defaultdict(list)
    unlabeled = 0

    for lbl_path in label_paths:
        labels = parse_yolo_label(lbl_path)
        if not labels:
            unlabeled += 1
            continue
        stem = lbl_path.stem
        img_path = IMAGES_DIR / f"{stem}.png"
        if not img_path.exists():
            continue

        case = parse_case(stem)
        thumb_path = THUMBS_DIR / f"{stem}.jpg"
        thumb_jobs.append((img_path, labels, class_names, thumb_path))

        present = sorted({class_names[c] for c, *_ in labels if c < len(class_names)})
        item = {
            "thumb": f"../thumbs/{stem}.jpg",
            "src": f"../images/{stem}.png",
            "caption": f"{stem}\n{', '.join(present)}",
        }
        for cls in present:
            class_index[cls].append(item)
        case_index[case].append(item)

    if not args.no_thumbs:
        print(f"Generating {len(thumb_jobs)} thumbnails ({args.workers} workers)...")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(make_thumb, j) for j in thumb_jobs]
            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass

    print("Writing HTML...")
    for cls, items in class_index.items():
        slug = cls.replace(" ", "_").replace("/", "_")
        items.sort(key=lambda it: it["caption"])
        write_grid_html(OUT_DIR / "class" / f"{slug}.html", cls, items)

    for case, items in case_index.items():
        items.sort(key=lambda it: it["caption"])
        write_grid_html(OUT_DIR / "case" / f"{case}.html", case, items)

    write_index_html(
        OUT_DIR / "index.html",
        {c: len(v) for c, v in class_index.items()},
        {c: len(v) for c, v in case_index.items()},
        len(label_paths),
        unlabeled,
    )

    print(f"\nSite ready at {OUT_DIR}")
    print("Open: http://localhost:8080/  (after starting http.server)")


if __name__ == "__main__":
    main()
