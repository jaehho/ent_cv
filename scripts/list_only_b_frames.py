"""Build a focused HTML showing the frames annotated only under Convention B.

These are frames where Convention A's label file is empty (treated as background)
but Convention B has at least one box — i.e. the catches from the empty-hand
re-review pass. Reuses the side-by-side thumbnails already rendered by
scripts/render_convention_pairs.py.

Run: uv run python scripts/list_only_b_frames.py
"""

from __future__ import annotations

from collections import Counter
from html import escape
from pathlib import Path

from ent_cv.config import DATASETS_DIR

REPO = Path(__file__).resolve().parents[1]
A_LABELS = DATASETS_DIR / "exports" / "tool-only" / "2026-05-08" / "labels" / "train"
B_LABELS = DATASETS_DIR / "exports" / "grasp-coupled" / "2026-05-08" / "labels" / "train"
THUMBS = REPO / "reports" / "compare_conventions" / "thumbs"
OUT = REPO / "reports" / "compare_conventions" / "only_b.html"

CLASS_NAMES = [
    "Forceps", "Non-forcep instruments", "Suction", "Navigation probe",
    "Navigation suction", "Energy-based hemostasis without suction",
    "Saline-enhanced energy delivery", "Suction bovie", "Microdebrider",
    "Drill", "Empty Hand", "Not Sure", "Patient",
]


def annotated_stems(d: Path) -> set[str]:
    out = set()
    for p in d.glob("*.txt"):
        if any(line.strip() for line in p.read_text().splitlines()):
            out.add(p.stem)
    return out


def b_classes(stem: str) -> list[int]:
    p = B_LABELS / f"{stem}.txt"
    return [int(line.split()[0]) for line in p.read_text().splitlines() if line.strip()]


def case_of(stem: str) -> str:
    return stem.rsplit("_f", 1)[0]


CSS = """
body { font-family: -apple-system, sans-serif; margin: 0; padding: 24px;
       background: #111; color: #eee; max-width: 1300px; }
h1 { font-size: 18px; margin: 0 0 4px; font-weight: 500; }
.sub { color: #888; font-size: 13px; margin: 0 0 20px; max-width: 90ch; line-height: 1.5; }
.case-header { color: #888; font-size: 13px; margin: 24px 0 6px;
               border-bottom: 1px solid #2a2a2a; padding-bottom: 4px; }
.row { background: #1a1a1a; border-radius: 6px; padding: 8px; margin-bottom: 12px; }
.row img { width: 100%; display: block; border-radius: 4px; }
.cap { font-family: 'SF Mono', monospace; font-size: 11.5px; color: #bbb;
       padding: 6px 4px 0; line-height: 1.5; }
.classpill { display: inline-block; padding: 1px 6px; margin: 0 4px 0 0;
             border-radius: 3px; background: #50e070; color: #111;
             font-weight: 600; font-size: 10.5px; }
.summary { background: #1a1a1a; border-radius: 6px; padding: 12px 16px;
           margin-bottom: 20px; font-size: 13px; line-height: 1.7; }
.summary b { color: #fff; }
"""


def main() -> None:
    a = annotated_stems(A_LABELS)
    b = annotated_stems(B_LABELS)
    only_b = sorted(b - a)

    # Class frequency among only-B
    cls_count: Counter[int] = Counter()
    for s in only_b:
        for c in b_classes(s):
            cls_count[c] += 1

    rows: list[str] = []
    last_case = None
    for stem in only_b:
        case = case_of(stem)
        if case != last_case:
            rows.append(f"<div class='case-header'>{escape(case)}</div>")
            last_case = case
        classes = b_classes(stem)
        pills = "".join(
            f"<span class='classpill'>{escape(CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'cls{c}')}</span>"
            for c in classes
        )
        rows.append(
            f"<div class='row'>"
            f"<img loading='lazy' src='thumbs/{stem}.jpg'>"
            f"<div class='cap'><b>{escape(stem)}</b> &middot; "
            f"{len(classes)} new B box(es): {pills}</div></div>"
        )

    cls_summary = " &middot; ".join(
        f"{CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'cls{c}'}: <b>{n}</b>"
        for c, n in cls_count.most_common()
    )
    intro = (
        "<div class='summary'>"
        f"<b>{len(only_b)}</b> frames annotated only under Convention B (grasp-coupled). "
        "Convention A's label file for each was empty, then the empty-hand re-review "
        f"pass added boxes. Total new boxes: <b>{sum(cls_count.values())}</b>.<br>"
        f"<span style='color:#888'>By class — {cls_summary}</span>"
        "</div>"
    )

    html = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Convention B only — newly annotated frames</title>",
        f"<style>{CSS}</style></head><body>",
        "<h1>Frames annotated only under Convention B</h1>",
        "<p class='sub'>Each row shows the same frame: A panel is dimmed/empty "
        "(no annotations under tool-only), B panel shows the boxes added by the "
        "empty-hand re-review pass.</p>",
        intro,
        *rows,
        "</body></html>",
    ]
    OUT.write_text("\n".join(html))
    print(f"Wrote {OUT} ({len(only_b)} frames)")


if __name__ == "__main__":
    main()
