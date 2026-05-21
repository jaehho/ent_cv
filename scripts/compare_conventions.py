"""Compare frame and annotation coverage between the two labeling conventions.

  Convention A (tool-only):    enclose metal/plastic instrument only; 10 classes.
  Convention B (grasp-coupled): include the surgeon's hand gripping the tool; 13 classes.

A frame is considered "in" a convention when it has at least one annotation under
that convention. Empty label files (no boxes) do not count as coverage — Convention
A's export writes empty .txt for background frames; Convention B's export omits them.
This script normalizes that asymmetry so the comparison reflects real annotation
coverage, not export artifacts.

Outputs both a frame-level and box-level summary plus a per-case breakdown.

Run: uv run python scripts/compare_conventions.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ent_cv.config import DATASETS_DIR

A_LABELS = DATASETS_DIR / "exports" / "tool-only" / "2026-05-08" / "labels" / "train"
B_LABELS = DATASETS_DIR / "exports" / "grasp-coupled" / "2026-05-08" / "labels" / "train"


def annotated_frames(d: Path) -> dict[str, int]:
    """Return {stem: box_count} for every label file with >=1 box."""
    out: dict[str, int] = {}
    for p in d.glob("*.txt"):
        n = sum(1 for line in p.read_text().splitlines() if line.strip())
        if n:
            out[p.stem] = n
    return out


def case_of(stem: str) -> str:
    return stem.rsplit("_f", 1)[0]


def main() -> None:
    a = annotated_frames(A_LABELS)
    b = annotated_frames(B_LABELS)
    a_files = {p.stem for p in A_LABELS.glob("*.txt")}
    b_files = {p.stem for p in B_LABELS.glob("*.txt")}
    a_keys, b_keys = set(a), set(b)
    both = a_keys & b_keys
    only_a = a_keys - b_keys
    only_b = b_keys - a_keys

    print("Frame counts (annotated = has >=1 box)")
    print(f"  Convention A: {len(a_keys):>5} annotated  "
          f"({len(a_files) - len(a_keys)} empty .txt, {len(a_files)} total files)")
    print(f"  Convention B: {len(b_keys):>5} annotated  "
          f"({len(b_files) - len(b_keys)} empty .txt, {len(b_files)} total files)")
    print(f"  Both:         {len(both):>5}")
    print(f"  Only A:       {len(only_a):>5}")
    print(f"  Only B:       {len(only_b):>5}\n")

    print("Box counts (total annotations)")
    print(f"  Convention A: {sum(a.values()):>6}")
    print(f"  Convention B: {sum(b.values()):>6}")
    print(f"  Δ (B − A):    {sum(b.values()) - sum(a.values()):>+6}\n")

    cases = sorted({case_of(s) for s in a_keys | b_keys})
    print(f"Per-case breakdown (frames | boxes)")
    print(f"{'case':<32} "
          f"{'A_fr':>5} {'B_fr':>5} {'both_fr':>7} {'oA_fr':>6} {'oB_fr':>6}  "
          f"{'A_box':>6} {'B_box':>6}")
    for c in cases:
        ca = {s: n for s, n in a.items() if case_of(s) == c}
        cb = {s: n for s, n in b.items() if case_of(s) == c}
        sa, sb = set(ca), set(cb)
        cboth = len(sa & sb)
        print(f"{c:<32} "
              f"{len(sa):>5} {len(sb):>5} {cboth:>7} "
              f"{len(sa) - cboth:>6} {len(sb) - cboth:>6}  "
              f"{sum(ca.values()):>6} {sum(cb.values()):>6}")


if __name__ == "__main__":
    main()
