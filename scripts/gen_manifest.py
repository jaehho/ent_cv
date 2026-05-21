"""Write manifest.yaml for an existing export directory.

For new exports, prefer `uv run ent-cv cvat export` — it writes the manifest
automatically. This script is for backfilling or repairing manifests.

Run: uv run python scripts/gen_manifest.py <export_dir> --convention grasp-coupled
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ent_cv.data.manifest import DEFAULT_CONVENTION, VALID_CONVENTIONS, write_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("--convention", default=DEFAULT_CONVENTION, choices=VALID_CONVENTIONS)
    ap.add_argument("--cvat-task-ids", nargs="*", type=int, default=[])
    ap.add_argument("--cvat-project-id", type=int, default=None)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    out = write_manifest(
        args.export_dir,
        convention=args.convention,
        cvat_project_id=args.cvat_project_id,
        cvat_task_ids=args.cvat_task_ids,
        notes=args.notes,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
