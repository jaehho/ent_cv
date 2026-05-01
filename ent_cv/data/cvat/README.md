# CVAT Integration

All scripts are accessible via `ent-cv cvat <command>`:

```bash
ent-cv cvat export              # Export annotations from CVAT as YOLO dataset
ent-cv cvat export-manual       # Export manual annotations from a CVAT job
ent-cv cvat combine             # Combine multiple YOLO datasets
ent-cv cvat import              # Import a YOLO dataset into CVAT
ent-ev cvat upload              # Upload images/videos to CVAT (before annotation)
ent-cv cvat upload-predictions  # Upload raw videos + YOLO prediction annotations
```

- **export** — Export CVAT annotations in Ultralytics YOLO format. Outputs to `datasets/exports/<task_name>/`.
- **export-manual** — Export only hand-drawn annotations from a CVAT job (filtering by `source=manual`). Useful when a task has both auto-imported predictions and manual corrections. Outputs to `datasets/exports/<task_name>_<source>/`.
- **combine** — Merge multiple exports into one training dataset. Handles class label remapping across datasets with interactive resolution. Outputs to `datasets/dataset/`.
- **import** — Import a YOLO-format dataset into a CVAT task.
- **upload** — Upload images/videos to CVAT without annotations, typically after frame extraction.
- **upload-predictions** — Register raw videos in CVAT and import YOLO predictions as annotations for review.

## Dataset Layout

```
/mnt/data/ent_cv/datasets/
├── exports/          # Immutable CVAT exports, one per batch
│   ├── batch1/
│   └── batch2/
└── dataset/          # Active training set, rebuilt from exports via `combine`
```

## Workflow

1. **Label in CVAT** — Upload frames/videos, annotate (or review model pre-annotations)
2. **Export** — `ent-cv cvat export` or `ent-cv cvat export-manual` → `exports/<name>/`
3. **Combine** — `ent-cv cvat combine` → `datasets/dataset/`
4. **Split** — `ent-cv prepare-dataset datasets/dataset/` → adds train/val split
5. **Train** — `ent-cv train --data datasets/dataset/data_with_val.yaml`
