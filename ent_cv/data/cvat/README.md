# Interacting with CVAT

- `combine.py` - Combine multiple datasets into a single dataset for import into CVAT. This is useful for merging datasets that were annotated separately or for creating a larger dataset from multiple sources.
- `import.py` - Import a dataset in Ultralytics YOLO format into CVAT.
- `export.py` - Export annotations from CVAT in Ultralytics YOLO format for use in training models. Refer to `../modeling/prepare_dataset.py` for preparing the exported data for training (e.g., splitting into train/val, creating data.yaml).
- `upload.py` - Upload images and videos to CVAT without annotations. Usually after preprocessing raw data (e.g., splitting videos into frames.) but before annotation.
- `filter_unlabeled.py` - Analyze a dataset and create a new one containing only the unlabeled frames (frames whose label file is absent or empty). Useful since you can't filter by unlabeled in CVAT.