# Data Pipeline

All data scripts are accessible via the unified CLI:

```bash
ent-cv data extract-frames  # Extract frames from surgical videos
ent-cv data tile            # Tile YOLO dataset into patches
ent-cv data download        # Download videos from SharePoint
ent-cv cvat <command>       # CVAT integration (see cvat/README.md)
```

## Typical workflow

1. Download videos from SharePoint:
    1. Get cookies using the `Get cookies.txt` browser extension and save to `cookies.txt`.
    2. Run `ent-cv data download` to scrape directories and download videos to `/mnt/data/ent_cv/raw/`.
2. Extract and process frames with `ent-cv data extract-frames`. Frames are saved to `/mnt/data/ent_cv/processed/`.
3. Upload frames to CVAT with `ent-cv cvat upload`.
4. After annotating in CVAT, export with `ent-cv cvat export`. Annotations are saved to `/mnt/data/ent_cv/datasets/`.
