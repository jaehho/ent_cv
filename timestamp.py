import os
import cv2
import pandas as pd
from pathlib import Path
import re
import time

video_path = Path("./data/20250911_01_5min.mp4")
label_dir = Path("runs/detect/predict2/labels")
output_dir = Path("timestamps"); output_dir.mkdir(exist_ok=True)
fps = 30


records = []
for fname in sorted(label_dir.glob("*.txt")):
    m = re.search(r"(\d+)$", fname.stem)
    frame_idx = int(m.group(1)) if m else 0
    timestamp_sec = frame_idx / fps
    hhmmss_ss = time.strftime("%H:%M:%S", time.gmtime(timestamp_sec)) + f".{int((timestamp_sec % 1)*100):02d}"
    with open(fname) as f:
        for line in f:
            parts = list(map(float, line.split()))
            if len(parts) == 6:
                cls, xc, yc, w, h, conf = parts
            elif len(parts) == 5:
                cls, xc, yc, w, h = parts
                conf = None
            else:
                continue
            records.append({
                "frame": frame_idx,
                "timestamp_hhmmss": hhmmss_ss,
                "class": int(cls),
                "confidence": conf,
                "bbox": [xc, yc, w, h]
            })

df = pd.DataFrame(records)
df.sort_values(by="frame", inplace=True)
output_csv = output_dir / video_path.with_suffix(".csv").name
df.to_csv(output_csv, index=False)
print(f"Saved: {output_csv}")
