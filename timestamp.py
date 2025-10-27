import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import re
import time

video_path = Path("./data/20251016_02")
label_dir = Path("runs/predict/20251016_02/labels")
mapping = {0: "Bovie", 1: "Frazier", 2: "Forceps", 3: "Microdebrider", 4: "Freer"}
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

if df.empty:
    output_csv = output_dir / video_path.with_suffix(".csv").name
    pd.DataFrame(columns=["second","timestamp_hhmmss","label","class","confidence","bbox","frame"]).to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")
    raise SystemExit

# one detection per second
df["second"] = (df["frame"] // fps).astype(int)
df["label"]  = df["class"].map(mapping)
df["conf_fill"] = df["confidence"].fillna(-1)
df.sort_values(["second","conf_fill","frame","class"], inplace=True)
df = df.groupby("second").tail(1).copy()
df["timestamp_hhmmss"] = df["second"].apply(lambda s: time.strftime("%H:%M:%S", time.gmtime(s)))
df.sort_values("second", inplace=True)

# find contiguous segments of the same class
df["change"] = (df["class"].ne(df["class"].shift()) | (df["second"].diff() != 1)).astype(int)
df["segment_id"] = df["change"].cumsum()

# compute segment lengths
seg_len = df.groupby("segment_id")["second"].transform("count")
df = df[seg_len >= 5].copy()

df = df[["second","timestamp_hhmmss","label","class","confidence","bbox","frame"]]

output_csv = output_dir / video_path.with_suffix(".csv").name
df.to_csv(output_csv, index=False)
print(f"Saved: {output_csv}")
