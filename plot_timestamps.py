import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

video_csv = Path("timestamps/20250911_01_5min.csv")
# names = {0: "Bovie", 1: "Frazier", 2: "Forceps", 3: "Microdebrider", 4: "Freer"}
names = None  # set dict above if you want a legend with names

df = pd.read_csv(video_csv)
assert {"frame", "class", "timestamp_sec"}.issubset(df.columns)

# preserve original order => “last detected” wins
df["ord"] = np.arange(len(df))
df = df.drop_duplicates(subset=["frame", "class"], keep="last")
df_one = df.sort_values(["frame", "ord"]).groupby("frame", as_index=False).tail(1).sort_values("frame")

# colors by class (let matplotlib assign)
classes = np.sort(df_one["class"].astype(int).unique())
cls_to_code = {c:i for i, c in enumerate(classes)}
cvals = df_one["class"].astype(int).map(cls_to_code).to_numpy()

# plotting
fig, ax = plt.subplots(figsize=(12, 2.2))
ax.scatter(df_one["frame"].to_numpy(), np.zeros(len(df_one)), c=cvals, s=12)
ax.get_yaxis().set_visible(False)
ax.set_xlabel("Time (HH:MM:SS.SS)")
ax.set_title("Detections timeline")

# build ticks in frame space, label by formatted video time
frames = df_one["frame"].to_numpy()
secs = df_one["timestamp_sec"].to_numpy()
fmin, fmax = int(frames.min()), int(frames.max())
nticks = min(12, fmax - fmin + 1)
tick_pos = np.linspace(fmin, fmax, nticks, dtype=int)

# interpolate seconds for any frames without a direct row
sec_at = np.interp(tick_pos, frames, secs)

def fmt_hhmmss_ss(x):
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    s = x % 60
    return f"{h:02d}:{m:02d}:{s:06.2f}"

tick_lbl = [fmt_hhmmss_ss(x) for x in sec_at]
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lbl, rotation=45, ha="right")

# optional legend
if names:
    handles, labels = [], []
    for cls in classes:
        if not (df_one["class"].astype(int) == cls).any():
            continue
        h = ax.scatter([], [], c=[cls_to_code[cls]], s=30)
        handles.append(h)
        labels.append(names.get(int(cls), str(int(cls))))
    ax.legend(handles, labels, title="Class", loc="upper right", frameon=False)

plt.tight_layout()
plt.savefig(video_csv.with_suffix(".png"), dpi=150)
