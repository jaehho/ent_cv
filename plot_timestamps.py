import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from pathlib import Path

# inputs
video_csv = Path("timestamps/20251016_02.csv")
names = {0: "Bovie", 1: "Frazier", 2: "Forceps", 3: "Microdebrider", 4: "Freer"}

# load
df = pd.read_csv(video_csv)
assert {"frame", "timestamp_hhmmss", "class"}.issubset(df.columns)

# keep last detection per (frame, class), then last per frame
df["ord"] = np.arange(len(df))
df = df.drop_duplicates(subset=["frame", "class"], keep="last")
df_one = (
    df.sort_values(["frame", "ord"])
      .groupby("frame", as_index=False)
      .tail(1)
      .sort_values("frame")
)

# class coding
classes = df_one["class"].astype(int).unique()
classes.sort()
cls_to_code = {c: i for i, c in enumerate(classes)}
cvals = df_one["class"].astype(int).map(cls_to_code).to_numpy()

# build 1×N pixel strip across full frame span
fmin, fmax = int(df_one["frame"].min()), int(df_one["frame"].max())
n = fmax - fmin + 1
strip = np.full(n, np.nan)
idx = (df_one["frame"].astype(int) - fmin).to_numpy()
strip[idx] = cvals  # class codes at detected frames

# figure
fig, ax = plt.subplots(figsize=(12, 1))

# colormap and normalization
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad(color="none", alpha=0.0)  # show gaps as transparent
norm = mcolors.Normalize(vmin=min(cls_to_code.values()), vmax=max(cls_to_code.values()))

# each frame renders as one pixel
ax.imshow(
    strip[np.newaxis, :],
    aspect="auto",
    interpolation="nearest",
    cmap=cmap,
    norm=norm,
    extent=[fmin - 0.5, fmax + 0.5, -0.5, 0.5],
)

# axes styling
ax.get_yaxis().set_visible(False)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("Video Time (HH:MM)")
ax.set_title(f"{video_csv.stem} Timeline")
# ax.set_title("Detection Timeline (1D)")

# compute ticks
t = pd.to_datetime(df_one["timestamp_hhmmss"], format="%H:%M:%S.%f", errors="coerce")
if t.notna().any():
    start, end = t.iloc[0], t.iloc[-1]
    tick_times = pd.date_range(start, end, freq="15min")
    # find nearest frame for each tick
    tick_frames = [
        df_one.loc[(t - tt).abs().idxmin(), "frame"] for tt in tick_times
    ]
    ax.set_xticks(tick_frames)
    ax.set_xticklabels(
        [tt.strftime("%H:%M") for tt in tick_times],
        rotation=45,
        ha="right"
    )

# legend via proxy handles
if names is not False:
    handles, labels = [], []
    for cls in classes:
        color = cmap(norm(cls_to_code[cls]))
        handles.append(
            Line2D([0], [0], marker="s", linestyle="", markersize=6,
                   markerfacecolor=color, markeredgecolor="none")
        )
        label = names.get(int(cls), str(int(cls))) if isinstance(names, dict) else str(int(cls))
        labels.append(label)
    ax.legend(
        handles, labels, title="Class",
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=True, framealpha=1, edgecolor="black"
    )

out_path = video_csv.parent / f"{video_csv.stem}_timeline.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2, facecolor="white")
print(f"saved: {out_path}")