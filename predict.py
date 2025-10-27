import os
from ultralytics import YOLO

model = YOLO("/home/jaeho/ent_cv/runs/detect/train/weights/best.pt")
mapping = {0: "Bovie", 1: "Frazier", 2: "Forceps", 3: "Microdebrider", 4: "Freer"}
assert len(mapping) == model.model.nc
model.model.names = mapping

source = "data/20251016_02"

# Create a folder name based on the source file
folder_name = os.path.splitext(os.path.basename(source))[0]
save_dir = os.path.join("runs", "predict", folder_name)

results = model.predict(
    source=source,
    stream=True,
    save=True,              # save images/videos with detections
    project="runs/predict", # parent directory
    name=folder_name,       # folder named after source
    save_txt=True,
    save_conf=True,
    conf=0.7,
    verbose=True,
)

for i, r in enumerate(results):
    for b in r.boxes:
        cls = int(b.cls[0])
        conf = float(b.conf[0])
