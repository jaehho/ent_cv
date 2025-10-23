from ultralytics import YOLO
import time

model = YOLO("/home/jaeho/ent_cv/runs/detect/train/weights/best.pt")
mapping = {0: "Bovie", 1: "Frazier", 2: "Forceps", 3: "Microdebrider", 4: "Freer"}
assert len(mapping) == model.model.nc
model.model.names = mapping

source = "./data/20250911_01.mp4"

results = model.predict(
    source=source,
    stream=True,
    save=False,           
    save_txt=True,
    save_conf=True,
    conf=0.7,
    verbose=True,
)

for i, r in enumerate(results):
    for b in r.boxes:
        cls = int(b.cls[0])
        conf = float(b.conf[0])
