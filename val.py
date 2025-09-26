from ultralytics import YOLO

# Load the model
model = YOLO("./runs/detect/train/weights/best.pt")
mapping = {0: "Bovie", 1: "Frazier", 2: "Forceps", 3: "Microdebrider", 4: "Freer"}
assert len(mapping) == model.model.nc, "names count must match number of classes"
model.model.names = mapping

# Validate the model
metrics = model.val(plots=True)
print(metrics.box.map)  # mAP50-95
print(metrics.confusion_matrix.summary())
print(metrics.confusion_matrix.to_df())