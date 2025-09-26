from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("/home/jaeho/ent_cv/runs/detect/train/weights/best.pt")
mapping = {0: "Bovie", 1: "Frazier", 2: "Forceps", 3: "Microdebrider", 4: "Freer"}
assert len(mapping) == model.model.nc, "names count must match number of classes"
model.model.names = mapping

# Define path to file
source = "/home/jaeho/footage/frame-*.png"

# Run inference on the source
results = model(source, save=True, save_txt=True, save_conf=True, conf=0.8)