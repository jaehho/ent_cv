from ultralytics import YOLO

# Load the model
model = YOLO("./runs/detect/train12/weights/best.pt")

# Validate the model
metrics = model.val()
print(metrics.box.map)  # mAP50-95