from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("/home/jaeho/ent_cv/runs/detect/train12/weights/best.pt")

# Define path to file
source = "/home/jaeho/footage/20250908_01.mp4_20250924_022955.216.jpg"

# Run inference on the source
results = model(source)