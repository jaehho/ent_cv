from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolo11n.pt")

# Train the model on your custom dataset
model.train(data="/home/jaeho/datasets/vid_1/data.yaml", epochs=100, imgsz=640)