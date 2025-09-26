from ultralytics import YOLO

# Load a model
model = YOLO("/home/jaeho/ent_cv/runs/detect/train12/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")