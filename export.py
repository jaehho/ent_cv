from ultralytics import YOLO

model = YOLO("/home/jaeho/ent_cv/runs/detect/train12/weights/best.pt")

model.export(format="onnx", imgsz=640, opset=17, dynamic=False, simplify=True, nms=True)
