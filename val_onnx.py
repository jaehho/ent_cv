from ultralytics import YOLO

MODEL = "/home/jaeho/ent_cv/runs/detect/train12/weights/best.onnx"
DATA = "/home/jaeho/datasets/vid_1/data.yaml" 

model = YOLO(MODEL)
metrics = model.val(data=DATA, imgsz=640, device=0) 
print(metrics.box.map)
