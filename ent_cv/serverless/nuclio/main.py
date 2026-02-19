import json
import base64
import io
from PIL import Image

from ultralytics import YOLO


WEIGHTS_PATH = "/opt/nuclio/best.pt"
CONFIDENCE_THRESHOLD = 0.5


def init_context(context):
    context.logger.info("Init context...  0%")
    model = YOLO(WEIGHTS_PATH)
    context.user_data.model = model
    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run YOLOv26x ENT CV model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", CONFIDENCE_THRESHOLD))
    image = Image.open(buf).convert("RGB")

    results = context.user_data.model(image, conf=threshold)

    best_detection = None
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if best_detection is None or conf > float(best_detection["confidence"]):
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                best_detection = {
                    "confidence": str(conf),
                    "label": label,
                    "points": xyxy,
                    "type": "rectangle",
                }

    detections = [best_detection] if best_detection is not None else []

    return context.Response(
        body=json.dumps(detections),
        headers={},
        content_type="application/json",
        status_code=200,
    )
