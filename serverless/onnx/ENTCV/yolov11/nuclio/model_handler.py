# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import cv2
import numpy as np
import onnxruntime as ort


class ModelHandler:
    def __init__(self, labels):
        self.model = None
        self.load_network(model="best.onnx")
        self.labels = labels

    def load_network(self, model):
        device = ort.get_device()
        cuda = True if device == 'GPU' else False
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            so = ort.SessionOptions()
            so.log_severity_level = 3

            self.model = ort.InferenceSession(model, providers=providers, sess_options=so)
            self.output_details = [i.name for i in self.model.get_outputs()]
            self.input_details = [i.name for i in self.model.get_inputs()]

            self.is_inititated = True
        except Exception as e:
            raise Exception(f"Cannot load model {model}: {e}")

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def _infer(self, inputs: np.ndarray):
        try:
            img = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
            image = img.copy()
            image, ratio, dwdh = self.letterbox(image, auto=False)
            image = image.transpose((2, 0, 1))
            image = np.expand_dims(image, 0)
            image = np.ascontiguousarray(image).astype(np.float32) / 255.0

            inp = {self.input_details[0]: image}
            outs = self.model.run(self.output_details, inp)

            # Prefer single-output models
            det = outs[0]
            det = np.squeeze(det)  # handle [1, N, 6] -> [N, 6]
            if det.ndim != 2 or det.size == 0:
                return None

            cols = det.shape[1]
            if cols == 6:
                # Ultralytics NMS: [x1, y1, x2, y2, score, class]
                boxes = det[:, 0:4]
                scores = det[:, 4]
                labels = det[:, 5].astype(np.int64)
            elif cols == 7:
                # YOLOv7 NMS: [image_id, x1, y1, x2, y2, class, score]
                boxes = det[:, 1:5]
                labels = det[:, 5].astype(np.int64)
                scores = det[:, 6]
            else:
                # Try multi-output naming fallback if exporter produced separate tensors
                out_map = {o.name: v for o, v in zip(self.model.get_outputs(), outs)}
                if {"boxes", "scores", "classes"}.issubset(out_map):
                    boxes = np.squeeze(out_map["boxes"])
                    scores = np.squeeze(out_map["scores"])
                    labels = np.squeeze(out_map["classes"]).astype(np.int64)
                else:
                    return None

            # De-letterbox to original image size
            boxes -= np.array(dwdh * 2)
            boxes /= ratio
            boxes = boxes.round().astype(np.int32)
            return [boxes, labels, scores]

        except Exception as e:
            print(e)
            return None


    def infer(self, image, threshold):
        image = np.array(image)[:, :, ::-1].copy()
        h, w, _ = image.shape
        detections = self._infer(image)

        results = []
        if detections:
            boxes, labels, scores = detections
            for lbl, score, box in zip(labels, scores, boxes):
                if float(score) >= float(threshold):
                    x1 = max(int(box[0]), 0); y1 = max(int(box[1]), 0)
                    x2 = min(int(box[2]), w); y2 = min(int(box[3]), h)
                    results.append({
                        "confidence": str(float(score)),
                        "label": self.labels.get(int(lbl), "unknown"),
                        "points": [x1, y1, x2, y2],
                        "type": "rectangle",
                    })
        return results
