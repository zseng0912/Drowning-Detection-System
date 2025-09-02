from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path, confidence):
        self.model = YOLO(model_path)
        self.classList = ["normal", "drowning"]
        self.confidence = confidence

    def detect(self, image):
        results = self.model.predict(image, conf=self.confidence)
        result = results[0]
        return self.make_detections(result)

    def make_detections(self, result):
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_number = int(box.cls[0])
            class_name = result.names[class_number]
            if class_name not in self.classList:
                continue
            conf = float(box.conf[0])
            detections.append((x1, y1, x2, y2, class_name, conf))
        return detections
