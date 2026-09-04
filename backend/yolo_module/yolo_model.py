from ultralytics import YOLO


class YOLODetector:

    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def predict(self, image):
        return self.model(
            image,
            imgsz=640,
            device="cpu",
            verbose=False
        )