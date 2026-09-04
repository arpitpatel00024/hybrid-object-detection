import torch
from ultralytics import YOLO

# Prevent excessive CPU thread overhead on small Render instances
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class YOLODetector:

    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def predict(self, image):
        return self.model(
            image,
            imgsz=320,
            device="cpu",
            verbose=False
        )