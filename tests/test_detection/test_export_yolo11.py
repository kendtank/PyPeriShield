import ultralytics
from ultralytics import YOLO
model = YOLO("/home/lyh/work/depoly/PyPeriShield-feature/weights/yolo11n.pt")

model.export(
        format='engine',
        device='cuda',
        # half=half
    )