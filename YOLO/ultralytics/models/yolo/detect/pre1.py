from ultralytics import YOLO

model = YOLO("E:/yolo/yolov8/ultralytics/models/yolo/detect/runs/detect/train28/weights/best.pt")
results = model.predict(source='E:/yolo/yolov8/ultralytics/assets', save=True, save_conf=True, save_txt=True)