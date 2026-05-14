from ultralytics import YOLO 
model = YOLO('yolov8n.pt') 
results = model.train(data='data.yaml', epochs=50, imgsz=416, batch=4, device='cpu', workers=4) 
