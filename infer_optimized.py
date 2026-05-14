import csv
import torch
from pathlib import Path
from ultralytics import YOLO
import numpy as np


def infer_with_tta(model, img_path, imgsz=416, device='cuda'):
    """
    使用测试时增强（TTA）进行推理
    """
    img = cv2.imread(str(img_path))
    all_boxes = []
    
    # 原始图像
    results = model.predict(
        source=img,
        imgsz=imgsz,
        conf=0.001,
        iou=0.5,
        device=device,
        max_det=300,
        verbose=False
    )
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                all_boxes.append({
                    'xywhn': box.xywhn[0].cpu().numpy(),
                    'cls': int(box.cls[0].item()),
                    'conf': float(box.conf[0].item())
                })
    
    return all_boxes


def nms_boxes(boxes, iou_threshold=0.5):
    """
    非极大值抑制
    """
    if len(boxes) == 0:
        return []
    
    boxes = sorted(boxes, key=lambda x: x['conf'], reverse=True)
    selected = []
    
    while len(boxes) > 0:
        current = boxes.pop(0)
        selected.append(current)
        
        boxes = [b for b in boxes if compute_iou(current['xywhn'], b['xywhn']) < iou_threshold]
    
    return selected


def compute_iou(box1, box2):
    """
    计算两个边界框的IoU
    """
    x1 = box1[0] - box1[2] / 2
    y1 = box1[1] - box1[3] / 2
    x2 = box1[0] + box1[2] / 2
    y2 = box1[1] + box1[3] / 2
    
    x3 = box2[0] - box2[2] / 2
    y3 = box2[1] - box2[3] / 2
    x4 = box2[0] + box2[2] / 2
    y4 = box2[1] + box2[3] / 2
    
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def generate_submission():
    """
    优化版推理脚本
    """
    # 模型路径
    model_path = 'runs/detect/runs/detect/train/weights/best.pt'
    
    # 加载模型
    model = YOLO(model_path)
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 测试图片目录
    test_dir = Path('test/images')
    image_paths = sorted([p for p in test_dir.iterdir() 
                         if p.is_file() and p.suffix.lower() in ('.jpg', '.png')])
    
    print(f"Found {len(image_paths)} test images")
    print(f"First few images: {[p.name for p in image_paths[:3]]}")
    
    # 输出文件
    output_path = 'submission_optimized.csv'
    
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"],
        )
        writer.writeheader()
        
        # 推理参数
        imgsz_list = [416, 512, 640]  # 多尺度测试
        conf_threshold = 0.05
        nms_iou = 0.6
        
        for i, img_path in enumerate(image_paths):
            image_id = img_path.name
            
            if (i + 1) % 20 == 0:
                print(f"Processing {i + 1}/{len(image_paths)} images...")
            
            # 多尺度推理
            all_detections = []
            
            for imgsz in imgsz_list:
                results = model.predict(
                    source=str(img_path),
                    imgsz=imgsz,
                    conf=0.001,
                    iou=0.5,
                    device=device,
                    max_det=500,
                    verbose=False,
                    augment=True  # 开启增强推理
                )
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            all_detections.append({
                                'xywhn': box.xywhn[0].cpu().numpy(),
                                'cls': int(box.cls[0].item()),
                                'conf': float(box.conf[0].item())
                            })
            
            # 按类别分组进行NMS
            final_detections = []
            classes = set([d['cls'] for d in all_detections])
            
            for cls in classes:
                cls_boxes = [d for d in all_detections if d['cls'] == cls]
                cls_boxes = sorted(cls_boxes, key=lambda x: x['conf'], reverse=True)
                
                # 简单的加权融合
                if len(cls_boxes) > 0:
                    final_detections.extend(cls_boxes[:10])
            
            # 最终NMS
            final_detections = nms_boxes(final_detections, iou_threshold=nms_iou)
            
            # 过滤低置信度
            for det in final_detections:
                if det['conf'] >= conf_threshold:
                    x_center, y_center, width, height = det['xywhn']
                    
                    # 确保坐标在[0, 1]范围内
                    x_center = np.clip(x_center, 0, 1)
                    y_center = np.clip(y_center, 0, 1)
                    width = np.clip(width, 0, 1)
                    height = np.clip(height, 0, 1)
                    
                    writer.writerow({
                        "image_id": image_id,
                        "class_id": det['cls'],
                        "x_center": float(x_center),
                        "y_center": float(y_center),
                        "width": float(width),
                        "height": float(height),
                        "confidence": det['conf'],
                    })
    
    print(f"Optimized submission saved to {output_path}")
    print(f"Processing completed!")


if __name__ == "__main__":
    import cv2
    generate_submission()
