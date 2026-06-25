import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ========== 配置 ==========
model_path = "E:/0909/WBF/best.pt"  # WBF YOLOv8 模型路径
data_dir = "E:/0909/WBF/images"  # 原始图像文件夹
anno_path = "E:/0909/WBF/WBF.json"  # GT标注路径 (COCO格式)
perturb_strengths = [0, 5, 10, 15, 20, 25]  # 扰动强度列表（如高斯模糊 kernel size）
output_dir = "./wbf_robust_eval"

os.makedirs(output_dir, exist_ok=True)

# ========== 加载模型 ==========
model = YOLO(model_path)

# ========== 加载 COCO GT ==========
coco_gt = COCO(anno_path)


# ========== 应用扰动函数 ==========
def apply_perturbation(img, strength):
    if strength == 0:
        return img
    return cv2.GaussianBlur(img, (strength | 1, strength | 1), 0)


# ========== 推理并生成 COCO 预测格式 ==========
def run_inference_and_eval(strength):
    img_ids = coco_gt.getImgIds()
    results = []

    for img_id in tqdm(img_ids, desc=f"Inference with strength={strength}"):
        img_info = coco_gt.loadImgs([img_id])[0]
        img_path = os.path.join(data_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        img = apply_perturbation(img, strength)

        # BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = model.predict(source=img_rgb, verbose=False)[0]

        for box, score, cls in zip(pred.boxes.xyxy.cpu().numpy(),
                                   pred.boxes.conf.cpu().numpy(),
                                   pred.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = box
            results.append({
                "image_id": img_id,
                "category_id": int(cls) + 1,  # COCO 类别索引从1开始
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score)
            })

    pred_path = os.path.join(output_dir, f"predictions_strength_{strength}.json")
    with open(pred_path, 'w') as f:
        import json
        json.dump(results, f)

    # 评估
    coco_dt = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map_50 = coco_eval.stats[1]
    map_50_95 = coco_eval.stats[0]
    return map_50, map_50_95


# ========== 主流程 ==========
map_50_list = []
map_50_95_list = []

for strength in perturb_strengths:
    map50, map50_95 = run_inference_and_eval(strength)
    map_50_list.append(map50)
    map_50_95_list.append(map50_95)

# ========== 绘图 ==========
plt.figure(figsize=(8, 5))
plt.plot(perturb_strengths, map_50_list, marker='o', label="mAP@0.5")
plt.plot(perturb_strengths, map_50_95_list, marker='s', label="mAP@0.5:0.95")
plt.title("Robustness Evaluation of WBF Model")
plt.xlabel("Perturbation Strength (Gaussian blur kernel size)")
plt.ylabel("mAP")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "robustness_curve.png"))
plt.show()
