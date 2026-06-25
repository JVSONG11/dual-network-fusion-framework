import json
from pycocotools.coco import COCO

# 加载COCO格式的标注文件
coco_gt = COCO('./dataset11.json')

# 检查检测结果文件格式
with open('./ssd11.json', 'r') as f:
    detection_results = json.load(f)

# 检查数据的整体结构
print(type(detection_results))  # 打印数据类型
print(detection_results[:5])    # 打印前5个检测结果

# 加载检测结果
coco_dt = coco_gt.loadRes('./ssd11.json')

# 继续执行COCO评估
from pycocotools.cocoeval import COCOeval

# 初始化COCOeval
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

# 过滤低置信度的检测结果
if isinstance(detection_results, list):
    high_conf_results = [det for det in detection_results if det['score'] > 0.5]
else:
    print("detection_results 的格式不正确。它不是一个列表。")


# 评估检测结果
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

