import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 加载COCO ground truth 和预测结果
cocoGt = COCO('../dataset11.json')
cocoDt = cocoGt.loadRes('../ssd11.json')

# 初始化COCOeval对象
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

# 运行COCO评价
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

# 通过COCOeval对象获得PR曲线的详细信息
precision = cocoEval.eval['precision']  # 5维数组 (iou_thresh, recall, category, area, max_dets)
recall = np.linspace(0, 1, len(precision[0, :, 0, 0, 0]))

# 以某个类别绘制PR曲线，假设类别ID为1
plt.plot(recall, precision[0, :, 0, 0, 2], label='Class 1 PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for COCO Class 1')
plt.show()



from scipy.interpolate import make_interp_spline

# 提取某个类别（如类别ID=1）的 precision 和 recall
precision = cocoEval.eval['precision'][0, :, 0, 0, 2]
recall = np.linspace(0, 1, len(precision))

# 使用make_interp_spline进行平滑插值
recall_smooth = np.linspace(recall.min(), recall.max(), 300)
precision_smooth = make_interp_spline(recall, precision)(recall_smooth)

# 绘制平滑后的P-R曲线
plt.plot(recall_smooth, precision_smooth, label='Smoothed PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Smoothed Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()

