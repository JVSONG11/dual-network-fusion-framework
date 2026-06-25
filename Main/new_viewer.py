import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import numpy as np
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ensemble_boxes import weighted_boxes_fusion
import cv2
from PIL import Image, ImageTk
from multiprocessing import Pool, Process, cpu_count, Manager
from ensemble_boxes import *


# Your existing functions (get_coco_annotations_data, get_coco_score, etc.) go here
# Make sure to modify `ensemble_predictions` and `ensemble` to support the GUI flow

def get_coco_score(csv_path, images):
    # Adjusted code to handle the images parameter passed from GUI
    s = pd.read_csv(csv_path, dtype={'img_id': np.str_, 'label': np.str_})

    out = np.zeros((len(s), 7), dtype=np.float64)
    out[:, 0] = s['img_id']
    ids = s['img_id'].astype(np.int32).values  # ids在后续未使用 这可能是个问题
    # 坐标看起来正确读取了
    x1 = s['x1'].values
    x2 = s['x2'].values
    y1 = s['y1'].values
    y2 = s['y2'].values
    for i in range(len(s)):
        width = images[ids[i]]['width']
        height = images[ids[i]]['height']
        out[i, 1] = x1[i] * width
        out[i, 2] = y1[i] * height
        out[i, 3] = (x2[i] - x1[i]) * width
        out[i, 4] = (y2[i] - y1[i]) * height
    out[:, 5] = s['score'].values
    out[:, 6] = s['label'].values

    filename = './WBF.json'
    coco_gt = COCO(filename)
    detections = out
    image_ids = list(set(detections[:, 0]))
    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_metrics = coco_eval.stats
    return coco_metrics, detections


def plot_image_with_boxes(image_path, boxes, scores, labels, canvas):
    # 加载图像
    image = Image.open(image_path)
    image = ImageTk.PhotoImage(image)

    # 在画布上绘制图像
    canvas.create_image(0, 0, anchor=tk.NW, image=image)

    # 绘制目标框
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        score = scores[i]
        label = labels[i]

        # 绘制矩形框
        canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)

        # 显示置信度
        canvas.create_text(x1, y1, text=f"{label} {score:.2f}", fill="white", font=('Helvetica', 12))

    # 保持对图像的引用
    canvas.image = image


def on_process_button_click(in_dir_entry, confidence_entry, result_label, canvas):
    try:
        # 解析输入的坐标和置信度
        in_dir = in_dir_entry.get().split(',')
        in_dir = list(map(float, in_dir))  # 转为浮点数 [x1, x2, y1, y2]
        confidence = float(confidence_entry.get())  # 转为浮动数

        # You can replace the ensemble predictions here to suit your logic
        # Example: assume this function returns the merged predictions
        predictions = ensemble_predictions(['path/to/your/predictions.csv'], [confidence], params={})

        # For simplicity, let's assume we directly have the boxes, scores, and labels
        boxes = predictions['boxes']
        scores = predictions['scores']
        labels = predictions['labels']

        # Update the result label
        result_label.config(text=f"Processed {len(boxes)} boxes.")

        # 显示图像和目标框
        image_path = './path_to_image.jpg'  # Replace with the actual image path
        plot_image_with_boxes(image_path, boxes, scores, labels, canvas)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


# Tkinter UI setup
def create_gui():
    root = tk.Tk()
    root.title("ImageSet Viewer")

    # 假设在 GUI 初始化部分添加这个输入框：
    image_path_label = tk.Label(root, text="Enter Image Path:")
    image_path_label.pack()

    image_path_entry = tk.Entry(root)
    image_path_entry.pack()

    # Input section for in_dir and confidence
    input_frame = tk.Frame(root)
    input_frame.pack(padx=10, pady=10)

    tk.Label(input_frame, text="Enter Box:").grid(row=0, column=0, padx=5, pady=5)  # (x1,x2,y1,y2)
    in_dir_entry = tk.Entry(input_frame)
    in_dir_entry.grid(row=0, column=1, padx=5, pady=5)

    tk.Label(input_frame, text="Enter Confidence:").grid(row=1, column=0, padx=5, pady=5)
    confidence_entry = tk.Entry(input_frame)
    confidence_entry.grid(row=1, column=1, padx=5, pady=5)

    # Result section to display the processed boxes
    result_label = tk.Label(root, text="Results will be shown here.")
    result_label.pack(pady=10)

    # Canvas to display image with bounding boxes
    canvas_frame = tk.Frame(root)
    canvas_frame.pack(padx=10, pady=10)

    canvas = tk.Canvas(canvas_frame, width=800, height=600)
    canvas.pack()

    # Process button
    process_button = tk.Button(root, text="Process",
                               command=lambda: on_process_button_click(in_dir_entry, confidence_entry, result_label,
                                                                       canvas))
    process_button.pack(pady=20)

    root.mainloop()


def process_single_id(id, res_boxes, weights, params):
    run_type = params['run_type']
    verbose = params['verbose']

    # print('Go for ID: {}'.format(id))
    boxes_list = []
    scores_list = []
    labels_list = []
    labels_to_use_forward = dict()
    labels_to_use_backward = dict()

    for i in range(len(res_boxes[id])):
        boxes = []
        scores = []
        labels = []

        dt = res_boxes[id][i]

        for j in range(0, len(dt)):
            lbl = dt[j][5]
            scr = float(dt[j][4])
            box_x1 = float(dt[j][0])
            box_y1 = float(dt[j][1])
            box_x2 = float(dt[j][2])
            box_y2 = float(dt[j][3])

            if box_x1 >= box_x2:
                if verbose:
                    print('Problem with box x1 and x2: {}. Skip it'.format(dt[j]))
                continue
            if box_y1 >= box_y2:
                if verbose:
                    print('Problem with box y1 and y2: {}. Skip it'.format(dt[j]))
                continue
            if scr <= 0:
                if verbose:
                    print('Problem with box score: {}. Skip it'.format(dt[j]))
                continue

            """
            注释掉了上述代码，因为在调试中发现，所有的数据坐标可能都是乱的
            对于上面注释的代码，因为x1>x2是一种错误的情况，故box_x1和box_x2以及y相关数据直接被continue（line 94）了
            没有执行下面的代码（line 110），这会导致boxes/scores等为空，请调整csv里面的格式；强烈建议取消掉我上面注释的代码，这样能保证你数据格式一定是正确的

            还有一个疑问就是那个基准的标签文件 尝试各种办法 都没办法生成，然后我就想用之前的类似它的一个文件就是这个2007_val 但我不确定这个文件能不能读出来
            """

            boxes.append([box_x1, box_y1, box_x2, box_y2])
            scores.append(scr)
            if lbl not in labels_to_use_forward:
                cur_point = len(labels_to_use_forward)
                labels_to_use_forward[lbl] = cur_point
                labels_to_use_backward[cur_point] = lbl
            labels.append(labels_to_use_forward[lbl])

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    # Empty predictions for all models
    if len(boxes_list) == 0:
        return np.array([]), np.array([]), np.array([])

    if run_type == 'wbf':
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                                                                           # list len=2;
                                                                           weights=weights,
                                                                           iou_thr=params['intersection_thr'],
                                                                           skip_box_thr=params['skip_box_thr'],
                                                                           conf_type=params['conf_type'])
    elif run_type == 'wbf_exp':
        # ipdb.set_trace()
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion_experimental(boxes_list, scores_list,
                                                                                        labels_list,
                                                                                        weights=weights, iou_thr=params[
                'intersection_thr'],
                                                                                        skip_box_thr=params[
                                                                                            'skip_box_thr'],
                                                                                        conf_type=params['conf_type'])

    # print(len(boxes_list), len(merged_boxes))
    if 'limit_boxes' in params:
        limit_boxes = params['limit_boxes']
        if len(merged_boxes) > limit_boxes:
            merged_boxes = merged_boxes[:limit_boxes]
            merged_scores = merged_scores[:limit_boxes]
            merged_labels = merged_labels[:limit_boxes]

    # Rename labels back
    merged_labels_string = []
    for m in merged_labels:
        merged_labels_string.append(labels_to_use_backward[m])
    merged_labels = np.array(merged_labels_string, dtype=np.str_)

    # Create IDs array
    ids_list = [id] * len(merged_labels)

    return merged_boxes.copy(), merged_scores.copy(), merged_labels.copy(), ids_list.copy()


def process_part_of_data(proc_number, return_dict, ids_to_use, res_boxes, weights, params):
    print('Start process: {} IDs to proc: {}'.format(proc_number, len(ids_to_use)))
    # ipdb.set_trace()
    result = []
    for id in ids_to_use:
        merged_boxes, merged_scores, merged_labels, ids_list = process_single_id(id, res_boxes, weights, params)
        # print(merged_boxes.shape, merged_scores.shape, merged_labels.shape, len(ids_list))
        result.append((merged_boxes, merged_scores, merged_labels, ids_list))
    return_dict[proc_number] = result.copy()


def ensemble_predictions(pred_filenames, weights, params):
    verbose = False
    if 'verbose' in params:
        verbose = params['verbose']

    start_time = time.time()
    procs_to_use = max(cpu_count() // 2, 1)
    # procs_to_use = 1
    # procs_to_use = 6
    print('Use processes: {}'.format(procs_to_use))
    weights = np.array(weights)

    res_boxes = dict()
    ref_ids = None
    for j in range(len(pred_filenames)):
        if weights[j] == 0:
            continue
        print('Read {}...'.format(pred_filenames[j]))
        s = pd.read_csv(pred_filenames[j], dtype={'img_id': np.str_, 'label': np.str_})
        s.sort_values('img_id', inplace=True)
        s.reset_index(drop=True, inplace=True)
        ids = s['img_id'].values
        unique_ids = sorted(s['img_id'].unique())
        # import ipdb
        # ipdb.set_trace()
        # ref_ids=None
        # ipdb.set_trace()
        if ref_ids is None:
            ref_ids = tuple(unique_ids)
        else:
            if ref_ids != tuple(unique_ids):
                print('Different IDs in ensembled CSVs! {} != {}'.format(len(ref_ids), len(unique_ids)))
                s = s[s['img_id'].isin(ref_ids)]
                s.sort_values('img_id', inplace=True)
                s.reset_index(drop=True, inplace=True)
                ids = s['img_id'].values
        preds = s[['x1', 'y1', 'x2', 'y2', 'score', 'label']].values
        single_res = dict()
        for i in range(len(ids)):
            id = ids[i]
            if id not in single_res:
                single_res[id] = []
            single_res[id].append(preds[i])
        for el in single_res:
            if el not in res_boxes:
                res_boxes[el] = []
            res_boxes[el].append(single_res[el])

    # Reduce weights if needed
    weights = weights[weights != 0]

    ids_to_use = sorted(list(res_boxes.keys()))
    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    # import ipdb
    # ipdb.set_trace()
    # procs_to_use=1

    for i in range(procs_to_use):
        start = i * len(ids_to_use) // procs_to_use
        end = (i+1) * len(ids_to_use) // procs_to_use
        if i == procs_to_use - 1:
            end = len(ids_to_use)
        # 多线程执行任务有问题
        # ipdb.set_trace()
        p = Process(target=process_part_of_data, args=(i, return_dict, ids_to_use[start:end], res_boxes, weights, params)) # i=0;
        jobs.append(p)
        p.start()

    for i in range(len(jobs)):
        jobs[i].join()

    results = []
    for i in range(len(jobs)):
        results += return_dict[i]

    # process_part_of_data(0, return_dict, ids_to_use, res_boxes, weights, params)
    # ipdb.set_trace()
    # p = Pool(processes=procs_to_use)
    # results = p.starmap(process_single_id, zip(ids_to_use, repeat(weights), repeat(params)))

    all_ids = []
    all_boxes = []
    all_scores = []
    all_labels = []
    for boxes, scores, labels, ids_list in results:
        if boxes is None:
            continue
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        all_ids.append(ids_list)

    all_ids = np.concatenate(all_ids)
    all_boxes = np.concatenate(all_boxes)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    if verbose:
        print(all_ids.shape, all_boxes.shape, all_scores.shape, all_labels.shape)

    res = pd.DataFrame(all_ids, columns=['img_id'])
    res['label'] = all_labels
    res['score'] = all_scores
    res['x1'] = all_boxes[:, 0]
    res['x2'] = all_boxes[:, 2]
    res['y1'] = all_boxes[:, 1]
    res['y2'] = all_boxes[:, 3]
    print('Run time: {:.2f}'.format(time.time() - start_time))
    return res


# Start the Tkinter application
if __name__ == '__main__':
    create_gui()
