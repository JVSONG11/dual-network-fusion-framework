import numpy as np
import pandas as pd
import json
import time
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from multiprocessing import Process, cpu_count, Manager
from ensemble_boxes.ciou_nms_module import ciou_nms  # 确保你正确实现了 ciou_nms_module

def get_coco_annotations_data():
    with open('./WBF.json') as f:
        data = json.load(f)
    return {str(img['id']).zfill(5): img for img in data['images']}  # 标准化 ID 为 '00016' 形式

def get_coco_score(csv_path):
    images = get_coco_annotations_data()
    s = pd.read_csv(csv_path, dtype={'img_id': str, 'label': str})
    out = np.zeros((len(s), 7), dtype=np.float64)
    valid_rows = []
    for i in range(len(s)):
        img_id = s.iloc[i]['img_id']
        if img_id not in images:
            print(f"Warning: image id {img_id} not found in annotation. Skipping.")
            continue
        width = images[img_id]['width']
        height = images[img_id]['height']
        valid_rows.append(i)
        out[i, 0] = float(img_id)
        out[i, 1] = s.iloc[i]['x1'] * width
        out[i, 2] = s.iloc[i]['y1'] * height
        out[i, 3] = (s.iloc[i]['x2'] - s.iloc[i]['x1']) * width
        out[i, 4] = (s.iloc[i]['y2'] - s.iloc[i]['y1']) * height
    out = out[valid_rows]
    out[:, 5] = s.iloc[valid_rows]['score'].values
    out[:, 6] = s.iloc[valid_rows]['label'].astype(float).values
    coco_gt = COCO('./WBF.json')
    try:
        coco_dt = coco_gt.loadRes(out)
    except Exception as e:
        print("Error loading results:", e)
        return None, None
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = list(set(out[:, 0]))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats, out

def process_single_id(id, res_boxes, weights, params):
    boxes_list, scores_list, labels_list = [], [], []
    labels_to_use_forward, labels_to_use_backward = {}, {}
    for model_preds in res_boxes[id]:
        boxes, scores, labels = [], [], []
        for dt in model_preds:
            box_x1, box_y1, box_x2, box_y2, scr, lbl = list(map(float, dt[:5])) + [dt[5]]
            if box_x1 >= box_x2 or box_y1 >= box_y2 or scr <= 0:
                continue
            boxes.append([box_x1, box_y1, box_x2, box_y2])
            scores.append(scr)
            if lbl not in labels_to_use_forward:
                idx = len(labels_to_use_forward)
                labels_to_use_forward[lbl] = idx
                labels_to_use_backward[idx] = lbl
            labels.append(labels_to_use_forward[lbl])
        boxes_list.append(np.array(boxes, dtype=np.float32))
        scores_list.append(np.array(scores, dtype=np.float32))
        labels_list.append(np.array(labels, dtype=np.int32))
    if not boxes_list:
        return np.array([]), np.array([]), np.array([]), []
    merged_boxes, merged_scores, merged_labels = ciou_nms(
        boxes_list, scores_list, labels_list, weights=weights, iou_thr=params['iou_thr']
    )
    merged_labels = [labels_to_use_backward[m] for m in merged_labels]
    return merged_boxes, merged_scores, np.array(merged_labels, dtype=str), [id] * len(merged_labels)

def process_part_of_data(proc_number, return_dict, ids_to_use, res_boxes, weights, params):
    result = []
    for id in ids_to_use:
        merged_boxes, merged_scores, merged_labels, ids_list = process_single_id(id, res_boxes, weights, params)
        result.append((merged_boxes, merged_scores, merged_labels, ids_list))
    return_dict[proc_number] = result

def ensemble_predictions(pred_filenames, weights, params):
    start_time = time.time()
    procs_to_use = max(cpu_count() // 2, 1)
    weights = np.array(weights)
    res_boxes = {}
    ref_ids = None
    for j, filename in enumerate(pred_filenames):
        if weights[j] == 0:
            continue
        s = pd.read_csv(filename, dtype={'img_id': str, 'label': str})
        s.sort_values('img_id', inplace=True)
        s = s.reset_index(drop=True)
        ids = s['img_id'].values
        preds = s[['x1', 'y1', 'x2', 'y2', 'score', 'label']].values
        single_res = {}
        for i in range(len(s)):
            img_id = ids[i]
            single_res.setdefault(img_id, []).append(preds[i])
        for img_id in single_res:
            res_boxes.setdefault(img_id, []).append(single_res[img_id])
    weights = weights[weights != 0]
    ids_to_use = sorted(res_boxes.keys())
    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(procs_to_use):
        start = i * len(ids_to_use) // procs_to_use
        end = (i + 1) * len(ids_to_use) // procs_to_use
        p = Process(target=process_part_of_data,
                    args=(i, return_dict, ids_to_use[start:end], res_boxes, weights, params))
        jobs.append(p)
        p.start()
    for job in jobs:
        job.join()
    results = []
    for i in range(len(jobs)):
        results.extend(return_dict[i])
    all_ids, all_boxes, all_scores, all_labels = [], [], [], []
    for boxes, scores, labels, ids_list in results:
        if len(boxes) == 0:
            continue
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        all_ids.append(ids_list)
    all_ids = np.concatenate(all_ids)
    all_boxes = np.concatenate(all_boxes)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    res = pd.DataFrame(all_ids, columns=['img_id'])
    res['label'] = all_labels
    res['score'] = all_scores
    res['x1'] = all_boxes[:, 0]
    res['x2'] = all_boxes[:, 2]
    res['y1'] = all_boxes[:, 1]
    res['y2'] = all_boxes[:, 3]
    fps = len(ids_to_use) / (time.time() - start_time)
    print(f'Processed {len(ids_to_use)} images -> FPS: {fps:.2f}')
    return res

def ensemble(benchmark_csv, weights, params, get_score_init=True):
    if get_score_init:
        for bcsv in benchmark_csv:
            print('Scoring original prediction:', bcsv)
            get_coco_score(bcsv)
    ensemble_preds = ensemble_predictions(benchmark_csv, weights, params)
    out_path = f"{params['run_type']}_results.csv"
    ensemble_preds.to_csv(out_path, index=False)
    get_coco_score(out_path)

if __name__ == '__main__':
    params = {
        'run_type': 'ciou-nms',
        'iou_thr': 0.5,
        'verbose': True,
    }
    in_dir = './redataset_last/'
    benchmark_csv = [
        in_dir + 'output_ssd.csv',
        in_dir + 'output_yolo.csv',
    ]
    weights = [1, 1]
    assert len(benchmark_csv) == len(weights)
    ensemble(benchmark_csv, weights, params, get_score_init=False)
