import numpy as np
import pandas as pd
import json
import time
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from multiprocessing import Pool, Process, cpu_count, Manager
from ensemble_boxes import *
from torchvision.ops import generalized_box_iou


def giou_nms(boxes, scores, iou_threshold):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64)

    scores, order = scores.sort(descending=True)
    boxes = boxes[order]

    keep = []
    while boxes.numel() > 0:
        keep.append(order[0].item())
        if boxes.size(0) == 1:
            break
        ious = generalized_box_iou(boxes[0].unsqueeze(0), boxes[1:]).squeeze(0)
        keep_idx = (ious <= iou_threshold).nonzero(as_tuple=False).squeeze(1) + 1
        boxes = boxes[keep_idx]
        order = order[keep_idx]

    return torch.tensor(keep, dtype=torch.long)


def get_coco_annotations_data():
    file_in = './WBF.json'
    images = dict()
    with open(file_in) as json_file:
        data = json.load(json_file)
    for i in range(len(data['images'])):
        image_id = data['images'][i]['id']
        images[image_id] = data['images'][i]
    return images


def get_coco_score(csv_path):
    images = get_coco_annotations_data()
    s = pd.read_csv(csv_path, dtype={'img_id': np.str_, 'label': np.str_})
    out = np.zeros((len(s), 7), dtype=np.float64)
    out[:, 0] = s['img_id']
    ids = s['img_id'].astype(np.int32).values
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
    return coco_eval.stats, detections


def process_single_id(id, res_boxes, weights, params):
    run_type = params['run_type']
    verbose = params['verbose']
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
            if box_x1 >= box_x2 or box_y1 >= box_y2 or scr <= 0:
                continue
            boxes.append([box_x1, box_y1, box_x2, box_y2])
            scores.append(scr)
            if lbl not in labels_to_use_forward:
                cur_point = len(labels_to_use_forward)
                labels_to_use_forward[lbl] = cur_point
                labels_to_use_backward[cur_point] = lbl
            labels.append(labels_to_use_forward[lbl])

        boxes_list.append(np.array(boxes, dtype=np.float32))
        scores_list.append(np.array(scores, dtype=np.float32))
        labels_list.append(np.array(labels, dtype=np.int32))

    if len(boxes_list) == 0:
        return np.array([]), np.array([]), np.array([]), []

    if run_type == 'giou-nms':
        boxes = boxes_list[0]
        scores = scores_list[0]
        labels = labels_list[0]
        keep_indices = giou_nms(torch.tensor(boxes), torch.tensor(scores), params['iou_thr'])
        merged_boxes = boxes[keep_indices.numpy()]
        merged_scores = scores[keep_indices.numpy()]
        merged_labels = labels[keep_indices.numpy()]
    else:
        raise NotImplementedError("Only 'giou-nms' is supported in this version.")

    merged_labels_string = [labels_to_use_backward[m] for m in merged_labels]
    merged_labels = np.array(merged_labels_string, dtype=np.str_)
    ids_list = [id] * len(merged_labels)

    return merged_boxes.copy(), merged_scores.copy(), merged_labels.copy(), ids_list.copy()


def process_part_of_data(proc_number, return_dict, ids_to_use, res_boxes, weights, params):
    result = []
    for id in ids_to_use:
        merged_boxes, merged_scores, merged_labels, ids_list = process_single_id(id, res_boxes, weights, params)
        result.append((merged_boxes, merged_scores, merged_labels, ids_list))
    return_dict[proc_number] = result.copy()


def ensemble_predictions(pred_filenames, weights, params):
    start_time = time.time()
    procs_to_use = max(cpu_count() // 2, 1)
    weights = np.array(weights)
    res_boxes = dict()
    ref_ids = None

    for j in range(len(pred_filenames)):
        if weights[j] == 0:
            continue
        s = pd.read_csv(pred_filenames[j], dtype={'img_id': np.str_, 'label': np.str_})
        s.sort_values('img_id', inplace=True)
        s.reset_index(drop=True, inplace=True)
        ids = s['img_id'].values
        unique_ids = sorted(s['img_id'].unique())
        if ref_ids is None:
            ref_ids = tuple(unique_ids)
        else:
            if ref_ids != tuple(unique_ids):
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

    weights = weights[weights != 0]
    ids_to_use = sorted(list(res_boxes.keys()))
    manager = Manager()
    return_dict = manager.dict()
    jobs = []

    for i in range(procs_to_use):
        start = i * len(ids_to_use) // procs_to_use
        end = (i + 1) * len(ids_to_use) // procs_to_use
        if i == procs_to_use - 1:
            end = len(ids_to_use)
        p = Process(target=process_part_of_data,
                    args=(i, return_dict, ids_to_use[start:end], res_boxes, weights, params))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    results = []
    for i in range(len(jobs)):
        results += return_dict[i]

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

    res = pd.DataFrame(all_ids, columns=['img_id'])
    res['label'] = all_labels
    res['score'] = all_scores
    res['x1'] = all_boxes[:, 0]
    res['x2'] = all_boxes[:, 2]
    res['y1'] = all_boxes[:, 1]
    res['y2'] = all_boxes[:, 3]

    total_time = time.time() - start_time
    fps = len(ids_to_use) / total_time
    print(f'Run time: {total_time:.2f} seconds')
    print(f'Processed {len(ids_to_use)} images -> FPS: {fps:.2f}')
    return res


def ensemble(benchmark_csv, weights, params, get_score_init=True):
    if get_score_init:
        for bcsv in benchmark_csv:
            print('Go for {}'.format(bcsv))
            get_coco_score(bcsv)
    ensemble_preds = ensemble_predictions(benchmark_csv, weights, params)
    ensemble_preds.to_csv("giou_nms.csv", index=False)
    get_coco_score("giou_nms.csv")


if __name__ == '__main__':
    params = {
        'run_type': 'giou-nms',
        'iou_thr': 0.5,
        'verbose': True,
    }

    in_dir = './'
    benchmark_csv = [
        in_dir + 'redataset_last/output_ssd.csv',
        in_dir + 'redataset_last/output_yolo.csv',
    ]
    weights = [1, 1]
    assert len(benchmark_csv) == len(weights)
    ensemble(benchmark_csv, weights, params, get_score_init=False)
