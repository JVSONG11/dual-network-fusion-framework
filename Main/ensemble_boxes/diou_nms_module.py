import numpy as np

def bbox_diou(box1, box2):
    """
    Compute the DIoU between two boxes.
    Boxes are [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    # Intersection
    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Union
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    union = area1 + area2 - inter_area + 1e-7

    iou = inter_area / union

    # Center distance
    center_x1 = (x1 + x2) / 2
    center_y1 = (y1 + y2) / 2
    center_x2 = (x1g + x2g) / 2
    center_y2 = (y1g + y2g) / 2
    center_dist = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2

    # Enclosing box diagonal
    enclose_x1 = min(x1, x1g)
    enclose_y1 = min(y1, y1g)
    enclose_x2 = max(x2, x2g)
    enclose_y2 = max(y2, y2g)
    c_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-7

    # DIoU
    diou = iou - center_dist / c_diag
    return diou

def diou_nms_single_class(boxes, scores, iou_thr):
    indices = scores.argsort()[::-1]
    keep = []
    while indices.size > 0:
        current = indices[0]
        keep.append(current)
        if indices.size == 1:
            break
        rest = indices[1:]
        ious = np.array([bbox_diou(boxes[current], boxes[i]) for i in rest])
        indices = rest[ious < iou_thr]
    return keep

def diou_nms(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.5):
    """
    boxes_list: list of numpy arrays of shape (N_i, 4)
    scores_list: list of numpy arrays of shape (N_i,)
    labels_list: list of numpy arrays of shape (N_i,)
    """
    all_boxes = np.concatenate(boxes_list, axis=0)
    all_scores = np.concatenate(scores_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    result_boxes = []
    result_scores = []
    result_labels = []

    for label in np.unique(all_labels):
        inds = np.where(all_labels == label)[0]
        boxes = all_boxes[inds]
        scores = all_scores[inds]

        keep = diou_nms_single_class(boxes, scores, iou_thr)

        result_boxes.extend(boxes[keep])
        result_scores.extend(scores[keep])
        result_labels.extend([label] * len(keep))

    return (
        np.array(result_boxes),
        np.array(result_scores),
        np.array(result_labels),
    )
