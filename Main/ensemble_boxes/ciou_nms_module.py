import numpy as np

def bbox_ciou(box1, box2, eps=1e-7):
    """Calculate the CIoU between two boxes (x1, y1, x2, y2 format)"""
    # Intersection
    inter_x1 = np.maximum(box1[0], box2[0])
    inter_y1 = np.maximum(box1[1], box2[1])
    inter_x2 = np.minimum(box1[2], box2[2])
    inter_y2 = np.minimum(box1[3], box2[3])

    inter_area = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)

    # Union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area + eps

    iou = inter_area / union_area

    # Enclosing box
    enclose_x1 = np.minimum(box1[0], box2[0])
    enclose_y1 = np.minimum(box1[1], box2[1])
    enclose_x2 = np.maximum(box1[2], box2[2])
    enclose_y2 = np.maximum(box1[3], box2[3])
    enclose_diagonal = (enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2 + eps

    # center distance
    c1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    c2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    center_dist = (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2

    # aspect ratio consistency
    v = (4 / (np.pi ** 2)) * np.power(
        np.arctan((box1[2] - box1[0]) / (box1[3] - box1[1] + eps)) -
        np.arctan((box2[2] - box2[0]) / (box2[3] - box2[1] + eps)), 2
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = v / (1 - iou + v + eps)
    ciou = iou - (center_dist / enclose_diagonal + alpha * v)
    return ciou

def ciou_nms(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.5, thresh=0.0):
    """Apply CIoU-based NMS."""
    merged_boxes = []
    merged_scores = []
    merged_labels = []

    all_boxes = np.concatenate(boxes_list, axis=0)
    all_scores = np.concatenate(scores_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    unique_labels = np.unique(all_labels)

    for lbl in unique_labels:
        inds = np.where(all_labels == lbl)[0]
        boxes = all_boxes[inds]
        scores = all_scores[inds]
        order = scores.argsort()[::-1]

        while len(order) > 0:
            i = order[0]
            merged_boxes.append(boxes[i])
            merged_scores.append(scores[i])
            merged_labels.append(lbl)

            rem_boxes = boxes[order[1:]]
            ious = np.array([bbox_ciou(boxes[i], b) for b in rem_boxes])
            keep_inds = np.where(ious < iou_thr)[0]
            order = order[keep_inds + 1]  # +1 to account for skipped i

    return np.array(merged_boxes), np.array(merged_scores), np.array(merged_labels)
