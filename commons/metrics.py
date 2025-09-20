import torch
import numpy as np
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def compute_ap(recall, precision, method="interp"):
    """
    Compute the average precision, given the recall and precision curves.
    Modified to support both continuous and interpolated (COCO-style) AP calculation.

    Args:
        recall:    The recall curve (list).
        precision: The precision curve (list).
        method:    "interp" (COCO 101-point interpolation) or "continuous" (Pascal VOC-style).
    Returns:
        The average precision.
    """
    # Append sentinel values to ensure the curves start at 0 and end at 1
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute the precision envelope (monotonically decreasing)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Choose calculation method
    if method == "interp":
        # 101-point interpolation (COCO)
        x = np.linspace(0, 1, 101)  # 101 evenly spaced recall points
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # Integrate using trapezoidal rule
    else:
        # Continuous (Pascal VOC-style)
        i = np.where(mrec[1:] != mrec[:-1])[0]  # Points where recall changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # Area under curve
    return ap

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.p, r, ap, f1------np.array
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes = np.unique(target_cls)
    ap, p, r, f1, class_ids, n_gt, n_p = [], [], [], [], [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt_c = (target_cls == c).sum()
        n_p_c = i.sum()
        class_ids.append(c)
        n_gt.append(n_gt_c)
        n_p.append(n_p_c)
        if n_p_c == 0 and n_gt_c == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
            f1.append(0)
        elif n_p_c == 0 or n_gt_c == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
            f1.append(0)
        else:
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()
            recall_curve = tpc / (n_gt_c + 1e-16)
            precision_curve = tpc / (tpc + fpc)
            r.append(recall_curve[-1])
            p.append(precision_curve[-1])
            ap.append(compute_ap(recall_curve, precision_curve))
            f1.append(2 * precision_curve[-1] * recall_curve[-1] / (precision_curve[-1] + recall_curve[-1] + 1e-16))
    p, r, ap, f1 = np.array(p), np.array(r), np.array(ap), np.array(f1)
    return np.mean(p), np.mean(r), np.mean(ap), np.mean(f1), p, r, ap, f1, np.array(class_ids), np.array(n_gt), np.array(n_p)
    
def get_batch_statistics(outputs, targets, device, iou_threshold=0.5):
    """_summary_Compute true positives, predicted scores and predicted labels per sample
    Args:
        outputs (YOLO-Results): outputs = model.predict()
        targets (tensor(gpu)): 所有预测的bbox
        device (torch.device()): 在哪里运行
        iou_threshold (float, optional): iou阈值. Defaults to 0.5.

    Returns:
        list: len为batch，每个元素是长为3的list，list的三个元素分别为：array表示TP，tensor表示预测分数，tensor表示预测类别
    """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes =  output.boxes.xyxy # bbox
        pred_scores = output.boxes.conf
        pred_labels = output.boxes.cls

        true_positives = np.zeros(pred_boxes.shape[0])
        if len(targets.shape) == 1:
            annotations = []
        else:
            annotations = targets[targets[:, 5] == sample_i][:, 0:5]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                
                pred_box = pred_box.to(device)
                pred_label = pred_label.to(device)

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label.to(device) not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores.cpu(), pred_labels.cpu()])
    return batch_metrics

def print_per_class_metrics(mp, mr, map50, mf1, p, r, ap, f1, class_ids, n_gt, n_p, class_names=None, dataset_len=None):
    print(f"{'Class':>15} {'Images':>8} {'Instances':>10} {'P':>10} {'R':>10} {'mAP50':>10} {'F1':>10}")
    print(f"{'all':>15} {dataset_len if dataset_len is not None else '':>8} {int(np.sum(n_gt)):>10} {mp:10.3f} {mr:10.3f} {map50:10.3f} {mf1:10.3f}")
    for i, cid in enumerate(class_ids):
        cname = class_names[int(cid)] if class_names is not None and cid < len(class_names) else str(cid)
        print(f"{cname:>15} {dataset_len if dataset_len is not None else '':>8} {n_gt[i]:>10} {p[i]:10.3f} {r[i]:10.3f} {ap[i]:10.3f} {f1[i]:10.3f}")