import torch.nn as nn
from torchvision.models import shufflenet_v2_x0_5
from config import *
from ultralytics import YOLO
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset
from commons.dataset import ClassifyDataset, DetectionDataset
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import time


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


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
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

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return np.mean(p), np.mean(r), np.mean(ap), np.mean(f1)

    
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

class ViDiscrimNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router = self._prepare_router()
        self.weak_det = YOLO(pestv3_config['weak_detector'])
        self.strong_det = YOLO(pestv3_config['strong_detector'])
    
    def _prepare_router(self):
        router = shufflenet_v2_x0_5()
        router.fc = torch.nn.Linear(router.fc.in_features, 2)
        router.load_state_dict(torch.load(classify_config['classifier']))
        router.eval()
        return router

    
    def forward(self, X, mode): 
        def _prepare_det(mode):
            if mode == 'dynamic':
                return self.weak_det, self.strong_det
            elif mode == 'edge':
                return self.weak_det, self.weak_det
            else:
                return self.strong_det, self.strong_det
        det1, det2 = _prepare_det(mode)
        outputs = self.router(X)
        rst = outputs.argmax(dim=1)
        easys = []
        diffs = []
        eouts = []
        douts = []
        for idx, elem in enumerate(rst):
            if elem == 0:  # easy
                easys.append(idx)
            else:  # difficult
                diffs.append(idx)
        if len(easys) > 0:
            eouts = det1.predict(X[easys], verbose=False)
        if len(diffs) > 0:
            douts = det2.predict(X[diffs], verbose=False)

        # 创建一个与输入大小相同的空列表
        outs = [None] * len(X)
        l = len(diffs)
        weights = {
            'dynamic': len(diffs),
            'edge': 0,
            'cloud': 1
        }
        offloading = weights[mode] * IMGSZ[0] * IMGSZ[1] * 3 
        # 将预测结果根据索引放回到对应位置
        for i, idx in enumerate(easys):
            outs[idx] = eouts[i]
        for i, idx in enumerate(diffs):
            outs[idx] = douts[i]

        return outs, offloading

    #模型评估
    def evaluation(self, val_dataloader, device, mode):
        labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)
        pbar = tqdm(val_dataloader)
        classes = []
        total_num = 0
        total_time = 0
        total_offloading = 0
        for imgs, targets in pbar:
            # Extract classes
            if len(targets.shape) == 1:
                classes += []
            else:
                classes += targets[:, 0].tolist()
                targets[:, 1:5] = xywh2xyxy(targets[:, 1:5])
                targets[:, 1:5] *= torch.tensor([*IMGSZ, *IMGSZ])

            labels = targets.to(device)
            imgs = imgs.to(device)
            # ====================== time begin =====================
            begin = time.time()
            output, offloading_num = self.forward(imgs, mode)
            end = time.time()
            total_num += imgs.shape[0]
            total_time += end - begin
            # ====================== time end =====================
            pbar.set_description("Evaluation model:") 
            sample_metrics += get_batch_statistics(output, labels, device)
            total_offloading += offloading_num
        if len(sample_metrics) == 0:  # No detections over whole validation set.
            print("---- No detections over whole validation set ----")
            return None

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, classes)
        fps = total_num / total_time
        return metrics_output, fps, total_offloading  

    
if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ViDiscrimNN().to(device)
    dataset = DetectionDataset(pestv3_config['source_images'], pestv3_config['source_labels'], 'val', open=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=DetectionDataset.collate_fn)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=16, collate_fn=DetectionDataset.collate_fn)
    modes = ['edge', 'cloud', 'dynamic']
    metrics = {
        'Precision': [],
        'Recall': [],
        'mAP50': [],
        'F1 Score': [],
        'FPS': [],
        'Uploading': []
    }

    for mode in modes:
        performance, fps, uploading = model.evaluation(dataloader, device, mode)
        print(f'----------------------execute {mode} mode:-----------------------')
        print("Precision: ", performance[0])
        print("Recall", performance[1])
        print("mAP50", performance[2])
        print("F1 Score: ", performance[3])
        print("FPS: ", fps)
        print("Uploading ", uploading)
        print(f'----------------------end evaluation:-----------------------')

        metrics['Precision'].append(performance[0])
        metrics['Recall'].append(performance[1])
        metrics['mAP50'].append(performance[2])
        metrics['F1 Score'].append(performance[3])
        metrics['FPS'].append(fps)
        metrics['Uploading'].append(uploading)

    # 绘制图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Performance in Different Modes', fontsize=16)

    for ax, (metric, values) in zip(axes.flatten(), metrics.items()):
        ax.bar(modes, values, color=['skyblue', 'orange', 'green'])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_xlabel('Mode')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = "figure/performance_metrics.png"
    plt.savefig(output_path)
    print(f"Performance metrics chart saved to {output_path}")

