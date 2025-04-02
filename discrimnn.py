import argparse
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
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights, shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
import random
from commons.proutils import parse
from commons.utils import resolve_npz
    
def max_edge(rs, clfs, idx):
    rs, clfs = np.array(rs), np.array(clfs)
    candidate_rs, candidate_clf = rs[idx], clfs[idx]
    candidate_edge_x = candidate_rs * candidate_clf + (1 - candidate_rs) * (1 - candidate_clf)
    max_indices = np.where(candidate_edge_x == np.max(candidate_edge_x))[0]
    return max_indices[-1], candidate_rs[max_indices[-1]]

def dynamic_ap(aps, clfs, r):
    emap, dmap, emapx, dmapx = aps[0][1], aps[1][1], aps[2][1], aps[3][1]
    candidate_n = len(emap)
    # dynamic_ap列表
    dynamic_aps = []
    r_index = 0
    r = [item/100 for item in r]
    # 将 r 转换为集合以提高查询效率
    r_set = set(r) if isinstance(r, list) else r
    for i in range(candidate_n):
        if emap[i][0] not in r_set:
            continue
        pcls = clfs[r_index]
        pwe, pwh, pse, psh = emap[i][1], dmap[i][1], emapx[i][1], dmapx[i][1]
        p_correct_e = r[r_index] * pcls * pwe
        p_wrong_h = (1 - r[r_index]) * (1 - pcls) * pwh
        p_correct_h = (1 - r[r_index]) * pcls * psh
        p_wrong_e = r[r_index] * (1 - pcls) * pse
        ptotal = p_correct_e + p_wrong_h + p_correct_h + p_wrong_e
        dynamic_aps.append(ptotal)
        r_index += 1
    return dynamic_aps
    
class cls_scheme:
    def __init__(self, dataset='pestv3', ratio=[30, 40, 50, 60, 70]):
        self.data = dataset
        self.rs = ratio
    
    def __get_model(self, id, device):
        if id == 1:
            cls = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.DEFAULT)
            cls.fc = torch.nn.Linear(cls.fc.in_features, 2)
            cls.to(device)
        else:
            cls = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
            cls.fc = torch.nn.Linear(cls.fc.in_features, 2)
            cls.to(device)
        return cls

    def items(self):
        clfs = []
        for r in self.rs:
            model_path = os.path.join('checkpoint/classifier', self.data, f'{r}', 'best_model.pth')
            clfs.append(model_path)
        return clfs


    def scores(self):
        clfs = []
        models = self.items()
        for idx, model in enumerate(models):
            val_accuracy = self.evaluate_model_accuracy(model, self.rs[idx])
            clfs.append(val_accuracy)
        return clfs

    def evaluate_model_accuracy(self, model_path, ratio):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # 加载模型
        model = self.__get_model(1, device)  # 使用训练时的相同模型ID（1表示shufflenet_v2_x0_5）
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # 准备验证集
        val_transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])
        val_dataset = ClassifyDataset(img_dir=os.path.join('out', self.data, str(ratio), 'trainval'), 
                                    transform=val_transform, train=False)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16)
        
        # 计算分类精度
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)  # 获取预测类别
                
                correct += (preds == labels).sum().item()  # 统计正确预测数
                total += labels.size(0)  # 统计总样本数
        
        accuracy = correct / total  # 计算分类精度
        return accuracy

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
    def __init__(self, weight, dconfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router = self._prepare_router(weight)
        self.weak_det = YOLO(dconfig['weak_detector'])
        self.strong_det = YOLO(dconfig['strong_detector'])
        self.cloud_flag = False # random scheme中要用到
    
    def _prepare_router(self, weight):
        router = shufflenet_v2_x0_5()
        router.fc = torch.nn.Linear(router.fc.in_features, 2)
        router.load_state_dict(torch.load(weight))
        router.eval()
        return router

    
    def forward(self, X, mode): 
        def _prepare_det(mode):
            if mode == 'dynamic':
                return self.weak_det, self.strong_det
            elif mode == 'edge':
                return self.weak_det, self.weak_det
            elif mode == 'random':
                choice = random.randint(0, 1)
                if choice == 1:
                    det1 = self.weak_det
                else:
                    det1 = self.strong_det
                    self.cloud_flag = True
                choice = random.randint(0, 1)
                if choice == 1:
                    det2 = self.weak_det
                else:
                    det2 = self.strong_det
                    self.cloud_flag = True
                return det1, det2
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
            'cloud': 1,
            'random': 1 if self.cloud_flag else 0
        }
        offloading = weights[mode] * IMGSZ[0] * IMGSZ[1] * 3 
        # 将预测结果根据索引放回到对应位置
        for i, idx in enumerate(easys):
            outs[idx] = eouts[i]
        for i, idx in enumerate(diffs):
            outs[idx] = douts[i]
        self.cloud_flag = False
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
    parser = argparse.ArgumentParser(description='discrimnn system design')
    parser.add_argument('--dataset', type=str, default='pestv3',  help='选择划分哪个数据集：voc12/voc07/coco/pestv3/visdrone/pestv1/ip102/pest24')
    parser.add_argument('--model_zoo', type=str, default='pestv3', help='选择用哪套模型来划分数据:voc12/voc07/coco/pestv3/visdrone/pestv1/ip102/pest24')
    parser.add_argument('--expected_ap', type=int, default=0.94, help='用户希望系统能够达到的精度')
    parser.add_argument('--iterdata', type=str, default='exp/data', help='保存outlier迭代输出文件的目录')
    parser.add_argument('--dataType', type=str, default='val', help='iter文件的类型, train or val')
    opt = parser.parse_args()
    dconfig, mconfig = parse(opt)

    # 获取划分比例
    r = [30, 40, 50, 60, 70]

    # 解析npz文件，用户获取ap的迭代曲线
    npz_path = os.path.join(opt.iterdata, f'{opt.dataType}_iter_map_{opt.dataset}.npz')
    iter_aps = resolve_npz(npz_path)

    # 训练分类器，获得分类策略
    scheme = cls_scheme(dataset=opt.dataset, ratio=r)
    cs = scheme.scores()
    c_models = scheme.items()
    # 计算ratio
    expected_ap = opt.expected_ap
    dynamic_aps = np.array(dynamic_ap(iter_aps, cs, r))
    # 找到dynamic_ap值大于用户值的下标
    idx = np.where(dynamic_aps > expected_ap)[0]
    loc, max_r = max_edge(r, cs, idx)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ViDiscrimNN(c_models[loc], dconfig).to(device)
    dataset = DetectionDataset(dconfig['source_images'], dconfig['source_labels'], 'val', open=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=DetectionDataset.collate_fn)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=16, collate_fn=DetectionDataset.collate_fn)
    modes = ['edge', 'cloud', 'dynamic', 'random']
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

    # 定义每个指标的纵坐标范围
    ylim_dict = {
        'Precision': (0.5, 1),      # Precision 范围 0 到 1
        'Recall': (0.5, 1),         # Recall 范围 0 到 1
        'mAP50': (0.5, 1),          # mAP50 范围 0 到 1
        'F1 Score': (0.5, 1),       # F1 Score 范围 0 到 1
        'FPS': (0, max(metrics['FPS']) + 10),  # FPS 范围 0 到最大值 + 10
        'Uploading': (0, max(metrics['Uploading']) + 10)  # Uploading 范围 0 到最大值 + 10
    }

    for ax, (metric, values) in zip(axes.flatten(), metrics.items()):
        ax.bar(modes, values, color=['skyblue', 'orange', 'red', 'green'])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_xlabel('Mode')
        ax.set_ylim(ylim_dict[metric])  # 设置纵坐标范围
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = "figure/performance_metrics.png"
    plt.savefig(output_path)
    print(f"Performance metrics chart saved to {output_path}")

