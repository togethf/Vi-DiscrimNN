from ultralytics import YOLO
from config_plus import det_pool, ds, model_predict_conf, model_predict_iou
from commons.metrics import bbox_iou, get_batch_statistics, ap_per_class, xywh2xyxy, print_per_class_metrics
from commons.dataset import DetectionDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Tuple
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

from mmdet.apis import init_detector, inference_detector

class MMDetWrapper:
    """
    让 MMDetection 3.x 模型表现得像 Ultralytics YOLO 模型。
    (修正版 v2：增加了 RGB -> BGR 转换，修复颜色通道不匹配导致的精度下降)
    """
    def __init__(self, config_path, checkpoint_path, device='cuda:0'):
        # 初始化模型
        self.model = init_detector(config_path, checkpoint_path, device=device)
        self.device = device
        self.names = self.model.dataset_meta.get('classes', {}) if hasattr(self.model, 'dataset_meta') else {}

    def __call__(self, source, conf=0.25, iou=0.7, verbose=False):
        """
        模拟 YOLO 的调用接口
        """
        # 1. 输入处理
        if isinstance(source, torch.Tensor):
            # source shape: [1, 3, H, W] or [3, H, W] (RGB, 0-1 normalized)
            img = source.cpu().numpy()
            if img.ndim == 4:
                img = img[0]
            
            # [C, H, W] -> [H, W, C]
            if img.shape[0] == 3: 
                img = img.transpose(1, 2, 0)
            
            # 反归一化: 0-1 float -> 0-255 uint8
            if img.max() <= 1.05:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            
            # === 关键修复：RGB 转 BGR ===
            # DetectionDataset (PIL) 读入的是 RGB
            # MMDetection (OpenCV) 期望的是 BGR
            img = img[..., ::-1] 
            # ==========================

        elif isinstance(source, (str, np.ndarray)):
            # 如果输入直接是路径或numpy array，通常假设已经是正确的格式
            # 但如果外部用 PIL 读取传入 numpy，这里可能也需要处理，视情况而定
            img = source
        else:
            raise TypeError(f"Unsupported input type: {type(source)}")

        # 2. 推理
        result = inference_detector(self.model, img)
        
        # 确保 result 是列表
        if not isinstance(result, (list, tuple)):
            result_list = [result]
        else:
            result_list = result

        # 3. 结果封装
        yolo_results = []
        for det_sample in result_list:
            pred = det_sample.pred_instances
            
            # 提取数据
            scores = pred.scores
            bboxes = pred.bboxes
            labels = pred.labels
            
            # 应用置信度过滤
            keep = scores > conf
            scores = scores[keep]
            bboxes = bboxes[keep]
            labels = labels[keep]
            
            # 构造结果对象
            yolo_results.append(self._make_yolo_result(bboxes, scores, labels))
            
        return yolo_results

    def predict(self, source, **kwargs):
        return self(source, **kwargs)[0]

    def _make_yolo_result(self, bboxes, scores, labels):
        class MockBoxes:
            def __init__(self, xyxy, conf, cls, dev):
                self.xyxy = xyxy.to(dev)
                self.conf = conf.to(dev)
                self.cls = cls.to(dev)
                if xyxy.shape[0] > 0:
                    self.data = torch.cat((self.xyxy, self.conf.unsqueeze(1), self.cls.unsqueeze(1)), dim=1)
                else:
                    self.data = torch.empty((0, 6), device=dev)

        class MockResult:
            def __init__(self, boxes, names):
                self.boxes = boxes
                self.names = names
                
        boxes = MockBoxes(bboxes, scores, labels, self.device)
        return MockResult(boxes, self.names)

def _compute_iou(box_a: np.ndarray, box_b: np.ndarray, device: torch.device) -> float:
    iou = bbox_iou(
        torch.tensor(box_a, dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor(box_b, dtype=torch.float32, device=device).unsqueeze(0)
    )[0].item()
    return float(iou)


def _fuse_clusters(
    clusters: List[List[Tuple[np.ndarray, float, float]]],
    method: str = 'max'
):
    fused = []
    method = method.lower()
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        # cluster elements: (box_xyxy, conf, cls)
        if method == 'wbf':
            weights = np.array([c[1] for c in cluster], dtype=np.float32)
            weights = weights / (weights.sum() + 1e-9)
            boxes = np.stack([c[0] for c in cluster], axis=0).astype(np.float32)
            fused_box = (boxes * weights[:, None]).sum(axis=0)
            # class: pick the one with max total weight per class
            classes = np.array([c[2] for c in cluster])
            # choose class of the max-weight member
            top_idx = int(np.argmax([c[1] for c in cluster]))
            fused_cls = float(cluster[top_idx][2])
            # confidence: max confidence in cluster
            fused_conf = float(np.max([c[1] for c in cluster]))
            fused.append((fused_box, fused_conf, fused_cls))
        else:
            # default: pick max confidence
            best = max(cluster, key=lambda x: x[1])
            fused.append(best)
    return fused


def ensemble_yolo_multi(
    img,
    models: List[YOLO],
    iou_thr: float = 0.5,
    conf_thr: float = 0.4,
    device: str = 'cpu',
    method: str = 'max'
):
    # 多模型单张图片推理并融合（类无关聚类）
    all_dets: List[Tuple[np.ndarray, float, float]] = []
    for model in models:
        results = model(img, verbose=False, iou=model_predict_iou, conf=model_predict_conf)[0]
        dets = list(zip(
            results.boxes.xyxy.detach().cpu().numpy(),
            results.boxes.conf.detach().cpu().numpy(),
            results.boxes.cls.detach().cpu().numpy(),
        ))
        # 只保留高于阈值的
        dets = [d for d in dets if d[1] >= conf_thr]
        all_dets.extend(dets)

    # 基于 IoU 的贪心聚类（不区分类别）
    clusters: List[List[Tuple[np.ndarray, float, float]]] = []
    for box, conf, cls in all_dets:
        assigned = False
        for cluster in clusters:
            # 与簇中任意一个框 IoU 超过阈值则归入该簇
            if any(_compute_iou(box, member_box, device) > iou_thr for member_box, _, _ in cluster):
                cluster.append((box, float(conf), float(cls)))
                assigned = True
                break
        if not assigned:
            clusters.append([(box, float(conf), float(cls))])

    final_dets = _fuse_clusters(clusters, method=method)
    return final_dets

class EnsembleYOLO:
    def __init__(self, model_paths: List[str], iou_thr=0.5, conf_thr=0.4, device='cpu', method: str = 'max'):
        self.models = [YOLO(p) for p in model_paths]
        self.iou_thr = iou_thr
        self.conf_thr = conf_thr
        self.device = device
        self.method = method

    def predict(self, img):
        dets = ensemble_yolo_multi(img, self.models, self.iou_thr, self.conf_thr, self.device, self.method)
        # 转为YOLO风格的结果对象，便于后续评估
        class DummyResult:
            pass
        result = DummyResult()
        if len(dets) == 0:
            result.boxes = type('', (), {})()
            result.boxes.xyxy = torch.empty((0, 4))
            result.boxes.conf = torch.empty((0,))
            result.boxes.cls = torch.empty((0,))
        else:
            xyxy = torch.tensor(np.array([d[0] for d in dets]))
            conf = torch.tensor(np.array([d[1] for d in dets]))
            cls = torch.tensor(np.array([d[2] for d in dets]))
            result.boxes = type('', (), {})()
            result.boxes.xyxy = xyxy
            result.boxes.conf = conf
            result.boxes.cls = cls
        return result

def evaluate_ensemble(model, dataloader, device):
    """
    修改：
    1. 运行一次dataloader，保存所有预测和目标。
    2. 循环10个IoU阈值，在内存中计算统计数据。
    3. 返回 12 个值，增加了 map50_95。
    """
    img_size = 640  # 或根据实际图片尺寸动态获取
    
    # 1. 运行一次 dataloader 获取所有预测和目标
    all_batch_outputs = []
    all_targets_processed = []
    all_classes = []
    
    for imgs, targets in tqdm(dataloader, desc="Evaluating ensemble (Model Inference)"):
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        # 归一化xywh转xyxy像素
        if len(targets.shape) > 1 and targets.shape[1] >= 5 and targets[:, 1:5].max() <= 1.0:
            targets[:, 1:5] = xywh2xyxy(targets[:, 1:5])
            targets[:, 1:5] *= torch.tensor([img_size, img_size, img_size, img_size], device=targets.device)
        
        if len(targets.shape) == 1:
            all_classes += []
        else:
            all_classes += targets[:, 0].tolist()
        
        batch_outputs = []
        for img in imgs:
            if img.ndim == 3:
                img = img.unsqueeze(0)
            res = model.predict(img)
            res.boxes.xyxy = res.boxes.xyxy.to(device)
            res.boxes.conf = res.boxes.conf.to(device)
            res.boxes.cls = res.boxes.cls.to(device)
            batch_outputs.append(res)
            
        all_batch_outputs.append(batch_outputs)
        all_targets_processed.append(targets)

    if len(all_classes) == 0:
        print("---- No ground truths found ----")
        return None

    # 2. 循环10个IoU阈值，计算mAP
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    map_scores = []
    metrics_output_50 = None # 存储IoU=0.5时的详细指标

    for iou_thresh in tqdm(iou_thresholds, desc="Calculating mAP@.5:.95"):
        sample_metrics = []
        for batch_outputs, targets in zip(all_batch_outputs, all_targets_processed):
            sample_metrics += get_batch_statistics(batch_outputs, targets, device, iou_threshold=iou_thresh)
        
        if len(sample_metrics) == 0:
            map_scores.append(0.0)
            continue
            
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        
        # 传入所有真实类别
        metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, all_classes)
        
        if metrics_output is not None:
            mp, mr, map50, mf1, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output
            map_scores.append(map50) # map50 在这里是当前 iou_thresh 的 mAP
            
            # 保存 IoU=0.5 时的完整指标
            if np.isclose(iou_thresh, 0.5):
                metrics_output_50 = metrics_output
        else:
            map_scores.append(0.0)

    if metrics_output_50 is None:
        print("---- No detections over whole validation set ----")
        return None

    # 3. 计算 mAP@0.5:0.95
    map50_95 = np.mean(map_scores)

    # 4. 返回 12 个值
    mp, mr, map50, mf1, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output_50
    return mp, mr, map50, mf1, map50_95, p, r, ap, f1, class_ids, n_gt, n_p


def evaluate_single(model_path, dataloader, device):
    """
    修改：
    1. 运行一次dataloader，保存所有预测和目标。
    2. 循环10个IoU阈值，在内存中计算统计数据。
    3. 返回 12 个值，增加了 map50_95。
    """

    """
    修改：评估函数，支持 YOLO 和 MMDetection
    """
    # === 修改开始 ===
    if isinstance(model_path, str) and ";" in model_path:
        # 解析 MMDetection 路径: "config.py;checkpoint.pth"
        config_file, checkpoint_file = model_path.split(";")
        print(f"Loading MMDetection model...\nConfig: {config_file}\nCheckpoint: {checkpoint_file}")
        # 使用我们上面定义的 Wrapper
        model = MMDetWrapper(config_file, checkpoint_file, device=device)
    else:
        # 原有的 YOLO 加载
        model = YOLO(model_path)
    # === 修改结束 ===
    img_size = 640  # 或根据实际图片尺寸动态获取

    # 1. 运行一次 dataloader 获取所有预测和目标
    all_batch_outputs = []
    all_targets_processed = []
    all_classes = []

    for imgs, targets in tqdm(dataloader, desc=f"Evaluating single yolo (Model Inference)"):
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        # 归一化xywh转xyxy像素
        if len(targets.shape) > 1 and targets.shape[1] >= 5 and targets[:, 1:5].max() <= 1.0:
            targets[:, 1:5] = xywh2xyxy(targets[:, 1:5])
            targets[:, 1:5] *= torch.tensor([img_size, img_size, img_size, img_size], device=targets.device)
        
        if len(targets.shape) == 1:
            all_classes += []
        else:
            all_classes += targets[:, 0].tolist()

        batch_outputs = []
        for img in imgs:
            if img.ndim == 3:
                img = img.unsqueeze(0)
            results = model(img, verbose=False, conf=model_predict_conf, iou=model_predict_iou)[0]
            batch_outputs.append(results)
            
        all_batch_outputs.append(batch_outputs)
        all_targets_processed.append(targets)

    if len(all_classes) == 0:
        print("---- No ground truths found ----")
        return None

    # 2. 循环10个IoU阈值，计算mAP
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    map_scores = []
    metrics_output_50 = None # 存储IoU=0.5时的详细指标

    for iou_thresh in tqdm(iou_thresholds, desc="Calculating mAP@.5:.95"):
        sample_metrics = []
        for batch_outputs, targets in zip(all_batch_outputs, all_targets_processed):
            sample_metrics += get_batch_statistics(batch_outputs, targets, device, iou_threshold=iou_thresh)
        
        if len(sample_metrics) == 0:
            map_scores.append(0.0)
            continue
            
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        
        # 传入所有真实类别
        metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, all_classes)
        
        if metrics_output is not None:
            mp, mr, map50, mf1, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output
            map_scores.append(map50) # map50 在这里是当前 iou_thresh 的 mAP
            
            # 保存 IoU=0.5 时的完整指标
            if np.isclose(iou_thresh, 0.5):
                metrics_output_50 = metrics_output
        else:
            map_scores.append(0.0)

    if metrics_output_50 is None:
        print("---- No detections over whole validation set ----")
        return None

    # 3. 计算 mAP@0.5:0.95
    map50_95 = np.mean(map_scores)

    # 4. 返回 12 个值
    mp, mr, map50, mf1, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output_50
    return mp, mr, map50, mf1, map50_95, p, r, ap, f1, class_ids, n_gt, n_p


def evaluate_ensemble_once(det_pool, num_models, method, iou_thr, conf_thr, dataloader, device):
    num_models = max(1, min(num_models, len(det_pool)))
    model_paths = det_pool[:num_models]
    ensemble_model = EnsembleYOLO(model_paths, iou_thr, conf_thr, device, method)
    return evaluate_ensemble(ensemble_model, dataloader, device)


def sweep_ensemble_size(det_pool, max_k, method, iou_thr, conf_thr, dataloader, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    xs = []
    map50s = []
    map50_95s = [] # 新增
    f1s = []
    csv_path = os.path.join(out_dir, 'sweep.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_models', 'mp', 'mr', 'map50', 'mf1', 'map50_95']) # 新增
        for k in range(1, min(max_k, len(det_pool)) + 1):
            print(f"=========== sweep ensemble size = {k} =============")
            metrics_output = evaluate_ensemble_once(det_pool, k, method, iou_thr, conf_thr, dataloader, device)
            if metrics_output is not None:
                # 解包 12 个值
                mp, mr, map50, mf1, map50_95, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output
                xs.append(k)
                map50s.append(map50)
                map50_95s.append(map50_95) # 新增
                f1s.append(mf1)
                writer.writerow([k, mp, mr, map50, mf1, map50_95]) # 新增
            print("=========== end sweep =============")
    if len(xs) > 0:
        plt.figure(figsize=(6,4))
        plt.plot(xs, map50s, marker='o', label='mAP@0.5')
        plt.plot(xs, map50_95s, marker='^', label='mAP@.5:.95') # 新增
        plt.plot(xs, f1s, marker='s', label='mF1')
        plt.xlabel('Number of models in ensemble')
        plt.ylabel('Score')
        plt.title('Ensemble size vs performance')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        fig_path = os.path.join(out_dir, 'sweep.png')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        print(f"Saved CSV to {csv_path} and figure to {fig_path}")


def sweep_iou_threshold(
    det_pool,
    ensemble_sets: List[List[int]],
    method: str,
    iou_values: List[float],
    conf_thr: float,
    dataloader,
    device,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    if not ensemble_sets:
        ensemble_sets = [[0, 1]] if len(det_pool) >= 2 else [[0]]

    for chosen in ensemble_sets:
        sub_pool = [det_pool[i] for i in chosen]
        label = 'ens[' + ','.join(map(str, chosen)) + ']'
        rows = []
        xs = []
        map50s = []
        map50_95s = [] # 新增
        f1s = []
        csv_path = os.path.join(out_dir, f'sweep_iou_{"_".join(map(str, chosen))}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iou', 'mp', 'mr', 'map50', 'mf1', 'map50_95']) # 新增
            for iou_thr in iou_values:
                print(f"=========== sweep IoU {iou_thr:.2f} on {label} =============")
                metrics_output = evaluate_ensemble_once(
                    det_pool=sub_pool,
                    num_models=len(sub_pool),
                    method=method,
                    iou_thr=float(iou_thr),
                    conf_thr=conf_thr,
                    dataloader=dataloader,
                    device=device,
                )
                if metrics_output is not None:
                    # 解包 12 个值
                    mp, mr, map50, mf1, map50_95, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output
                    xs.append(float(iou_thr))
                    map50s.append(map50)
                    map50_95s.append(map50_95) # 新增
                    f1s.append(mf1)
                    writer.writerow([float(iou_thr), mp, mr, map50, mf1, map50_95]) # 新增
                print("=========== end sweep =============")
        if len(xs) > 0:
            plt.figure(figsize=(6,4))
            plt.plot(xs, map50s, marker='o', label='mAP@0.5')
            plt.plot(xs, map50_95s, marker='^', label='mAP@.5:.95') # 新增
            plt.plot(xs, f1s, marker='s', label='mF1')
            plt.xlabel('IoU threshold for clustering')
            plt.ylabel('Score')
            plt.title(f'IoU vs performance {label}')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            fig_path = os.path.join(out_dir, f'sweep_iou_{"_".join(map(str, chosen))}.png')
            plt.tight_layout()
            plt.savefig(fig_path, dpi=200)
            best_map50_idx = int(np.argmax(map50s))
            best_f1_idx = int(np.argmax(f1s))
            best_map50_95_idx = int(np.argmax(map50_95s)) # 新增
            print(f"Best mAP@0.5 at IoU={xs[best_map50_idx]:.3f}: {map50s[best_map50_idx]:.4f}")
            print(f"Best mAP@.5:.95 at IoU={xs[best_map50_95_idx]:.3f}: {map50_95s[best_map50_95_idx]:.4f}") # 新增
            print(f"Best mF1 at IoU={xs[best_f1_idx]:.3f}: {f1s[best_f1_idx]:.4f}")
            print(f"Saved CSV to {csv_path} and figure to {fig_path}")


def sweep_conf_threshold(
    det_pool,
    ensemble_sets: List[List[int]],
    method: str,
    conf_values: List[float],
    iou_thr: float,
    dataloader,
    device,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    if not ensemble_sets:
        ensemble_sets = [[0, 1]] if len(det_pool) >= 2 else [[0]]

    for chosen in ensemble_sets:
        sub_pool = [det_pool[i] for i in chosen]
        label = 'ens[' + ','.join(map(str, chosen)) + ']'
        rows = []
        xs = []
        map50s = []
        map50_95s = [] # 新增
        f1s = []
        csv_path = os.path.join(out_dir, f'sweep_conf_{"_".join(map(str, chosen))}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['conf', 'mp', 'mr', 'map50', 'mf1', 'map50_95']) # 新增
            for conf_thr in conf_values:
                print(f"=========== sweep Conf {conf_thr:.2f} on {label} =============")
                metrics_output = evaluate_ensemble_once(
                    det_pool=sub_pool,
                    num_models=len(sub_pool),
                    method=method,
                    iou_thr=float(iou_thr),
                    conf_thr=float(conf_thr),
                    dataloader=dataloader,
                    device=device,
                )
                if metrics_output is not None:
                    # 解包 12 个值
                    mp, mr, map50, mf1, map50_95, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output
                    xs.append(float(conf_thr))
                    map50s.append(map50)
                    map50_95s.append(map50_95) # 新增
                    f1s.append(mf1)
                    writer.writerow([float(conf_thr), mp, mr, map50, mf1, map50_95]) # 新增
                print("=========== end sweep =============")
        if len(xs) > 0:
            plt.figure(figsize=(6,4))
            plt.plot(xs, map50s, marker='o', label='mAP@0.5')
            plt.plot(xs, map50_95s, marker='^', label='mAP@.5:.95') # 新增
            plt.plot(xs, f1s, marker='s', label='mF1')
            plt.xlabel('Confidence threshold')
            plt.ylabel('Score')
            plt.title(f'Conf vs performance {label}')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            fig_path = os.path.join(out_dir, f'sweep_conf_{"_".join(map(str, chosen))}.png')
            plt.tight_layout()
            plt.savefig(fig_path, dpi=200)
            best_map50_idx = int(np.argmax(map50s))
            best_f1_idx = int(np.argmax(f1s))
            best_map50_95_idx = int(np.argmax(map50_95s)) # 新增
            print(f"Best mAP@0.5 at Conf={xs[best_map50_idx]:.3f}: {map50s[best_map50_idx]:.4f}")
            print(f"Best mAP@.5:.95 at Conf={xs[best_map50_95_idx]:.3f}: {map50_95s[best_map50_95_idx]:.4f}") # 新增
            print(f"Best mF1 at Conf={xs[best_f1_idx]:.3f}: {f1s[best_f1_idx]:.4f}")
            print(f"Saved CSV to {csv_path} and figure to {fig_path}")

def _parse_indices(s: str, total: int):
    if not s:
        return list(range(total))
    parts = [p.strip() for p in s.split(',') if p.strip() != '']
    idxs = []
    for p in parts:
        try:
            v = int(p)
            if 0 <= v < total:
                idxs.append(v)
        except Exception:
            pass
    # 去重并保持顺序
    seen = set()
    out = []
    for v in idxs:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out

def compare_base_vs_ensemble(det_pool, idxs, ensemble_k, method, iou_thr, conf_thr, dataloader, device, out_dir, ens_idxs=None):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    labels = []
    map50_vals = []
    map50_95_vals = [] # 新增
    f1_vals = []
    # 单模型评测
    for i in idxs:
        print(f"=========== evaluate single model[{i}] =============")
        metrics_output = evaluate_single(det_pool[i], dataloader, device)
        if metrics_output is not None:
            # 解包 12 个值
            mp, mr, map50, mf1, map50_95, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output
            rows.append(['single', i, mp, mr, map50, mf1, map50_95]) # 新增
            labels.append(f"m{i}")
            map50_vals.append(map50)
            map50_95_vals.append(map50_95) # 新增
            f1_vals.append(mf1)
        print("=========== end single =============")
    # ensemble 评测
    if ens_idxs is not None and len(ens_idxs) > 0:
        chosen = ens_idxs
        print(f"=========== evaluate ensemble over indices {chosen} =============")
        sub_pool = [det_pool[i] for i in chosen]
        k = len(chosen)
        metrics_output = evaluate_ensemble_once(sub_pool, k, method, iou_thr, conf_thr, dataloader, device)
        ens_label = "ens[" + ",".join(map(str, chosen)) + "]"
    else:
        k = ensemble_k if ensemble_k > 0 else len(idxs)
        k = max(1, min(k, len(idxs)))
        print(f"=========== evaluate ensemble over first {k} of {idxs} =============")
        sub_pool = [det_pool[i] for i in idxs]
        metrics_output = evaluate_ensemble_once(sub_pool, k, method, iou_thr, conf_thr, dataloader, device)
        ens_label = f"ens{k}"
    if metrics_output is not None:
        # 解包 12 个值
        mp, mr, map50, mf1, map50_95, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output
        rows.append(['ensemble', k, mp, mr, map50, mf1, map50_95]) # 新增
        labels.append(ens_label)
        map50_vals.append(map50)
        map50_95_vals.append(map50_95) # 新增
        f1_vals.append(mf1)
    print("=========== end ensemble =============")
    # 写CSV
    csv_path = os.path.join(out_dir, 'compare.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'idx_or_k', 'mp', 'mr', 'map50', 'mf1', 'map50_95']) # 新增
        writer.writerows(rows)
    # 画对比条形图
    if len(labels) > 0:
        x = np.arange(len(labels))
        width = 0.25 # 调窄
        plt.figure(figsize=(max(6, len(labels)*0.8), 4)) # 调整
        plt.bar(x - width, map50_vals, width, label='mAP@0.5')
        plt.bar(x, map50_95_vals, width, label='mAP@.5:.95') # 新增
        plt.bar(x + width, f1_vals, width, label='mF1')
        plt.xticks(x, labels, rotation=0)
        plt.ylabel('Score')
        plt.title('Base models vs Ensemble')
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.legend()
        fig_path = os.path.join(out_dir, 'compare.png')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        print(f"Saved compare CSV to {csv_path} and figure to {fig_path}")


def _parse_ensemble_sets(s: str, total: int):
    if not s or not s.strip():
        return []
    groups = []
    for grp in s.split(';'):
        grp = grp.strip()
        if not grp:
            continue
        idxs = _parse_indices(grp, total)
        if len(idxs) > 0:
            groups.append(idxs)
    return groups


def compare_multiple_ensembles(det_pool, base_idxs, ensemble_sets, method, iou_thr, conf_thr, dataloader, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    labels = []
    map50_vals = []
    map50_95_vals = [] # 新增
    f1_vals = []
    for i in base_idxs:
        print(f"=========== evaluate single model[{i}] =============")
        metrics_output = evaluate_single(det_pool[i], dataloader, device)
        if metrics_output is not None:
            # 解包 12 个值
            mp, mr, map50, mf1, map50_95, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output
            rows.append(['single', i, mp, mr, map50, mf1, map50_95]) # 新增
            labels.append(f"m{i}")
            map50_vals.append(map50)
            map50_95_vals.append(map50_95) # 新增
            f1_vals.append(mf1)
        print("=========== end single =============")
    for chosen in ensemble_sets:
        print(f"=========== evaluate ensemble over indices {chosen} =============")
        sub_pool = [det_pool[i] for i in chosen]
        metrics_output = evaluate_ensemble_once(sub_pool, len(sub_pool), method, iou_thr, conf_thr, dataloader, device)
        if metrics_output is not None:
            # 解包 12 个值
            mp, mr, map50, mf1, map50_95, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output
            rows.append(['ensemble', ','.join(map(str, chosen)), mp, mr, map50, mf1, map50_95]) # 新增
            labels.append('ens[' + ','.join(map(str, chosen)) + ']')
            map50_vals.append(map50)
            map50_95_vals.append(map50_95) # 新增
            f1_vals.append(mf1)
        print("=========== end ensemble =============")
    csv_path = os.path.join(out_dir, 'compare_multi.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'idx_or_set', 'mp', 'mr', 'map50', 'mf1', 'map50_95']) # 新增
        writer.writerows(rows)
    if len(labels) > 0:
        x = np.arange(len(labels))
        width = 0.25 # 调窄
        plt.figure(figsize=(max(6, len(labels)*0.8), 4)) # 调整
        plt.bar(x - width, map50_vals, width, label='mAP@0.5')
        plt.bar(x, map50_95_vals, width, label='mAP@.5:.95') # 新增
        plt.bar(x + width, f1_vals, width, label='mF1')
        plt.xticks(x, labels, rotation=0)
        plt.ylabel('Score')
        plt.title('Base models vs Multiple Ensemble Sets')
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.legend()
        fig_path = os.path.join(out_dir, 'compare_multi.png')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        print(f"Saved compare CSV to {csv_path} and figure to {fig_path}")


def sweep_custom_ensemble_sets(det_pool, ensemble_sets, method, iou_thr, conf_thr, dataloader, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    labels = []
    map50_vals = []
    map50_95_vals = [] # 新增
    f1_vals = []
    for chosen in ensemble_sets:
        print(f"=========== sweep custom ensemble over indices {chosen} =============")
        sub_pool = [det_pool[i] for i in chosen]
        metrics_output = evaluate_ensemble_once(
            det_pool=sub_pool,
            num_models=len(sub_pool),
            method=method,
            iou_thr=iou_thr,
            conf_thr=conf_thr,
            dataloader=dataloader,
            device=device,
        )
        if metrics_output is not None:
            # 解包 12 个值
            mp, mr, map50, mf1, map50_95, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output
            rows.append([','.join(map(str, chosen)), mp, mr, map50, mf1, map50_95]) # 新增
            labels.append('[' + ','.join(map(str, chosen)) + ']')
            map50_vals.append(map50)
            map50_95_vals.append(map50_95) # 新增
            f1_vals.append(mf1)
        print("=========== end custom ensemble =============")
    csv_path = os.path.join(out_dir, 'sweep_custom.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['set', 'mp', 'mr', 'map50', 'mf1', 'map50_95']) # 新增
        writer.writerows(rows)
    if len(labels) > 0:
        x = np.arange(len(labels))
        width = 0.25 # 调窄
        plt.figure(figsize=(max(6, len(labels)*0.8), 4)) # 调整
        plt.bar(x - width, map50_vals, width, label='mAP@0.5')
        plt.bar(x, map50_95_vals, width, label='mAP@.5:.95') # 新增
        plt.bar(x + width, f1_vals, width, label='mF1')
        plt.xticks(x, labels, rotation=0)
        plt.ylabel('Score')
        plt.title('Custom Ensemble Sets')
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.legend()
        fig_path = os.path.join(out_dir, 'sweep_custom.png')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        print(f"Saved custom sweep CSV to {csv_path} and figure to {fig_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default= ds, help='验证集图片目录')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--iou_thr', type=float, default=0.5, help='IoU阈值')
    parser.add_argument('--conf_thr', type=float, default=0.3, help='置信度阈值')
    parser.add_argument('--ensemble', action='store_true', help='是否使用ensemble')
    parser.add_argument('--num_models', type=int, default=2, help='参与融合的模型数量')
    parser.add_argument('--ensemble_models', type=str, default='', help='逗号分隔的模型索引（优先于num_models），如 "0,2,5"')
    parser.add_argument('--method', type=str, default='wbf', choices=['max', 'wbf'], help='融合方式: max 或 wbf')

    parser.add_argument('--sweep_max', type=int, default=0, help='若>0，则从1..sweep_max做ensemble规模扫参并可视化')
    parser.add_argument('--out_dir', type=str, default='./runs/ensemble_sweep', help='结果保存目录')
    parser.add_argument('--sweep_sets', type=str, default='', help='自定义多组ensemble索引，组间用;分隔，如 "0,2;1,3,5"')
    parser.add_argument('--sweep_iou', action='store_true', help='对IoU阈值进行扫参并可视化')
    parser.add_argument('--sweep_iou_values', type=str, default='0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7', help='逗号分隔的IoU阈值列表，如 "0.3,0.4,0.5,0.6,0.7"')
    parser.add_argument('--sweep_iou_sets', type=str, default='', help='对哪些ensemble组合进行IoU扫参，格式与 --sweep_sets 相同；为空则默认[0,1]或[0]')
    parser.add_argument('--sweep_conf', action='store_true', help='对置信度阈值进行扫参并可视化')
    parser.add_argument('--sweep_conf_values', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6', help='逗号分隔的置信度阈值列表，如 "0.2,0.3,0.4,0.5"')
    parser.add_argument('--sweep_conf_sets', type=str, default='', help='对哪些ensemble组合进行Conf扫参，格式与 --sweep_sets 相同；为空则默认[0,1]或[0]')

    parser.add_argument('--compare', action='store_true', help='对比单个基础模型与ensemble性能')
    parser.add_argument('--compare_models', type=str, default='', help='逗号分隔的模型索引，如 "0,1,2"；为空表示使用全部')
    parser.add_argument('--compare_k', type=int, default=2, help='对比时ensemble的模型数量；0表示使用compare_models长度')
    parser.add_argument('--compare_out_dir', type=str, default='./runs/compare', help='对比结果保存目录')
    parser.add_argument('--compare_ensemble_models', type=str, default='', help='对比时ensemble使用的模型索引，逗号分隔；优先于 compare_k. 多组时用;分割。示例："0,2;1,3,5;0,1,2" ')
    args = parser.parse_args()

    model_single_path = det_pool[2] if len(det_pool) > 2 else det_pool[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = DetectionDataset(args.data, 'val', use_open=True)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=False, collate_fn=DetectionDataset.collate_fn)
    class_names = getattr(dataset, 'class_names', None)
    
    # ... [compare, sweep等逻辑, 它们现在会自动处理12个返回] ...
    # [compare 和 sweep 部分的代码不需要改动，因为它们只是传递 metrics_output]
    if args.compare:
        idxs = _parse_indices(args.compare_models, len(det_pool))
        if len(idxs) == 0:
            idxs = list(range(len(det_pool)))
        ens_sets = _parse_ensemble_sets(args.compare_ensemble_sets, len(det_pool)) if 'compare_ensemble_sets' in args else []
        if args.compare_ensemble_models:
            chosen = _parse_indices(args.compare_ensemble_models, len(det_pool))
            if len(chosen) > 0:
                ens_sets = [chosen]
        if len(ens_sets) > 1:
            compare_multiple_ensembles(
                det_pool=det_pool,
                base_idxs=idxs,
                ensemble_sets=ens_sets,
                method=args.method,
                iou_thr=args.iou_thr,
                conf_thr=args.conf_thr,
                dataloader=dataloader,
                device=device,
                out_dir=args.compare_out_dir,
            )
        else:
            ens_idxs = ens_sets[0] if len(ens_sets) == 1 else []
            compare_base_vs_ensemble(
                det_pool=det_pool,
                idxs=idxs,
                ensemble_k=args.compare_k,
                method=args.method,
                iou_thr=args.iou_thr,
                conf_thr=args.conf_thr,
                dataloader=dataloader,
                device=device,
                out_dir=args.compare_out_dir,
                ens_idxs=ens_idxs if len(ens_idxs) > 0 else None,
            )
    elif args.ensemble and args.sweep_max > 0:
        if args.sweep_sets and args.sweep_sets.strip():
            ens_sets = _parse_ensemble_sets(args.sweep_sets, len(det_pool))
            if len(ens_sets) == 0:
                print('[Warn] --sweep_sets 解析为空，回退到规模扫参')
                sweep_ensemble_size(
                    det_pool=det_pool,
                    max_k=args.sweep_max,
                    method=args.method,
                    iou_thr=args.iou_thr,
                    conf_thr=args.conf_thr,
                    dataloader=dataloader,
                    device=device,
                    out_dir=args.out_dir,
                )
            else:
                sweep_custom_ensemble_sets(
                    det_pool=det_pool,
                    ensemble_sets=ens_sets,
                    method=args.method,
                    iou_thr=args.iou_thr,
                    conf_thr=args.conf_thr,
                    dataloader=dataloader,
                    device=device,
                    out_dir=args.out_dir,
                )
        else:
            sweep_ensemble_size(
                det_pool=det_pool,
                max_k=args.sweep_max,
                method=args.method,
                iou_thr=args.iou_thr,
                conf_thr=args.conf_thr,
                dataloader=dataloader,
                device=device,
                out_dir=args.out_dir,
            )
    elif args.ensemble and args.sweep_iou:
        try:
            iou_values = [float(x.strip()) for x in args.sweep_iou_values.split(',') if x.strip()]
        except Exception:
            iou_values = [0.3, 0.4, 0.5, 0.6, 0.7]
        ens_sets = _parse_ensemble_sets(args.sweep_iou_sets, len(det_pool)) if args.sweep_iou_sets else []
        sweep_iou_threshold(
            det_pool=det_pool,
            ensemble_sets=ens_sets,
            method=args.method,
            iou_values=iou_values,
            conf_thr=args.conf_thr,
            dataloader=dataloader,
            device=device,
            out_dir=args.out_dir,
        )
    elif args.ensemble and args.sweep_conf:
        try:
            conf_values = [float(x.strip()) for x in args.sweep_conf_values.split(',') if x.strip()]
        except Exception:
            conf_values = [0.2, 0.3, 0.4, 0.5]
        ens_sets = _parse_ensemble_sets(args.sweep_conf_sets, len(det_pool)) if args.sweep_conf_sets else []
        sweep_conf_threshold(
            det_pool=det_pool,
            ensemble_sets=ens_sets,
            method=args.method,
            conf_values=conf_values,
            iou_thr=args.iou_thr,
            dataloader=dataloader,
            device=device,
            out_dir=args.out_dir,
        )
    elif args.ensemble:
        print("===========evaluation ensemble =============")
        if args.ensemble_models and args.ensemble_models.strip():
            idxs = _parse_indices(args.ensemble_models, len(det_pool))
            if len(idxs) == 0:
                print("[Warn] --ensemble_models 无有效索引，回退到 num_models 模式")
                metrics_output = evaluate_ensemble_once(
                    det_pool=det_pool,
                    num_models=args.num_models,
                    method=args.method,
                    iou_thr=args.iou_thr,
                    conf_thr=args.conf_thr,
                    dataloader=dataloader,
                    device=device,
                )
            else:
                sub_pool = [det_pool[i] for i in idxs]
                metrics_output = evaluate_ensemble_once(
                    det_pool=sub_pool,
                    num_models=len(sub_pool),
                    method=args.method,
                    iou_thr=args.iou_thr,
                    conf_thr=args.conf_thr,
                    dataloader=dataloader,
                    device=device,
                )
        else:
            metrics_output = evaluate_ensemble_once(
                det_pool=det_pool,
                num_models=args.num_models,
                method=args.method,
                iou_thr=args.iou_thr,
                conf_thr=args.conf_thr,
                dataloader=dataloader,
                device=device,
            )
        if metrics_output is not None:
            # 解包 12 个值
            mp, mr, map50, mf1, map50_95, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output
            print(f"mAP@0.5:0.95: {map50_95:.4f}") # 打印新指标
            print_per_class_metrics(mp, mr, map50, mf1, p, r, ap, f1, class_ids, n_gt, n_p, class_names=class_names, dataset_len=len(dataset))
        print("===========end evaluation=============")
    else:
        for i in range(0, 10):
            model_single_path = det_pool[i]
            print(f"===========evaluation single yolo[{i}]=============")
            metrics_output = evaluate_single(model_single_path, dataloader, device)
            if metrics_output is not None:
                # 解包 12 个值
                mp, mr, map50, mf1, map50_95, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output
                print(f"mAP@0.5:0.95: {map50_95:.4f}") # 打印新指标
                print_per_class_metrics(mp, mr, map50, mf1, p, r, ap, f1, class_ids, n_gt, n_p, class_names=class_names, dataset_len=len(dataset))
            print("===========end evaluation=============")