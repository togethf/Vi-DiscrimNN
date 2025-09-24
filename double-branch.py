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
    sample_metrics = []
    classes = []
    img_size = 640  # 或根据实际图片尺寸动态获取
    for imgs, targets in tqdm(dataloader, desc="Evaluating ensemble"):
        imgs = imgs.to(device)
        targets = targets.to(device)
        # 归一化xywh转xyxy像素（与discrimnn.py一致）
        if len(targets.shape) > 1 and targets.shape[1] >= 5 and targets[:, 1:5].max() <= 1.0:
            targets[:, 1:5] = xywh2xyxy(targets[:, 1:5])
            targets[:, 1:5] *= torch.tensor([img_size, img_size, img_size, img_size], device=targets.device)
        if len(targets.shape) == 1:
            classes += []
        else:
            classes += targets[:, 0].tolist()
        batch_outputs = []
        for img in imgs:
            if img.ndim == 3:
                img = img.unsqueeze(0)
            res = model.predict(img)
            res.boxes.xyxy = res.boxes.xyxy.to(device)
            res.boxes.conf = res.boxes.conf.to(device)
            res.boxes.cls = res.boxes.cls.to(device)
            batch_outputs.append(res)
        sample_metrics += get_batch_statistics(batch_outputs, targets, device)
    if len(sample_metrics) == 0:
        print("---- No detections over whole validation set ----")
        return None
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, classes)
    return metrics_output


def evaluate_single(model_path, dataloader, device):
    model = YOLO(model_path)
    sample_metrics = []
    classes = []
    img_size = 640  # 或根据实际图片尺寸动态获取
    for imgs, targets in tqdm(dataloader, desc="Evaluating single yolo"):
        imgs = imgs.to(device)
        targets = targets.to(device)
        # 归一化xywh转xyxy像素（与ensemble一致）
        if len(targets.shape) > 1 and targets.shape[1] >= 5 and targets[:, 1:5].max() <= 1.0:
            targets[:, 1:5] = xywh2xyxy(targets[:, 1:5])
            targets[:, 1:5] *= torch.tensor([img_size, img_size, img_size, img_size], device=targets.device)
        # 将类别标签添加到classes列表中
        if len(targets.shape) == 1:
            classes += []
        else:
            classes += targets[:, 0].tolist()
        batch_outputs = []
        for img in imgs:
            if img.ndim == 3:
                img = img.unsqueeze(0)
            results = model(img, verbose=False, conf=model_predict_conf, iou=model_predict_iou)[0]
            batch_outputs.append(results)
        sample_metrics += get_batch_statistics(batch_outputs, targets, device)
    if len(sample_metrics) == 0:
        print("---- No detections ----")
        return None
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, classes)
    return metrics_output


def evaluate_ensemble_once(det_pool, num_models, method, iou_thr, conf_thr, dataloader, device):
    num_models = max(1, min(num_models, len(det_pool)))
    model_paths = det_pool[:num_models]
    ensemble_model = EnsembleYOLO(model_paths, iou_thr, conf_thr, device, method)
    return evaluate_ensemble(ensemble_model, dataloader, device)


def sweep_ensemble_size(det_pool, max_k, method, iou_thr, conf_thr, dataloader, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    xs = []
    map50s = []
    f1s = []
    csv_path = os.path.join(out_dir, 'sweep.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_models', 'mp', 'mr', 'map50', 'mf1'])
        for k in range(1, min(max_k, len(det_pool)) + 1):
            print(f"=========== sweep ensemble size = {k} =============")
            metrics_output = evaluate_ensemble_once(det_pool, k, method, iou_thr, conf_thr, dataloader, device)
            if metrics_output is not None:
                mp, mr, map50, mf1, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output
                xs.append(k)
                map50s.append(map50)
                f1s.append(mf1)
                writer.writerow([k, mp, mr, map50, mf1])
            print("=========== end sweep =============")
    if len(xs) > 0:
        plt.figure(figsize=(6,4))
        plt.plot(xs, map50s, marker='o', label='mAP@0.5')
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
    f1_vals = []
    # 单模型评测
    for i in idxs:
        print(f"=========== evaluate single model[{i}] =============")
        metrics_output = evaluate_single(det_pool[i], dataloader, device)
        if metrics_output is not None:
            mp, mr, map50, mf1, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output
            rows.append(['single', i, mp, mr, map50, mf1])
            labels.append(f"m{i}")
            map50_vals.append(map50)
            f1_vals.append(mf1)
        print("=========== end single =============")
    # ensemble 评测：若 ens_idxs 提供则用其索引，否则使用所选 idxs 的前 k 个
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
        mp, mr, map50, mf1, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output
        rows.append(['ensemble', k, mp, mr, map50, mf1])
        labels.append(ens_label)
        map50_vals.append(map50)
        f1_vals.append(mf1)
    print("=========== end ensemble =============")
    # 写CSV
    csv_path = os.path.join(out_dir, 'compare.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'idx_or_k', 'mp', 'mr', 'map50', 'mf1'])
        writer.writerows(rows)
    # 画对比条形图
    if len(labels) > 0:
        x = np.arange(len(labels))
        width = 0.38
        plt.figure(figsize=(max(6, len(labels)*0.6), 4))
        plt.bar(x - width/2, map50_vals, width, label='mAP@0.5')
        plt.bar(x + width/2, f1_vals, width, label='mF1')
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
    f1_vals = []
    for i in base_idxs:
        print(f"=========== evaluate single model[{i}] =============")
        metrics_output = evaluate_single(det_pool[i], dataloader, device)
        if metrics_output is not None:
            mp, mr, map50, mf1, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output
            rows.append(['single', i, mp, mr, map50, mf1])
            labels.append(f"m{i}")
            map50_vals.append(map50)
            f1_vals.append(mf1)
        print("=========== end single =============")
    for chosen in ensemble_sets:
        print(f"=========== evaluate ensemble over indices {chosen} =============")
        sub_pool = [det_pool[i] for i in chosen]
        metrics_output = evaluate_ensemble_once(sub_pool, len(sub_pool), method, iou_thr, conf_thr, dataloader, device)
        if metrics_output is not None:
            mp, mr, map50, mf1, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output
            rows.append(['ensemble', ','.join(map(str, chosen)), mp, mr, map50, mf1])
            labels.append('ens[' + ','.join(map(str, chosen)) + ']')
            map50_vals.append(map50)
            f1_vals.append(mf1)
        print("=========== end ensemble =============")
    csv_path = os.path.join(out_dir, 'compare_multi.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'idx_or_set', 'mp', 'mr', 'map50', 'mf1'])
        writer.writerows(rows)
    if len(labels) > 0:
        x = np.arange(len(labels))
        width = 0.38
        plt.figure(figsize=(max(6, len(labels)*0.6), 4))
        plt.bar(x - width/2, map50_vals, width, label='mAP@0.5')
        plt.bar(x + width/2, f1_vals, width, label='mF1')
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
            mp, mr, map50, mf1, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output
            rows.append([','.join(map(str, chosen)), mp, mr, map50, mf1])
            labels.append('[' + ','.join(map(str, chosen)) + ']')
            map50_vals.append(map50)
            f1_vals.append(mf1)
        print("=========== end custom ensemble =============")
    csv_path = os.path.join(out_dir, 'sweep_custom.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['set', 'mp', 'mr', 'map50', 'mf1'])
        writer.writerows(rows)
    if len(labels) > 0:
        x = np.arange(len(labels))
        width = 0.38
        plt.figure(figsize=(max(6, len(labels)*0.6), 4))
        plt.bar(x - width/2, map50_vals, width, label='mAP@0.5')
        plt.bar(x + width/2, f1_vals, width, label='mF1')
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
    parser.add_argument('--conf_thr', type=float, default=0.4, help='置信度阈值')
    parser.add_argument('--ensemble', action='store_true', help='是否使用ensemble')
    parser.add_argument('--num_models', type=int, default=2, help='参与融合的模型数量')
    parser.add_argument('--ensemble_models', type=str, default='', help='逗号分隔的模型索引（优先于num_models），如 "0,2,5"')
    parser.add_argument('--method', type=str, default='max', choices=['max', 'wbf'], help='融合方式: max 或 wbf')

    parser.add_argument('--sweep_max', type=int, default=0, help='若>0，则从1..sweep_max做ensemble规模扫参并可视化')
    parser.add_argument('--out_dir', type=str, default='./runs/ensemble_sweep', help='结果保存目录')
    parser.add_argument('--sweep_sets', type=str, default='', help='自定义多组ensemble索引，组间用;分隔，如 "0,2;1,3,5"')

    parser.add_argument('--compare', action='store_true', help='对比单个基础模型与ensemble性能')
    parser.add_argument('--compare_models', type=str, default='', help='逗号分隔的模型索引，如 "0,1,2"；为空表示使用全部')
    parser.add_argument('--compare_k', type=int, default=2, help='对比时ensemble的模型数量；0表示使用compare_models长度')
    parser.add_argument('--compare_out_dir', type=str, default='./runs/compare', help='对比结果保存目录')
    parser.add_argument('--compare_ensemble_models', type=str, default='', help='对比时ensemble使用的模型索引，逗号分隔；优先于 compare_k. 多组时用;分割。示例："0,2;1,3,5;0,1,2" ')
    args = parser.parse_args()

    # det_pool在config_plus.py中定义，包含多个模型权重路径
    model_single_path = det_pool[2] if len(det_pool) > 2 else det_pool[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = DetectionDataset(args.data, 'val', use_open=True)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=False, collate_fn=DetectionDataset.collate_fn)
    class_names = getattr(dataset, 'class_names', None)
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
        # 若提供 --sweep_sets，则按自定义组合评测；否则按规模扫参
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
            mp, mr, map50, mf1, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output
            print_per_class_metrics(mp, mr, map50, mf1, p, r, ap, f1, class_ids, n_gt, n_p, class_names=class_names, dataset_len=len(dataset))
        print("===========end evaluation=============")
    else:
        for i in [3,4,5]:
            model_single_path = det_pool[i]
            print(f"===========evaluation single yolo[{i}]=============")
            metrics_output = evaluate_single(model_single_path, dataloader, device)
            if metrics_output is not None:
                mp, mr, map50, mf1, p, r, ap, f1, class_ids, n_gt, n_p = metrics_output
                print_per_class_metrics(mp, mr, map50, mf1, p, r, ap, f1, class_ids, n_gt, n_p, class_names=class_names, dataset_len=len(dataset))
            print("===========end evaluation=============")
