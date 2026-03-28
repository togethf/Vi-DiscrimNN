import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.structures import InstanceData
from ultralytics.utils.metrics import box_iou
from config_plus import model_predict_conf, model_predict_iou

def calculate_pr_f1(tp, fp, fn):
    """Calculate Precision, Recall, and F1-score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def validate_single_mmdet(cfg_file, ckpt_file, conf_thr=0.5, iou_thr=0.7, verbose=True):
    """Validate a single MMDetection model and return metrics."""
    init_default_scope('mmdet')
    
    cfg = Config.fromfile(cfg_file)
    cfg.load_from = ckpt_file
    
    # Ensure work_dir exists
    if not hasattr(cfg, 'work_dir'):
        cfg.work_dir = './work_dirs/val_temp'
    
    runner = Runner.from_cfg(cfg)
    
    # Run standard evaluation to get mAP
    if verbose:
        print(f"\n>>> Validating: {os.path.basename(cfg_file)}")
        print("Running standard MMDetection evaluation...")
    
    metrics = runner.test()
    
    map50 = metrics.get('coco/bbox_mAP_50', metrics.get('bbox_mAP_50', 0))
    map50_95 = metrics.get('coco/bbox_mAP', metrics.get('coco/bbox_mAP', metrics.get('bbox_mAP', 0)))
    
    # Manual calculation of Precision and Recall
    if verbose:
        print(f"Calculating Precision and Recall (Conf: {conf_thr}, IoU: {iou_thr})...")
    
    model = runner.model
    model.eval()
    
    dataloader = runner.test_dataloader
    dataset = dataloader.dataset
    metainfo = getattr(dataset, 'metainfo', {})
    class_names = metainfo.get('classes', [])
    num_classes = len(class_names)
    
    class_stats = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(num_classes)}
    
    with torch.no_grad():
        for data in tqdm(dataloader, disable=not verbose):
            # Move data to device
            if hasattr(runner, 'device'):
                data = runner.model.to_tester_prepare_data(data)
            
            outputs = model.test_step(data)
            
            for i, output in enumerate(outputs):
                # Ground truth
                gt_instances = data['data_samples'][i].gt_instances
                gt_bboxes = gt_instances.bboxes
                if hasattr(gt_bboxes, 'tensor'):
                    gt_bboxes = gt_bboxes.tensor
                gt_labels = gt_instances.labels
                
                # Predictions
                pred_instances = output.pred_instances
                pred_bboxes = pred_instances.bboxes
                if hasattr(pred_bboxes, 'tensor'):
                    pred_bboxes = pred_bboxes.tensor
                pred_scores = pred_instances.scores
                pred_labels = pred_instances.labels
                
                # Filter predictions by confidence threshold
                keep = pred_scores >= conf_thr
                pred_bboxes = pred_bboxes[keep]
                pred_labels = pred_labels[keep]
                pred_scores = pred_scores[keep]
                
                # Match predictions to ground truth per class
                for cls_idx in range(num_classes):
                    cls_gt_bboxes = gt_bboxes[gt_labels == cls_idx]
                    cls_pred_bboxes = pred_bboxes[pred_labels == cls_idx]
                    cls_pred_scores = pred_scores[pred_labels == cls_idx]
                    
                    num_gt = len(cls_gt_bboxes)
                    num_pred = len(cls_pred_bboxes)
                    
                    if num_gt == 0:
                        class_stats[cls_idx]['fp'] += num_pred
                        continue
                    
                    if num_pred == 0:
                        class_stats[cls_idx]['fn'] += num_gt
                        continue
                    
                    # Sort predictions by score
                    sort_inds = torch.argsort(cls_pred_scores, descending=True)
                    cls_pred_bboxes = cls_pred_bboxes[sort_inds]
                    
                    # Ensure both are on same device (CPU) and are floats
                    cls_pred_bboxes = cls_pred_bboxes.to(device='cpu').float()
                    cls_gt_bboxes = cls_gt_bboxes.to(device='cpu').float()
                    
                    # Compute IoU matrix [num_pred, num_gt]
                    ious = box_iou(cls_pred_bboxes, cls_gt_bboxes)
                    
                    matched_gt = torch.zeros(num_gt, dtype=torch.bool)
                    tps = 0
                    for p_idx in range(num_pred):
                        iou_max, g_idx = ious[p_idx].max(0)
                        if iou_max >= iou_thr and not matched_gt[g_idx]:
                            tps += 1
                            matched_gt[g_idx] = True
                    
                    fps = num_pred - tps
                    fns = num_gt - tps
                    
                    class_stats[cls_idx]['tp'] += tps
                    class_stats[cls_idx]['fp'] += fps
                    class_stats[cls_idx]['fn'] += fns
    
    # Aggregate results
    total_tp = sum(s['tp'] for s in class_stats.values())
    total_fp = sum(s['fp'] for s in class_stats.values())
    total_fn = sum(s['fn'] for s in class_stats.values())
    
    overall_p, overall_r, overall_f1 = calculate_pr_f1(total_tp, total_fp, total_fn)
    
    if verbose:
        print("\n" + "-"*30 + " Per-Class Stats " + "-"*30)
        print(f"{'Class':<20} | {'P':<8} | {'R':<8} | {'F1':<8} | {'TP':<6} | {'FP':<6} | {'FN':<6}")
        print("-" * 80)
        for i, name in enumerate(class_names):
            stats = class_stats[i]
            p, r, f1 = calculate_pr_f1(stats['tp'], stats['fp'], stats['fn'])
            print(f"{name:<20} | {p:<8.4f} | {r:<8.4f} | {f1:<8.4f} | {stats['tp']:<6} | {stats['fp']:<6} | {stats['fn']:<6}")
        print("-" * 80)

    return {
        'P': overall_p,
        'R': overall_r,
        'F1': overall_f1,
        'mAP50': map50,
        'mAP50-95': map50_95,
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn
    }

def print_summary_table(results_list):
    """Print a summary table of all validated models."""
    print("\n" + "="*95)
    print(f"{'Model':<40} | {'P':<8} | {'R':<8} | {'F1':<8} | {'mAP50':<8} | {'mAP50-95':<8}")
    print("-" * 95)
    for res in results_list:
        name = res['name']
        p, r, f1 = res['P'], res['R'], res['F1']
        m50, m95 = res['mAP50'], res['mAP50-95']
        print(f"{name:<40} | {p:<8.4f} | {r:<8.4f} | {f1:<8.4f} | {m50:<8.4f} | {m95:<8.4f}")
    print("="*95 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate MMDetection models with P/R/F1")
    parser.add_argument("--config", type=str, help="Path to MMDetection config file")
    parser.add_argument("--checkpoint", type=str, help="Path to MMDetection checkpoint file")
    parser.add_argument("--model_idx", type=int, help="Index of a single model in det_pool")
    parser.add_argument("--model_indices", type=str, help="Comma-separated indices in det_pool (e.g. 5,6,7)")
    parser.add_argument("--all_mmdet", action="store_true", help="Validate all MMDetection models in det_pool")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="IoU threshold for matching")
    
    args = parser.parse_args()
    
    conf = args.conf if args.conf is not None else model_predict_conf
    iou = args.iou if args.iou is not None else model_predict_iou
    
    from config_plus import det_pool
    
    targets = []
    
    if args.config and args.checkpoint:
        targets.append({'cfg': args.config, 'ckpt': args.checkpoint, 'name': os.path.basename(args.config)})
    elif args.model_idx is not None:
        model_str = det_pool[args.model_idx]
        if ";" in model_str:
            c, ck = model_str.split(";")
            targets.append({'cfg': c, 'ckpt': ck, 'name': f"det_pool[{args.model_idx}]"})
    elif args.model_indices:
        indices = [int(i.strip()) for i in args.model_indices.split(",")]
        for idx in indices:
            model_str = det_pool[idx]
            if ";" in model_str:
                c, ck = model_str.split(";")
                targets.append({'cfg': c, 'ckpt': ck, 'name': f"det_pool[{idx}]"})
    elif args.all_mmdet:
        for idx, model_str in enumerate(det_pool):
            if ";" in model_str:
                c, ck = model_str.split(";")
                targets.append({'cfg': c, 'ckpt': ck, 'name': f"det_pool[{idx}]"})
    
    if not targets:
        print("Error: No models specified. Use --config/--checkpoint, --model_idx, --model_indices, or --all_mmdet.")
        exit(1)
        
    all_results = []
    for target in targets:
        try:
            res = validate_single_mmdet(target['cfg'], target['ckpt'], conf_thr=conf, iou_thr=iou)
            res['name'] = target['name']
            all_results.append(res)
        except Exception as e:
            print(f"Error validating {target['name']}: {e}")
            
    if len(all_results) > 0:
        print_summary_table(all_results)
