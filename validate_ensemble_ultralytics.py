"""
在 Ultralytics 原生验证管道中评估 Ensemble/双分支 模型的 mAP。

功能:
    1. 加载多个 YOLO 权重，对每张图用多模型推理并融合检测框
    2. 融合结果交给 Ultralytics 原生的 DetectionValidator 计算 mAP
    3. 保证 mAP 数值与 model.val() 完全对齐（相同的预处理、坐标还原、TP匹配、AP计算）

用法:
    # 单模型验证（用于与 model.val() 对比确认管道一致）
    python validate_ensemble_ultralytics.py \
        --models /path/to/yolo11n.pt \
        --data cfg/datasets/pest/v3.yaml

    # 多模型 ensemble 验证
    python validate_ensemble_ultralytics.py \
        --models /path/to/model1.pt /path/to/model2.pt /path/to/model3.pt \
        --data cfg/datasets/pest/v3.yaml \
        --method wbf \
        --ensemble_iou 0.5 \
        --ensemble_conf 0.3

    # 使用 config_plus.py 中的模型池
    python validate_ensemble_ultralytics.py \
        --use_det_pool \
        --data cfg/datasets/pest/v3.yaml
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode


# ─────────────────────────────── Ensemble 融合工具函数 ─────────────────────────────── #

def _compute_iou_single(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    """计算两个 xyxy 框的 IoU (都是 1-d tensor, shape [4])."""
    iou = box_iou(box_a.unsqueeze(0), box_b.unsqueeze(0))
    return float(iou[0, 0].item())


def _fuse_detections(
    all_dets: List[Tuple[torch.Tensor, float, float, int, float]],
    iou_thr: float = 0.5,
    method: str = "max",
    min_models: int = 1,
    score_power: float = 1.0,
) -> List[Tuple[torch.Tensor, float, float]]:
    """
    将多模型的检测结果通过 IoU 贪心聚类 + 融合合并为最终检测。
    ★ 按类别分组后再融合，避免跨类合并。

    Args:
        all_dets: list of (box_xyxy [4], conf, cls, model_id, model_weight)
        iou_thr: 聚类的 IoU 阈值
        method: 'max' (取最高置信度框) 或 'wbf' (加权平均框)
        min_models: 一个 cluster 至少来自多少个模型才保留
        score_power: 置信度权重指数，>1 会更偏向高置信度框

    Returns:
        融合后的 list of (box_xyxy [4], conf, cls)
    """
    if len(all_dets) == 0:
        return []

    # ★ 按类别分组
    from collections import defaultdict
    class_groups: Dict[int, List[Tuple[torch.Tensor, float, int, float]]] = defaultdict(list)
    for box, conf, cls, model_id, model_weight in all_dets:
        class_groups[int(cls)].append((box, conf, model_id, model_weight))

    fused = []
    method = method.lower()

    for cls_id, dets in class_groups.items():
        # 在同一类别内做 IoU 贪心聚类
        clusters: List[List[Tuple[torch.Tensor, float, int, float]]] = []
        for box, conf, model_id, model_weight in dets:
            assigned = False
            for cluster in clusters:
                if any(_compute_iou_single(box, m[0]) > iou_thr for m in cluster):
                    cluster.append((box, conf, model_id, model_weight))
                    assigned = True
                    break
            if not assigned:
                clusters.append([(box, conf, model_id, model_weight)])

        # 融合每个 cluster
        for cluster in clusters:
            if len(cluster) == 0:
                continue
            cluster_model_ids = {c[2] for c in cluster}
            if len(cluster_model_ids) < min_models:
                continue
            if method == "wbf" and len(cluster) > 1:
                raw_scores = torch.tensor([c[1] for c in cluster], dtype=torch.float32)
                model_weights = torch.tensor([c[3] for c in cluster], dtype=torch.float32)
                weights = (raw_scores.clamp(min=0) ** score_power) * model_weights
                weights = weights / (weights.sum() + 1e-9)
                boxes = torch.stack([c[0] for c in cluster], dim=0).float()
                weights = weights.to(boxes.device)
                fused_box = (boxes * weights[:, None]).sum(dim=0)
                scaled_conf = [
                    min(c[1] * c[3], 1.0) for c in cluster
                ]
                fused_conf = float(max(scaled_conf))
                fused.append((fused_box, fused_conf, float(cls_id)))
            else:
                best = max(cluster, key=lambda x: x[1] * x[3])
                fused_conf = float(min(best[1] * best[3], 1.0))
                fused.append((best[0], fused_conf, float(cls_id)))

    return fused


def _classwise_nms(
    bboxes: torch.Tensor,
    confs: torch.Tensor,
    clss: torch.Tensor,
    iou_thr: float,
) -> torch.Tensor:
    """简单的 class-wise NMS，返回保留索引."""
    if bboxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=bboxes.device)

    keep_indices: List[int] = []
    classes = clss.unique()
    for cls_id in classes:
        idx = (clss == cls_id).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        boxes_c = bboxes[idx]
        scores_c = confs[idx]
        order = scores_c.argsort(descending=True)
        while order.numel() > 0:
            i = order[0].item()
            keep_indices.append(idx[i].item())
            if order.numel() == 1:
                break
            rest = order[1:]
            ious = box_iou(
                boxes_c[i].unsqueeze(0).float(),
                boxes_c[rest].float(),
            ).squeeze(0)
            order = rest[ious <= iou_thr]

    return torch.tensor(keep_indices, device=bboxes.device, dtype=torch.long)


@contextmanager
def _temp_log_level(level: int):
    prev = LOGGER.level
    LOGGER.setLevel(level)
    try:
        yield
    finally:
        LOGGER.setLevel(prev)


# ─────────────────────────── EnsembleDetectionValidator ─────────────────────────── #

class EnsembleDetectionValidator(DetectionValidator):
    """
    继承 Ultralytics DetectionValidator，用多模型 ensemble 替换单模型推理，
    其余的预处理、坐标还原、TP 匹配和 mAP 计算全部使用 Ultralytics 原生逻辑。
    """

    def __init__(
        self,
        model_paths: List[str],
        ensemble_iou: float = 0.5,
        ensemble_conf: float = 0.0,  # ensemble 内部的 conf 过滤（建议设0，让AP曲线完整）
        ensemble_method: str = "max",
        model_weights: List[float] | None = None,
        min_models: int = 1,
        score_power: float = 1.0,
        final_nms_iou: float | None = None,
        final_nms_conf: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_paths = model_paths
        self.ensemble_iou = ensemble_iou
        self.ensemble_conf = ensemble_conf
        self.ensemble_method = ensemble_method
        self.model_weights = model_weights or [1.0 for _ in model_paths]
        self.min_models = min_models
        self.score_power = score_power
        self.final_nms_iou = final_nms_iou
        self.final_nms_conf = final_nms_conf
        self._models: List[AutoBackend] = []

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        重写验证主循环:
        - 单模型: 使用与 model.val() 完全相同的推理路径 (batch inference + postprocess)
        - 多模型: 每个模型分别推理 + postprocess，然后融合检测结果
        - 融合结果交给 Ultralytics 的 update_metrics (坐标还原 + TP 匹配 + mAP 计算)
        """
        self.training = False
        callbacks.add_integration_callbacks(self)

        # 用第一个模型初始化 AutoBackend（主要为了获取 stride, names 等元数据）
        primary_model = AutoBackend(
            weights=self.model_paths[0],
            device=select_device(self.args.device, self.args.batch),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
        )
        self.device = primary_model.device
        self.args.half = primary_model.fp16
        stride = primary_model.stride
        imgsz = check_imgsz(self.args.imgsz, stride=stride)

        # 加载所有模型
        self._models = [primary_model]
        for mp in self.model_paths[1:]:
            m = AutoBackend(
                weights=mp,
                device=self.device,
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self._models.append(m)

        self._is_single = len(self._models) == 1
        LOGGER.info(f"Loaded {len(self._models)} model(s) for ensemble validation")
        if len(self.model_weights) != len(self._models):
            raise ValueError(
                f"model_weights 数量({len(self.model_weights)})与模型数量({len(self._models)})不一致"
            )

        # 检查数据集
        from ultralytics.data.utils import check_det_dataset
        if str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
            self.data = check_det_dataset(self.args.data)
        else:
            raise FileNotFoundError(f"Dataset '{self.args.data}' not found")

        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0

        self.stride = stride
        self.dataloader = self.dataloader or self.get_dataloader(
            self.data.get(self.args.split), self.args.batch
        )

        # Eval & warmup
        for m in self._models:
            m.eval()
            m.warmup(imgsz=(1, self.data.get("channels", 3), imgsz, imgsz))

        self.run_callbacks("on_val_start")

        from ultralytics.utils.ops import Profile
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )

        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(primary_model))
        self.jdict = []

        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i

            # Preprocess (Ultralytics 标准: letterbox, normalize, to device)
            with dt[0]:
                batch = self.preprocess(batch)
            img = batch["img"]

            # ──── 推理 + 后处理 ────
            with dt[1]:
                if self._is_single:
                    # ★ 单模型: 完全复用标准 DetectionValidator 的代码路径
                    raw_preds = self._models[0](img)
                    preds = self.postprocess(raw_preds)
                else:
                    # ★ 多模型: 逐图推理 → postprocess → 融合
                    batch_size = img.shape[0]
                    preds = []
                    for si in range(batch_size):
                        single_img = img[si : si + 1]  # [1, C, H, W]

                        # 收集所有模型的 post-NMS 检测结果
                        all_dets: List[Tuple[torch.Tensor, float, float, int, float]] = []
                        for mi, m in enumerate(self._models):
                            raw = m(single_img)
                            processed = self.postprocess(raw)  # 标准 NMS，返回 List[Dict]
                            pred_dict = processed[0]  # 单图, {bboxes, conf, cls, extra}
                            for j in range(len(pred_dict["cls"])):
                                box = pred_dict["bboxes"][j]    # tensor [4], 在模型输入空间
                                conf = float(pred_dict["conf"][j])
                                cls = float(pred_dict["cls"][j])
                                if conf >= self.ensemble_conf:
                                    all_dets.append((box, conf, cls, mi, float(self.model_weights[mi])))

                        # 融合多模型检测
                        fused = _fuse_detections(
                            all_dets,
                            iou_thr=self.ensemble_iou,
                            method=self.ensemble_method,
                            min_models=self.min_models,
                            score_power=self.score_power,
                        )

                        # 转回 Dict 格式
                        if len(fused) > 0:
                            bboxes = torch.stack([f[0] for f in fused])
                            confs = torch.tensor(
                                [f[1] for f in fused], device=self.device, dtype=img.dtype
                            )
                            clss = torch.tensor(
                                [f[2] for f in fused], device=self.device, dtype=img.dtype
                            )
                            if self.final_nms_iou is not None and self.final_nms_iou > 0:
                                keep = _classwise_nms(bboxes, confs, clss, self.final_nms_iou)
                                if keep.numel() > 0:
                                    bboxes = bboxes[keep]
                                    confs = confs[keep]
                                    clss = clss[keep]
                                else:
                                    bboxes = bboxes[:0]
                                    confs = confs[:0]
                                    clss = clss[:0]
                            if self.final_nms_conf > 0:
                                keep = confs >= self.final_nms_conf
                                bboxes = bboxes[keep]
                                confs = confs[keep]
                                clss = clss[keep]
                            preds.append({
                                "bboxes": bboxes,
                                "conf": confs,
                                "cls": clss,
                                "extra": torch.empty((len(bboxes), 0), device=self.device),
                            })
                        else:
                            preds.append({
                                "bboxes": torch.zeros((0, 4), device=self.device),
                                "conf": torch.zeros((0,), device=self.device),
                                "cls": torch.zeros((0,), device=self.device),
                                "extra": torch.empty((0, 0), device=self.device),
                            })

            # 用 Ultralytics 原生的 update_metrics (坐标还原 + TP 匹配)
            self.update_metrics(preds, batch)

            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")

        stats = self.get_stats()
        self.speed = dict(
            zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt))
        )
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")

        LOGGER.info(
            "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                *tuple(self.speed.values())
            )
        )

        return stats


# ─────────────────────────────── CLI ─────────────────────────────── #

def _is_yolo_model(path: str) -> bool:
    """判断是否为 YOLO/Ultralytics 模型 (排除 MMDetection 等)."""
    return path.strip().endswith(".pt") and ";" not in path


def _is_mmdet_model(path: str) -> bool:
    """判断是否为 MMDetection 模型 (config;checkpoint)."""
    return ";" in path


def _parse_mmdet_model(path: str) -> Tuple[str, str]:
    cfg, ckpt = [p.strip() for p in path.split(";", 1)]
    return cfg, ckpt


def _validate_mmdet(
    cfg_path: str,
    ckpt_path: str,
    pr_iou: float = 0.5,
    pr_conf: float = 0.001,
    pr_avg: str = "micro",
) -> Dict[str, float | None]:
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"MMDet config not found: {cfg_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"MMDet checkpoint not found: {ckpt_path}")

    try:
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmengine.runner import Runner
        from mmengine.structures import InstanceData
    except Exception as e:
        raise RuntimeError(
            "未检测到 MMDetection 环境，请先安装 mmdet+mmengine"
        ) from e

    init_default_scope("mmdet")
    cfg = Config.fromfile(cfg_path)
    cfg.load_from = ckpt_path
    if not getattr(cfg, "work_dir", None):
        cfg.work_dir = "work_dirs/codex_mmdet_eval"

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    def _mean_ignore_neg(x: np.ndarray) -> float | None:
        valid = x[x > -1]
        if valid.size == 0:
            return None
        return float(valid.mean())

    def _extract_pr_from_coco_eval(coco_eval: Any) -> Tuple[float | None, float | None]:
        if coco_eval is None:
            return None, None
        eval_dict = None
        if isinstance(coco_eval, dict):
            eval_dict = coco_eval
        elif hasattr(coco_eval, "eval"):
            eval_dict = coco_eval.eval

        if not eval_dict or "precision" not in eval_dict or "recall" not in eval_dict:
            return None, None

        precision = eval_dict["precision"]
        recall = eval_dict["recall"]
        if precision is None or recall is None:
            return None, None

        iou_thrs = None
        area_labels = None
        max_dets = None
        if hasattr(coco_eval, "params"):
            params = coco_eval.params
            iou_thrs = getattr(params, "iouThrs", None)
            area_labels = getattr(params, "areaRngLbl", None)
            max_dets = getattr(params, "maxDets", None)
        if iou_thrs is None and isinstance(eval_dict, dict):
            iou_thrs = eval_dict.get("iouThrs", None)

        iou_idx = 0
        if iou_thrs is not None:
            iou_thrs = np.array(iou_thrs)
            match = np.where(np.isclose(iou_thrs, 0.5))[0]
            if match.size > 0:
                iou_idx = int(match[0])

        area_idx = 0
        if area_labels is not None and "all" in area_labels:
            area_idx = int(area_labels.index("all"))

        maxdet_idx = -1
        if max_dets is not None and len(max_dets) > 0:
            maxdet_idx = int(len(max_dets) - 1)

        # precision shape: [TxRxKxAxM]
        p = precision[iou_idx, :, :, area_idx, maxdet_idx]
        # recall shape: [TxKxAxM]
        r = recall[iou_idx, :, area_idx, maxdet_idx]
        return _mean_ignore_neg(p), _mean_ignore_neg(r)

    def _get_coco_metric_from_runner() -> Any:
        evaluator = getattr(runner, "test_evaluator", None)
        if evaluator is None:
            return None
        metrics_list = getattr(evaluator, "metrics", None) or getattr(evaluator, "_metrics", None)
        if not metrics_list:
            return None
        for m in metrics_list:
            name = m.__class__.__name__.lower()
            if "coco" in name:
                return m
        return None

    coco_metric = _get_coco_metric_from_runner()
    coco_eval = None
    if coco_metric is not None:
        coco_eval = getattr(coco_metric, "coco_eval", None) or getattr(coco_metric, "_coco_eval", None)
        if coco_eval is None and hasattr(coco_metric, "results"):
            try:
                coco_metric.compute_metrics(coco_metric.results)
                coco_eval = getattr(coco_metric, "coco_eval", None) or getattr(coco_metric, "_coco_eval", None)
            except Exception:
                coco_eval = None
    pr_from_coco = _extract_pr_from_coco_eval(coco_eval)
    def _compute_pr_from_confmat() -> Tuple[float | None, float | None]:
        evaluator = getattr(runner, "test_evaluator", None)
        dataloader = getattr(runner, "test_dataloader", None)
        model = getattr(runner, "model", None)
        if dataloader is None or model is None:
            return None, None

        dataset = getattr(dataloader, "dataset", None)
        class_names = None
        if dataset is not None:
            meta = getattr(dataset, "metainfo", None) or getattr(dataset, "METAINFO", None)
            if meta and "classes" in meta:
                class_names = list(meta["classes"])
        num_classes = len(class_names) if class_names else None

        tp = None
        fp = None
        fn = None

        def _ensure_class_buffers(n: int):
            nonlocal tp, fp, fn
            if tp is None:
                tp = torch.zeros((n,), dtype=torch.long)
                fp = torch.zeros((n,), dtype=torch.long)
                fn = torch.zeros((n,), dtype=torch.long)

        model.eval()
        with torch.no_grad():
            for data in dataloader:
                outputs = model.test_step(data)
                # outputs: list of DetDataSample
                for sample, out in zip(data["data_samples"], outputs):
                    gt_instances = sample.gt_instances
                    pred_instances = out.pred_instances

                    gt_bboxes = gt_instances.bboxes
                    gt_labels = gt_instances.labels
                    pred_bboxes = pred_instances.bboxes
                    pred_scores = pred_instances.scores
                    pred_labels = pred_instances.labels

                    if hasattr(gt_bboxes, "tensor"):
                        gt_bboxes = gt_bboxes.tensor
                    if hasattr(pred_bboxes, "tensor"):
                        pred_bboxes = pred_bboxes.tensor

                    if num_classes is None:
                        max_label = 0
                        if gt_labels.numel() > 0:
                            max_label = max(max_label, int(gt_labels.max().item()))
                        if pred_labels.numel() > 0:
                            max_label = max(max_label, int(pred_labels.max().item()))
                        _ensure_class_buffers(max_label + 1)
                    else:
                        _ensure_class_buffers(num_classes)

                    # filter by conf
                    keep = pred_scores >= pr_conf
                    pred_bboxes = pred_bboxes[keep]
                    pred_scores = pred_scores[keep]
                    pred_labels = pred_labels[keep]

                    for c in range(tp.numel()):
                        gt_mask = gt_labels == c
                        pred_mask = pred_labels == c
                        gt_c = gt_bboxes[gt_mask].cpu()
                        pred_c = pred_bboxes[pred_mask].cpu()
                        pred_s = pred_scores[pred_mask].cpu()
                        if gt_c.numel() == 0 and pred_c.numel() == 0:
                            continue
                        if gt_c.numel() == 0:
                            fp[c] += int(pred_c.shape[0])
                            continue
                        if pred_c.numel() == 0:
                            fn[c] += int(gt_c.shape[0])
                            continue

                        if pred_c.numel() > 0:
                            order = torch.argsort(pred_s, descending=True)
                            pred_c = pred_c[order]
                        ious = box_iou(pred_c.float(), gt_c.float())
                        # greedy match by IoU
                        matched_gt = torch.zeros((gt_c.shape[0],), dtype=torch.bool)
                        for pi in range(pred_c.shape[0]):
                            iou_row = ious[pi]
                            max_iou, gi = iou_row.max(dim=0)
                            if max_iou >= pr_iou and not matched_gt[gi]:
                                tp[c] += 1
                                matched_gt[gi] = True
                            else:
                                fp[c] += 1
                        fn[c] += int((~matched_gt).sum().item())

        if tp is None:
            return None, None

        if pr_avg == "micro":
            tps = int(tp.sum().item())
            fps = int(fp.sum().item())
            fns = int(fn.sum().item())
            if tps + fps == 0 or tps + fns == 0:
                return None, None
            mp = tps / max(tps + fps, 1)
            mr = tps / max(tps + fns, 1)
            return float(mp), float(mr)
        else:
            valid = (tp + fn) > 0
            if valid.sum().item() == 0:
                return None, None
            precision = tp.float() / (tp + fp).clamp(min=1)
            recall = tp.float() / (tp + fn).clamp(min=1)
            mp = precision[valid].mean().item()
            mr = recall[valid].mean().item()
            return mp, mr

    if pr_from_coco == (None, None):
        try:
            evaluator = getattr(runner, "test_evaluator", None)
            dataloader = getattr(runner, "test_dataloader", None)
            model = getattr(runner, "model", None)
            if evaluator is not None and dataloader is not None and model is not None:
                if hasattr(evaluator, "reset"):
                    evaluator.reset()
                model.eval()
                with torch.no_grad():
                    for data in dataloader:
                        outputs = model.test_step(data)
                        evaluator.process(data, outputs)
                evaluator.evaluate(len(dataloader.dataset))
                coco_metric = _get_coco_metric_from_runner()
                if coco_metric is not None:
                    coco_eval = getattr(coco_metric, "coco_eval", None) or getattr(coco_metric, "_coco_eval", None)
                    pr_from_coco = _extract_pr_from_coco_eval(coco_eval)
        except Exception:
            pass
    if pr_from_coco == (None, None):
        pr_from_coco = _compute_pr_from_confmat()

    def _pick(keys: List[str]) -> float | None:
        for k in keys:
            if k in metrics:
                return float(metrics[k])
        return None

    map50 = _pick(["coco/bbox_mAP_50", "bbox_mAP_50", "mAP_50"])
    map50_95 = _pick(["coco/bbox_mAP", "bbox_mAP", "mAP"])
    mar = _pick(["coco/bbox_mAR", "bbox_mAR", "mAR"])

    return {
        "P": pr_from_coco[0],
        "R": pr_from_coco[1] if pr_from_coco[1] is not None else mar,
        "mAP50": map50,
        "mAP50-95": map50_95,
    }


def _model_short_name(path: str) -> str:
    """从模型路径中提取简短名称."""
    import os
    parts = path.replace("\\", "/").rstrip("/").split("/")
    # 尝试提取有意义的目录名 (例如 yolov11n-C3k2-HetConv2)
    for i, p in enumerate(parts):
        if p == "weights" and i > 0:
            return parts[i - 1]
    return os.path.basename(path)


def main():
    parser = argparse.ArgumentParser(
        description="使用 Ultralytics 原生验证管道评估 Ensemble 模型的 mAP"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="模型权重路径列表，如 model1.pt model2.pt model3.pt"
    )
    parser.add_argument(
        "--use_det_pool", action="store_true",
        help="使用 config_plus.py 中 det_pool 定义的模型"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Ultralytics 数据集 YAML 路径，如 cfg/datasets/pest/v3.yaml"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="推理图片尺寸")
    parser.add_argument("--batch", type=int, default=1, help="batch size（ensemble 模式建议 1）")
    parser.add_argument("--device", type=str, default="", help="设备, 如 0 或 cpu")
    parser.add_argument(
        "--method", type=str, default="max", choices=["max", "wbf"],
        help="多模型融合方式: max(取最高置信度) 或 wbf(加权框融合)"
    )
    parser.add_argument("--ensemble_iou", type=float, default=0.5, help="Ensemble 融合的 IoU 阈值")
    parser.add_argument("--ensemble_conf", type=float, default=0.0, help="Ensemble 内部 conf 过滤阈值 (建议设 0)")
    parser.add_argument(
        "--model_weights", type=str, default="",
        help="模型权重(逗号分隔), 如 '1.0,0.8,1.2'，为空则全为 1.0"
    )
    parser.add_argument(
        "--min_models", type=int, default=1,
        help="一个融合 cluster 至少来自多少个模型才保留"
    )
    parser.add_argument(
        "--score_power", type=float, default=1.0,
        help="WBF 权重里的置信度指数，>1 更偏向高置信度"
    )
    parser.add_argument(
        "--final_nms_iou", type=float, default=-1.0,
        help="融合后再做一次 class-wise NMS 的 IoU 阈值，<=0 表示关闭"
    )
    parser.add_argument(
        "--final_nms_conf", type=float, default=0.0,
        help="融合后再做一次 conf 过滤阈值"
    )
    parser.add_argument(
        "--conf", type=float, default=0.001,
        help="NMS 置信度阈值（与 Ultralytics val 对齐默认 0.001）"
    )
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--half", action="store_true", help="使用 FP16")
    parser.add_argument("--plots", action="store_true", help="保存验证可视化图")
    parser.add_argument("--single_val", action="store_true", help="同时运行单模型 model.val() 对比")
    parser.add_argument(
        "--validate_each", action="store_true",
        help="对列表中每个模型分别运行 model.val() 并汇总结果"
    )
    parser.add_argument(
        "--model_indices", type=str, default="",
        help="选择 det_pool 中的模型索引 (逗号分隔), 如 '0,3' 表示双分支"
    )
    parser.add_argument(
        "--auto_tune", action="store_true",
        help="自动搜索融合参数(以 mAP50-95 最大为目标, 同时 mAP50 不低于单模型)"
    )
    parser.add_argument(
        "--baseline_map50", type=float, default=-1.0,
        help="单模型 baseline mAP50(若提供则不重复跑单模型)"
    )
    parser.add_argument(
        "--grid_ensemble_iou", type=str, default="0.4,0.5,0.6",
        help="ensemble_iou 搜索范围"
    )
    parser.add_argument(
        "--grid_min_models", type=str, default="1,2",
        help="min_models 搜索范围"
    )
    parser.add_argument(
        "--grid_score_power", type=str, default="1.0,1.5,2.0",
        help="score_power 搜索范围"
    )
    parser.add_argument(
        "--grid_final_nms_iou", type=str, default="0.45,0.5,0.55",
        help="final_nms_iou 搜索范围(<=0 表示关闭)"
    )
    parser.add_argument(
        "--grid_final_nms_conf", type=str, default="0.0,0.001",
        help="final_nms_conf 搜索范围"
    )
    parser.add_argument(
        "--mmdet_pr_iou", type=float, default=0.5,
        help="MMDet 计算 P/R 的 IoU 阈值(混淆矩阵)"
    )
    parser.add_argument(
        "--mmdet_pr_conf", type=float, default=0.001,
        help="MMDet 计算 P/R 的置信度阈值(混淆矩阵)"
    )
    parser.add_argument(
        "--mmdet_pr_avg", type=str, default="micro", choices=["micro", "macro"],
        help="MMDet 计算 P/R 的平均方式: micro(全局) 或 macro(按类平均)"
    )

    args = parser.parse_args()

    # 获取模型列表
    if args.use_det_pool:
        from config_plus import det_pool
        all_model_paths = det_pool
    elif args.models:
        all_model_paths = args.models
    else:
        parser.error("请指定 --models 或 --use_det_pool")

    # 筛选模型索引
    if args.model_indices:
        indices = [int(x.strip()) for x in args.model_indices.split(",") if x.strip()]
        model_paths = [all_model_paths[i] for i in indices if 0 <= i < len(all_model_paths)]
    else:
        model_paths = all_model_paths


    # ─── 模式 1: 逐个模型验证 ───
    if args.validate_each:
        # 过滤出 YOLO / MMDet 模型
        yolo_models = [(i, p) for i, p in enumerate(all_model_paths) if _is_yolo_model(p)]
        mmdet_models = [(i, p) for i, p in enumerate(all_model_paths) if _is_mmdet_model(p)]

        print("=" * 80)
        print("逐个模型验证 (Ultralytics model.val())")
        print("=" * 80)
        print(f"YOLO 模型: {len(yolo_models)} 个")
        print(f"MMDet 模型: {len(mmdet_models)} 个")
        print()

        results_table = []
        for idx, path in yolo_models:
            name = _model_short_name(path)
            print(f"\n{'━' * 60}")
            print(f"[{idx}] {name}")
            print(f"    {path}")
            print(f"{'━' * 60}")
            try:
                m = YOLO(path)
                res = m.val(
                    data=args.data,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device=args.device or None,
                    conf=args.conf,
                    iou=args.iou,
                    half=args.half,
                )
                results_table.append({
                    "idx": idx,
                    "name": name,
                    "backend": "YOLO",
                    "P": res.box.mp,
                    "R": res.box.mr,
                    "mAP50": res.box.map50,
                    "mAP50-95": res.box.map,
                })
            except Exception as e:
                print(f"  ✗ 验证失败: {e}")
                results_table.append({
                    "idx": idx,
                    "name": name,
                    "backend": "YOLO",
                    "P": 0, "R": 0, "mAP50": 0, "mAP50-95": 0,
                    "error": str(e),
                })

        for idx, path in mmdet_models:
            name = _model_short_name(path)
            cfg_path, ckpt_path = _parse_mmdet_model(path)
            print(f"\n{'━' * 60}")
            print(f"[{idx}] {name} (MMDet)")
            print(f"    {cfg_path}")
            print(f"    {ckpt_path}")
            print(f"{'━' * 60}")
            try:
                res = _validate_mmdet(
                    cfg_path,
                    ckpt_path,
                    pr_iou=args.mmdet_pr_iou,
                    pr_conf=args.mmdet_pr_conf,
                    pr_avg=args.mmdet_pr_avg,
                )
                results_table.append({
                    "idx": idx,
                    "name": name,
                    "backend": "MMDet",
                    "P": res.get("P"),
                    "R": res.get("R"),
                    "mAP50": res.get("mAP50"),
                    "mAP50-95": res.get("mAP50-95"),
                })
            except Exception as e:
                print(f"  ✗ 验证失败: {e}")
                results_table.append({
                    "idx": idx,
                    "name": name,
                    "backend": "MMDet",
                    "P": None, "R": None, "mAP50": None, "mAP50-95": None,
                    "error": str(e),
                })

        # 打印汇总表格
        print("\n\n" + "=" * 80)
        print("验证结果汇总")
        print("=" * 80)
        print(f"{'Idx':>4} {'Model':<35} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}")
        print("-" * 80)
        def _fmt(v: float | None, width: int) -> str:
            if v is None:
                return f"{'-':>{width}}"
            return f"{v:>{width}.4f}"
        for r in results_table:
            if "error" in r:
                print(f"{r['idx']:>4} {r['name']:<35} {'FAILED':>8}")
            else:
                print(
                    f"{r['idx']:>4} {r['name']:<35} "
                    f"{_fmt(r['P'],8)} {_fmt(r['R'],8)} {_fmt(r['mAP50'],8)} {_fmt(r['mAP50-95'],10)}"
                )
        print("=" * 80)

        return

    # ─── 过滤非 YOLO 模型 ───
    non_yolo = [p for p in model_paths if not _is_yolo_model(p)]
    model_paths = [p for p in model_paths if _is_yolo_model(p)]
    if non_yolo:
        print(f"⚠ 跳过 {len(non_yolo)} 个非 YOLO 模型:")
        for p in non_yolo:
            print(f"   {p}")
        print()

    if len(model_paths) == 0:
        print("✗ 没有可用的 YOLO 模型")
        return

    # 解析模型权重
    if args.model_weights:
        model_weights = [float(x.strip()) for x in args.model_weights.split(",") if x.strip()]
        if len(model_weights) != len(model_paths):
            raise ValueError(
                f"--model_weights 数量({len(model_weights)})与模型数量({len(model_paths)})不一致"
            )
    else:
        model_weights = [1.0 for _ in model_paths]

    print("=" * 60)
    print("Ensemble 模型验证 (Ultralytics 原生管道)")
    print("=" * 60)
    print(f"模型数量: {len(model_paths)}")
    for i, p in enumerate(model_paths):
        print(f"  [{i}] {_model_short_name(p):30s}  {p}")
    print(f"融合方法: {args.method}")
    print(f"数据集: {args.data}")
    print(f"NMS conf: {args.conf}, NMS iou: {args.iou}")
    print(f"Ensemble IoU: {args.ensemble_iou}, Ensemble conf filter: {args.ensemble_conf}")
    print(f"Model weights: {model_weights}")
    print(f"Min models per cluster: {args.min_models}, Score power: {args.score_power}")
    if args.final_nms_iou > 0:
        print(f"Final NMS: iou={args.final_nms_iou}, conf={args.final_nms_conf}")
    print("=" * 60)

    # ─── 单模型对照测试 (可选) ───
    if args.single_val and len(model_paths) >= 1:
        print("\n━━━ 单模型 model.val() 结果 (对照) ━━━")
        single_model = YOLO(model_paths[0])
        single_results = single_model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device or None,
            conf=args.conf,
            iou=args.iou,
            half=args.half,
            plots=args.plots,
        )
        print(f"model.val() mAP50:     {single_results.box.map50:.4f}")
        print(f"model.val() mAP50-95:  {single_results.box.map:.4f}")
        print()

    def _build_cfg():
        overrides = {
            "data": args.data,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device or None,
            "conf": args.conf,
            "iou": args.iou,
            "half": args.half,
            "plots": args.plots,
            "rect": True,  # ★ 关键: 与 YOLO.val() 的默认值对齐, 否则 mAP 会偏低
            "task": "detect",
            "mode": "val",
            "model": model_paths[0],  # 用于 cfg 解析
        }
        return get_cfg(overrides=overrides)

    if args.auto_tune:
        print("\n━━━ 自动搜索融合参数 ━━━")

        if args.baseline_map50 > 0:
            baseline_map50 = args.baseline_map50
        else:
            print("计算单模型 baseline mAP50 ...")
            baseline_map50 = 0.0
            for p in model_paths:
                m = YOLO(p)
                res = m.val(
                    data=args.data,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device=args.device or None,
                    conf=args.conf,
                    iou=args.iou,
                    half=args.half,
                )
                baseline_map50 = max(baseline_map50, float(res.box.map50))

        def _parse_floats(s: str) -> List[float]:
            return [float(x.strip()) for x in s.split(",") if x.strip()]

        def _parse_ints(s: str) -> List[int]:
            return [int(x.strip()) for x in s.split(",") if x.strip()]

        grid_ensemble_iou = _parse_floats(args.grid_ensemble_iou)
        grid_min_models = _parse_ints(args.grid_min_models)
        grid_score_power = _parse_floats(args.grid_score_power)
        grid_final_nms_iou = _parse_floats(args.grid_final_nms_iou)
        grid_final_nms_conf = _parse_floats(args.grid_final_nms_conf)

        best = None
        best_stats = None
        total = (
            len(grid_ensemble_iou)
            * len(grid_min_models)
            * len(grid_score_power)
            * len(grid_final_nms_iou)
            * len(grid_final_nms_conf)
        )
        tried = 0
        for ensemble_iou in grid_ensemble_iou:
            for min_models in grid_min_models:
                for score_power in grid_score_power:
                    for final_nms_iou in grid_final_nms_iou:
                        for final_nms_conf in grid_final_nms_conf:
                            tried += 1
                            cfg = _build_cfg()
                            validator = EnsembleDetectionValidator(
                                model_paths=model_paths,
                                ensemble_iou=ensemble_iou,
                                ensemble_conf=args.ensemble_conf,
                                ensemble_method=args.method,
                                model_weights=model_weights,
                                min_models=min_models,
                                score_power=score_power,
                                final_nms_iou=final_nms_iou if final_nms_iou > 0 else None,
                                final_nms_conf=final_nms_conf,
                                args=cfg,
                            )
                            with _temp_log_level(30):
                                stats = validator()

                            map50 = stats.get("metrics/mAP50(B)", 0)
                            map50_95 = stats.get("metrics/mAP50-95(B)", 0)
                            ok = map50 >= baseline_map50
                            tag = "OK " if ok else "LOW"
                            print(
                                f"[{tried}/{total}] {tag} "
                                f"mAP50={map50:.4f} mAP50-95={map50_95:.4f} | "
                                f"ensemble_iou={ensemble_iou} min_models={min_models} "
                                f"score_power={score_power} final_nms_iou={final_nms_iou} "
                                f"final_nms_conf={final_nms_conf}"
                            )

                            if ok:
                                if best_stats is None or map50_95 > best_stats["map50_95"]:
                                    best = (ensemble_iou, min_models, score_power, final_nms_iou, final_nms_conf)
                                    best_stats = {"map50": map50, "map50_95": map50_95}

        print("\n=== 搜索完成 ===")
        print(f"baseline mAP50: {baseline_map50:.4f}")
        if best_stats is None:
            print("未找到满足 mAP50 >= baseline 的组合")
        else:
            print(
                f"Best mAP50-95={best_stats['map50_95']:.4f} "
                f"mAP50={best_stats['map50']:.4f}"
            )
            print(
                f"Params: ensemble_iou={best[0]}, min_models={best[1]}, "
                f"score_power={best[2]}, final_nms_iou={best[3]}, final_nms_conf={best[4]}"
            )
        return

    # ─── Ensemble 验证 ───
    print("\n━━━ Ensemble 验证 (Ultralytics 原生管道) ━━━")

    # 构建 cfg
    cfg = _build_cfg()

    validator = EnsembleDetectionValidator(
        model_paths=model_paths,
        ensemble_iou=args.ensemble_iou,
        ensemble_conf=args.ensemble_conf,
        ensemble_method=args.method,
        model_weights=model_weights,
        min_models=args.min_models,
        score_power=args.score_power,
        final_nms_iou=args.final_nms_iou if args.final_nms_iou > 0 else None,
        final_nms_conf=args.final_nms_conf,
        args=cfg,
    )

    stats = validator()

    print("\n" + "=" * 60)
    print("Ensemble 验证结果:")
    print("=" * 60)
    if stats:
        map50 = stats.get("metrics/mAP50(B)", 0)
        map50_95 = stats.get("metrics/mAP50-95(B)", 0)
        precision = stats.get("metrics/precision(B)", 0)
        recall = stats.get("metrics/recall(B)", 0)
        print(f"  Precision:   {precision:.4f}")
        print(f"  Recall:      {recall:.4f}")
        print(f"  mAP@0.5:     {map50:.4f}")
        print(f"  mAP@0.5:0.95:{map50_95:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
