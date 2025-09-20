from ultralytics import YOLO
from config_plus import det_pool, ds, model_predict_conf, model_predict_iou
from commons.metrics import bbox_iou, get_batch_statistics, ap_per_class, xywh2xyxy
from commons.dataset import DetectionDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm

def ensemble_yolo(img, model1, model2, iou_thr=0.5, conf_thr=0.4, device='cpu'):
    # 单张图片推理并融合
    results1 = model1(img, verbose=False, iou=model_predict_iou, conf=model_predict_conf)[0]
    results2 = model2(img, verbose=False, iou=model_predict_iou, conf=model_predict_conf)[0]
    dets1 = list(zip(results1.boxes.xyxy.cpu().numpy(), results1.boxes.conf.cpu().numpy(), results1.boxes.cls.cpu().numpy()))
    dets2 = list(zip(results2.boxes.xyxy.cpu().numpy(), results2.boxes.conf.cpu().numpy(), results2.boxes.cls.cpu().numpy()))

    used2 = set()
    final_dets = []

    for box1, conf1, cls1 in dets1:
        matched = False
        for idx2, (box2, conf2, cls2) in enumerate(dets2):
            if idx2 in used2:
                continue
            # 使用metrics.py的bbox_iou
            iou = bbox_iou(
                torch.tensor(box1, dtype=torch.float32, device=device).unsqueeze(0),
                torch.tensor(box2, dtype=torch.float32, device=device).unsqueeze(0)
            )[0].item()
            if iou > iou_thr:
                matched = True
                used2.add(idx2)
                # 类别一致或不一致都保留置信度高的
                if conf1 >= conf2:
                    final_dets.append((box1, conf1, cls1))
                else:
                    final_dets.append((box2, conf2, cls2))
                break
        if not matched and conf1 >= conf_thr:
            final_dets.append((box1, conf1, cls1))
    for idx2, (box2, conf2, cls2) in enumerate(dets2):
        if idx2 not in used2 and conf2 >= conf_thr:
            final_dets.append((box2, conf2, cls2))
    return final_dets

class EnsembleYOLO:
    def __init__(self, model1_path, model2_path, iou_thr=0.5, conf_thr=0.4, device='cpu'):
        self.model1 = YOLO(model1_path)
        self.model2 = YOLO(model2_path)
        self.iou_thr = iou_thr
        self.conf_thr = conf_thr
        self.device = device

    def predict(self, img):
        dets = ensemble_yolo(img, self.model1, self.model2, self.iou_thr, self.conf_thr, self.device)
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
            xyxy = torch.tensor([d[0] for d in dets])
            conf = torch.tensor([d[1] for d in dets])
            cls = torch.tensor([d[2] for d in dets])
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default= ds, help='验证集图片目录')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--iou_thr', type=float, default=0.5, help='IoU阈值')
    parser.add_argument('--conf_thr', type=float, default=0.4, help='置信度阈值')
    parser.add_argument('--ensemble', type=bool, default=False, help='是否使用ensemble')
    args = parser.parse_args()

    # det_pool在config_plus.py中定义，包含两个模型权重路径
    model1_path = det_pool[0]
    model2_path = det_pool[1]
    model_single_path = det_pool[2]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = DetectionDataset(args.data, 'val', open=True)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=False, collate_fn=DetectionDataset.collate_fn)
    if args.ensemble:
        print("===========evaluation ensemble =============")
        ensemble_model = EnsembleYOLO(model1_path, model2_path, args.iou_thr, args.conf_thr, device)
        metrics_output = evaluate_ensemble(ensemble_model, dataloader, device)
        if metrics_output is not None:
            print("Precision: ", metrics_output[0])
            print("Recall: ", metrics_output[1])
            print("mAP50: ", metrics_output[2])
            print("F1 Score: ", metrics_output[3])
        print("===========end evaluation=============")
    else:
        for i in [2, 3, 4]:
            model_single_path = det_pool[i]
            print(f"===========evaluation single yolo[{i}]=============")
            metrics_output = evaluate_single(model_single_path, dataloader, device)
            if metrics_output is not None:
                print("Precision: ", metrics_output[0])
                print("Recall: ", metrics_output[1])
                print("mAP50: ", metrics_output[2])
                print("F1 Score: ", metrics_output[3])
            print("===========end evaluation=============")
