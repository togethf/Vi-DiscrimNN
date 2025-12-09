import numpy as np
from commons.dataset import DetectionDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from ultralytics import YOLO
from tqdm import tqdm
import os
from commons.det_utils import cal_iou
from PIL import Image
from config import *
from commons.utils import save, extract_label_full
import argparse
from config import *
def check_val_model(opt):
    """

    Args:
        opt (_type_): _description_

    Raises:
        ValueError: _description_
    """
    version_list = [5, 11]
    if opt in version_list:
        return
    else:
        raise ValueError(f"Invalid val model version {opt.val_model}")
 
def parse(opt):
    """解析命令行参数

    Args:
        opt (_type_): 命令行参数

    Returns:
        tuple: data_config, model_config - 划分使用的配置（voc 或 pest）
    """
    # 定义数据配置映射字典
    data_config_map = {
        'voc12': voc12_config,
        'voc07': voc07_config,
        'pestv3': pestv3_config
    }

    # 根据 opt.dataset 获取对应的数据配置
    if opt.dataset in data_config_map:
        data_config = data_config_map[opt.dataset]
    else:
        raise ValueError(f"Invalid dataset: {opt.dataset}. Available options are {', '.join(data_config_map.keys())}.")

    # 定义模型配置映射字典
    model_config_map = {
        'voc12': voc12_config,
        'voc07': voc07_config,
        'pestv3': pestv3_config
    }

    # 根据 opt.model 获取对应的模型配置
    if opt.model in model_config_map:
        model_config = model_config_map[opt.model]
    else:
        raise ValueError(f"Invalid model: {opt.model}. Available options are {', '.join(model_config_map.keys())}.")

    return data_config, model_config

def prepare_det(weight):
    model = YOLO(weight)
    return model

# 格式化模型预测结果
def format_result(results):
    """
    将 YOLO 模型的预测结果格式化为按类别存储的字典。
    
    参数:
    - results: 模型的预测结果
    
    返回值:
    - 按类别存储的预测框字典
    """
    output_per_class = {}
    for result in results:
        sample = dict()
        pred_boxes = result.boxes.xyxy # bbox
        pred_cls = result.boxes.cls
        pred_conf = result.boxes.conf
        for index in range(pred_cls.shape[0]):
            cls_id = int(pred_cls[index].item())
            sample = {
                'conf': float(pred_conf[index].item()),
                'bbox': [float(x) for x in pred_boxes[index].tolist()]
            }
            # 按类别存储预测结果
            if cls_id not in output_per_class:
                output_per_class[cls_id] = []
            output_per_class[cls_id].append(sample)
    return output_per_class

# 计算TP, FP, FN
def calculate_precision_recall(pred_boxes, true_boxes, iou_threshold=0.5):
    """
    计算预测结果的精度（Precision）和召回率（Recall）。
    
    参数:
    - pred_boxes: 预测的边界框列表
    - true_boxes: 真实的边界框列表
    - iou_threshold: 用于判断匹配的 IOU 阈值
    
    返回值:
    - precision: 精度
    - recall: 召回率
    """
    TP, FP, FN = 0, 0, 0
    
    matched_true_boxes = []
    
    for pred in pred_boxes:
        iou_max = 0
        match = None
        for true in true_boxes:
            iou = cal_iou(pred['bbox'], true['bbox'])
            if iou > iou_max:
                iou_max = iou
                match = true
                
        if iou_max >= iou_threshold and match not in matched_true_boxes:
            TP += 1
            matched_true_boxes.append(match)
        else:
            FP += 1

    FN = len(true_boxes) - len(matched_true_boxes)
    
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    
    return precision, recall

# ultralytics采用的是这种计算ap的方式，实际使用发现精度更高，最终使用这种方式，最后return 的内容我作了修改，注释未改
def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    # return ap, mpre, mrec
    return ap

# 计算mAP
def calculate_map(pred_boxes_per_class, true_boxes_per_class, iou_threshold=0.5):
    """
    计算所有类别的平均平均精度（mAP）。
    
    参数:
    - pred_boxes_per_class: 每个类别的预测框列表
    - true_boxes_per_class: 每个类别的真实框列表
    - iou_threshold: 用于匹配的 IOU 阈值
    
    返回值:
    - mAP 值
    """
    aps = []
    
    for class_id in pred_boxes_per_class.keys():
        pred_boxes = pred_boxes_per_class[class_id]

        # 检查是否有对应的真实框
        true_boxes = true_boxes_per_class.get(class_id, [])
        
        precisions, recalls = [], []
        
        # 排序预测框根据置信度降序
        pred_boxes = sorted(pred_boxes, key=lambda x: x['conf'], reverse=True)
        
        for i in range(1, len(pred_boxes) + 1):
            pred_filtered = pred_boxes[:i]
            precision, recall = calculate_precision_recall(pred_filtered, true_boxes, iou_threshold)
            precisions.append(precision)
            recalls.append(recall)
            
        ap = compute_ap(np.array(precisions), np.array(recalls))
        aps.append(ap)
    
    return np.mean(aps) if aps else 0.0  # 如果没有有效的 AP，返回 0

# 格式化真实标签
def format_gt(gt_labels, img_width, img_height):
    """
    将 YOLO 标签格式化为绝对坐标的边界框，并按类别存储。
    
    参数:
    - gt_labels: 原始 YOLO 标签
    - img_width: 图像宽度
    - img_height: 图像高度
    
    返回值:
    - 按类别存储的真实边界框字典
    """
    true_boxes_per_class = {}

    for label in gt_labels:
        class_id, x_center_norm, y_center_norm, width_norm, height_norm = label

        # 转换归一化坐标为绝对坐标
        x_center = x_center_norm * img_width
        y_center = y_center_norm * img_height
        width = width_norm * img_width
        height = height_norm * img_height

        # 计算 x_min, y_min, x_max, y_max
        x_min = x_center - (width / 2)
        y_min = y_center - (height / 2)
        x_max = x_center + (width / 2)
        y_max = y_center + (height / 2)

        # 创建 bbox 和类的映射
        bbox = [x_min, y_min, x_max, y_max]
        if class_id not in true_boxes_per_class:
            true_boxes_per_class[class_id] = []
        
        true_boxes_per_class[class_id].append({'bbox': bbox})

    return true_boxes_per_class

def validate(model, dconfig):
    """验证YOLO模型在难易数据集和整个数据集上的表现

    Args:
        model (YOLO): 加载好的YOLO模型
        dconfig (dict): config类中配置好的数据字典
    """

    model.val(data=dconfig['cfg'])
    print("validate on whole done ")

    model.val(data=dconfig['cfg'].replace('.yaml', '_easy.yaml'))
    print("validate on easy done")

    model.val(data=dconfig['cfg'].replace('.yaml', '_diff.yaml'))
    print("validate on diff done")

def main():
    parser = argparse.ArgumentParser(description='Calculate mAP for each image in a dataset')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IOU threshold for mAP calculation')
    parser.add_argument('--dataset', type=str, default='pestv3', help='选择划分哪个数据集：voc12/voc07/pestv3')
    parser.add_argument('--model', type=str, default='pestv3', help='选择用哪个系列的检测器来划分数据集:voc12/voc07/pestv3')
    parser.add_argument('--validate', type=bool, default=False, help='会决定是划分数据集还是验证')
    parser.add_argument('--series', type=int, default=11, help='采用的YOLO Version')
    opt = parser.parse_args()
    dconfig, mconfig = parse(opt)
    # 判断是进行数据划分还是验证效果
    if opt.validate:
        val_wmodel_dir = os.path.dirname(mconfig['weak_detector'])
        val_smodel_dir = os.path.dirname(mconfig['strong_detector'])
        val_model_series = opt.series
        check_val_model(val_model_series)
        weak_val = YOLO(os.path.join(val_wmodel_dir, f"{val_model_series}n.pt"))
        strong_val = YOLO(os.path.join(val_smodel_dir, f"{val_model_series}m.pt"))
        print("=================== begin to validate weak model on datasets ==================")
        validate(weak_val, dconfig)
        print("=================== begin to validate strong model on datasets ==================")
        validate(strong_val, dconfig)
        print("=================== end validation ==================")  
    else:
        # 加载模型和数据集
        weak = prepare_det(mconfig['weak_detector'])
        strong = prepare_det(mconfig['strong_detector'])
        dataset = DetectionDataset(dconfig['source_images'], dconfig['source_labels'], 'val')
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

        easy_imgs = []
        easy_labels = []
        diff_imgs = []
        diff_labels = []

        # 划分数据集 begin
        for images, labels in dataloader:
            def single_map(model, image, label, pred_boxes_per_class, true_boxes_per_class):
                # 模型预测
                results = model.predict(image, save=False, verbose=False)
                output = format_result(results)
                lbs = extract_label_full(label)

                target_shape = Image.open(image).size
                
                gt = format_gt(lbs, *target_shape)
                for class_id, boxes in output.items():
                    if class_id not in pred_boxes_per_class:
                        pred_boxes_per_class[class_id] = []
                    pred_boxes_per_class[class_id].extend(boxes)
                for class_id, boxes in gt.items():
                    if class_id not in true_boxes_per_class:
                        true_boxes_per_class[class_id] = []
                    true_boxes_per_class[class_id].extend(boxes)
                # 计算单张图片的 mAP
                map_50 = calculate_map(pred_boxes_per_class, true_boxes_per_class, iou_threshold=0.5)
                return map_50
                # 初始化结果存储
            weak_pred_boxes_per_class = {}
            weak_true_boxes_per_class = {}
            strong_pred_boxes_per_class = {}
            strong_true_boxes_per_class = {}
            image = images[0]
            label = labels[0]
            weak_map = single_map(weak, image, label,weak_pred_boxes_per_class, weak_true_boxes_per_class)
            strong_map = single_map(strong, image, label, strong_pred_boxes_per_class, strong_true_boxes_per_class)
            if strong_map > weak_map:
                print(f"weak: {weak_map} < strong: {strong_map}------difficult")
                diff_imgs.append(image)
                diff_labels.append(label)
            else:
                print(f"weak: {weak_map} >= strong: {strong_map}------easy")
                easy_imgs.append(image)
                easy_labels.append(label)
        # 保存划分好的难易数据集
        save(easy_imgs, easy_labels, tag_config['output_easy_dir'])
        save(diff_imgs, diff_labels, tag_config['output_diff_dir']) 


if __name__ == '__main__':
    main()