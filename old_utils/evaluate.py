import numpy as np
from ultralytics import YOLO
from .utils import extract_label_full
import matplotlib.pyplot as plt
import cv2
import os
from joblib import Parallel, delayed
from tqdm import tqdm

# 计算 IOU（Intersection over Union）
def compute_iou(box1, box2):
    """
    计算两个边界框的 IOU（交并比）。
    
    参数:
    - box1: 第一个边界框，格式为 [x_min, y_min, x_max, y_max]
    - box2: 第二个边界框，格式为 [x_min, y_min, x_max, y_max]
    
    返回值:
    - 两个边界框的 IOU 值
    """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

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
            iou = compute_iou(pred['bbox'], true['bbox'])
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

# 这种ap计算方式的精度会更低点，最终没有采用
def calculate_ap(precisions, recalls):
    """
    计算平均精度（AP）。
    
    参数:
    - precisions: 精度列表
    - recalls: 召回率列表
    
    返回值:
    - AP 值
    """
    precisions = np.concatenate(([0], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))
    
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
        
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap

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
    
    for class_id in tqdm(pred_boxes_per_class.keys()):
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

# 计算 mAP @50-90
def calculate_map_50_90(pred_boxes_per_class, true_boxes_per_class):
    aps = []
    for iou_threshold in np.linspace(0.5, 0.9, 5):
        map_iou = calculate_map(pred_boxes_per_class, true_boxes_per_class, iou_threshold=iou_threshold)
        aps.append(map_iou)
    return np.mean(aps)



# 可视化预测结果和真实框
def visualize_predictions(image_path, true_boxes_per_class, pred_boxes_per_class):
    """
    可视化真实边界框和预测边界框。
    
    参数:
    - image_path: 图像文件路径
    - true_boxes_per_class: 每个类别的真实框
    - pred_boxes_per_class: 每个类别的预测框
    """
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 绘制真实框
    for class_id, boxes in true_boxes_per_class.items():
        for box in boxes:
            bbox = box['bbox']
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)  # 绿色

    # 绘制预测框
    for class_id, boxes in pred_boxes_per_class.items():
        for box in boxes:
            bbox = box['bbox']
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)  # 蓝色

    plt.imshow(image)
    plt.axis('off')
    plt.title('True Boxes (Green) and Predicted Boxes (Blue)')
    plt.show()

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

def val_single_image(model, img_path, label_path, verbose=True):
    """
    对单张图片进行预测并计算 mAP@50、mAP@50-90、Precision 和 Recall。
    
    参数:
    - model: YOLO 模型
    - img_path: 图片路径
    - label_path: 标签路径
    
    返回值:
    - map_50: mAP@50
    - map_50_90: mAP@50-90
    - precision: 精度
    - recall: 召回率
    """
    # 读取图片的宽度和高度
    image = cv2.imread(img_path)
    img_height, img_width = image.shape[:2]

    # 预测
    results = model.predict(img_path, save=False, verbose=False)
    pred_boxes_per_class = format_result(results)

    # 获取标签并根据实际宽高转换
    labels = extract_label_full(label_path)
    true_boxes_per_class = format_gt(labels, img_width, img_height)

    # 计算 mAP@50 和 mAP@50-90
    map_50 = calculate_map(pred_boxes_per_class, true_boxes_per_class, iou_threshold=0.5)
    map_50_90 = calculate_map_50_90(pred_boxes_per_class, true_boxes_per_class)

    # 计算 Precision 和 Recall（使用 IOU 阈值为 0.5）
    precision, recall = calculate_precision_recall(
        pred_boxes=[box for class_boxes in pred_boxes_per_class.values() for box in class_boxes], 
        true_boxes=[box for class_boxes in true_boxes_per_class.values() for box in class_boxes],
        iou_threshold=0.5
    )
    if verbose:
        print(f'Single Image mAP@50: {map_50:.4f}')
        print(f'Single Image mAP@50-90: {map_50_90:.4f}')
        print(f'Single Image Precision: {precision:.4f}')
        print(f'Single Image Recall: {recall:.4f}')
        visualize_predictions(img_path, true_boxes_per_class, pred_boxes_per_class)

    return map_50, map_50_90, precision, recall


def val(model, img_paths, label_paths):
    all_pred_boxes_per_class = {}
    all_true_boxes_per_class = {}

    def process_image(img_path, label_path):
        image = cv2.imread(img_path)
        img_height, img_width = image.shape[:2]
        results = model.predict(img_path, save=False, verbose=False)
        output = format_result(results)
        labels = extract_label_full(label_path)
        gt = format_gt(labels, img_width, img_height)
        return output, gt

    results = Parallel(n_jobs=-1)(delayed(process_image)(img, label) for img, label in zip(img_paths, label_paths))

    for output, gt in results:
        for class_id, boxes in output.items():
            if class_id not in all_pred_boxes_per_class:
                all_pred_boxes_per_class[class_id] = []
            all_pred_boxes_per_class[class_id].extend(boxes)
        for class_id, boxes in gt.items():
            if class_id not in all_true_boxes_per_class:
                all_true_boxes_per_class[class_id] = []
            all_true_boxes_per_class[class_id].extend(boxes)

    # Calculate mAP
    map_50 = calculate_map(all_pred_boxes_per_class, all_true_boxes_per_class, iou_threshold=0.5)
    map_50_90 = calculate_map_50_90(all_pred_boxes_per_class, all_true_boxes_per_class)
    return map_50, map_50_90


def val_dataset(model, images_dir, labels_dir):
    """
    测试整个数据集并计算 mAP。
    
    参数:
    - model: YOLO 模型
    - images_dir: 图片文件夹路径
    - labels_dir: 标签文件夹路径
    
    返回值:
    - 数据集的 mAP@50 和 mAP@50-90
    """
    all_pred_boxes_per_class = {}
    all_true_boxes_per_class = {}
    
    img_files = os.listdir(images_dir)
    
    for img_file in img_files:
        # 获取图片路径和标签路径
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, img_file.replace('.jpg', '.txt'))
        
        # 读取图片的宽度和高度
        image = cv2.imread(img_path)
        img_height, img_width = image.shape[:2]

        # 预测
        results = model.predict(img_path, save=False, verbose=False)
        output = format_result(results)

        # 获取标签并根据实际宽高转换
        labels = extract_label_full(label_path)
        gt = format_gt(labels, img_width, img_height)
        
        # 累积所有图片的预测结果
        for class_id, boxes in output.items():
            if class_id not in all_pred_boxes_per_class:
                all_pred_boxes_per_class[class_id] = []
            all_pred_boxes_per_class[class_id].extend(boxes)
        
        # 累积所有图片的真实标签
        for class_id, boxes in gt.items():
            if class_id not in all_true_boxes_per_class:
                all_true_boxes_per_class[class_id] = []
            all_true_boxes_per_class[class_id].extend(boxes)

    # 计算 mAP@50 和 mAP@50-90
    map_50 = calculate_map(all_pred_boxes_per_class, all_true_boxes_per_class, iou_threshold=0.5)
    map_50_90 = calculate_map_50_90(all_pred_boxes_per_class, all_true_boxes_per_class)

    return map_50, map_50_90

def __example_single_image__():
    img_path = r"E:\code\exp\animals\valid\images\216_jpgrf9eebf7c5f9c6c981e8eb31d607b348b4.jpg"
    label_path = r"E:\code\exp\animals\valid\labels\216_jpgrf9eebf7c5f9c6c981e8eb31d607b348b4.txt"
    
    model = YOLO(r'E:\code\mainline\runs\detect\animal8n\weights\best.pt')
    
    map_50, map_50_90, precision, recall = val_single_image(model, img_path, label_path)

# 示例使用
def __example_dataset__():
    images_dir = r"G:\science_data\datasets\RicePestsV3\VOCdevkit\VOC2007\images"
    labels_dir = r"G:\science_data\datasets\RicePestsV3\VOCdevkit\VOC2007\images"
    
    model = YOLO(r'E:\code\mainline\runs\detect\v3\train3\weights\best.pt')
    
    map_50, map_50_90 = val_dataset(model, images_dir, labels_dir)
    
    print(f'Dataset mAP@50: {map_50:.4f}')
    print(f'Dataset mAP@50-90: {map_50_90:.4f}')


def __example_check_onebyone__():
    images_dir = r"E:\code\exp\animals\valid\images"
    labels_dir = r"E:\code\exp\animals\valid\labels"
    
    model = YOLO(r'E:\code\mainline\runs\detect\animal8n\weights\best.pt')
    img_paths = [os.path.join(images_dir, img) for img in os.listdir(images_dir)]
    label_paths = [os.path.join(labels_dir, label) for label in os.listdir(labels_dir)]
    for img_path, label_path in zip(img_paths, label_paths):
        val_single_image(model, img_path, label_path)
        print("======================== next loop =========================")
