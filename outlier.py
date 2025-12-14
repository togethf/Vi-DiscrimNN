import numpy as np
import os
import argparse
import os
from commons.dataset import DetectionDataset
from commons.utils import save, extract_label_full
from config import IMGSZ
from commons.det_utils import cal_iou
from tqdm import tqdm
from commons.proutils import get_model, parse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from commons.metrics import *
import torch

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
        
        true_boxes_per_class[class_id].append(bbox)

    return true_boxes_per_class

def find_outliers_dict(models, judge=None, iou_threshold=0.5): 
    """
    检测多个模型的输出中，基于字典格式的离群值
    :param models: List of models' outputs in dictionary format {"class": [bbox1, bbox2, ...]}
    :param judge: List of gt in dictionary format {"class": [bbox1, bbox2, ...]}
    :param iou_threshold: IoU threshold for matching
    :return: List of outliers for each model in dictionary format
    """
    all_outliers = [{} for _ in models]  # 每个模型的离群值，用字典存储
    if judge:
        # 遍历每个模型
        for model_idx, model_outputs in enumerate(models):
            for cls_a, bboxes_a in model_outputs.items():
                for box_a in bboxes_a:
                    matched = False

                    # 检查judge有无这个类别
                    if cls_a in judge:
                        for box_b in judge[cls_a]:
                            if cal_iou(box_a, box_b) >= iou_threshold:
                                matched = True
                                break
                    if not matched:
                        if cls_a not in all_outliers[model_idx]:
                            all_outliers[model_idx][cls_a] = []
                        all_outliers[model_idx][cls_a].append(box_a)
                    
    else:
        # 遍历每个模型
        for model_idx, model_outputs in enumerate(models):
            for cls_a, bboxes_a in model_outputs.items():
                for box_a in bboxes_a:
                    matched = False
                    
                    # 检查其他模型的对应类别
                    for other_idx, other_model in enumerate(models):
                        if other_idx == model_idx:
                            continue
                        if cls_a in other_model:  # 类别存在
                            for box_b in other_model[cls_a]:
                                if cal_iou(box_a, box_b) >= iou_threshold:
                                    matched = True
                                    break
                        if matched:
                            break
                    
                    # 如果没有匹配，将该框记录为离群值
                    if not matched:
                        if cls_a not in all_outliers[model_idx]:
                            all_outliers[model_idx][cls_a] = []
                        all_outliers[model_idx][cls_a].append(box_a)

    return all_outliers

def format_result(results):
    """将YOLO模型的检测结果转换成期望的字典格式

    Args:
        results (YOLO Results):YOLO模型的检测结果

    Returns:
        dict: dict[cls_id] -> bbox:list
    """
    output_per_class = {}
    for result in results:
        sample = dict()
        pred_boxes = result.boxes.xyxy
        pred_cls = result.boxes.cls
        for index in range(pred_cls.shape[0]):
            cls_id = int(pred_cls[index].item())
            sample = [float(x) for x in pred_boxes[index].tolist()]
            if cls_id not in output_per_class:
                output_per_class[cls_id] = []
            output_per_class[cls_id].append(sample)
    return output_per_class

def detect(img, format=True, *models):
    """汇总所有model的检测结果

    Args:
        img (str): 图片路径
        format (bool, optional): 是否将检测结果格式成字典的形式. Defaults to True.

    Returns:
        list: list of dict if format, else list of Result
    """
    results = []
    for model in models:
        result = model.predict(img, verbose=False)
        if format:
            result = format_result(result)
        results.append(result)
    return results

    
def get_median(data):
    """计算data的中位数

    Args:
        data (list): 需要计算中位数的list

    Returns:
        int: 计算出的中位数
    """
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half]) / 2

def validate(model, dconfig):
    """验证YOLO模型在难易数据集和整个数据集上的表现

    Args:
        model (YOLO): 加载好的YOLO模型
        dconfig (dict): config类中配置好的数据字典
    """

    # metrics_w = model.val(data=dconfig['cfg'], verbose=False)
    # print("validate on whole done, map50: ", metrics_w.box.map50)

    metrics_e = model.val(data=dconfig['cfg'].replace('.yaml', '_easy.yaml'), verbose=False)
    print("validate on easy done, map50: ", metrics_e.box.map50)

    metrics_d = model.val(data=dconfig['cfg'].replace('.yaml', '_diff.yaml'), verbose=False)
    print("validate on diff done, map50: ", metrics_d.box.map50)
    return metrics_e.box.map50, metrics_d.box.map50

def __evaluation(model, val_dataloader, device):
    """主要由本文件内部调用
    """
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    pbar = tqdm(val_dataloader)
    classes = []
    total_num = 0
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
        output = model.predict(imgs, verbose=False)
        total_num += imgs.shape[0]
        pbar.set_description("Evaluation model:") 
        sample_metrics += get_batch_statistics(output, labels, device)
    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, classes)
    return metrics_output

def val(model, dconfig):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    X_e = dconfig['output_easy_dir'].replace('easy', os.path.join('easy', 'images'))
    dataset = DetectionDataset(X_e, 'val', use_open=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=DetectionDataset.collate_fn)
    performance_e = __evaluation(model, dataloader, device)
    print("validate on easy done, map50: ", performance_e[2])
    X_d = dconfig['output_diff_dir'].replace('diff', os.path.join('diff', 'images'))
    dataset = DetectionDataset(X_d, 'val', use_open=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=DetectionDataset.collate_fn)
    performance_d = __evaluation(model, dataloader, device)
    print("validate on diff done, map50: ", performance_d[2])
    return performance_e[2], performance_d[2]

def dynamic_mAP50(edge_score, cloud_score, ratio):
    return edge_score * ratio + cloud_score * (1-ratio)

def save_map_curves(e_map, d_map, e_map_x, d_map_x, save_path):
    # 解构数据
    ratios_easy, map_easy = zip(*e_map)
    ratios_easy, map_diff = zip(*d_map)
    ratios_easy_x, map_easy_x = zip(*e_map_x)
    ratios_easy_x, map_diff_x = zip(*d_map_x)

    dynamic_ap = []
    for ratio, ape, apd in zip(ratios_easy, map_easy, map_diff_x):
        dynamic_ap.append(dynamic_mAP50(ape, apd, ratio))
    
    # 计算差值并找到最大差值的 ratio
    diff_easy = [abs(de - ee) for de, ee in zip(map_diff, map_easy)]
    diff_easy_x = [abs(dx - ex) for dx, ex in zip(map_diff_x, map_easy_x)]
    max_diff_easy_idx = diff_easy.index(max(diff_easy))
    max_diff_easy_x_idx = diff_easy_x.index(max(diff_easy_x))

    # 找到两个差值和最大的 ratio
    combined_diff = [de + dx for de, dx in zip(diff_easy, diff_easy_x)]
    max_combined_diff_idx = combined_diff.index(max(combined_diff))

    print(f"map_diff 和 map_easy 差值最大时的 ratio: {ratios_easy[max_diff_easy_idx]}, 差值: {diff_easy[max_diff_easy_idx]}")
    print(f"map_diff_x 和 map_easy_x 差值最大时的 ratio: {ratios_easy_x[max_diff_easy_x_idx]}, 差值: {diff_easy_x[max_diff_easy_x_idx]}")
    print(f"两个差值和最大时的 ratio: {ratios_easy[max_combined_diff_idx]}, 差值和: {combined_diff[max_combined_diff_idx]}")


    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制简单图片的 mAP 曲线
    plt.plot(ratios_easy, map_easy, label='Easy - weak', marker='o')
    plt.plot(ratios_easy_x, map_easy_x, label='Easy - strong', marker='x', linestyle='--')
    
    # 绘制困难图片的 mAP 曲线
    plt.plot(ratios_easy, map_diff, label='Hard - weak', marker='o')
    plt.plot(ratios_easy_x, map_diff_x, label='Hard - strong', marker='x', linestyle='--')

    # 绘制dynamic_map50
    # plt.plot(ratios_easy, dynamic_ap, label='Dynamic_mAP', marker='*')
    
    # 添加标题和标签
    # plt.title("mAP Curves for different ratio of easy samples")
    plt.xlabel("Ratio of easy Images")
    plt.ylabel("mAP")
    plt.legend()
    plt.grid(True)
    
    # 保存图像到本地
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保保存路径存在
    plt.savefig(save_path, dpi=300)  # 高分辨率保存
    plt.close()  # 关闭图像，释放内存

    print(f"图像已保存至: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='find outlier based method to tag the difficulty of imgs')
    parser.add_argument('--dataset', type=str, default='pestv3', help='选择划分哪个数据集：voc12/voc07/coco/pestv3/visdrone/pestv1/ip102/pest24')
    parser.add_argument('--model_zoo', type=str, default='pestv3', help='选择用哪套模型来划分数据:voc12/voc07/coco/pestv3/visdrone/pestv1/ip102/pest24')
    parser.add_argument('--validate', type=int, default=None, help='是否验证所有baseline模型在完整验证集上的表现')
    parser.add_argument('--judge', type=str, default=None,help='是否启用judge')
    parser.add_argument('--iter', type=str, default=None, help='是否通过遍历找到最佳的划分点，保存图像')
    parser.add_argument('--iter_weak_model', type=int, default=0, help='iter的时候会用强弱两个检测器去跑ap, 这个值指定了弱检测器使用哪个, 0: n, 1: m, 2: l')
    parser.add_argument('--iter_strong_model', type=int, default=3, help='iter的时候会用强弱两个检测器去跑ap, 这个值指定了强检测器使用哪个, 0: n, 1: m, 2: l')
    parser.add_argument('--keep_dir', action="store_false", help="是否清除原先的目录，不输入时为True")
    parser.add_argument('--dataType', type=str, default='val', help='选择验证集还是训练集')
    parser.add_argument('--out', type=str, default='expN', help='output dir')
    opt = parser.parse_args()
    dconfig, mconfig = parse(opt)
    img_dir = dconfig['source_images'] + opt.dataType
    threshold, judge_model_list = get_model(mconfig, judge=True)
    _, model_list = get_model(mconfig, judge=False)
    if opt.validate:
        ap_baseline = []
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        dataset = DetectionDataset(dconfig['source_images'], 'val', use_open=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=DetectionDataset.collate_fn)
        for model in model_list:
            print("validate name: ", model.model_name)
            performance = __evaluation(model, dataloader, device)
            print("Precision: ", performance[0])
            print("Recall", performance[1])
            print(f"ap50: {performance[2]}")
            print("F1 Score: ", performance[3])
            ap_baseline.append(performance[2])
        np.save(f'{opt.out}/data/{opt.dataType}_baseline_{opt.dataset}', ap_baseline)
    else:
        trace = []
        diff = []
        ldiff = []
        easy = []
        leasy = []
        for file in tqdm(os.listdir(img_dir)):
            path = os.path.join(img_dir, file)
            lpath = path.replace('images', 'labels').replace('jpg', 'txt')
            labels = extract_label_full(lpath)
            gt = format_gt(labels, *IMGSZ) if opt.judge else None
            results = detect(path, True, *judge_model_list)
            outliers = find_outliers_dict(results, gt)
            total_outliers = sum(len(bboxes) for model_outliers in outliers for bboxes in model_outliers.values())
            trace.append((path, lpath, total_outliers))
        #     if total_outliers > threshold:
        #         diff.append(path)
        #         ldiff.append(path.replace('images', 'labels').replace('jpg', 'txt'))
        #     else:
        #         easy.append(path)
        #         leasy.append(path.replace('images', 'labels').replace('jpg', 'txt'))
        #     if total_outliers != 0:
        #         trace.append(total_outliers)
        # median = get_median(trace)
        # print("median num of outliers: ", median)
        trace.sort(key=lambda x: x[2])
        num_total = len(trace)
        # 判断是否要进行遍历，可视化简单困难在不同的划分情况下的map曲线
        if opt.iter:
            weak_model_id = opt.iter_weak_model
            strong_model_id = opt.iter_strong_model
            e_map = []
            d_map = []
            e_map_x = []
            d_map_x = []
            # 通过遍历，找到适合的比例。
            for n in np.arange(1, 10, 0.5) :
                diff = []
                ldiff = []
                easy = []
                leasy = []
                num_easy = int(n/10 * num_total)  # 30% 难度大的图像
                for i, l, _ in trace[:num_easy]:
                    easy.append(i)
                    leasy.append(l)
                for i, l, _ in trace[num_easy:]:
                    diff.append(i)
                    ldiff.append(l)

                save(easy, leasy, dconfig['output_easy_dir'], clear_dir=True)
                save(diff, ldiff, dconfig['output_diff_dir'], clear_dir=True)
                # 使用字典来存储不同 idx 对应的结果列表和处理逻辑
                result_maps = {
                    weak_model_id: (e_map, d_map),
                    strong_model_id: (e_map_x, d_map_x)
                }

                for idx, model in enumerate(model_list):
                    if idx in result_maps:
                        print("validate name: ", model.model_name)
                        eap, dap = val(model, dconfig)
                        e_map_result, d_map_result = result_maps[idx]
                        e_map_result.append((n/10, eap))
                        d_map_result.append((n/10, dap))
            # 保存迭代ap结果
            np.savez(f'{opt.out}/data/{opt.dataType}_iter_map_model{weak_model_id}_{opt.dataset}', e_map=e_map, d_map=d_map, e_map_x=e_map_x, d_map_x=d_map_x)
            # 调用绘图函数
            save_map_curves(e_map, d_map, e_map_x, d_map_x, save_path=f'{opt.out}/figure/{opt.dataType}_ratio_iter_model{weak_model_id}_{opt.dataset}.png')
        else:
            num_easy = int(threshold * num_total)
            for i, l, _ in trace[:num_easy]:
                easy.append(i)
                leasy.append(l)
            for i, l, _ in trace[num_easy:]:
                diff.append(i)
                ldiff.append(l)
            save(easy, leasy, dconfig['output_easy_dir'], clear_dir=opt.keep_dir)
            save(diff, ldiff, dconfig['output_diff_dir'], clear_dir=opt.keep_dir)
        

if __name__ == '__main__':
    main()