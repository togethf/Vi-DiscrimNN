import numpy as np
import os
import argparse
import os
from ultralytics import YOLO
from commons.utils import save
from config import *
from commons.det_utils import cal_iou


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
        'pestv3': pestv3_config,
        'coco': coco_config
    }

    # 根据 opt.dataset 获取对应的数据配置
    if opt.dataset in data_config_map:
        data_config = data_config_map[opt.dataset]
    else:
        raise ValueError(f"Invalid dataset: {opt.dataset}. Available options are {', '.join(data_config_map.keys())}.")

    # 定义模型配置映射字典
    model_config_map = {
        'voc12': judge_config['voc12'],
        'voc07': judge_config['voc07'],
        'pestv3': judge_config['pestv3'],
        'coco': judge_config['coco']
    }

    # 根据 opt.model 获取对应的模型配置
    if opt.model_zoo in model_config_map:
        model_config = model_config_map[opt.model_zoo]
    else:
        raise ValueError(f"Invalid model: {opt.model}. Available options are {', '.join(model_config_map.keys())}.")

    return data_config, model_config

def get_model(mconfig):
    """获取用于judge的模型

    Args:
        mconfig (dict): config.py['which']
    """
    model_list = []
    for model in mconfig['models']:
        model_list.append(YOLO(model))
    threshold = mconfig['threshold']
    return threshold, model_list

def find_outliers_dict(models, iou_threshold=0.5): 
    """
    检测多个模型的输出中，基于字典格式的离群值
    :param models: List of models' outputs in dictionary format {"class": [bbox1, bbox2, ...]}
    :param iou_threshold: IoU threshold for matching
    :return: List of outliers for each model in dictionary format
    """
    all_outliers = [{} for _ in models]  # 每个模型的离群值，用字典存储

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
        result = model.predict(img)
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

    model.val(data=dconfig['cfg'])
    print("validate on whole done ")

    model.val(data=dconfig['cfg'].replace('.yaml', '_easy.yaml'))
    print("validate on easy done")

    model.val(data=dconfig['cfg'].replace('.yaml', '_diff.yaml'))
    print("validate on diff done")

def main():
    parser = argparse.ArgumentParser(description='find outlier based method to tag the difficulty of imgs')
    parser.add_argument('--dataset', type=str, default='pestv3', help='选择划分哪个数据集：voc12/voc07/pestv3')
    parser.add_argument('--model_zoo', type=str, default='pestv3', help='选择用哪套模型来划分数据:voc12/voc07/pestv3')
    parser.add_argument('--validate', type=str, default=None, help='会决定是划分数据集还是验证')
    opt = parser.parse_args()
    dconfig, mconfig = parse(opt)  
    img_dir = dconfig['source_images'] + 'val'
    threshold, model_list = get_model(mconfig)
    if opt.validate:
        for model in model_list:
            print("validate name: ", model.model_name)
            validate(model, dconfig)
    else:
        trace = []
        diff = []
        ldiff = []
        easy = []
        leasy = []
        for file in os.listdir(img_dir):
            path = os.path.join(img_dir, file)
            results = detect(path, True, *model_list)
            outliers = find_outliers_dict(results)
            total_outliers = sum(len(bboxes) for model_outliers in outliers for bboxes in model_outliers.values())
            if total_outliers > threshold:
                diff.append(path)
                ldiff.append(path.replace('images', 'labels').replace('jpg', 'txt'))
            else:
                easy.append(path)
                leasy.append(path.replace('images', 'labels').replace('jpg', 'txt'))
            if total_outliers != 0:
                trace.append(total_outliers)
        median = get_median(trace)
        print("median num of outliers: ", median)

        save(easy, leasy, tag_config['output_easy_dir'])
        save(diff, ldiff, tag_config['output_diff_dir'])
if __name__ == '__main__':
    main()