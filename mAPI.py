import numpy as np
from commons.dataset import DetectionDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from ultralytics import YOLO
from tqdm import tqdm
import os
from PIL import Image
from config import *
from commons.utils import save, extract_label_full
import argparse
from config import *
from commons.det_utils import get_batch_statistics, xywh2xyxy, ap_per_class


 
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
        'voc12': voc12_config if not opt.validate else judge_config['voc12'],
        'voc07': voc07_config if not opt.validate else judge_config['voc07'],
        'pestv3': pestv3_config if not opt.validate else judge_config['pestv3'],
        'coco': coco_config if not opt.validate else judge_config['coco']
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

def get_model(mconfig):
    """获取用于judge的模型

    Args:
        mconfig (dict): config.py['which']
    """
    model_list = []
    for model in mconfig['models']:
        model_list.append(YOLO(model))
    return model_list

def validate(model, dconfig, func=None):
    """验证YOLO模型在难易数据集和整个数据集上的表现

    Args:
        model (YOLO): 加载好的YOLO模型
        dconfig (dict): config类中配置好的数据字典
    """

    model.val(data=dconfig['cfg'])
    print("validate on whole done ")

    if func:
        model.val(data=dconfig['cfg'].replace('.yaml', '_easy.yaml'))
        print("validate on easy done")

        model.val(data=dconfig['cfg'].replace('.yaml', '_diff.yaml'))
        print("validate on diff done")
    else:
        model.val(data=dconfig['cfg'].replace('.yaml', '_api_easy.yaml'))
        print("validate on easy done")

        model.val(data=dconfig['cfg'].replace('.yaml', '_api_diff.yaml'))
        print("validate on diff done")

def mAPI(outs, labels, classes, device):
    """计算单张图片的map

    Args:
        outs (YOLO results: YOLO模型的predict结果
        labels (tensor(gpu)): 一个batch所有的bbox汇总，(number of bbox, dimension)
        classes (list): True object classes
        device (torch.device): 

    Returns:
        _type_: mapi
    """
    sample_metrics = get_batch_statistics(outs, labels, device=device)
    if len(sample_metrics) == 0:
        return 0
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, classes)
    return metrics_output[3] 

def main():
    parser = argparse.ArgumentParser(description='Calculate mAP for each image in a dataset')
    parser.add_argument('--dataset', type=str, default='coco', help='选择划分哪个数据集：voc12/voc07/pestv3')
    parser.add_argument('--model', type=str, default='coco', help='选择用哪个系列的检测器pair来划分数据集，而在验证阶段则是用哪套模型来验证:voc12/voc07/pestv3')
    parser.add_argument('--validate', type=str, default=True, help='会决定是划分数据集还是验证')
    parser.add_argument('--my', type=str, default='yes', help='用什么方法进行难易划分, 默认是mapi, 传值就是用我自己的方法')
    opt = parser.parse_args()
    dconfig, mconfig = parse(opt)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # 判断是进行数据划分还是验证效果
    if opt.validate:
        model_list = get_model(mconfig)
        for model in model_list:
            print("validate name: ", model.model_name)
            validate(model, dconfig, opt.my)
    else:
        # 加载模型和数据集
        weak = prepare_det(mconfig['weak_detector'])
        strong = prepare_det(mconfig['strong_detector'])
        dataset = DetectionDataset(dconfig['source_images'], 'val')
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=DetectionDataset.collate_fn)

        easy_imgs = []
        easy_labels = []
        diff_imgs = []
        diff_labels = []
        classes = []
        tsf = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])
        # 划分数据集 begin
        for images, labels in tqdm(dataloader):
            if len(labels.shape) == 1:
                classes += []
            else:
                classes += labels[:, 0].tolist()
                labels[:, 1:5] = xywh2xyxy(labels[:, 1:5])
                labels[:, 1:5] *= torch.tensor([*IMGSZ, *IMGSZ])
            image = Image.open(images[0]).convert('RGB')
            image = tsf(image)
            image.to(device)
            wouts = weak(image, verbose=False)
            souts = strong(image, verbose=False)

            labels = labels.to(device)

            wmapi = mAPI(wouts, labels, classes, device)
            smapi = mAPI(souts, labels, classes, device)
            

            if smapi > wmapi:
                print(f"weak: {wmapi} < strong: {smapi}------difficult")
                diff_imgs.append(images[0])
                diff_labels.append(images[0].replace('images', 'labels').replace('jpg', 'txt'))
            else:
                print(f"weak: {wmapi} >= strong: {smapi}------easy")
                easy_imgs.append(images[0])
                easy_labels.append(images[0].replace('images', 'labels').replace('jpg', 'txt'))
        # 保存划分好的难易数据集
        save(easy_imgs, easy_labels, tag_config['output_easy_dir'])
        save(diff_imgs, diff_labels, tag_config['output_diff_dir']) 


if __name__ == '__main__':
    main()