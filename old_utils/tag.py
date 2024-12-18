import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
from sklearn.cluster import KMeans
from ICNet import ICNet, infer_one_image
import torch
from torchvision import transforms
from ultralytics import YOLO
from tqdm import tqdm
import os
import sys
import time
sys.path.append("..")
from togethf_toolkits.obj_detection.evaluate import val
from togethf_toolkits.obj_detection.utils import save
import argparse
from config import tag_config


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# 获取数据集的所有图像路径
def get_paths(input):
    files = os.listdir(input)
    paths = [os.path.join(input, file) for file in files]
    return paths

# 返回一个用于得到样本难度分数的模型
def get_score_model(device):
    model = ICNet()
    model.load_state_dict(torch.load(tag_config.get('classfier_weight'),map_location=torch.device('cpu')))
    return model

# 返回每个样本的难度分数，例如 0 到 1 之间
def classify_sample_difficulty(samples):
    # 返回每个样本的难度分数，例如 0 到 1 之间
    model = get_score_model(device)
    model.eval()
    model.to(device)

    img_scores = []
    for img in tqdm(samples):
        inference_transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        score = infer_one_image(model, img, inference_transform, device)
        img_scores.append(score)
    return np.array(img_scores)


# 弱检测器在简单样本上的表现
def eval_weak_detector(samples, labels):
    # 使用 YOLO 模型对简单样本进行检测
    weaker = YOLO(tag_config['weak_detection'])  # 加载你预训练的弱 YOLO 模型
    map50, map50_90 = val(weaker, samples, labels)
    # 打印或返回结果
    evaluation = {
        'map@50': map50,
        'map@50-90': map50_90,
    }
    print(f"    mAP@50: {map50:.4f}")
    print(f"    mAP@50-90: {map50_90:.4f}")
    return evaluation

#初步分配样本为简单还是困难
def cluster_samples_by_difficulty(samples, scores):
    # 根据难度分数进行聚类
    kmeans = KMeans(n_clusters=2)  # 两类：简单与困难
    labels = kmeans.fit_predict(scores.reshape(-1, 1))

    # 计算每个类别的平均分数
    avg_score_0 = np.mean(scores[labels == 0])
    avg_score_1 = np.mean(scores[labels == 1])

    # 确定简单样本的类别
    if avg_score_0 < avg_score_1:
        simple_mask = labels == 0
    else:
        simple_mask = labels == 1

    return simple_mask  # 返回布尔数组，简单样本为 True，困难样本为 False

def iterative_detection(images, labels, desired_accuracy):
    print("Try to execute only on edge: ")
    print("---------------------------------------------------------------")
    iteration = 0
    final_hard = np.array([])
    final_hard_labels = np.array([])
    images = np.array(images)
    labels = np.array(labels)
    first_rst = eval_weak_detector(images, labels)

    if first_rst['map@50'] >= desired_accuracy:
        print("===============================================================")
        print("you can all samples run on the edge")
        return images, labels, final_hard, final_hard_labels
    else:
        while True:
            print("===============================================================")
            print("partition samples begin!")
            iteration += 1
            print(f"Iteration {iteration}:")
            print("---------------------------------------------------------------")
            # Step 1: 对样本进行难度打分
            scores = classify_sample_difficulty(images)
            
            # Step 2: 聚类样本，分为简单和困难样本
            bool_flag = cluster_samples_by_difficulty(images, scores)
            simple_samples = images[bool_flag]
            hard_samples = images[~bool_flag]
            simple_labels = labels[bool_flag]
            hard_labels = labels[~bool_flag]

            #把当前批次困难样本记录下来
            final_hard = np.concatenate((final_hard, hard_samples))
            final_hard_labels = np.concatenate((final_hard_labels, hard_labels))

            print(f"Simple samples: {len(simple_samples)}, Hard samples: {len(hard_samples)}")
            
            # Step 3: 使用弱检测器检测简单样本
            result = eval_weak_detector(simple_samples, simple_labels)


            if len(simple_samples) < tag_config['min_simple_samples']:
                print("===============================================================")
                print(f"Number of simple samples ({len(simple_samples)}) is below the threshold ({tag_config['min_simple_samples']}). Stopping...")
                break
            
            # Step 4: 检查检测结果是否达到用户期望
            if result['map@50'] >= desired_accuracy:
                print("===============================================================")
                print("Desired accuracy achieved.")
                break
            else:
                # 如果未达到预期精度，进一步将简单样本划分为简单和困难
                print("===============================================================")
                print("Desired accuracy not achieved, refining simple samples...")
                images = simple_samples  # 继续对简单样本进行难度打分并聚类
                labels = simple_labels
                
        return simple_samples, simple_labels, final_hard, final_hard_labels
def main():
    parser = argparse.ArgumentParser(description='difficult/easy tag program')
    parser.add_argument('--desired_mAP50', type=float, default=0.8, help='desired_mAP@50')
    opt = parser.parse_args()


    images = get_paths(tag_config['source_images'])
    labels = get_paths(tag_config['source_labels'])
    # 执行过程
    simple_samples, simple_labels, final_hard, final_hard_labels = iterative_detection(images, labels, opt.desired_mAP50)

    print("Final simple samples:", len(simple_samples))
    print("Final hard samples:", len(final_hard))

    print('**************** saving new dataset *********************')
    save(simple_samples, simple_labels, tag_config['output_easy_dir'])
    save(final_hard, final_hard_labels, tag_config['output_diff_dir'])
    print('*********************** end  *********************')

if __name__ == '__main__':
    main()
