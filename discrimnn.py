import argparse
import torch.nn as nn
from torchvision.models import shufflenet_v2_x0_5
from config import *
from ultralytics import YOLO
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset
from commons.dataset import ClassifyDataset, DetectionDataset
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import time
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights, shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
import random
from commons.proutils import parse
from commons.utils import resolve_npz
from commons.metrics import *
from torch.nn.functional import softmax
    
def max_edge(rs, clfs, idx):
    rs, clfs = np.array(rs), np.array(clfs)
    candidate_rs, candidate_clf = rs[idx], clfs[idx]
    candidate_edge_x = candidate_rs * candidate_clf + (1 - candidate_rs) * (1 - candidate_clf)
    max_indices = np.where(candidate_edge_x == np.max(candidate_edge_x))[0]
    return max_indices[-1], candidate_rs[max_indices[-1]]

def dynamic_ap(aps, clfs, r):
    emap, dmap, emapx, dmapx = aps[0][1], aps[1][1], aps[2][1], aps[3][1]
    candidate_n = len(emap)
    # dynamic_ap列表
    dynamic_aps = []
    r_index = 0
    r = [item/100 for item in r]
    # 将 r 转换为集合以提高查询效率
    r_set = set(r) if isinstance(r, list) else r
    for i in range(candidate_n):
        if emap[i][0] not in r_set:
            continue
        pcls = clfs[r_index]
        pwe, pwh, pse, psh = emap[i][1], dmap[i][1], emapx[i][1], dmapx[i][1]
        p_correct_e = r[r_index] * pcls * pwe
        p_wrong_h = (1 - r[r_index]) * (1 - pcls) * pwh
        p_correct_h = (1 - r[r_index]) * pcls * psh
        p_wrong_e = r[r_index] * (1 - pcls) * pse
        ptotal = p_correct_e + p_wrong_h + p_correct_h + p_wrong_e
        dynamic_aps.append(ptotal)
        r_index += 1
    return dynamic_aps
    
class cls_scheme:
    def __init__(self, dataset='pestv3', ratio=[30, 40, 50, 60, 70]):
        self.data = dataset
        self.rs = ratio
    
    def __get_model(self, id, device):
        if id == 1:
            cls = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.DEFAULT)
            cls.fc = torch.nn.Linear(cls.fc.in_features, 2)
            cls.to(device)
        else:
            cls = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
            cls.fc = torch.nn.Linear(cls.fc.in_features, 2)
            cls.to(device)
        return cls

    def items(self):
        clfs = []
        for r in self.rs:
            model_path = os.path.join('checkpoint/classifier', self.data, f'{r}', 'best_model.pth')
            clfs.append(model_path)
        return clfs


    def scores(self):
        clfs = []
        models = self.items()
        for idx, model in enumerate(models):
            val_accuracy = self.evaluate_model_accuracy(model, self.rs[idx])
            clfs.append(val_accuracy)
        return clfs

    def evaluate_model_accuracy(self, model_path, ratio):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # 加载模型
        model = self.__get_model(1, device)  # 使用训练时的相同模型ID（1表示shufflenet_v2_x0_5）
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # 准备验证集
        val_transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])
        val_dataset = ClassifyDataset(img_dir=os.path.join('out', self.data, str(ratio), 'trainval'), 
                                    transform=val_transform, train=False)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16)
        
        # 计算分类精度
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)  # 获取预测类别
                
                correct += (preds == labels).sum().item()  # 统计正确预测数
                total += labels.size(0)  # 统计总样本数
        
        accuracy = correct / total  # 计算分类精度
        return accuracy

class ViDiscrimNN(nn.Module):
    def __init__(self, weight, dconfig, mode, bias=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router = self._prepare_router(weight)
        self.weak_det = YOLO(dconfig['weak_detector'])
        self.strong_det = YOLO(dconfig['strong_detector'])
        self.cloud_flag = False # random scheme中要用到
        self.mode = mode
        self.bias = bias
    
    def _prepare_router(self, weight):
        router = shufflenet_v2_x0_5()
        router.fc = torch.nn.Linear(router.fc.in_features, 2)
        router.load_state_dict(torch.load(weight))
        router.eval()
        return router

    
    def forward(self, X): 
        def _prepare_det(mode):
            if mode == 'dynamic':
                return self.weak_det, self.strong_det
            elif mode == 'edge':
                return self.weak_det, self.weak_det
            elif mode == 'random':
                choice = random.randint(0, 1)
                if choice == 1:
                    det1 = self.weak_det
                else:
                    det1 = self.strong_det
                    self.cloud_flag = True
                choice = random.randint(0, 1)
                if choice == 1:
                    det2 = self.weak_det
                else:
                    det2 = self.strong_det
                    self.cloud_flag = True
                return det1, det2
            else:
                return self.strong_det, self.strong_det
        det1, det2 = _prepare_det(self.mode)
        outputs = self.router(X)
        outputs = softmax(outputs, dim=1)
        rst = outputs.argmax(dim=1)
        if self.bias and torch.max(outputs, 1)[0] < self.bias:
            rst = torch.ones(outputs.shape[0]).to(outputs.device)
        easys = []
        diffs = []
        eouts = []
        douts = []
        for idx, elem in enumerate(rst):
            if elem == 0:  # easy
                easys.append(idx)
            else:  # difficult
                diffs.append(idx)
        if len(easys) > 0:
            eouts = det1.predict(X[easys], verbose=False)
        if len(diffs) > 0:
            douts = det2.predict(X[diffs], verbose=False)

        # 创建一个与输入大小相同的空列表
        outs = [None] * len(X)
        l = len(diffs)
        weights = {
            'dynamic': len(diffs),
            'edge': 0,
            'cloud': 1,
            'random': 1 if self.cloud_flag else 0
        }
        offloading = weights[mode] * IMGSZ[0] * IMGSZ[1] * 24 
        # 将预测结果根据索引放回到对应位置
        for i, idx in enumerate(easys):
            outs[idx] = eouts[i]
        for i, idx in enumerate(diffs):
            outs[idx] = douts[i]
        self.cloud_flag = False
        return outs, offloading

    #模型评估
    def evaluation(self, val_dataloader, device):
        labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)
        pbar = tqdm(val_dataloader)
        classes = []
        total_num = 0
        total_time = 0
        total_offloading = 0
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
            # ====================== time begin =====================
            begin = time.time()
            output, offloading_num = self.forward(imgs)
            end = time.time()
            total_num += imgs.shape[0]
            total_time += end - begin
            # ====================== time end =====================
            pbar.set_description(f"Evaluation model in {self.mode} mode") 
            sample_metrics += get_batch_statistics(output, labels, device)
            total_offloading += offloading_num
        if len(sample_metrics) == 0:  # No detections over whole validation set.
            print("---- No detections over whole validation set ----")
            return None

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, classes)
        fps = total_num / total_time
        return metrics_output, fps, total_offloading  

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='discrimnn system design')
    parser.add_argument('--dataset', type=str, default='pestv3',  help='选择划分哪个数据集：voc12/voc07/coco/pestv3/visdrone/pestv1/ip102/pest24')
    parser.add_argument('--model_zoo', type=str, default='pestv3', help='选择用哪套模型来划分数据:voc12/voc07/coco/pestv3/visdrone/pestv1/ip102/pest24')
    parser.add_argument('--expected_ap', type=float, default=0.85, help='用户希望系统能够达到的精度')
    parser.add_argument('--edge_level', type=int, default=0, help='边缘设备算力所能承载的最大模型, 0为n, 以此类推')
    parser.add_argument('--iterdata', type=str, default='expN/data', help='保存outlier迭代输出文件的目录')
    parser.add_argument('--baseline_data', type=str, default='expN/data', help='保存baseline性能文件的目录')
    parser.add_argument('--dataType', type=str, default='val', help='iter文件的类型, train or val')
    opt = parser.parse_args()

    dconfig, mconfig = parse(opt)
    expected_ap = opt.expected_ap
    baseline_path = os.path.join(opt.baseline_data, f'{opt.dataType}_baseline_{opt.dataset}.npy')
    baseline_aps = np.load(baseline_path)
    print(f"system performance ranging from {baseline_aps[0]} to {baseline_aps[-1]}:")
    # 获取可用的边缘detector
    eg_level = min(opt.edge_level, len(baseline_aps-1))
    assert eg_level >= 0, '边缘算力过低, 不足以支持DNN推理.'
    print('edge model juding......')
    for m_i in range(eg_level+1):
        if baseline_aps[m_i] >= expected_ap:
            print(f"edge model{m_i} mAP50: {baseline_aps[m_i]} >= user expected {expected_ap}")
            print("all samples run on the edge.")
            sys.exit(0)
    print(f"edge model{eg_level} chosen, mAP50: {baseline_aps[eg_level]}")

    # 获取划分比例
    r = [30, 35, 40, 45, 50, 55, 60, 65, 70]
    # 解析npz文件，用户获取ap的迭代曲线
    iterap_path = os.path.join(opt.iterdata, f'{opt.dataType}_iter_map_model{eg_level}_{opt.dataset}.npz')
    iter_aps = resolve_npz(iterap_path)

    # 训练分类器，获得分类策略
    scheme = cls_scheme(dataset=opt.dataset, ratio=r)
    cs = scheme.scores()
    c_models = scheme.items()
    # 计算ratio
    dynamic_aps = np.array(dynamic_ap(iter_aps, cs, r))
    # 找到dynamic_ap值大于用户值的下标
    idx = np.where(dynamic_aps > expected_ap)[0]
    if len(idx) == 0:
        mode = 'cloud'
        loc, max_r = 0, 0 # 这两个参数这种情况下没有意义，传入ViDiscrimNN的c_models[loc]不会生效，因为mode = 'cloud'
    else:
        mode = 'dynamic'
        loc, max_r = max_edge(r, cs, idx)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ViDiscrimNN(c_models[loc], dconfig, mode).to(device)
    dataset = DetectionDataset(dconfig['source_images'], 'val', open=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=DetectionDataset.collate_fn)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=16, collate_fn=DetectionDataset.collate_fn)

    performance, fps, uploading = model.evaluation(dataloader, device)
    print("Precision: ", performance[0])
    print("Recall", performance[1])
    print("mAP50", performance[2])
    print("F1 Score: ", performance[3])
    print("FPS: ", fps)
    print(f"Total params: {(IMGSZ[0] * IMGSZ[1] * 24 * 1801) / 8 / 1024 / 1024:.2f}MB, Uploading {(uploading) / 8 / 1024 / 1024:.2f}MB")


    # modes = ['edge', 'cloud', 'dynamic', 'random']
    # metrics = {
    #     'Precision': [],
    #     'Recall': [],
    #     'mAP50': [],
    #     'F1 Score': [],
    #     'FPS': [],
    #     'Uploading': []
    # }

    # for mode in modes:
    #     performance, fps, uploading = model.evaluation(dataloader, device)
    #     print(f'----------------------execute {mode} mode:-----------------------')
    #     print("Precision: ", performance[0])
    #     print("Recall", performance[1])
    #     print("mAP50", performance[2])
    #     print("F1 Score: ", performance[3])
    #     print("FPS: ", fps)
    #     print("Uploading ", uploading)
    #     print(f'----------------------end evaluation:-----------------------')

    #     metrics['Precision'].append(performance[0])
    #     metrics['Recall'].append(performance[1])
    #     metrics['mAP50'].append(performance[2])
    #     metrics['F1 Score'].append(performance[3])
    #     metrics['FPS'].append(fps)
    #     metrics['Uploading'].append(uploading)

    # # 绘制图表
    # fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # fig.suptitle('Model Performance in Different Modes', fontsize=16)

    # # 定义每个指标的纵坐标范围
    # ylim_dict = {
    #     'Precision': (0.5, 1),      # Precision 范围 0 到 1
    #     'Recall': (0.5, 1),         # Recall 范围 0 到 1
    #     'mAP50': (0.5, 1),          # mAP50 范围 0 到 1
    #     'F1 Score': (0.5, 1),       # F1 Score 范围 0 到 1
    #     'FPS': (0, max(metrics['FPS']) + 10),  # FPS 范围 0 到最大值 + 10
    #     'Uploading': (0, max(metrics['Uploading']) + 10)  # Uploading 范围 0 到最大值 + 10
    # }

    # for ax, (metric, values) in zip(axes.flatten(), metrics.items()):
    #     ax.bar(modes, values, color=['skyblue', 'orange', 'red', 'green'])
    #     ax.set_title(metric)
    #     ax.set_ylabel(metric)
    #     ax.set_xlabel('Mode')
    #     ax.set_ylim(ylim_dict[metric])  # 设置纵坐标范围
    #     ax.grid(axis='y', linestyle='--', alpha=0.7)

    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # output_path = "figure/performance_metrics.png"
    # plt.savefig(output_path)
    # print(f"Performance metrics chart saved to {output_path}")

