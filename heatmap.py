import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os

# ================= 配置区域 =================
# 1. 你的模型路径
names = [
    'yolo11n',
    'yolov11n-C3k2-HetConv2',
    'yolov11n-VAE-SE2',
]

model_name = names[0]
m = os.path.join('/home','insslab','paper3','runs', 'lightweight', 'pest_v3', model_name, 'weights', 'best.pt')
MODEL_PATH = m

# 2. 你的测试图片路径
IMG_PATH = '/home/insslab/Desktop/datasets/RicePestsV3/VOCdevkit/images/val/147645_22-07-04-00-22-11_1_2560_3000_1024.jpg'


# 3. 输出热力图保存路径
OUTPUT_PATH = "heatmap_result.png"
# ===========================================

class EigenCAM:
    """
    内置 EigenCAM 核心算法 (无需 pip install)
    """
    def __init__(self, model, target_layers=None):
        self.model = model.model
        self.target_layers = target_layers
        self.activations = []
        # 注册钩子提取特征
        for layer in self.target_layers:
            layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def __call__(self, img_tensor):
        self.activations = []
        with torch.no_grad():
            self.model(img_tensor)
        
        if not self.activations:
            return None
            
        activation = self.activations[-1] # [B, C, H, W]
        b, c, h, w = activation.shape
        # 展平并转置
        activation = activation.squeeze(0).view(c, -1).permute(1, 0) 
        # 中心化
        activation = activation - activation.mean(dim=0, keepdim=True)
        # SVD 计算主成分
        U, S, V = torch.svd(activation)
        # 第一主成分
        cam = activation @ V[:, 0].unsqueeze(1)
        # 归一化
        cam = cam.view(h, w)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        return cam.cpu().numpy()

def show_cam_on_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """将热力图叠加到原图上"""
    # 1. 生成彩色热力图 (蓝->红)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # 2. 归一化原图
    if np.max(img) > 1:
        img = np.float32(img) / 255
        
    # 3. 叠加 (权重可调，这里是 1:1)
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def main():
    print(f"正在加载模型: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    

    # 9 或 10: Backbone末端 (语义强，位置准，推荐首选)
    # 15, 18, 21: Neck层 (特征更细，但可能噪点多)
    TARGET_LAYER_INDEX = 10
    # 自动锁定检测头之前的层 (通常是 Neck 的输出)
    target_layer = model.model.model[TARGET_LAYER_INDEX]
    print(f"目标层级: {target_layer.__class__.__name__}")

    # 读取并预处理图片
    img_raw = cv2.imread(IMG_PATH)
    if img_raw is None:
        print(f"错误: 找不到图片 {IMG_PATH}")
        return

    # 缩放到 640x640 进行推理 (防止尺寸不匹配报错)
    img_resized = cv2.resize(img_raw, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # 计算热力图
    print("正在计算 EigenCAM ...")
    eigencam = EigenCAM(model, target_layers=[target_layer])
    mask = eigencam(img_tensor)

    if mask is None:
        print("错误: 未能提取到特征图")
        return

    # 将热力图缩放回原图尺寸
    mask_resized = cv2.resize(mask, (img_raw.shape[1], img_raw.shape[0]))
    
    # 叠加显示
    result_image = show_cam_on_image(img_raw, mask_resized)
    
    # 保存
    cv2.imwrite(OUTPUT_PATH, result_image)
    print(f"✅ 热力图已保存: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()