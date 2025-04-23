import os
from PIL import Image
from thop import profile
import torch
import os
import shutil
from PIL import Image
import numpy as np


def validate_probabilities(outputs):
    """验证概率输出格式的完整函数"""
    # 类型检查
    if not isinstance(outputs, torch.Tensor):
        raise TypeError(f"输出应是torch.Tensor，实际是{type(outputs)}")
    
    # 形状检查
    if outputs.ndim != 2 or outputs.shape[1] != 2:
        raise ValueError(f"输出形状应为(N,2)，实际是{outputs.shape}")
    
    # 概率值检查
    if not torch.all((outputs >= 0) & (outputs <= 1)).item():
        bad_values = outputs[(outputs < 0) | (outputs > 1)]
        raise ValueError(f"概率值超出[0,1]范围：{bad_values.cpu().numpy()}")
    
    # 概率和检查
    sums = torch.sum(outputs, dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
        bad_sums = sums[~torch.isclose(sums, torch.ones_like(sums), atol=1e-5)]
        raise ValueError(f"概率和不为1的错误值：{bad_sums.cpu().numpy()}")
    
def resolve_npz(npz_file):
    fs = np.load(npz_file)
    result = []
    for key in fs.files:
        result.append((key, fs[key]))
    return result

def extract_label_full(file):
    """提取yolo的标签文件，返回[cls, x, y, w, h]

    Args:
        file (str): 完整路径地址

    Returns:
        _type_: _description_
    """
    rst = []
    with open(file, 'r') as f:
        for line in f.readlines():
            i = 0
            elems = []
            for elem in line.strip().split():
                if i != 0:
                    elems.append(float(elem))
                else:
                    elems.append(int(elem))
                i = i + 1
            rst.append(elems)
    return rst

def model_summary(model, input):
    """打印模型的参数量和浮点运算

    Args:
        model (_type_): 创建好的模型
        input (_type_): 输入的shape如：[1, 3, 640, 640]
    """
    img = torch.randn(*input)
    flops, params = profile(model, img)
    print("FLOPs: {:.2f}G".format(flops / 1e9)) 
    print("params: ", params)

def save(images, labels, output_dir, clear_dir=False):
    """保存图像和标签，并在需要时清空目标目录。

    Args:
        images (list): 所有图片路径
        labels (list): 所有的标签路径
        output_dir (str): 要保存的位置
        clear_dir (bool): 是否清空目标目录，默认为 False
    """
    # 删除图像和标签文件夹及其内容
    image_dir = os.path.join(output_dir, 'images', 'val')
    label_dir = os.path.join(output_dir, 'labels', 'val')
# 如果清空目录
    if clear_dir:
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir)
        if os.path.exists(label_dir):
            shutil.rmtree(label_dir)
        
        # 重新创建文件夹
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        print(f"Cleared the directories: {image_dir} and {label_dir}")

    # 如果不清空，直接确保目录存在
    else:
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

    saved_count = 0  # 计数器，记录保存的图像和标签数量

    for img_path, label_path in zip(images, labels):
        # 加载图像
        img = Image.open(img_path).convert('RGB') 
        
        # 获取文件名（不带扩展名）
        filename = os.path.splitext(os.path.basename(img_path))[0]

        # 保存图像
        img.save(os.path.join(image_dir, f"{filename}.jpg"))  # 保存为jpg格式
        
        # 保存标签
        with open(label_path, 'r') as label_file:
            label_data = label_file.read()
        
        with open(os.path.join(label_dir, f"{filename}.txt"), 'w') as out_label_file:
            out_label_file.write(label_data)
        
        saved_count += 1  # 更新保存计数

    print(f"Saved {saved_count} images and labels to {output_dir}")

def count(dir):
    """返回dir目录下有多少文件

    Args:
        dir (dir_path): 想要统计的文件数量

    Returns:
        int: dir目录下的文件数量
    """
    return len(os.listdir(dir))
