import os
from PIL import Image
from thop import profile
import torch

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

def save(images, labels, output_dir):
    """_summary_

    Args:
        images (list)): 所有图片路径
        labels (list): 所有的标签路径
        output_dir (str): 要保存的位置
    """

    # 确保输出目录存在
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)

    saved_count = 0  # 计数器，记录保存的图像和标签数量

    for img_path, label_path in zip(images, labels):
        # 加载图像
        img = Image.open(img_path).convert('RGB') 
        
        # 获取文件名（不带扩展名）
        filename = os.path.splitext(os.path.basename(img_path))[0]

        # 保存图像
        img.save(os.path.join(output_dir, 'images', 'val', f"{filename}.jpg"))  # 或者使用jpg，取决于需求
        
        # 保存标签
        with open(label_path, 'r') as label_file:
            label_data = label_file.read()
        
        with open(os.path.join(output_dir, 'labels', 'val', f"{filename}.txt"), 'w') as out_label_file:
            out_label_file.write(label_data)
        
        saved_count += 1  # 更新保存计数

    print(f"Saved {saved_count} images and labels to {output_dir}")
