# MMDetection 训练和使用指南

本指南介绍如何使用 MMDetection 框架训练和使用 Faster-RCNN 等主流目标检测模型进行对比实验。

## 目录

- [安装](#安装)
- [数据集准备](#数据集准备)
- [训练模型](#训练模型)
- [测试模型](#测试模型)
- [使用预训练模型进行推理](#使用预训练模型进行推理)
- [模型对比实验](#模型对比实验)

## 安装

### 1. 安装 MMDetection

MMDetection 已经包含在项目中。如果需要重新安装或更新：

```bash
cd mmdetection
pip install -e .
```

### 2. 安装依赖

```bash
pip install mmcv mmengine
```

## 数据集准备

MMDetection 支持多种数据集格式，推荐使用 COCO 格式。

### COCO 格式

数据集目录结构：
```
RicePestsV3/
├── annotations/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── train/
│   ├── image1.jpg
│   └── ...
├── val/
│   ├── image1.jpg
│   └── ...
└── test/
    ├── image1.jpg
    └── ...
```

### 转换数据集格式

如果你的数据集是 VOC 格式，可以使用 MMDetection 提供的转换工具：

```bash
python mmdetection/tools/dataset_converters/pascal_voc.py \
    --devkit_path /home/insslab/Desktop/datasets/PureRicePestsV3/VOCdevkit/ \
    --out-dir mmdet_training/dataset
```

## 训练模型

### 1. 使用配置文件训练

我们提供了针对 PestV3 数据集的配置文件：

```bash
# 训练 Faster R-CNN
python mmdet_training/train_mmdet.py \
    --config mmdet_training/configs/faster_rcnn_r50_fpn_pestv3.py \
    --work-dir work_dirs/faster_rcnn_pestv3

# 训练 RetinaNet
python mmdet_training/train_mmdet.py \
    --config mmdet_training/configs/retinanet_r50_fpn_pestv3.py \
    --work-dir work_dirs/retinanet_pestv3
```

### 2. 从预训练模型微调

```bash
python mmdet_training/train_mmdet.py \
    --config mmdet_training/configs/faster_rcnn_r50_fpn_pestv3.py \
    --load-from https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    --work-dir work_dirs/faster_rcnn_pestv3
```

### 3. 恢复训练

```bash
python mmdet_training/train_mmdet.py \
    --config mmdet_training/configs/faster_rcnn_r50_fpn_pestv3.py \
    --resume work_dirs/faster_rcnn_pestv3/latest.pth
```

### 4. 自定义训练参数

```bash
python mmdet_training/train_mmdet.py \
    --config mmdet_training/configs/faster_rcnn_r50_fpn_pestv3.py \
    --cfg-options train_cfg.max_epochs=20 optim_wrapper.optimizer.lr=0.01
```

## 测试模型

### 评估模型性能

```bash
python mmdet_training/test_mmdet.py \
    --config mmdet_training/configs/faster_rcnn_r50_fpn_pestv3.py \
    --checkpoint work_dirs/faster_rcnn_pestv3/best.pth \
    --eval mAP
```

### 生成可视化结果

```bash
python mmdet_training/test_mmdet.py \
    --config mmdet_training/configs/faster_rcnn_r50_fpn_pestv3.py \
    --checkpoint work_dirs/faster_rcnn_pestv3/best.pth \
    --show-dir results/visualizations
```

## 使用预训练模型进行推理

### 使用 MMDetAdapter

```python
from mmdet_integration.mmdet_adapter import MMDetModelRegistry

# 加载预训练的 Faster R-CNN 模型
model = MMDetModelRegistry.get_model(
    'faster-rcnn-r50',
    device='cuda:0',
    score_threshold=0.3
)

# 进行推理
result = model.predict('path/to/image.jpg')
print(result)
```

### 使用自定义配置

```python
from mmdet_integration.mmdet_adapter import MMDetAdapter

# 使用自定义配置和权重
adapter = MMDetAdapter(
    config_file='mmdet_training/configs/faster_rcnn_r50_fpn_pestv3.py',
    checkpoint_file='work_dirs/faster_rcnn_pestv3/best.pth',
    device='cuda:0'
)

result = adapter.predict('path/to/image.jpg')
```

## 模型对比实验

### 使用对比实验脚本

项目中的 `experiments/mmdet_comparison.py` 提供了模型对比功能：

```bash
python experiments/mmdet_comparison.py \
    --models faster-rcnn-r50 retinanet-r50 rtmdet-s \
    --device cuda:0
```

### 可用的模型

通过 `MMDetModelRegistry` 可以访问以下模型：

- **Faster R-CNN**: `faster-rcnn-r50`, `faster-rcnn-r101`
- **RetinaNet**: `retinanet-r50`
- **Cascade R-CNN**: `cascade-rcnn-r50`
- **YOLOX**: `yolox-s`, `yolox-m`
- **RTMDet**: `rtmdet-tiny`, `rtmdet-s`
- **DETR**: `detr-r50`

查看所有可用模型：

```python
from mmdet_integration.mmdet_adapter import MMDetModelRegistry

models = MMDetModelRegistry.list_models()
for name, desc in models.items():
    print(f"{name}: {desc}")
```

## 创建自定义配置文件

### 基于现有配置

1. 复制基础配置文件：
```bash
cp mmdet_training/configs/faster_rcnn_r50_fpn_pestv3.py mmdet_training/configs/my_custom_config.py
```

2. 修改配置参数：
   - 数据集路径
   - 类别数量
   - 训练参数（学习率、epochs等）
   - 模型结构

### 配置文件结构

```python
_base_ = ['../../mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py']

# 数据集设置
data_root = '/path/to/your/dataset/'
classes = ['class1', 'class2', ...]

# 模型设置
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=len(classes))
    )
)

# 训练设置
train_cfg = dict(max_epochs=12)
optim_wrapper = dict(optimizer=dict(lr=0.02))
```

## 常见问题

### 1. 内存不足

减少 batch size：
```python
data = dict(samples_per_gpu=1)  # 原来是 2
```

### 2. 训练速度慢

- 使用更少的 workers：`workers_per_gpu=1`
- 使用混合精度训练：添加 `fp16=dict(loss_scale=512.0)` 到配置

### 3. 模型不收敛

- 降低学习率
- 增加训练轮数
- 检查数据标注是否正确

## 参考资源

- [MMDetection 官方文档](https://mmdetection.readthedocs.io/)
- [MMDetection GitHub](https://github.com/open-mmlab/mmdetection)
- [模型库](https://mmdetection.readthedocs.io/en/latest/model_zoo.html)

## 示例：完整训练流程

```bash
# 1. 准备数据集（确保是 COCO 格式）

# 2. 训练模型
python mmdet_training/train_mmdet.py \
    --config mmdet_training/configs/faster_rcnn_r50_fpn_pestv3.py \
    --work-dir work_dirs/faster_rcnn_pestv3

# 3. 评估模型
python mmdet_training/test_mmdet.py \
    --config mmdet_training/configs/faster_rcnn_r50_fpn_pestv3.py \
    --checkpoint work_dirs/faster_rcnn_pestv3/best.pth \
    --eval mAP

# 4. 进行推理
python -c "
from mmdet_integration.mmdet_adapter import MMDetAdapter
adapter = MMDetAdapter(
    'mmdet_training/configs/faster_rcnn_r50_fpn_pestv3.py',
    'work_dirs/faster_rcnn_pestv3/best.pth'
)
result = adapter.predict('test_image.jpg')
print(result)
"
```









