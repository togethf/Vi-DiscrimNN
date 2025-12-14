# MMDetection 快速开始指南

## 简介

本指南帮助您快速使用 MMDetection 框架训练和使用 Faster-RCNN 等主流目标检测模型进行对比实验。

## 一、快速开始（使用预训练模型）

### 1. 查看可用模型

```python
python mmdet_training/quick_start.py
```

或者：

```python
from mmdet_integration.mmdet_adapter import MMDetModelRegistry

# 查看所有可用模型
models = MMDetModelRegistry.list_models()
for name, desc in models.items():
    print(f"{name}: {desc}")
```

### 2. 使用预训练模型进行推理

```python
from mmdet_integration.mmdet_adapter import MMDetModelRegistry

# 加载 Faster R-CNN 模型
model = MMDetModelRegistry.get_model('faster-rcnn-r50', device='cuda:0')

# 进行推理
result = model.predict('path/to/image.jpg')
print(f"检测到 {result['num_detections']} 个目标")
for det in result['detections']:
    print(f"  {det['class_name']}: {det['score']:.2f}")
```

## 二、训练自定义模型

### 1. 准备数据集

确保数据集是 COCO 格式，目录结构如下：

```
RicePestsV3/
├── annotations/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── train/
│   └── *.jpg
├── val/
│   └── *.jpg
└── test/
    └── *.jpg
```

### 2. 修改配置文件

编辑 `mmdet_training/configs/faster_rcnn_r50_fpn_pestv3.py`，修改：
- `data_root`: 数据集路径
- `ann_file`: 标注文件路径
- `img_prefix`: 图像目录路径

### 3. 开始训练

```bash
# 基础训练
python mmdet_training/train_mmdet.py \
    --config mmdet_training/configs/faster_rcnn_r50_fpn_pestv3.py \
    --work-dir work_dirs/faster_rcnn_pestv3

# 从预训练模型微调
python mmdet_training/train_mmdet.py \
    --config mmdet_training/configs/faster_rcnn_r50_fpn_pestv3.py \
    --load-from https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    --work-dir work_dirs/faster_rcnn_pestv3
```

### 4. 评估模型

```bash
python mmdet_training/test_mmdet.py \
    --config mmdet_training/configs/faster_rcnn_r50_fpn_pestv3.py \
    --checkpoint work_dirs/faster_rcnn_pestv3/best.pth \
    --eval mAP
```

## 三、模型对比实验

### 使用对比脚本

```bash
python experiments/mmdet_comparison.py \
    --models faster-rcnn-r50 retinanet-r50 rtmdet-s \
    --device cuda:0
```

### 手动对比多个模型

```python
from mmdet_integration.mmdet_adapter import MMDetModelRegistry

models_to_compare = ['faster-rcnn-r50', 'retinanet-r50', 'rtmdet-s']

results = {}
for model_name in models_to_compare:
    print(f"Loading {model_name}...")
    model = MMDetModelRegistry.get_model(model_name, device='cuda:0')
    
    # 运行推理
    result = model.predict('test_image.jpg')
    results[model_name] = result
    
    print(f"{model_name}: {result['num_detections']} detections")

# 比较结果
for name, result in results.items():
    print(f"\n{name}:")
    print(f"  Detections: {result['num_detections']}")
```

## 四、常用模型配置

### Faster R-CNN

```python
# 使用预训练模型
model = MMDetModelRegistry.get_model('faster-rcnn-r50')

# 或使用自定义配置
from mmdet_integration.mmdet_adapter import MMDetAdapter
adapter = MMDetAdapter(
    config_file='mmdet_training/configs/faster_rcnn_r50_fpn_pestv3.py',
    checkpoint_file='work_dirs/faster_rcnn_pestv3/best.pth'
)
```

### RetinaNet

```python
model = MMDetModelRegistry.get_model('retinanet-r50')
```

### YOLOX

```python
model = MMDetModelRegistry.get_model('yolox-s')  # 或 'yolox-m'
```

### RTMDet

```python
model = MMDetModelRegistry.get_model('rtmdet-s')  # 或 'rtmdet-tiny'
```

## 五、常见问题

### Q: 如何安装 MMDetection？

A: MMDetection 已经包含在项目中。如果需要重新安装：

```bash
cd mmdetection
pip install -e .
pip install mmcv mmengine
```

### Q: 如何转换数据集格式？

A: 如果数据集是 VOC 格式，可以使用：

```bash
python mmdetection/tools/dataset_converters/pascal_voc.py \
    --devkit_path /path/to/VOCdevkit \
    --out-dir /path/to/output
```

### Q: 训练时内存不足怎么办？

A: 在配置文件中减少 batch size：

```python
data = dict(samples_per_gpu=1)  # 原来是 2
```

### Q: 如何调整检测阈值？

A: 在创建模型时设置：

```python
model = MMDetModelRegistry.get_model(
    'faster-rcnn-r50',
    score_threshold=0.5  # 调整阈值
)
```

## 六、完整示例

```python
#!/usr/bin/env python3
"""完整的使用示例"""

from mmdet_integration.mmdet_adapter import MMDetModelRegistry

# 1. 加载模型
print("加载 Faster R-CNN 模型...")
model = MMDetModelRegistry.get_model(
    'faster-rcnn-r50',
    device='cuda:0',
    score_threshold=0.3
)

# 2. 进行推理
print("进行推理...")
result = model.predict('test_image.jpg')

# 3. 显示结果
print(f"\n检测结果:")
print(f"  检测到 {result['num_detections']} 个目标")
for i, det in enumerate(result['detections'][:5], 1):  # 显示前5个
    print(f"  {i}. {det['class_name']}: {det['score']:.2f}")
    print(f"     位置: {det['bbox']}")
```

## 七、更多资源

- 详细文档：`mmdet_training/README.md`
- MMDetection 官方文档：https://mmdetection.readthedocs.io/
- 模型库：https://mmdetection.readthedocs.io/en/latest/model_zoo.html









