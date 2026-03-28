"""
RTMDet-L for PestV3 (COCO-format)
Based on: mmdetection/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py
"""

_base_ = ['../../mmdetection/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py']

# ================= 1. 预训练权重 (关键！) =================
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'

# ================= 2. 数据集设置 =================
dataset_type = 'CocoDataset'
data_root = '/home/insslab/Desktop/datasets/RicePestsV3/VOCdevkit/images/'

classes = [
    'DaoZhong', 'ErHua', 'DaMingShen', 'HeiBai', 'DaoMingLing',
    'YuMiMing', 'YangXue', 'LouGu', 'JinGui'
]

# ================= 3. 模型头设置 =================
model = dict(
    bbox_head=dict(
        num_classes=len(classes),
    )
)

# ================= 4. 训练参数 =================
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img=''),
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        backend_args=None
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        metainfo=dict(classes=classes),
        backend_args=None
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    classwise=True,
    backend_args=None
)

test_evaluator = val_evaluator

# ================= 5. 优化器与学习率 (适配 BS=8) =================
# RTMDet 默认配置针对 BS=256，这里必须手动调整以适应小 BS
optim_wrapper = dict(
    _delete_=True,  # 删除继承的优化器配置，完全使用下面的配置
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 降低学习率 (原版是 0.004)
        weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True)
)

# ================= 6. 调度策略 =================
train_cfg = dict(max_epochs=200, val_interval=5)

# 学习率调度 (Cosine)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=1.0e-5,
        begin=100,      # max_epochs // 2
        end=200,        # max_epochs
        T_max=100,      # max_epochs // 2
        by_epoch=True,
        convert_to_iter_based=True)
]

work_dir = './work_dirs/rtmdet_l_pestv3'