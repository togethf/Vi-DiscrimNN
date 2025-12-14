"""
Faster R-CNN configuration for PestV3 dataset.

This config is based on the standard Faster R-CNN R50 FPN config,
adapted for the PestV3 dataset with 10 classes.
"""

_base_ = [
    '../../mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
]

# Dataset settings (override COCO defaults)
dataset_type = 'CocoDataset'
data_root = '/home/insslab/Desktop/datasets/RicePestsV3/VOCdevkit/images/'

# Class names for PestV3 (must match classes.txt)
classes = [
    'DaoZhong',
    'ErHua',
    'DaMingShen',
    'HeiBai',
    'DaoMingLing',
    'YuMiMing',
    'YangXue',
    'LouGu',
    'JinGui'
]

# Model settings
model = dict(roi_head=dict(bbox_head=dict(num_classes=len(classes))))

# Pipelines inherit from base; we only override dataloader/evaluator paths.

# Dataloaders (MMDet 3.x style)
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        # file_name in annotations already contains split folder (e.g., train/xxx)
        data_prefix=dict(img=''),
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        backend_args=None))

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        metainfo=dict(classes=classes),
        backend_args=None))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    backend_args=None)

test_evaluator = val_evaluator

# Training settings
train_cfg = dict(
    max_epochs=400,  # match YOLO epochs
    val_interval=1   # keep per-epoch eval; raise if too slow
)

# Optimizer settings
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.005,          # lower LR for stability
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True
    ),
    clip_grad=dict(max_norm=35, norm_type=2)  # prevent exploding grads
)

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=5e-4,  # gentler warmup
        by_epoch=True,
        begin=0,
        end=3
    ),
    dict(
        type='CosineAnnealingLR',
        begin=3,
        end=400,
        by_epoch=True,
        eta_min=1e-4
    )
]

# Runtime settings
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,  # Save checkpoint every epoch
        max_keep_ckpts=3,  # Keep only the latest 3 checkpoints
        save_best='auto'  # Save best model based on validation metric
    ),
    logger=dict(type='LoggerHook', interval=50)
)

# Work directory
work_dir = './work_dirs/faster_rcnn_r50_fpn_pestv3'

