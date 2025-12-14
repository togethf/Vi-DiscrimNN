"""
RetinaNet R50-FPN for PestV3 (COCO-format), stable training setup.
"""

_base_ = ['../../mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py']

# Dataset
dataset_type = 'CocoDataset'
data_root = '/home/insslab/Desktop/datasets/RicePestsV3/VOCdevkit/images/'

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

# Model
model = dict(bbox_head=dict(num_classes=len(classes)))

# Dataloaders
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
        data_prefix=dict(img=''),  # file_name already includes split folder
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

# Train settings
train_cfg = dict(
    max_epochs=400,
    val_interval=1
)

# Optimizer (stable)
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.005,          # lower LR for stability with batch=16
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# LR schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=5e-4,
        by_epoch=True,
        begin=0,
        end=3),
    dict(
        type='CosineAnnealingLR',
        begin=3,
        end=400,
        by_epoch=True,
        eta_min=1e-4)
]

# Hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='auto'),
    logger=dict(type='LoggerHook', interval=50)
)

# Work dir
work_dir = './work_dirs/retinanet_r50_fpn_pestv3'

