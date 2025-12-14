#!/usr/bin/env python3
"""
Training script for MMDetection models.

This script provides a convenient way to train MMDetection models on custom datasets.
Supports training Faster-RCNN, RetinaNet, YOLOX, RTMDet, and other models.

Usage:
python mmdet_training/train_mmdet.py \
  --config mmdet_training/configs/faster_rcnn_r50_fpn_pestv3.py \
  --work-dir work_dirs/faster_rcnn_pestv3
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'mmdetection'))

try:
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.registry import RUNNERS
    MMDET_AVAILABLE = True
except ImportError as e:
    MMDET_AVAILABLE = False
    print(f"Error: MMDetection not available: {e}")
    print("Please install MMDetection first:")
    print("  cd mmdetection && pip install -e .")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Train MMDetection model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        default=None,
        help='Working directory for saving logs and checkpoints'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--load-from',
        type=str,
        default=None,
        help='Load pretrained weights (for fine-tuning)'
    )
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU ID to use'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action='append',
        help='Override config options (e.g., model.backbone.depth=50)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = Config.fromfile(args.config)
    
    # Override config options
    if args.cfg_options:
        for option in args.cfg_options:
            for key, value in option:
                keys = key.split('.')
                cfg = config
                for k in keys[:-1]:
                    cfg = getattr(cfg, k)
                setattr(cfg, keys[-1], value)
    
    # Set work directory
    if args.work_dir:
        config.work_dir = args.work_dir
    elif config.get('work_dir', None) is None:
        # Default work directory
        config.work_dir = f"work_dirs/{Path(args.config).stem}"
    
    # Set device
    if torch.cuda.is_available():
        config.device = f'cuda:{args.gpu_id}'
    else:
        config.device = 'cpu'
    
    # Set random seed
    if args.seed is not None:
        config.randomness = dict(seed=args.seed, deterministic=True)
    
    # Create work directory
    os.makedirs(config.work_dir, exist_ok=True)
    
    # Build runner
    runner = Runner.from_cfg(config)
    
    # Load checkpoint if specified
    if args.load_from:
        runner.load_checkpoint(args.load_from)
        print(f"✓ Loaded pretrained weights from {args.load_from}")
    
    # Resume training if specified
    if args.resume:
        runner.resume(args.resume)
        print(f"✓ Resumed training from {args.resume}")
    
    # Start training
    print(f"\n{'='*80}")
    print(f"Starting training...")
    print(f"Config: {args.config}")
    print(f"Work directory: {config.work_dir}")
    print(f"Device: {config.device}")
    print(f"{'='*80}\n")
    
    runner.train()
    
    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"Results saved to: {config.work_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

