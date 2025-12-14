#!/usr/bin/env python3
"""
Testing/Evaluation script for MMDetection models.

This script evaluates trained MMDetection models on test datasets.

Usage:
    python mmdet_training/test_mmdet.py \
        --config configs/faster_rcnn_r50_fpn_pestv3.py \
        --checkpoint work_dirs/faster_rcnn_pestv3/latest.pth \
        --eval mAP
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'mmdetection'))

try:
    from mmengine.config import Config
    from mmengine.runner import Runner
    import torch
    MMDET_AVAILABLE = True
except ImportError as e:
    MMDET_AVAILABLE = False
    print(f"Error: MMDetection not available: {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Test MMDetection model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--eval',
        type=str,
        default='mAP',
        choices=['mAP', 'bbox', 'segm'],
        help='Evaluation metric'
    )
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU ID to use'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show results'
    )
    parser.add_argument(
        '--show-dir',
        type=str,
        default=None,
        help='Directory to save visualization results'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action='append',
        help='Override config options'
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
    
    # Set device
    if torch.cuda.is_available():
        config.device = f'cuda:{args.gpu_id}'
    else:
        config.device = 'cpu'
    
    # Build runner
    runner = Runner.from_cfg(config)
    
    # Load checkpoint
    runner.load_checkpoint(args.checkpoint)
    print(f"✓ Loaded checkpoint from {args.checkpoint}")
    
    # Run evaluation
    print(f"\n{'='*80}")
    print(f"Starting evaluation...")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {config.device}")
    print(f"Metric: {args.eval}")
    print(f"{'='*80}\n")
    
    # Evaluate
    metrics = runner.test()
    
    # Print results
    print(f"\n{'='*80}")
    print("Evaluation Results:")
    print(f"{'='*80}")
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            print(f"{key}: {value}")
    else:
        print(metrics)
    print(f"{'='*80}\n")
    
    # Save visualization if requested
    if args.show or args.show_dir:
        print("Generating visualizations...")
        # This would require additional code to generate visualizations
        # For now, we'll just note that it's requested
        if args.show_dir:
            print(f"Visualizations would be saved to: {args.show_dir}")


if __name__ == '__main__':
    main()









