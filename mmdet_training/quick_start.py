#!/usr/bin/env python3
"""
Quick start example for using MMDetection models.

This script demonstrates how to:
1. Load a pre-trained MMDetection model
2. Run inference on an image
3. Compare multiple models
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mmdet_integration.mmdet_adapter import MMDetModelRegistry, MMDetAdapter


def example_1_load_pretrained_model():
    """Example 1: Load a pre-trained model and run inference."""
    print("\n" + "="*80)
    print("Example 1: Loading pre-trained Faster R-CNN model")
    print("="*80)
    
    try:
        # Load pre-trained Faster R-CNN
        model = MMDetModelRegistry.get_model(
            'faster-rcnn-r50',
            device='cuda:0' if sys.argv[1:] else 'cpu',
            score_threshold=0.3
        )
        
        print("\nModel loaded successfully!")
        print(f"Class names: {model.class_names[:5]}...")  # Show first 5 classes
        
        # Example: Run inference (uncomment when you have an image)
        # result = model.predict('path/to/your/image.jpg')
        # print(f"\nDetections: {result['num_detections']}")
        # for det in result['detections'][:3]:  # Show first 3 detections
        #     print(f"  - {det['class_name']}: {det['score']:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: This requires MMDetection to be properly installed.")
        print("If you see import errors, try:")
        print("  cd mmdetection && pip install -e .")


def example_2_list_available_models():
    """Example 2: List all available models."""
    print("\n" + "="*80)
    print("Example 2: Available MMDetection models")
    print("="*80)
    
    models = MMDetModelRegistry.list_models()
    for name, desc in models.items():
        print(f"  • {name:20s} - {desc}")


def example_3_use_custom_config():
    """Example 3: Use custom trained model."""
    print("\n" + "="*80)
    print("Example 3: Using custom trained model")
    print("="*80)
    
    config_file = project_root / 'mmdet_training' / 'configs' / 'faster_rcnn_r50_fpn_pestv3.py'
    checkpoint_file = project_root / 'work_dirs' / 'faster_rcnn_pestv3' / 'best.pth'
    
    if config_file.exists():
        try:
            adapter = MMDetAdapter(
                config_file=str(config_file),
                checkpoint_file=str(checkpoint_file) if checkpoint_file.exists() else None,
                device='cuda:0' if sys.argv[1:] else 'cpu'
            )
            print("Custom model loaded successfully!")
        except Exception as e:
            print(f"Error loading custom model: {e}")
            print("Make sure you have trained a model first.")
    else:
        print(f"Config file not found: {config_file}")
        print("Train a model first using:")
        print("  python mmdet_training/train_mmdet.py --config mmdet_training/configs/faster_rcnn_r50_fpn_pestv3.py")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("MMDetection Quick Start Examples")
    print("="*80)
    
    example_2_list_available_models()
    example_1_load_pretrained_model()
    example_3_use_custom_config()
    
    print("\n" + "="*80)
    print("For more information, see: mmdet_training/README.md")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()









