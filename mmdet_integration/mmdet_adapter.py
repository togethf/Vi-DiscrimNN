#!/usr/bin/env python3
"""
MMDetection Adapter for easy model loading and inference.

This module provides a simple interface to load and use MMDetection models
for object detection tasks, especially for comparison experiments.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Add mmdetection to path
project_root = Path(__file__).parent.parent
mmdet_path = project_root / 'mmdetection'
if str(mmdet_path) not in sys.path:
    sys.path.insert(0, str(mmdet_path))

try:
    from mmdet.apis import init_detector, inference_detector
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.registry import MODELS
    MMDET_AVAILABLE = True
except ImportError as e:
    MMDET_AVAILABLE = False
    warnings.warn(f"MMDetection not available: {e}")


class MMDetAdapter:
    """
    Adapter class for MMDetection models.
    Provides a unified interface for loading and using MMDetection models.
    """
    
    def __init__(self, 
                 config_file: Union[str, Path],
                 checkpoint_file: Optional[Union[str, Path]] = None,
                 device: str = 'cuda:0',
                 score_threshold: float = 0.3):
        """
        Initialize MMDetection model.
        
        Args:
            config_file: Path to MMDetection config file
            checkpoint_file: Path to checkpoint file (optional, will download if None)
            device: Device to run inference on
            score_threshold: Confidence threshold for predictions
        """
        if not MMDET_AVAILABLE:
            raise ImportError("MMDetection is not installed. Please install it first.")
        
        self.config_file = str(config_file)
        self.checkpoint_file = str(checkpoint_file) if checkpoint_file else None
        self.device = device
        self.score_threshold = score_threshold
        
        # Load model
        self.model = init_detector(
            config_file=self.config_file,
            checkpoint=self.checkpoint_file,
            device=self.device
        )
        
        # Get class names
        if hasattr(self.model, 'dataset_meta') and 'classes' in self.model.dataset_meta:
            self.class_names = self.model.dataset_meta['classes']
        else:
            self.class_names = None
        
        print(f"✓ Loaded MMDetection model from {config_file}")
        if self.checkpoint_file:
            print(f"✓ Loaded checkpoint from {checkpoint_file}")
        print(f"✓ Device: {device}")
        print(f"✓ Number of classes: {len(self.class_names) if self.class_names else 'Unknown'}")
    
    def predict(self, 
                image: Union[np.ndarray, str, Path, torch.Tensor],
                return_vis: bool = False) -> Union[Dict, Tuple[Dict, np.ndarray]]:
        """
        Run inference on an image.
        
        Args:
            image: Input image (numpy array, file path, or torch tensor)
            return_vis: Whether to return visualization
            
        Returns:
            Dictionary with predictions, or tuple of (predictions, visualization) if return_vis=True
        """
        # Convert torch tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # Batch dimension
                image = image[0]
            image = image.cpu().numpy()
            # Convert from CHW to HWC
            if image.shape[0] == 3 or image.shape[0] == 1:
                image = image.transpose(1, 2, 0)
            # Denormalize if needed (assuming normalized to [0, 1])
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Run inference
        result = inference_detector(self.model, image)
        
        # Format results
        predictions = self._format_results(result)
        
        if return_vis:
            from mmdet.apis import DetInferencer
            inferencer = DetInferencer(
                model=self.config_file,
                weights=self.checkpoint_file,
                device=self.device
            )
            vis_image = inferencer(image, return_vis=True)[1]['visualization']
            return predictions, vis_image
        
        return predictions
    
    def _format_results(self, result) -> Dict:
        """
        Format MMDetection results into a standard format.
        
        Args:
            result: MMDetection inference result
            
        Returns:
            Dictionary with formatted predictions
        """
        if isinstance(result, tuple):
            # Instance segmentation result (bboxes, masks)
            bboxes, masks = result
        else:
            # Detection result (bboxes only)
            bboxes = result
            masks = None
        
        # Convert to list of detections
        detections = []
        
        if isinstance(bboxes, np.ndarray):
            # Single class case
            bboxes = [bboxes]
        
        for class_id, class_boxes in enumerate(bboxes):
            if len(class_boxes) == 0:
                continue
            
            for box in class_boxes:
                if len(box) >= 5:  # x1, y1, x2, y2, score
                    x1, y1, x2, y2, score = box[:5]
                    
                    if score >= self.score_threshold:
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'score': float(score),
                            'class_id': int(class_id),
                            'class_name': self.class_names[class_id] if self.class_names else f'class_{class_id}'
                        })
        
        return {
            'detections': detections,
            'num_detections': len(detections)
        }
    
    def predict_batch(self, images: List[Union[np.ndarray, str, Path]]) -> List[Dict]:
        """
        Run inference on a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results


class MMDetModelRegistry:
    """
    Registry for common MMDetection models.
    Provides easy access to popular detection models.
    """
    
    # Model configurations mapping
    MODEL_CONFIGS = {
        # Faster R-CNN variants
        'faster-rcnn-r50': {
            'config': 'faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py',
            'checkpoint': 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
            'description': 'Faster R-CNN with ResNet-50 backbone'
        },
        'faster-rcnn-r101': {
            'config': 'faster_rcnn/faster-rcnn_r101_fpn_1x_coco.py',
            'checkpoint': 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_1x_coco/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth',
            'description': 'Faster R-CNN with ResNet-101 backbone'
        },
        
        # RetinaNet
        'retinanet-r50': {
            'config': 'retinanet/retinanet_r50_fpn_1x_coco.py',
            'checkpoint': 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth',
            'description': 'RetinaNet with ResNet-50 backbone'
        },
        
        # Cascade R-CNN
        'cascade-rcnn-r50': {
            'config': 'cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py',
            'checkpoint': 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56de8.pth',
            'description': 'Cascade R-CNN with ResNet-50 backbone'
        },
        
        # YOLOX
        'yolox-s': {
            'config': 'yolox/yolox_s_8xb8-300e_coco.py',
            'checkpoint': 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth',
            'description': 'YOLOX-Small'
        },
        'yolox-m': {
            'config': 'yolox/yolox_m_8xb8-300e_coco.py',
            'checkpoint': 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_m_8x8_300e_coco/yolox_m_8x8_300e_coco_20211121_101425-83e3e0fd.pth',
            'description': 'YOLOX-Medium'
        },
        
        # RTMDet
        'rtmdet-tiny': {
            'config': 'rtmdet/rtmdet_tiny_8xb32-300e_coco.py',
            'checkpoint': 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth',
            'description': 'RTMDet-Tiny'
        },
        'rtmdet-s': {
            'config': 'rtmdet/rtmdet_s_8xb32-300e_coco.py',
            'checkpoint': 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth',
            'description': 'RTMDet-Small'
        },
        
        # DETR
        'detr-r50': {
            'config': 'detr/detr_r50_8xb2-150e_coco.py',
            'checkpoint': 'https://download.openmmlab.com/mmdetection/v3.0/detr/detr_r50_8xb2-150e_coco/detr_r50_8xb2-150e_coco_20221023_153551-436af07f.pth',
            'description': 'DETR with ResNet-50 backbone'
        },
    }
    
    @classmethod
    def get_model(cls, 
                  model_name: str,
                  device: str = 'cuda:0',
                  score_threshold: float = 0.3,
                  custom_config: Optional[str] = None,
                  custom_checkpoint: Optional[str] = None) -> MMDetAdapter:
        """
        Get a MMDetection model by name.
        
        Args:
            model_name: Name of the model (e.g., 'faster-rcnn-r50')
            device: Device to run inference on
            score_threshold: Confidence threshold
            custom_config: Optional custom config file path
            custom_checkpoint: Optional custom checkpoint path
            
        Returns:
            MMDetAdapter instance
        """
        if model_name not in cls.MODEL_CONFIGS and not custom_config:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(cls.MODEL_CONFIGS.keys())}"
            )
        
        project_root = Path(__file__).parent.parent
        mmdet_configs = project_root / 'mmdetection' / 'configs'
        
        if custom_config:
            config_file = custom_config
            checkpoint_file = custom_checkpoint
        else:
            model_info = cls.MODEL_CONFIGS[model_name]
            config_file = mmdet_configs / model_info['config']
            checkpoint_file = model_info['checkpoint'] if not custom_checkpoint else custom_checkpoint
        
        return MMDetAdapter(
            config_file=str(config_file),
            checkpoint_file=checkpoint_file,
            device=device,
            score_threshold=score_threshold
        )
    
    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """
        List all available models.
        
        Returns:
            Dictionary mapping model names to descriptions
        """
        return {name: info['description'] for name, info in cls.MODEL_CONFIGS.items()}


if __name__ == '__main__':
    # Test the adapter
    print("Available models:")
    for name, desc in MMDetModelRegistry.list_models().items():
        print(f"  - {name}: {desc}")
    
    print("\nExample usage:")
    print("  adapter = MMDetModelRegistry.get_model('faster-rcnn-r50', device='cuda:0')")
    print("  result = adapter.predict('path/to/image.jpg')")









