import sys
import os

cwd = os.getcwd()
IMGSZ = (640, 640)

judge_config = {
    'pest24': {
        'threshold': 0.4, # 
        'models': [
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'pest24', '11n.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'pest24', '11m.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'pest24', '11l.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'pest24', '11x.pt'),
        ]
    },
    'ip102': {
        'threshold': 0.4, # 
        'models': [
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'ip102', '11n.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'ip102', '11m.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'ip102', '11l.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'ip102', '11x.pt'),
        ]
    },
    'pestv1': {
        'threshold': 0.4, # 
        'models': [
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'pestv1', '11n.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'pestv1', '11m.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'pestv1', '11l.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'pestv1', '11x.pt'),
        ]
    },
    'visdrone': {
        'threshold': 0.4, # 
        'models': [
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'visdrone', '11n.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'visdrone', '11m.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'visdrone', '11l.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'visdrone', '11x.pt'),
        ]
    },
    'pestv3': {
        'threshold': 0.35, # 
        'models': [
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'pestv3', '11n.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'pestv3', '11m.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'pestv3', '11l.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'pestv3', '11x.pt'),
        ]
    },
    'voc07': {
        'threshold': 0.8, # 
        'models': [
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'voc07', '11n.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'voc07', '11m.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'voc07', '11l.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'voc07', '11x.pt'),
        ]
    },
    'voc12': {
        'threshold': 0.6, # 
        'models': [
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'voc12', '11n.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'voc12', '11m.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'voc12', '11l.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'voc12', '11x.pt'),
        ]
    },
    'coco': {
        'threshold': 0.5,
        'models': [
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'coco', '11n.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'coco', '11m.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'coco', '11l.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'coco', '11x.pt'),
        ]
    }
}

pestv1_config = {
    'weak_detector': os.path.join(cwd, 'checkpoint', 'weak_det', 'pestv1', '11n.pt'),
    'strong_detector': os.path.join(cwd, 'checkpoint', 'strong_det', 'pestv1', '11x.pt'),
    # means where stores train/val/test
    'source_images': r'/home/insslab/Desktop/datasets/RicePestsV1/VOCdevkit/images/',
    'source_labels': r'/home/insslab/Desktop/datasets/RicePestsV1/VOCdevkit/labels/',
    'cfg': os.path.join(cwd, 'cfg', 'datasets', 'pest', 'v1.yaml'),
    'output_easy_dir': os.path.join(cwd, 'out', 'pestv1', 'easy'),
    'output_diff_dir': os.path.join(cwd, 'out', 'pestv1', 'diff'),
}

pest24_config = {
    'weak_detector': os.path.join(cwd, 'checkpoint', 'weak_det', 'pest24', '11n.pt'),
    'strong_detector': os.path.join(cwd, 'checkpoint', 'strong_det', 'pest24', '11x.pt'),
    # means where stores train/val/test
    'source_images': r'/home/insslab/Desktop/datasets/pest24/Pest24/VOCdevkit/voc2007/split/images/',
    'source_labels': r'/home/insslab/Desktop/datasets/pest24/Pest24/VOCdevkit/voc2007/split/labels/',
    'cfg': os.path.join(cwd, 'cfg', 'datasets', 'pest24', 'pest24.yaml'),
    'output_easy_dir': os.path.join(cwd, 'out', 'pest24', 'easy'),
    'output_diff_dir': os.path.join(cwd, 'out', 'pest24', 'diff'),
}

ip102_config = {
    'weak_detector': os.path.join(cwd, 'checkpoint', 'weak_det', 'ip102', '11n.pt'),
    'strong_detector': os.path.join(cwd, 'checkpoint', 'strong_det', 'ip102', '11x.pt'),
    # means where stores train/val/test
    'source_images': r'/home/insslab/Desktop/datasets/ip102/IP102_YOLOv5/images/',
    'source_labels': r'/home/insslab/Desktop/datasets/ip102/IP102_YOLOv5/labels/',
    'cfg': os.path.join(cwd, 'cfg', 'datasets', 'ip102', 'ip102.yaml'),
    'output_easy_dir': os.path.join(cwd, 'out', 'ip102', 'easy'),
    'output_diff_dir': os.path.join(cwd, 'out', 'ip102', 'diff'),
}

visdrone_config = {
    'weak_detector': os.path.join(cwd, 'checkpoint', 'weak_det', 'visdrone', '11n.pt'),
    'strong_detector': os.path.join(cwd, 'checkpoint', 'strong_det', 'visdrone', '11x.pt'),
    # means where stores train/val/test
    'source_images': r'/home/insslab/Desktop/datasets/visdrone/images/',
    'source_labels': r'/home/insslab/Desktop/datasets/visdrone/labels/',
    'cfg': os.path.join(cwd, 'cfg', 'datasets', 'visdrone', 'visdrone.yaml'),
    'output_easy_dir': os.path.join(cwd, 'out', 'visdrone', 'easy'),
    'output_diff_dir': os.path.join(cwd, 'out', 'visdrone', 'diff'),
}

coco_config = {
    'weak_detector': os.path.join(cwd, 'checkpoint', 'weak_det', 'coco', '11n.pt'),
    'strong_detector': os.path.join(cwd, 'checkpoint', 'strong_det', 'coco', '11x.pt'),
    # means where stores train/val/test
    'source_images': r'/home/insslab/Desktop/datasets/coco2014/images/',
    'source_labels': r'/home/insslab/Desktop/datasets/coco2014/labels/',
    'cfg': os.path.join(cwd, 'cfg', 'datasets', 'coco', 'coco.yaml'),
    'output_easy_dir': os.path.join(cwd, 'out', 'coco', 'easy'),
    'output_diff_dir': os.path.join(cwd, 'out', 'coco', 'diff'),
}

voc07_config = {
    'weak_detector': os.path.join(cwd, 'checkpoint', 'weak_det', 'voc07', '11n.pt'),
    'strong_detector': os.path.join(cwd, 'checkpoint', 'strong_det', 'voc07', '11x.pt'),
    # means where stores train/val/test
    'source_images': r'/home/insslab/Desktop/datasets/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/images/',
    'source_labels': r'/home/insslab/Desktop/datasets/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/labels/',
    'cfg': os.path.join(cwd, 'cfg', 'datasets', 'voc07', 'voc07.yaml'),
    'output_easy_dir': os.path.join(cwd, 'out', 'voc07', 'easy'),
    'output_diff_dir': os.path.join(cwd, 'out', 'voc07', 'diff'),
}

voc12_config = {
    'weak_detector': os.path.join(cwd, 'checkpoint', 'weak_det', 'voc12', '11n.pt'),
    'strong_detector': os.path.join(cwd, 'checkpoint', 'strong_det', 'voc12', '11x.pt'),
    # means where stores train/val/test
    'source_images': r'/home/insslab/Desktop/datasets/object_detection/images/',
    'source_labels': r'/home/insslab/Desktop/datasets/object_detection/labels/',
    'cfg': os.path.join(cwd, 'cfg', 'datasets', 'voc12', 'voc12.yaml'),
    'output_easy_dir': os.path.join(cwd, 'out', 'voc12', 'easy'),
    'output_diff_dir': os.path.join(cwd, 'out', 'voc12', 'diff'),
}

pestv3_config = {
    'weak_detector': os.path.join(cwd, 'checkpoint', 'weak_det', 'pestv3', '11n.pt'),
    'strong_detector': os.path.join(cwd, 'checkpoint', 'strong_det', 'pestv3', '11x.pt'),
    # means where stores train/val/test
    'source_images': r'/home/insslab/Desktop/datasets/RicePestsV3/VOCdevkit/images/',
    'source_labels': r'/home/insslab/Desktop/datasets/RicePestsV3/VOCdevkit/labels/',
    # yolo的cfg文件
    'cfg': os.path.join(cwd, 'cfg', 'datasets', 'pest', 'v3.yaml'),
    'output_easy_dir': os.path.join(cwd, 'out', 'pestv3', 'easy'),
    'output_diff_dir': os.path.join(cwd, 'out', 'pestv3', 'diff'),
}

tag_config = {
    'output_dir': os.path.join(cwd, 'out'),
    'output_easy_dir': os.path.join(cwd, 'out', 'easy'),
    'output_diff_dir': os.path.join(cwd, 'out', 'diff'),
    'classfier_weight': os.path.join(cwd, 'checkpoint', 'ICNet', 'ck.pth'),
    'desired_mAP50': None, # 这里不设置，需要通过终端输入
    'min_simple_samples': 10,
}

classify_config = {
    'classifier': os.path.join(cwd, 'checkpoint', 'classifier', 'v3_shufflenet05.pth'),
}