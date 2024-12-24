import sys
import os

cwd = os.getcwd()
IMGSZ = (640, 640)

judge_config = {
    'pestv3': {
        'threshold': 1-0.5, # 2是中位数，-0.5是为了尽可能让模棱两可的样本是困难样本
        'models': [
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'pestv3', '11n.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'pestv3', '11m.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'pestv3', '11l.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'pestv3', '11x.pt'),
        ]
    },
    'voc07': {
        'threshold': 2-0.5, # 2是中位数，-0.5是为了尽可能让模棱两可的样本是困难样本
        'models': [
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'voc07', '11n.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'voc07', '11m.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'voc07', '11l.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'voc07', '11x.pt'),
        ]
    },
    'voc12': {
        'threshold': 0.6, # 2是中位数，-0.5是为了尽可能让模棱两可的样本是困难样本
        'models': [
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'voc12', '11n.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'voc12', '11m.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'voc12', '11l.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'voc12', '11x.pt'),
        ]
    },
    'coco': {
        'threshold': 2-0.5,
        'models': [
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'coco', '11n.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'coco', '11m.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'coco', '11l.pt'),
            os.path.join(cwd, 'checkpoint', 'model_zoo', 'coco', '11x.pt'),
        ]
    }
}
coco_config = {
    'weak_detector': os.path.join(cwd, 'checkpoint', 'weak_det', 'coco', '11n.pt'),
    'strong_detector': os.path.join(cwd, 'checkpoint', 'strong_det', 'coco', '11x.pt'),
    # means where stores train/val/test
    'source_images': r'/home/insslab/Desktop/datasets/coco2014/images/',
    'source_labels': r'/home/insslab/Desktop/datasets/coco2014/labels/',
    'cfg': os.path.join(cwd, 'cfg', 'datasets', 'coco', 'coco.yaml'),
}

voc07_config = {
    'weak_detector': os.path.join(cwd, 'checkpoint', 'weak_det', 'voc07', '11n.pt'),
    'strong_detector': os.path.join(cwd, 'checkpoint', 'strong_det', 'voc07', '11x.pt'),
    # means where stores train/val/test
    'source_images': r'/home/insslab/Desktop/datasets/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/images/',
    'source_labels': r'/home/insslab/Desktop/datasets/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/labels/',
    'cfg': os.path.join(cwd, 'cfg', 'datasets', 'voc07', 'voc07.yaml'),
}

voc12_config = {
    'weak_detector': os.path.join(cwd, 'checkpoint', 'weak_det', 'voc12', '11n.pt'),
    'strong_detector': os.path.join(cwd, 'checkpoint', 'strong_det', 'voc12', '11x.pt'),
    # means where stores train/val/test
    'source_images': r'/home/insslab/Desktop/datasets/object_detection/images/',
    'source_labels': r'/home/insslab/Desktop/datasets/object_detection/labels/',
    'cfg': os.path.join(cwd, 'cfg', 'datasets', 'voc12', 'voc12.yaml'),
}

pestv3_config = {
    'weak_detector': os.path.join(cwd, 'checkpoint', 'weak_det', 'pestv3', '11n.pt'),
    'strong_detector': os.path.join(cwd, 'checkpoint', 'strong_det', 'pestv3', '11x.pt'),
    # means where stores train/val/test
    'source_images': r'/home/insslab/Desktop/datasets/RicePestsV3/VOCdevkit/images/',
    'source_labels': r'/home/insslab/Desktop/datasets/RicePestsV3/VOCdevkit/labels/',
    # yolo的cfg文件
    'cfg': os.path.join(cwd, 'cfg', 'datasets', 'pest', 'v3.yaml'),
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
    'classifier': os.path.join(cwd, 'checkpoint', 'classifier', 'best_model.pth'),
}