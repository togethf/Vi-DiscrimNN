det_pool = [
    '/home/insslab/paper3/runs/lightweight/pest_v3/yolov11n-C3k2-HetConv2/weights/best.pt', # hetconv2, yumiming 0.745
    '/home/insslab/paper3/runs/lightweight/pest_v3/yolo11n-C2PSA-iEMA/weights/best.pt',
    '/home/insslab/paper3/runs/lightweight/pest_v3/yolo11n/weights/best.pt',
    # '/home/insslab/paper3/runs/lightweight/pest_v3/yolo11n-C2PSA-iEMA-HetConv2/weights/best.pt',
    '/home/insslab/paper3/runs/lightweight/pest_v3/yolov11n-VAE-SE2/weights/best.pt',
    'mmdet_training/configs/faster_rcnn_r50_fpn_pestv3.py;work_dirs/faster_rcnn_r50_fpn_pestv3/best_coco_bbox_mAP_epoch_212.pth',
    'mmdet_training/configs/retinanet_r50_fpn_pestv3.py;work_dirs/retinanet_r50_fpn_pestv3/best_coco_bbox_mAP_epoch_78.pth',
    'mmdet_training/configs/rtmdet_l_pestv3.py;work_dirs/rtmdet_l_pestv3/epoch_50.pth',
    '/home/insslab/paper3/runs/lightweight/pest_v3/sota/rtdetr-l/weights/best.pt',
    '/home/insslab/paper3/runs/lightweight/pest_v3/sota/pest-yolo/weights/best.pt',
    '/home/insslab/paper3/runs/lightweight/pest_v3/sota/paddy-yolo/weights/best.pt'
]

ds = '/home/insslab/Desktop/datasets/RicePestsV3/VOCdevkit/images/'
# model_predict_conf = 0.25
# model_predict_conf = 0.01 # vae效果不错
# model_predict_conf = 0.001 # 初版只看map50
model_predict_conf = 0.5
model_predict_iou = 0.7