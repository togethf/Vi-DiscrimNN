det_pool = [
    '/home/insslab/paper3/runs/lightweight/pest_v3/yolov11n-C3k2-HetConv2/weights/best.pt', # hetconv2, yumiming 0.745
    '/home/insslab/paper3/runs/lightweight/pest_v3/yolo11n-C2PSA-iEMA/weights/best.pt',
    '/home/insslab/paper3/runs/lightweight/pest_v3/yolo11n/weights/best.pt',
    '/home/insslab/paper3/runs/lightweight/pest_v3/yolov11n-C3k2-HetConv2/weights/best.pt', # hetconv2, yumiming 0.745
    '/home/insslab/paper3/runs/lightweight/pest_v3/yolo11n-C2PSA-iEMA/weights/best.pt',
    '/home/insslab/paper3/runs/lightweight/pest_v3/yolo11n-C2PSA-iEMA-HetConv2/weights/best.pt'
]

ds = '/home/insslab/Desktop/datasets/RicePestsV3/VOCdevkit/images/'
model_predict_conf = 0.25
model_predict_iou = 0.7