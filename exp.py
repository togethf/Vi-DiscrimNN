# python validate_ensemble_ultralytics.py   --models '/home/insslab/paper3/runs/lightweight/pest_v3/yolov11n-C3k2-HetConv2/weights/best.pt' '/home/insslab/paper3/runs/lightweight/pest_v3/yolov11n-VAE-SE2/weights/best.pt'  --data cfg/datasets/pest/v3.yaml   --method wbf   --model_weights "0.8,1.0"   --min_models 2   --ensemble_iou 0.5   --score_power 1.5   --final_nms_iou 0.5   --final_nms_conf 0.001


python validate_ensemble_ultralytics.py \
  --models '/home/insslab/paper3/runs/lightweight/pest_v3/yolov11n-C3k2-HetConv2/weights/best.pt' '/home/insslab/paper3/runs/lightweight/pest_v3/yolov11n-VAE-SE2/weights/best.pt' \
  --data cfg/datasets/pest/v3.yaml \
  --method wbf \
  --auto_tune \
  --baseline_map50 0.945
