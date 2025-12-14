from pathlib import Path
import json

root = Path("/home/insslab/Desktop/datasets/RicePestsV3/VOCdevkit/images")
ann = json.load(open(root/"annotations/train.json"))
coco_boxes = {}
for a in ann["annotations"]:
    coco_boxes[a["image_id"]] = coco_boxes.get(a["image_id"], 0) + 1
img_id_to_file = {img["id"]: img["file_name"] for img in ann["images"]}

missing = []
for img_id, fname in img_id_to_file.items():
    yolo_txt = (root / fname).with_suffix(".txt")
    # 先看同级 txt，不存在则 images→labels（会得到 labels/train/xxx.txt）
    if not yolo_txt.exists():
        alt = Path(str(yolo_txt).replace("images", "labels"))
        yolo_txt = alt if alt.exists() else yolo_txt
    yolo_cnt = 0
    if yolo_txt.exists():
        yolo_cnt = sum(1 for line in open(yolo_txt) if line.strip())
    coco_cnt = coco_boxes.get(img_id, 0)
    if yolo_cnt != coco_cnt:
        missing.append((fname, yolo_cnt, coco_cnt))

print("total images:", len(img_id_to_file))
print("diff count images:", len(missing))
print("examples:", missing[:20])