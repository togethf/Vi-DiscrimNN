#!/usr/bin/env python3
import json, os, glob
from pathlib import Path
from typing import List, Dict

# 配置路径
root = Path("/home/insslab/Desktop/datasets/RicePestsV3/VOCdevkit/images")
splits = ["train", "val"]   # 若有 test 可加上
classes = [c.strip() for c in open(root / "classes.txt").read().splitlines() if c.strip()]
name2id = {n: i for i, n in enumerate(classes)}
id2name = {i: n for n, i in name2id.items()}

def load_yolo_labels(label_path: Path, img_w: int, img_h: int):
    anns = []
    if not label_path.exists():
        return anns
    for line_id, line in enumerate(open(label_path)):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cid, x, y, w, h = map(float, parts)
        cid = int(cid)
        # YOLO (cx, cy, w, h) normalized -> COCO (x1, y1, w, h) absolute
        x1 = (x - w / 2) * img_w
        y1 = (y - h / 2) * img_h
        bw = w * img_w
        bh = h * img_h
        anns.append((cid, x1, y1, bw, bh))
    return anns

def convert_split(split: str):
    images, annotations = [], []
    ann_id = 1
    img_id = 1
    img_dir = root / split
    for img_file in sorted(glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png"))):
        img_file = Path(img_file)
        # 读取尺寸
        from PIL import Image
        with Image.open(img_file) as im:
            w, h = im.size
        images.append({
            "id": img_id,
            "file_name": str(img_file.relative_to(root)),
            "width": w,
            "height": h
        })
        label_file = img_file.with_suffix(".txt")
        # 若标签在 labels 子目录：
        alt_label = Path(str(label_file).replace("images", "labels"))
        if not label_file.exists() and alt_label.exists():
            label_file = alt_label
        for cid, x1, y1, bw, bh in load_yolo_labels(label_file, w, h):
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cid + 1,  # COCO 类别从 1 开始
                "bbox": [x1, y1, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_id += 1
        img_id += 1

    coco = {
        # Minimal required COCO fields to satisfy pycocotools
        "info": {"description": "PestV3 YOLO to COCO", "version": "1.0"},
        "licenses": [{"id": 1, "name": "unknown", "url": ""}],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i + 1, "name": name, "supercategory": "object"}
            for i, name in enumerate(classes)
        ]
    }
    out_dir = root / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{split}.json"
    with open(out_file, "w") as f:
        json.dump(coco, f)
    print(f"Saved {out_file}, images={len(images)}, anns={len(annotations)}")

if __name__ == "__main__":
    for s in splits:
        convert_split(s)