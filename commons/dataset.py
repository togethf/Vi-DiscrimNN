from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from config import IMGSZ

def extract_label_full(file):
    """提取yolo的标签文件，返回[cls, x, y, w, h]

    Args:
        file (str): 完整路径地址

    Returns:
        _type_: _description_
    """
    rst = []
    with open(file, 'r') as f:
        for line in f.readlines():
            i = 0
            elems = []
            for elem in line.strip().split():
                if i != 0:
                    elems.append(float(elem))
                else:
                    elems.append(int(elem))
                i = i + 1
            rst.append(elems)
    return rst

class ClassifyDataset(Dataset):
    def __init__(self, img_dir, transform=None, train=True):
        self.img_dir = img_dir
        self.transform = transform
        self.train = train  # A flag to distinguish between train and validation sets
        
        # Load paths and labels along with difficulty information
        self.paths, self.labels = self._load_data()
        self.length = len(self.paths)

    def _load_data(self):
        paths = []
        labels = []

        # Define directories based on train/val split
        easy_images_dir = os.path.join(self.img_dir, 'train' if self.train else 'val', 'easy', 'images')
        diff_images_dir = os.path.join(self.img_dir, 'train' if self.train else 'val', 'diff', 'images')

        # Load easy images
        if os.path.exists(easy_images_dir):
            easy_image_files = [os.path.join(easy_images_dir, name) for name in os.listdir(easy_images_dir) if name.lower().endswith(('.png', '.jpg', '.jpeg'))]
            paths.extend(easy_image_files)
            labels.extend([0] * len(easy_image_files))  # Assuming 0 for easy

        # Load difficult images
        if os.path.exists(diff_images_dir):
            diff_image_files = [os.path.join(diff_images_dir, name) for name in os.listdir(diff_images_dir) if name.lower().endswith(('.png', '.jpg', '.jpeg'))]
            paths.extend(diff_image_files)
            labels.extend([1] * len(diff_image_files))  # Assuming 1 for difficult

        return paths, labels

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label # Return difficulty level as well



class DetectionDataset(Dataset):
    def __init__(self, img_dir, label_dir, mode, open=None):
        self.open = open
        self.mode = mode
        self.img_paths = [os.path.join(img_dir, mode, name) for name in os.listdir(os.path.join(img_dir, mode))]
        self.label_paths = [path.replace('images', 'labels').replace('jpg', 'txt') for path in self.img_paths]
        self.length = len(self.img_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = self.img_paths[idx]
        # image = Image.open(img_path).convert('RGB')
        if self.open:
            image = Image.open(image).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize(IMGSZ),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image = transform(image)
        label = self.label_paths[idx]
        return image, label
    
    def collate_fn(batch):
        img, label_path = zip(*batch)
        labels = []
        for i, path in enumerate(label_path):
            l_list = extract_label_full(path)
            for item in l_list:
                item.append(i)  # 将图像索引附加到 bbox
            labels.append(torch.tensor(l_list))
        if isinstance(img[0], str):
            return img, torch.cat(labels, dim=0)
        else:
            return torch.stack(img), torch.cat(labels, dim=0)
