import os
import random
from shutil import copy2

# Define paths
base_dir = 'out'
diff_images_dir = os.path.join(base_dir, 'diff', 'images', 'val')
diff_labels_dir = os.path.join(base_dir, 'diff', 'labels', 'val')
easy_images_dir = os.path.join(base_dir, 'easy', 'images', 'val')
easy_labels_dir = os.path.join(base_dir, 'easy', 'labels', 'val')

train_ratio = 0.8  # 80% of the data will be used for training
val_ratio = 1 - train_ratio  # The remaining 20% for validation

# Create output directories with subdirectories for each difficulty level
output_dirs = {
    'train_easy_images': os.path.join(base_dir, 'trainval', 'easy', 'images', 'train',),
    'train_easy_labels': os.path.join(base_dir, 'trainval', 'easy', 'labels', 'train',),
    'train_diff_images': os.path.join(base_dir, 'trainval', 'diff', 'images', 'train',),
    'train_diff_labels': os.path.join(base_dir, 'trainval', 'diff', 'labels', 'train',),
    'val_easy_images': os.path.join(base_dir, 'trainval', 'easy', 'images', 'val'),
    'val_easy_labels': os.path.join(base_dir, 'trainval', 'easy', 'labels', 'val'),
    'val_diff_images': os.path.join(base_dir, 'trainval', 'diff', 'images', 'val'),
    'val_diff_labels': os.path.join(base_dir, 'trainval', 'diff', 'labels', 'val')
}

for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

def split_and_copy(files, labels, train_ratio, output_dirs, difficulty):
    # Shuffle files while keeping corresponding labels in sync
    combined = list(zip(files, labels))
    random.seed(42)  # For reproducibility
    random.shuffle(combined)
    files[:], labels[:] = zip(*combined)

    # Split indices
    split_index = int(len(files) * train_ratio)
    
    # Copy training files
    for file, label in zip(files[:split_index], labels[:split_index]):
        copy2(file, output_dirs[f'train_{difficulty}_images'])
        copy2(label, output_dirs[f'train_{difficulty}_labels'])

    # Copy validation files
    for file, label in zip(files[split_index:], labels[split_index:]):
        copy2(file, output_dirs[f'val_{difficulty}_images'])
        copy2(label, output_dirs[f'val_{difficulty}_labels'])

# Load image and label paths
diff_image_files = [os.path.join(diff_images_dir, f) for f in os.listdir(diff_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
diff_label_files = [os.path.join(diff_labels_dir, f) for f in os.listdir(diff_labels_dir) if f.lower().endswith('.txt')]  # Assuming labels are .txt files

easy_image_files = [os.path.join(easy_images_dir, f) for f in os.listdir(easy_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
easy_label_files = [os.path.join(easy_labels_dir, f) for f in os.listdir(easy_labels_dir) if f.lower().endswith('.txt')]

# Ensure that there is a corresponding label for each image and vice versa
diff_image_files.sort()
diff_label_files.sort()
easy_image_files.sort()
easy_label_files.sort()

# Split and copy both difficulty levels
split_and_copy(diff_image_files, diff_label_files, train_ratio, output_dirs, 'diff')
split_and_copy(easy_image_files, easy_label_files, train_ratio, output_dirs, 'easy')

print("Dataset splitting completed.")