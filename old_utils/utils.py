import difflib
import os
from collections import Counter
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def suffix_change(name, t):
    """替换文件的后缀名

    Args:
        names (str): 需要替换的文件
        t (str): 替换的格式

    Returns:
        替换后的文件名: _description_
    """
    prefix = name.split('.')[0]
    prefix = prefix + '.' + t
    return prefix


def label_delete(file, src, verbose=False):
    """将yolo标签文件中的某些类别删除

    Args:
        file (str): yolo标签文件
        src (iterable[int]): 要删除的类别
        verbose (bool): 是否打印替换信息
    """
    rst = []
    with open(file, 'r') as f:
        for line in f.readlines():
            label = int(line.split()[0])
            if label in src:
                if (label in src) and verbose:
                    print(f'delete {label} once in {file}')
                continue
            rst.append(line)
    with open(file, 'w') as f:
        f.writelines(rst)


def label_replace(file, src, target, verbose=False):
    """将yolo标签文件中的某些类别替换成指定类别

    Args:
        file (str): yolo标签文件, 完整路径
        src (iterable[int]): 被替换的类别
        target (int): 替换的类别
        verbose (bool): 是否打印替换信息
    """
    rst = []
    with open(file, 'r') as f:
        for line in f.readlines():
            label = int(line.split()[0])
            newline = line.replace(str(label), str(target), 1) if label in src else line
            rst.append(newline)
            if (label in src) and verbose:
                print(f'replace {label} to {target} in {file}')
    with open(file, 'w') as f:
        f.writelines(rst)


def move(src, dst, data=None):
    """将src中的data数据移动到dst，data可以是一个字符串列表

    Args:
        src (str): src目录
        dst (str): dst 目录
        data (iterable): src目录下的特定文件list
    """
    files = data if data else os.listdir(src)
    if not os.path.exists(dst):
        os.makedirs(dst)
    for file in files:
        shutil.copyfile(os.path.join(src, file), os.path.join(dst, file))


def make_yoloPath(img_dir, dst):
    """制作yolo的txt文件，根据img_dir中的图片名称，在dst文件中写入

    Args:
        img_dir (_type_): _description_
        dst (_type_): _description_
    """
    files = os.listdir(img_dir)
    with open(dst, 'w') as f:
        for file in files:
            file_path = os.path.join(img_dir, file)
            f.write(file_path + '\n')

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

def _extract_label(file):
    """从yolo标签文件中提取出所有的标签，不去重

    Args:
        file (str): yolo label文件

    Returns:
        list: label list
    """
    rst = []
    with open(file, 'r') as f:
        for line in f.readlines():
            label = line.strip().split()[0]
            rst.append(int(label))
    return rst


def extract_pic_byclass(label_dir, category, yolo=False):
    """
    找出标签目录下（存在了txt标签的目录）指定类别的文件
    :param label_dir:
    :param category:
    :return: list，指定类别的样本的路径列表
    """
    rst = []
    if yolo:
        with open(label_dir, 'r') as f:
            files = [line.strip() for line in f.readlines()]
    else:
        files = os.listdir(label_dir)
    for file in files:
        label_path = os.path.join(label_dir, file)
        if category in _extract_label(label_path):
            rst.append(label_path)
    return rst


def compare_file(f1, f2):
    with open(f1, 'r', encoding='UTF-8') as file1, open(f2, 'r', encoding='UTF-8') as file2:
        diff = difflib.ndiff(file1.readlines(), file2.readlines())
    print('\n'.join(diff))


def read_file2list(file):
    results = []
    with open(file, 'r', encoding='UTF-8') as f:
        rst = f.readlines()
        for line in rst:
            line = line.strip()
            results.append(line)
    return results


def class_inGroup(path):
    """
    对path路径下的label按照类别进行划分，得到counter()，key是类别，value是这个类别的paths，虽然能得到路径，但是由于//
    一个图片会有多个物体，所以一个label路径会多次出现在counter中，这意味着数据集中一个图片会多次使用。
    :param path:
    :return:
    """
    files = os.listdir(path)
    category_paths = Counter()
    for file in tqdm(files):
        label = os.path.join(path, file)
        with open(label, 'r') as f:
            for line in f.readlines():
                category_id = int(line.split()[0])
                if category_paths[category_id] == 0:
                    category_paths[category_id] = []
                category_paths[category_id].append(label)
    return category_paths


def _count_label(label_file, mapping=None):
    """
    统计一个yolo label文件中各个类别的数量
    :param label_file:
    :param mapping:
    :return: counter
    """
    c = Counter()
    with open(label_file) as f:
        lines = f.readlines()
        for line in lines:
            category_id = int(line.split()[0])
            if mapping:
                category_id = mapping[category_id]
            c[category_id] += 1
    return c

def class_count_bytxt(file, mapping=None):
    """
    统计yolo的txt文件指定的数据集的类被情况
    :param file:
    :param mapping:
    :return: Counter，包含了每个类别的数量
    """
    if mapping:
        mapping = read_file2list(mapping)
    with open(file, 'r') as f:
        c = Counter()
        lines = f.readlines()
        for line in lines:
            path = line.strip().replace('images', 'labels').replace('.jpg', '.txt')
            c += _count_label(path, mapping)
    return c



def class_count(path, class_mapping=None):
    """
        统计指定路径下（标签）各个类别的数量

        args:
            data_path: 类似这样的路径，'G:\science_data\datasets\RicePestsv3_tail\VOCdevkit\labels'
        return:
            先yield训练集的结果，再yield验证集的结果
            print(next(class_count(data_path)))
    """
    if class_mapping:
        class_mapping = read_file2list(class_mapping)
    label_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]
    # 统计每个类别的样本数量
    category_counts = Counter()

    for label_file in tqdm(label_files):
        category_counts += _count_label(label_file, class_mapping)

    # print(f"{t} data count: ", category_counts)
    return category_counts


def extract_tail_datasets(images_dir, labels_dir, new_images_dir, new_labels_dir, target_classes):
    """
        从原来的数据集中创建子数据，只由部分类别构成
    """
    # 创建新数据集目录
    os.makedirs(new_images_dir, exist_ok=True)
    os.makedirs(new_labels_dir, exist_ok=True)
    # 遍历标签文件，提取目标类别的数据
    for label_file in tqdm(os.listdir(labels_dir)):
        if label_file.endswith('.txt'):
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'r') as f:
                lines = f.readlines()

            # 过滤出包含目标类别的行
            filtered_lines = [line for line in lines if int(line.split()[0]) in target_classes]

            if filtered_lines:
                # 复制图像文件
                image_file = label_file.replace('.txt', '.jpg')
                image_path = os.path.join(images_dir, image_file)
                new_image_path = os.path.join(new_images_dir, image_file)
                shutil.copy(image_path, new_image_path)

                # 创建新的标签文件
                new_label_path = os.path.join(new_labels_dir, label_file)
                with open(new_label_path, 'w') as f:
                    f.writelines(filtered_lines)

    print("新数据集创建完成")
    g = class_count(new_labels_dir.split('train')[0])
    print("class count:", next(g))


def make_tail_datasets():
    old_img = 'G:\science_data\datasets\RicePestsv3\VOCdevkit\images\\train'
    old_labels = 'G:\science_data\datasets\RicePestsv3\VOCdevkit\labels\\train'
    new_img = 'G:\science_data\datasets\RicePestsv3_tail\VOCdevkit\images\\train'
    new_labels = 'G:\science_data\datasets\RicePestsv3_tail\VOCdevkit\labels\\train'
    extract_tail_datasets(old_img, old_labels, new_img, new_labels, [3, 4, 5, 6, 7, 8])


def visual_counts(*counter):
    """
    对Counter对象中的类别数量进行一个可视化呈现。
    :param counter:
    :return: None
    """
    for n, c in enumerate(counter):
        print(f'{n}: {c}')
        length = len(counter)
        plt.subplot(length, 1, n+1)
        plt.xlabel('class')
        plt.ylabel('count')
        plt.barh(c.keys(), c.values())
    plt.show()

def make_datasets_category(src, dst, mapping):
    """
    将src数据集按类别重新打包成新的数据集目录
    Args:
        src (str): 存放原始数据的label目录， 如G:\science_data\datasets\RicePestsv3\VOCdevkit\labels\\val
        dst (str): 新创建的数据集目录，如G:\science_data\datasets\RicePestv3_category
        mapping (iterable): classes.txt的路径
    """
    category = read_file2list(mapping)
    for i in range(len(category)):
        paths = extract_pic_byclass(src, i)
        for path in paths:
            path = path.replace('labels', 'images').replace('txt', 'jpg')
            if not os.path.exists(os.path.join(dst, category[i])):
                os.makedirs(os.path.join(dst, category[i]))
            shutil.copyfile(path, os.path.join(dst, category[i], path.split(os.path.sep)[-1]))

def yolopath_version_change(txt, old, new):
    lines = [item.replace(old, new) for item in read_file2list(txt)]
    with open(txt, 'w') as f:
        for line in lines:
            f.write(line.strip() + '\n')

def remove_chars(s, chars_to_remove, limit=None):
    for char in chars_to_remove:
        if limit:
            s = s.replace(char, '', limit)
        else:
            s = s.replace(char, '')
    return s



def add_prefix_files(dir, prefix):
     # 检查文件是否已经有 "mask_" 前缀
    for filename in os.listdir(dir):
        # 构建原始文件路径
        original_path = os.path.join(dir, filename)

        # 检查文件是否已经有 "mask_" 前缀
        if not filename.startswith(prefix):
            # 构建新的文件路径
            new_filename = prefix + filename
            new_path = os.path.join(dir, new_filename)

            # 重命名文件
            os.rename(original_path, new_path)
            print(f"Renamed '{filename}' to '{new_filename}'")
        else:
            print(f"Skipped '{filename}' (already has prefix)")
    
if __name__ == '__main__':
    # print(extract_label_full(r'E:\code\exp\animals\train\labels\108_jpgrf1d80b6f5f62039cbe57e0f9f51de082c.txt'))


    # 'G:\science_data\datasets\RicePestsv3\VOCdevkit\VOC2007\labels'
    # easy = [suffix_change(item, 'txt') for item in np.load('E:\code\IC9600\out\easy.npy')]
    # move('G:\science_data\datasets\RicePestsv3\VOCdevkit\VOC2007\labels', 'G:\science_data\datasets\RicePestsv3_easy\labels', easy)
    # for f in os.listdir('G:\science_data\datasets\RicePestsV1_less\VOCdevkit\VOC2007\labels'):
    #     f = 'G:\science_data\datasets\RicePestsV1_less\VOCdevkit\VOC2007\labels\\' + f
    #     label_delete(f, [6])
    #     label_replace(f, [7], 6)
    #     label_replace(f, [8], 7)
    #     label_replace(f, [9], 8)

    # yolopath_version_change('E:\code\mainline\datasets\RicePestV1\\train.txt', 'RicePestsv1', 'RicePestsV1')
    # label_path = 'G:\science_data\datasets\RicePestsV2\labels'
    # for file in os.listdir(label_path):
    #     file_path = os.path.join(label_path, file)
    #     label_replace(file_path, [0, 1], 0)
    #     label_replace(file_path, [2, 3, 4, 5], 1)
    #     label_replace(file_path, [6, 7, 8, 9], 2)
    #     label_replace(file_path, [10, 11], 3)
    #     label_replace(file_path, [12, 13], 4)
    #     label_replace(file_path, [14, 15], 5)
    #     label_replace(file_path, [16, 17], 6)
    #     label_replace(file_path, [18, 19], 7)

    # first
    # dir_path = r'E:\code\exp\animals\valid'
    # mode = 'valid'
    # for tp in ['images', 'labels']: 
    #     for file in os.listdir(dir_path + '/' + tp):
    #         os.rename(os.path.join(dir_path + '/' + tp, file), os.path.join(dir_path + '/' + tp, remove_chars(file, '.', 2)))
    # second IC9600

    # third move
    # difficult = np.load('E:\code\IC9600\out\difficult.npy').tolist()
    # easy = np.load('E:\code\IC9600\out\easy.npy').tolist()
    # move(dir_path + '/images', dir_path + '/difficult' + '/images/val', difficult)
    # move(dir_path + '/images', dir_path + '/easy' + '/images/val', easy)
    # difficult = [suffix_change(item, 'txt') for item in difficult]
    # easy = [suffix_change(item, 'txt') for item in easy]
    # move(dir_path + '/labels', dir_path + '/difficult' + '/labels/val', difficult)
    # move(dir_path + '/labels', dir_path + '/easy' + '/labels/val', easy)

    # fourth datasetUtils.py renameDirfiles
    # fifth yolo2coco
    # make_yoloPath(r'E:\code\exp\animals\valid\images', r'E:\code\exp\animals\valid\val.txt')

    add_prefix_files(r'G:\science_data\datasets\RicePestsV4\VOCdevkit\crop_version\labels', 'crop_')