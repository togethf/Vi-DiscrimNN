from skimage import io
from skimage.measure import shannon_entropy
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os
import matplotlib.pyplot as plt
from commons import dataset
# Image Entropy
def calculate_entropy(image):
    entropy = shannon_entropy(image)
    return entropy

# Edge Density
def calculate_edge_density(image):
    edges = cv2.Canny(image, 100, 200)
    edge_density = np.sum(edges) / edges.size  # 边缘像素占总像素的比例
    return edge_density

# Texture Complexity
def calculate_texture_complexity(image):
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))
    return contrast, entropy



# Complexity Score based on JPEG Compression
def jpeg_compression_complexity(image_path):
    original_size = os.path.getsize(image_path)
    compressed_image_path = 'compressed_image.jpg'
    image = cv2.imread(image_path)
    cv2.imwrite(compressed_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])  # 压缩图像
    compressed_size = os.path.getsize(compressed_image_path)
    compression_ratio = compressed_size / original_size
    os.remove(compressed_image_path)  # 删除临时文件
    return compression_ratio

# Laplacian Variance
def calculate_laplacian_variance(image):
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var

# Image Gradient
def calculate_image_gradient(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return np.mean(gradient_magnitude)

# Color Complexity
def calculate_color_complexity(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 转换为HSV颜色空间
    h, s, v = cv2.split(image)
    color_std = (np.std(h), np.std(s), np.std(v))  # 计算颜色通道的标准差
    return np.mean(color_std)

# Frequency Complexity
def calculate_frequency_complexity(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift))
    return np.mean(magnitude_spectrum)

# Hough Lines Complexity
def calculate_hough_lines_complexity(image):
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    return len(lines) if lines is not None else 0

# Block Variance
def calculate_block_variance(image, block_size=16):
    h, w = image.shape
    variance_list = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            variance_list.append(np.var(block))
    return np.mean(variance_list)

# Calculate all indicators
def calc_indicators(image_path, verbose=False):
    image = cv2.imread(image_path)
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_sk = io.imread(image_path, as_gray=True)

    # Store all indicators in a dictionary
    indicators = {
        "entropy": calculate_entropy(image_sk),
        "edge_density": calculate_edge_density(gray_image),
        "contrast": calculate_texture_complexity(image_sk.astype('uint8'))[0],
        "texture_entropy": calculate_texture_complexity(image_sk.astype('uint8'))[1],
        "jpeg_compression_score": jpeg_compression_complexity(image_path),
        "laplacian_variance": calculate_laplacian_variance(gray_image),
        "image_gradient": calculate_image_gradient(image),
        "color_complexity": calculate_color_complexity(image),
        "frequency_complexity": calculate_frequency_complexity(gray_image),
        "hough_lines_complexity": calculate_hough_lines_complexity(gray_image),
        "block_variance": calculate_block_variance(gray_image, block_size=16)
    }
    if verbose:
        # Print the results
        for key, value in indicators.items():
            print(f"{key}: {value}")

    # Return the dictionary
    return indicators

# 绘制图像及其复杂性指标汇总
def show_image_and_indicators(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 计算复杂性指标
    indicators = calc_indicators(image_path)

    # 创建图形并显示图像
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')  # 隐藏坐标轴
    plt.title("Input Image with Complexity Indicators")

    # 将复杂性指标绘制在图像上
    text_position_y = 10  # 初始文本的y轴位置
    for name, value in indicators.items():
        plt.text(10, text_position_y, f"{name}: {value:.4f}", fontsize=12, color='white', 
                 bbox=dict(facecolor='black', alpha=0.6))  # 黑色背景、白色字体的文本框
        text_position_y += 10  # 每个指标在y轴上向下移动一些

    # 显示带有指标的图像
    plt.show()

    # 返回指标，以便后续使用
    return indicators


# --- 保存分类结果图像到本地 ---
def save_images(image_paths, labels, save_dir):
    # 在保存目录下创建 `easy` 和 `difficult` 子目录
    save_dir_easy = os.path.join(save_dir, "easy", "images")
    save_dir_difficult = os.path.join(save_dir, "difficult", "images")
    
    # 如果目录不存在，创建它们
    if not os.path.exists(save_dir_easy):
        os.makedirs(save_dir_easy)
    if not os.path.exists(save_dir_difficult):
        os.makedirs(save_dir_difficult)

    easy_count = 0
    difficult_count = 0

    # 遍历图像路径和标签，根据标签分类保存
    for img_path, label in zip(image_paths, labels):
        image = cv2.imread(img_path)
        original_file_name = os.path.basename(img_path)  # 获取原始文件名
        
        if label == 0:  # 简易图像
            save_path = os.path.join(save_dir_easy, original_file_name)
            cv2.imwrite(save_path, image)
            easy_count += 1
            print(f"Saved easy image {easy_count} to {save_path}")
            
        elif label == 1:  # 困难图像
            save_path = os.path.join(save_dir_difficult, original_file_name)
            cv2.imwrite(save_path, image)
            difficult_count += 1
            print(f"Saved difficult image {difficult_count} to {save_path}")

    print(f"Total easy images saved: {easy_count}")
    print(f"Total difficult images saved: {difficult_count}")


if __name__ == '__main__':

# =========================== example =================================
    dataset.ClassifyDataset()

    # 计算指标
    calc_indicators(img_path,verbose=True)
    
    # 显示图像及其复杂性指标汇总
    # indicators_summary = show_image_and_indicators(img_path)