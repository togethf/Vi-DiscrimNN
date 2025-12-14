import os
import cv2
import numpy as np
import pickle
from skimage.feature import hog, graycomatrix, graycoprops
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import matplotlib.pyplot as plt

# Paths for saving and loading data
FEATURES_PATH = "features.pkl"
LABELS_PATH = "labels.pkl"
RESULTS_PATH = "results.txt"

def extract_hog_features(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """Extract HOG features from an image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(
        gray_image,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        visualize=True
    )
    return features

def extract_glcm_features(image, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """Extract texture features using GLCM."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    return [contrast, correlation, energy, homogeneity]

def extract_color_histogram_features(image):
    """Extract color histogram features (mean, std, skew, energy, entropy)."""
    channels = cv2.split(image)
    features = []
    for channel in channels:
        mean = np.mean(channel)
        std = np.std(channel)
        skewness = skew(channel.flatten())
        energy = np.sum(channel.flatten() ** 2) / channel.size
        hist, _ = np.histogram(channel, bins=256, range=(0, 256), density=True)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Add small value to avoid log(0)
        features.extend([mean, std, skewness, energy, entropy])
    return features

def extract_shape_features(image):
    """Extract shape features based on contours."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area = [cv2.contourArea(c) for c in contours]
    perimeter = [cv2.arcLength(c, True) for c in contours]
    area_mean = np.mean(area) if area else 0
    area_std = np.std(area) if area else 0
    perimeter_mean = np.mean(perimeter) if perimeter else 0
    perimeter_std = np.std(perimeter) if perimeter else 0
    
    return [area_mean, area_std, perimeter_mean, perimeter_std]

def extract_features(image):
    """Combine all features into a single feature vector."""
    features = []
    features.extend(extract_hog_features(image))
    features.extend(extract_glcm_features(image))
    features.extend(extract_color_histogram_features(image))
    features.extend(extract_shape_features(image))
    return np.array(features)

def load_dataset(image_dir, labels):
    """Load images and extract features."""
    features, target = [], []
    for label, sub_dir in labels.items():
        img_dir = os.path.join(image_dir, sub_dir, 'images', 'val')
        for img_name in tqdm(os.listdir(img_dir), desc=f"Processing {label}"):
            img_path = os.path.join(img_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue
            features.append(extract_features(image))
            target.append(label)
    return np.array(features), np.array(target)

def main():
    # Define paths and labels
    image_dir = './out'  # Path to your dataset
    labels = {0: 'easy', 1: 'diff'}  # Define labels (easy=0, difficult=1)

    # Check if features and labels are already saved
    if os.path.exists(FEATURES_PATH) and os.path.exists(LABELS_PATH):
        print("Loading saved features and labels...")
        with open(FEATURES_PATH, 'rb') as f:
            X = pickle.load(f)
        with open(LABELS_PATH, 'rb') as f:
            y = pickle.load(f)
    else:
        print("Extracting features...")
        X, y = load_dataset(image_dir, labels)
        with open(FEATURES_PATH, 'wb') as f:
            pickle.dump(X, f)
        with open(LABELS_PATH, 'wb') as f:
            pickle.dump(y, f)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE for balancing the dataset
    print("Applying SMOTE...")
    smote = SMOTE(sampling_strategy='auto', k_neighbors=10, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train a Random Forest Classifier
    print("Training the model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_resampled, y_train_resampled)

    # Evaluate the model
    print("Evaluating the model...")
    y_proba = clf.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    # Find the best threshold for F1-score
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]
    y_pred_adjusted = (y_proba > best_threshold).astype(int)

    # Output results
    report = classification_report(y_test, y_pred_adjusted)
    confusion = confusion_matrix(y_test, y_pred_adjusted)
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(confusion)
    print(f"Best Threshold: {best_threshold}")

    # Save results to a file
    with open(RESULTS_PATH, 'w') as f:
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(confusion) + "\n")
        f.write(f"Best Threshold: {best_threshold}\n")

    # Plot Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.scatter(recall[np.argmax(f1_scores)], precision[np.argmax(f1_scores)], color='red', label="Best Threshold")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()