import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from skimage.feature import hog
import random

# Configuration
BASE_DIR = r"C:\Users\evanr\OneDrive\Documents\GT SPRING 2025 SCHOLAR MODE 6\BME7785 Intro to Robo Research\IntroRobo Lab 6 Image Classification"
DATA_DIRS = [
    os.path.join(BASE_DIR, "2025S_imgs"),
    os.path.join(BASE_DIR, "2024F_imgs"),
    os.path.join(BASE_DIR, "2024F_Gimgs")
]
IMG_SIZE = (64, 64)
RANDOM_SEED = 123  # Changed seed to avoid data leakage issues
TEST_SIZE = 0.2
MODEL_SAVE_PATH = r"C:\Users\evanr\OneDrive\Documents\GT SPRING 2025 SCHOLAR MODE 6\BME7785 Intro to Robo Research\7785_Spring2025\knn_model_color.pkl"

# Color ranges in HSV for different sign colors
COLOR_RANGES = {
    'red1': ([0, 100, 100], [10, 255, 255]),
    'red2': ([160, 100, 100], [180, 255, 255]),
    'blue': ([100, 100, 100], [140, 255, 255]),
    'green': ([40, 100, 100], [80, 255, 255])
}

def isolate_sign_by_color(image):
    """Identify sign regions using color thresholding"""
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    best_mask = None
    best_color = None
    best_area = 0
    
    # Try each color range
    for color_name, (lower, upper) in COLOR_RANGES.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv_img, lower, upper)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > best_area and area > 100:
                best_area = area
                best_mask = mask
                best_color = color_name
    
    # Special handling for red (which spans two HSV ranges)
    if best_color in ['red1', 'red2']:
        # Combine both red masks
        red1_lower = np.array(COLOR_RANGES['red1'][0])
        red1_upper = np.array(COLOR_RANGES['red1'][1])
        red2_lower = np.array(COLOR_RANGES['red2'][0])
        red2_upper = np.array(COLOR_RANGES['red2'][1])
        
        mask1 = cv2.inRange(hsv_img, red1_lower, red1_upper)
        mask2 = cv2.inRange(hsv_img, red2_lower, red2_upper)
        best_mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = np.ones((5, 5), np.uint8)
        best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, kernel)
        best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel)
        
        best_color = 'red'
    
    return best_mask, best_color

def find_sign_region(image, mask):
    """Extract the sign region using the color mask"""
    if mask is None:
        return None
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    if area < 100:
        return None
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add a small margin
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2*margin)
    h = min(image.shape[0] - y, h + 2*margin)
    
    sign_region = image[y:y+h, x:x+w]
    
    if sign_region.size == 0:
        return None
    
    return sign_region

def extract_features(image):
    """Extract HOG and color features from an image"""
    resized = cv2.resize(image, IMG_SIZE)
    
    # Convert to grayscale for HOG
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    
    # Extract HOG features
    hog_features, _ = hog(
        gray, 
        orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        visualize=True, 
        block_norm='L2-Hys'
    )
    
    # Extract color features if image is color
    if len(image.shape) == 3:
        hsv_img = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv_img[:,:,0]], [0], None, [16], [0, 180])
        s_hist = cv2.calcHist([hsv_img[:,:,1]], [0], None, [16], [0, 256])
        v_hist = cv2.calcHist([hsv_img[:,:,2]], [0], None, [16], [0, 256])
        
        # Normalize and combine features
        h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
        s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
        v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
        
        color_features = np.concatenate([h_hist, s_hist, v_hist]).flatten()
        return np.concatenate([hog_features, color_features])
    
    return hog_features

def process_directory(data_dir):
    """Load images, extract features, and prepare for training"""
    label_file = os.path.join(data_dir, "labels.txt")
    
    if not os.path.exists(label_file):
        print(f"Warning: Label file not found at {label_file}")
        return [], []
    
    # Read labels
    image_data = []
    with open(label_file, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) >= 1:
                if len(line) == 1 and ',' in line[0]:
                    parts = line[0].split(',')
                    file_num = parts[0].strip()
                    label = int(parts[1].strip())
                elif len(line) >= 2:
                    file_num = line[0].strip()
                    label = int(line[1].strip())
                else:
                    continue
                
                filename = f"{file_num}.png"
                img_path = os.path.join(data_dir, filename)
                image_data.append((img_path, label))
    
    # Process images
    features = []
    labels = []
    
    print(f"Processing {len(image_data)} images from {data_dir}")
    
    for img_path, label in image_data:
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Try to isolate sign by color
        mask, _ = isolate_sign_by_color(image)
        
        # Process features from sign region or whole image
        if mask is not None:
            sign_region = find_sign_region(image, mask)
            if sign_region is not None:
                # Original sign region
                features.append(extract_features(sign_region))
                labels.append(label)
                
                # Augmentations (reduced to 2 for conciseness)
                for i in range(7):
                    angle = random.uniform(-15, 15)
                    height, width = sign_region.shape[:2]
                    M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
                    rotated = cv2.warpAffine(sign_region, M, (width, height))
                    
                    features.append(extract_features(rotated))
                    labels.append(label)
                
                continue
        
        # Fallback: process whole image
        features.append(extract_features(image))
        labels.append(label)
    
    print(f"Processed {len(features)} samples (including augmentations)")
    return features, labels

def train_and_evaluate():
    """Train model and generate evaluation plots"""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Load and process data
    all_features = []
    all_labels = []
    
    for data_dir in DATA_DIRS:
        features, labels = process_directory(data_dir)
        all_features.extend(features)
        all_labels.extend(labels)
    
    if len(all_features) == 0:
        print("Error: No images were successfully processed.")
        return
    
    # Convert to arrays and get class info
    X = np.array(all_features)
    y = np.array(all_labels)
    class_names = sorted(list(set(all_labels)))
    
    print(f"Total dataset: {len(X)} samples with {len(class_names)} classes")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    # Save test set
    with open('test_set.pkl', 'wb') as f:
        pickle.dump({'features': X_test, 'labels': y_test}, f)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with k=2
    print("Training KNN model with k=2...")
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Print classification report
    report = classification_report(y_test, y_pred, target_names=[str(c) for c in class_names])
    print("\nClassification Report:")
    print(report)
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump({
            'model': knn,
            'scaler': scaler,
            'class_names': class_names,
            'target_size': IMG_SIZE,
            'best_k': 5
        }, f)
    
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # Generate evaluation plots
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # 2. Class Distribution
    plt.figure(figsize=(10, 6))
    counts = np.bincount(y)
    plt.bar(class_names, counts[class_names])
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    plt.xticks(class_names)
    plt.savefig('class_distribution.png')
    
    # 3. Identify incorrectly classified samples
    incorrect_indices = np.where(y_test != y_pred)[0]
    if len(incorrect_indices) > 0:
        print(f"\nFound {len(incorrect_indices)} misclassified samples")
        print(f"Accuracy would be {(len(y_test) - len(incorrect_indices))/len(y_test):.4f} if these were corrected")
    else:
        print("\nAll test samples classified correctly!")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()