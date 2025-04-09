import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_labeled_images(data_dir, label_file, target_size=(64, 64), grayscale=True):
    """
    Load images and labels from a directory using your specific label file format.
    """
    # Read the label file - specific to your format "000, 1" etc.
    image_labels = {}
    class_set = set()
    
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Parse each line like "000, 1"
            parts = line.split(',')
            if len(parts) == 2:
                file_num = parts[0].strip()
                label = int(parts[1].strip())
                
                # The filename is the number with .png extension
                filename = f"{file_num}.png"
                image_labels[filename] = label
                class_set.add(label)
    
    # Get sorted list of unique class labels
    class_names = sorted(list(class_set))
    
    # Load images
    images = []
    labels = []
    filenames = []  # Keep track of filenames for debugging
    
    for filename, label in image_labels.items():
        img_path = os.path.join(data_dir, filename)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, target_size)
                if grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_flat = img.flatten()
                images.append(img_flat)
                labels.append(label)
                filenames.append(filename)
            else:
                print(f"Warning: Could not read image {img_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    if len(images) == 0:
        raise ValueError("No images were successfully loaded!")
        
    print(f"Successfully loaded {len(images)} images with labels: {', '.join(map(str, class_names))}")
    
    return np.array(images), np.array(labels), class_names

def prepare_data(data_dir, label_file, target_size=(64, 64), grayscale=True, test_size=0.2, random_state=42):
    """Prepare data for training: load, split, and scale."""
    print("Loading images and labels...")
    X, y, class_names = load_labeled_images(data_dir, label_file, target_size, grayscale)
    
    print(f"Loaded {len(X)} images with {len(class_names)} classes")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, class_names