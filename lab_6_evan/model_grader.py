#!/usr/bin/env python3
import os
import sys
import argparse
import csv
import cv2
import numpy as np
import pickle
from skimage.feature import hog
from train import isolate_sign_by_color, find_sign_region, COLOR_RANGES  # Import from train.py

# ------------------------------------------------------------------------------
#                  DO NOT MODIFY FUNCTION NAMES OR ARGUMENTS
# ------------------------------------------------------------------------------

def initialize_model(model_path=None):
    """
    Initialize and return your trained model.
    You MUST modify this function to load and/or construct your model.
    DO NOT change the function name or its input/output.
    
    Args:
        model_path: The path to your pretrained model file (if one is needed).
    Returns:
        model: Your trained model.
    """
    if model_path is None:
        raise ValueError("Model path must be provided to load the pretrained KNN model.")

    # Load the saved model and associated objects
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
    
    # Extract the model, scaler, and target size from the saved data
    model = {
        'knn': saved_data['model'],
        'scaler': saved_data['scaler'],
        'target_size': saved_data['target_size']
    }

    return model

def predict(model, image):
    """
    Run inference on a single image using your model.
    You MUST modify this function to perform prediction.
    DO NOT change the function signature.
    
    Args:
        model: The model object returned by initialize_model().
        image: The input image (as a NumPy array) to classify.
    
    Returns:
        int: The predicted class label.
    """
    # Extract components from the model dictionary
    knn = model['knn']
    scaler = model['scaler']
    IMG_SIZE = model['target_size']

    # Preprocess the image to isolate the sign region (using imported functions)
    mask, _ = isolate_sign_by_color(image)
    sign_region = find_sign_region(image, mask)
    
    # Use sign region if found, otherwise fall back to full image
    if sign_region is not None:
        image_to_process = sign_region
    else:
        image_to_process = image
    
    # Extract features (matching extract_features from train.py)
    resized = cv2.resize(image_to_process, IMG_SIZE)
    
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
    if len(image_to_process.shape) == 3:
        hsv_img = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv_img[:,:,0]], [0], None, [16], [0, 180])
        s_hist = cv2.calcHist([hsv_img[:,:,1]], [0], None, [16], [0, 256])
        v_hist = cv2.calcHist([hsv_img[:,:,2]], [0], None, [16], [0, 256])
        
        # Normalize and combine features
        h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
        s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
        v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
        
        color_features = np.concatenate([h_hist, s_hist, v_hist]).flatten()
        features = np.concatenate([hog_features, color_features])
    else:
        features = hog_features
    
    # Scale the features using the saved scaler
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Predict using the KNN model
    prediction = knn.predict(features_scaled)[0]
    
    return int(prediction)

# ------------------------------------------------------------------------------
#                      DO NOT MODIFY ANY CODE BELOW THIS LINE
# ------------------------------------------------------------------------------

def load_validation_data(data_path):
    """
    Load validation images and labels from the given directory.
    Expects a 'labels.txt' file in the directory and images in .png format.
    
    Args:
        data_path (str): Path to the validation dataset.
    
    Returns:
        list of tuples: Each tuple contains (image_path, true_label)
    """
    labels_file = os.path.join(data_path, "labels.txt")
    data = []
    with open(labels_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            # Assumes row[0] is the image filename (without extension) and row[1] is the label.
            image_file = os.path.join(data_path, row[0] + ".png")  # Modify if images use a different extension.
            data.append((image_file, int(row[1])))
    return data

def evaluate_model(model, validation_data):
    """
    Evaluate the model on the validation dataset.
    Computes and prints the confusion matrix and overall accuracy.
    
    Args:
        model: The model object.
        validation_data (list): List of tuples (image_path, true_label).
    """
    num_classes = 6  # Number of classes (adjust if needed)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    correct = 0
    total = len(validation_data)
    
    for image_path, true_label in validation_data:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print("Warning: Could not load image:", image_path)
            continue
        # Get the predicted label using the student's implementation.
        predicted_label = predict(model, image)
        
        if predicted_label == true_label:
            correct += 1
        confusion_matrix[true_label][predicted_label] += 1
        print(f"Image: {os.path.basename(image_path)} - True: {true_label}, Predicted: {predicted_label}")
    
    accuracy = correct / total if total > 0 else 0
    print("\nTotal accuracy:", accuracy)
    print("Confusion Matrix:")
    print(confusion_matrix)

def main():
    parser = argparse.ArgumentParser(description="Model Grader for Lab 6")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the validation dataset directory (must contain labels.txt and images)")
    parser.add_argument("--model_path", type=str, required=False,
                        help="Path to the trained model file (if applicable)")
    args = parser.parse_args()
    
    # Path to the validation dataset directory from command line argument.
    VALIDATION_DATASET_PATH = args.data_path

    # Path to the trained model file from command line argument.
    MODEL_PATH = args.model_path
    
    # Load validation data.
    validation_data = load_validation_data(VALIDATION_DATASET_PATH)
    
    # Initialize the model using the student's implementation.
    model = initialize_model(MODEL_PATH) if MODEL_PATH else initialize_model()
    
    # Evaluate the model on the validation dataset.
    evaluate_model(model, validation_data)

if __name__ == "__main__":
    main()
