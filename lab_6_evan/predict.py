import cv2
import numpy as np
import pickle
import sys
from skimage.feature import hog

# Load model
def predict_sign(image_path):
    # Load trained model
    with open(r'C:\Users\evanr\OneDrive\Documents\GT SPRING 2025 SCHOLAR MODE 6\BME7785 Intro to Robo Research\7785_Spring2025\knn_model_color.pkl', 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        scaler = data['scaler']
        IMG_SIZE = data['target_size']
        
    # Color ranges in HSV for sign detection
    COLOR_RANGES = {
        'red1': ([0, 100, 100], [10, 255, 255]),
        'red2': ([160, 100, 100], [180, 255, 255]),
        'blue': ([100, 100, 100], [140, 255, 255]),
        'green': ([40, 100, 100], [80, 255, 255])
    }
    
    # Load and process image
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Could not read image"
    
    # Detect sign by color
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    best_mask, best_area = None, 0
    
    # Try each color range
    for color, (lower, upper) in COLOR_RANGES.items():
        if color == 'red2' and best_mask is not None:
            continue  # Skip if we already found red1
            
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            area = max(cv2.contourArea(c) for c in contours)
            if area > best_area and area > 100:
                best_mask, best_area = mask, area
    
    # Extract sign region if detected
    if best_mask is not None:
        contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            margin = 10
            x, y = max(0, x - margin), max(0, y - margin)
            w = min(image.shape[1] - x, w + 2*margin)
            h = min(image.shape[0] - y, h + 2*margin)
            image = image[y:y+h, x:x+w]
    
    # Extract features
    resized = cv2.resize(image, IMG_SIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # HOG features
    hog_features, _ = hog(
        gray, 
        orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        visualize=True, 
        block_norm='L2-Hys'
    )
    
    # Color features
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
    
    # Scale features and predict
    features_scaled = scaler.transform(features.reshape(1, -1))
    prediction = model.predict(features_scaled)[0]
    
    return prediction

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
    else:
        result = predict_sign(sys.argv[1])
        print(f"Predicted class: {result}")