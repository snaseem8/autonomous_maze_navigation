import pickle
import cv2
import numpy as np
from skimage.feature import hog
from train import isolate_sign_by_color, find_sign_region, COLOR_RANGES

def initialize_model(model_path):
    # Same as above
    if model_path is None:
        raise ValueError("Model path must be provided to load the pretrained KNN model.")
    
    try:
        with open(model_path, 'rb') as file:
            saved_data = pickle.load(file)
        model = {
            'knn': saved_data['model'],
            'scaler': saved_data['scaler'],
            'target_size': saved_data['target_size']
        }
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        raise Exception(f"Failed to load model from {model_path}: {str(e)}")

def predict(model, image):
    # Same as above
    knn = model['knn']
    scaler = model['scaler']
    IMG_SIZE = model['target_size']
    
    mask, _ = isolate_sign_by_color(image)
    sign_region = find_sign_region(image, mask)
    image_to_process = sign_region if sign_region is not None else image
    resized = cv2.resize(image_to_process, IMG_SIZE)
    
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    
    hog_features, _ = hog(
        gray, 
        orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        visualize=True, 
        block_norm='L2-Hys'
    )
    
    if len(image_to_process.shape) == 3:
        hsv_img = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv_img[:,:,0]], [0], None, [16], [0, 180])
        s_hist = cv2.calcHist([hsv_img[:,:,1]], [0], None, [16], [0, 256])
        v_hist = cv2.calcHist([hsv_img[:,:,2]], [0], None, [16], [0, 256])
        h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
        s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
        v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
        color_features = np.concatenate([h_hist, s_hist, v_hist]).flatten()
        features = np.concatenate([hog_features, color_features])
    else:
        features = hog_features
    
    features_scaled = scaler.transform(features.reshape(1, -1))
    prediction = knn.predict(features_scaled)[0]
    
    return int(prediction)