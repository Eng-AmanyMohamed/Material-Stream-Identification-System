import cv2
import numpy as np
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load MobileNetV2 once (global)
mobilenet_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
print("✓ MobileNetV2 (1280-dim) loaded.")

def extract_features_for_image(image_path):
    """Extract 1280-dim features from a single image path."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = mobilenet_model.predict(img, verbose=0)
    return features.flatten()

def extract_features_from_dataset(data_path, classes):
    """Extract features for all images in augmented_dataset."""
    X, y = [], []
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    for cls in classes:
        class_path = os.path.join(data_path, cls)
        if not os.path.exists(class_path):
            continue
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        for img_name in sorted(images):  # ← fixed order!
            img_path = os.path.join(class_path, img_name)
            feat = extract_features_for_image(img_path)
            if feat is not None:
                X.append(feat)
                y.append(class_to_idx[cls])
    
    return np.array(X), np.array(y)

# For real-time
def extract_features_from_frame(frame):
    """Extract features from live camera frame (numpy array)."""
    if frame is None:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    features = mobilenet_model.predict(frame, verbose=0)
    return features.flatten()