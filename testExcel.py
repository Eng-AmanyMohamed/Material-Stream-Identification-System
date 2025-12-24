import os
import cv2
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def predict_and_export(dataFilePath, bestModelPath="models/svm_model.pkl", output_excel="predictions.xlsx"):

    
    # 1. Load model pipeline
    pipeline = joblib.load(bestModelPath)
    model = pipeline["model"]
    scaler = pipeline["scaler"]
    threshold = pipeline.get("rejection_threshold", 0.7)
    
    # 2. Load MobileNetV2 for feature extraction
    mobilenet = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    
    # 3. Get sorted list of image files
    image_files = sorted([
        f for f in os.listdir(dataFilePath)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ], key=str.lower)
    
    image_names = []
    predictions = []
    
    for img_name in image_files:
        img_path = os.path.join(dataFilePath, img_name)
        
        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            image_names.append(img_name)
            predictions.append(6)  # Unknown if unreadable
            continue
        
        # Convert to MobileNetV2 input
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        # Extract features
        features = mobilenet.predict(img, verbose=0).flatten()
        features = features.reshape(1, -1)
        features = scaler.transform(features)
        CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

        # Predict
        proba = model.predict_proba(features)
        conf = np.max(proba)
        pred_id = np.argmax(proba)
        
        if conf < threshold:
            predictions.append("Unknown")  # Unknown (ID=6)
        else:
            predictions.append(CLASSES[int(pred_id)])
        
        image_names.append(img_name)
    
    # 4. Create DataFrame and export to Excel
    df = pd.DataFrame({
        'Image Name': image_names,
        'Predicted Label': predictions
    })
    
    df.to_excel(output_excel, index=False)
    print(f"Predictions exported to {output_excel}")
    
    return df
#ق


# Usage example:
if __name__ == "__main__":
    # Specify your image folder path
    image_folder = "testImages/"
    
    # Run prediction and export
    results_df = predict_and_export(
        dataFilePath=image_folder,
        bestModelPath="models/svm_model.pkl",
        output_excel="predictions.xlsx"
    )
    
    # Display results
    print(results_df)