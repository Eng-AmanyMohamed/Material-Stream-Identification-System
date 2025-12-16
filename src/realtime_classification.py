import cv2
import numpy as np
import os
import sys
from unknown_logic import predict_with_rejection 
import joblib


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from feature_extraction import extract_features_from_frame

# Load model pipeline
MODEL_PATH = os.path.join(project_root, "models", "svm_model.pkl")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}")
    print("Please run 'python src/train_svm.py' first.")
    exit(1)

pipeline = joblib.load(MODEL_PATH)
model = pipeline["model"]
scaler = pipeline["scaler"]
classes = pipeline["classes"]  + ["Unknown"]
threshold = pipeline["rejection_threshold"]

def classify_frame(frame):
    """Returns (label, confidence)"""
    if frame is None:
        return "Invalid Frame", 0.0

    # Extract features
    feat = extract_features_from_frame(frame)
    if feat is None:
        return "No Features", 0.0

    # Preprocess
    feat = feat.reshape(1, -1)
    feat = scaler.transform(feat)

    # Predict
    pred_id, conf = predict_with_rejection(model, feat, threshold=threshold)
    label = classes[pred_id]  # classes[6] = "Unknown"

    return label, conf

def main():
    print("=== Real-Time Waste Classifier ===")
    print("1. Live Camera")
    print("2. Image File")
    print("3. Exit")

    while True:
        choice = input("\nEnter choice (1/2/3): ").strip()
        if choice == "1":
            # Live camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Cannot open camera.")
                continue

            print("Press 'q' to quit.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                label, conf = classify_frame(frame)
                color = (0, 0, 255) if label == "Unknown" else (0, 255, 0)
                cv2.putText(frame, f"{label} ({conf:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imshow("Waste Classifier", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        elif choice == "2":
            path = input("Enter image path: ").strip()
            if not os.path.exists(path):
                print("File not found.")
                continue
            img = cv2.imread(path)
            label, conf = classify_frame(img)
            print(f"Prediction: {label} (Confidence: {conf:.2f})")

        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()