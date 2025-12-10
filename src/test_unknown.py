from models.svm_model import SVMModel
from src.feature_extraction import extract_feature_vector_single

# load model
svm = SVMModel()
svm.load("models/svm_model.pkl")

image_path = "images/metal.jpg"

features = extract_feature_vector_single(image_path).reshape(1, -1)

pred, confidence = svm.predict_with_unknown(features)

classes = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

if pred == "Unknown":
    print("\nPredicted: UNKNOWN MATERIAL")
else:
    print(f"\nPredicted: {classes[pred]}")
# print(f"Confidence: {confidence:.4f}")

