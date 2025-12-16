import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from feature_extraction import extract_features_from_dataset

DATA_PATH = "data/augmented_dataset"
CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load features
print("Extracting features using MobileNetV2...")
X, y = extract_features_from_dataset(DATA_PATH, CLASSES)
print(f"Features shape: {X.shape}")

# 2. Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train SVM
svm = SVC(kernel="rbf", C=50, gamma="scale", probability=True)
svm.fit(X_train, y_train)

# 5. Evaluate
y_pred = svm.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"\nSVM Validation Accuracy: {acc:.4f}")
print("\n" + classification_report(y_val, y_pred, target_names=CLASSES))

# 6. Save
joblib.dump({
    "model": svm,
    "scaler": scaler,
    "classes": CLASSES,
    'rejection_threshold': 0.8
}, os.path.join(MODEL_DIR, "svm_model.pkl"))
print("SVM model saved.")