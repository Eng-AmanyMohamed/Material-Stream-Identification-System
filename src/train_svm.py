import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models.svm_model import SVMModel
from src.feature_extraction import exctract_feature_vectors

# Step 1 — Load features
classes = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
X, y = exctract_feature_vectors("data/augmented_dataset", classes)

print("Feature vectors loaded:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Step 2 — Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3 — Create SVM model
svm = SVMModel(kernel="rbf", C=10, gamma="scale")

# Step 4 — Train
svm.train(X_train, y_train)

# Step 5 — Validate
y_pred = svm.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print(f"\nValidation Accuracy: {acc:.4f}")

# Step 6 — Save trained model
svm.save("models/svm_model.pkl")
