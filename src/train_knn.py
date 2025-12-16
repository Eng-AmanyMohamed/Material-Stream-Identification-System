import os
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
from feature_extraction import extract_features_from_dataset


# Settings
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
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

#  4. Hyperparameter Tuning for k-NN

# param_grid = {
#     'n_neighbors': [3, 5, 7, 9, 11],
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean']
# }

# knn = KNeighborsClassifier()
# cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# grid_search = GridSearchCV(
#     knn, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1
# )
# grid_search.fit(X_train, y_train)

# best_knn = grid_search.best_estimator_
# print(f"Best params: {grid_search.best_params_}")
# print(f"Best CV accuracy: {grid_search.best_score_:.4f}")


# Use BEST PARAMETERS found during tuning

# 5. Train k-NN
knn = KNeighborsClassifier(n_neighbors=3, weights="distance", metric='euclidean')
knn.fit(X_train, y_train)

# 6. Evaluate
y_pred = knn.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n k-NN Validation Accuracy: {acc:.4f}")
print("\n" + classification_report(y_test, y_pred, target_names=CLASSES))

# 7. Save
joblib.dump({
    "model": knn,
    "scaler": scaler,
    "classes": CLASSES,
    'rejection_threshold': 0.6
}, os.path.join(MODEL_DIR, "knn_model.pkl"))
print("k-NN model saved.")



