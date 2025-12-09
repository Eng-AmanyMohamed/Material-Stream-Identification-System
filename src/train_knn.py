
import os
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from feature_extraction import exctract_feature_vectors

DATA_PATH = "data/augmented_dataset"
CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]  # IDs 0–5
SAVE_MODEL_PATH = "models/knn_model.pkl"
os.makedirs("models", exist_ok=True)

# 1. Load & Extract Features
print("Extracting features from augmented dataset...")
X, y = exctract_feature_vectors(DATA_PATH, CLASSES)
print(f" Features shape: {X.shape}, Labels shape: {y.shape}")

# 2. Preprocessing: Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=100)  # Keep 100 principal components
X_pca = pca.fit_transform(X_scaled)


# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

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
# 5. create KNN model
best_knn = KNeighborsClassifier(n_neighbors=3,weights='distance', metric='euclidean')

# 6. Train  KNN model
best_knn.fit(X_train, y_train)

# 7. validate model

y_pred = best_knn.predict(X_test)
val_acc = accuracy_score(y_test, y_pred)
print(f"\n Validation Accuracy: {val_acc:.4f}")
print("\n Classification Report :")
print(classification_report(y_test, y_pred, target_names=CLASSES))

# Save model + preprocessing objects

import joblib
os.makedirs("models", exist_ok=True)
joblib.dump({
    'model': best_knn,
    'scaler': scaler,
    'pca': pca,
    'classes': CLASSES
}, SAVE_MODEL_PATH)
