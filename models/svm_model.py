import joblib
from sklearn.svm import SVC

class SVMModel:
    def __init__(self, kernel="rbf", C=1.0, gamma="scale"):
        """
        Initialize SVM classifier.
        """
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True  # needed for Unknown class thresholding
        )

    def train(self, X_train, y_train):
        """
        Train SVM on feature vectors.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Predict class label.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict probability (used for Unknown class).
        """
        return self.model.predict_proba(X)

    def save(self, path="svm_model.pkl"):
        """
        Save model to file.
        """
        joblib.dump(self.model, path)
        print(f"SVM model saved at {path}")

    def load(self, path="svm_model.pkl"):
        """
        Load model from file.
        """
        self.model = joblib.load(path)
        print(f"SVM model loaded from {path}")
