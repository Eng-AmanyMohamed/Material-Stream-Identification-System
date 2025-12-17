import numpy as np

def predict_with_rejection(model, X, threshold=0.7):
    """
    Generic rejection function for any scikit-learn classifier with predict_proba().
    
    Parameters:
        model: trained classifier (SVM or k-NN) that supports .predict_proba()
        X: 2D array of shape (1, n_features) — preprocessed input
        threshold: float in (0, 1), confidence cutoff for rejection

    Returns:
        pred_id: int (0–5 for known classes, 6 for "Unknown")
        confidence: float (max predicted probability)
    """
    proba = model.predict_proba(X)
    confidence = np.max(proba)            
    pred_id = np.argmax(proba)              

    # Reject if confidence too low
    if confidence < threshold:
        return 6, confidence  # "Unknown"
    else:
        return int(pred_id), confidence