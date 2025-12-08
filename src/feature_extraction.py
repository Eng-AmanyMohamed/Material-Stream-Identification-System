import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import os
from skimage.feature import hog

def extract_features_for_image(image_path):
   image = cv2.imread(image_path)
   image = cv2.resize(image, (128, 128))

   gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

   HOG_features = hog(gray_image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2) ,block_norm='L2-Hys')
   hist_HSV_features = cv2.calcHist([hsv_image],[0, 1, 2],None, [6,6,6], [0,180, 0,256, 0,256] ).flatten()
   LPb = local_binary_pattern(gray_image,P=6 , R=1 , method='uniform')
   (LPb_hist_freatures ,_ ) = np.histogram(LPb.ravel(),bins=np.arange(0,29) , range=(0, 58) )
   LPb_hist_freatures = (LPb_hist_freatures.astype('float')) / (LPb_hist_freatures.sum() +1e-7)

   return np.hstack([HOG_features , hist_HSV_features , LPb_hist_freatures])




def exctract_feature_vectors(data_path, classes ):
    y = []
    X = []
    class_num = 0
    for c in classes:
        original_class_path = os.path.join(data_path, c)
        if not os.path.exists(original_class_path):
            print(f"Folder not found: {original_class_path}")
            continue

        original_images = [f for f in os.listdir(original_class_path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for imgName in original_images:
            src_path = os.path.join(original_class_path, imgName)
            image_feature = extract_features_for_image(src_path)
            X.append(image_feature)
            y.append(class_num)
        class_num += 1
    X = np.array(X)
    y = np.array(y)
    return X, y


X, y = exctract_feature_vectors("data/augmented_dataset" , ["glass", "paper", "cardboard", "plastic", "metal", "trash"])
print(X.shape)
print(y.shape)
