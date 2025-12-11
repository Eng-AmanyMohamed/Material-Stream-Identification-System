import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import os
from skimage.feature import hog

def extract_features_for_image(image_path):
   image = cv2.imread(image_path)
   image = cv2.resize(image, (64, 64))

   gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

   hist_HSV_features = cv2.calcHist([hsv_image],[0, 1, 2],None, [8,8,8], [0,180, 0,256, 0,256] ).flatten()

   #Local Binary Pattern (LBP) Captures fine-grained texture ( smooth glass vs fibrous cardboard)
   LPb = local_binary_pattern(gray_image,P=8 , R=1 , method='uniform')
   (LPb_hist_freatures ,_ ) = np.histogram(LPb.ravel(),bins=np.arange(0,26) , range=(0, 58) )
   LPb_hist_freatures = (LPb_hist_freatures.astype('float')) / (LPb_hist_freatures.sum() +1e-7)


   grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
   grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
   all_grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
   edge_features = np.mean(all_grad)

   return np.hstack([edge_features, hist_HSV_features , LPb_hist_freatures])

def extract_feature_vector_single(image_path):
    features = extract_features_for_image(image_path)
    return features.reshape(1, -1)

# make 2 functions to extract features from frame for real time application
def extract_features_from_frame(image):
   if image is None: 
       return None
   image = cv2.resize(image, (64, 64))

   gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

   hist_HSV_features = cv2.calcHist([hsv_image],[0, 1, 2],None, [8,8,8], [0,180, 0,256, 0,256] ).flatten()

   #Local Binary Pattern (LBP) Captures fine-grained texture ( smooth glass vs fibrous cardboard)
   LPb = local_binary_pattern(gray_image,P=8 , R=1 , method='uniform')
   (LPb_hist_freatures ,_ ) = np.histogram(LPb.ravel(),bins=np.arange(0,26) , range=(0, 58) )
   LPb_hist_freatures = (LPb_hist_freatures.astype('float')) / (LPb_hist_freatures.sum() +1e-7)


   grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
   grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
   all_grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
   edge_features = np.mean(all_grad)

   return np.hstack([edge_features, hist_HSV_features , LPb_hist_freatures])

def extract_feature_frame_single(image):
    features = extract_features_from_frame(image)
    return features.reshape(1, -1)

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


X, y = exctract_feature_vectors("data/augmented_dataset" ,["glass", "paper", "cardboard", "plastic", "metal", "trash"])
# print(X.shape)
# print(y.shape)
