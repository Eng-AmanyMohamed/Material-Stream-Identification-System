import os
import cv2
import random
from skimage import transform, exposure
import numpy as np

original_data_path = 'data/dataset'
final_data_path = 'data/augmented_dataset'
classes = ["paper", "plastic", "metal", "glass", "cardboard", "trash"]
target_images_per_class = 500

os.makedirs(final_data_path, exist_ok=True)

for c in classes:
    print(f"\n processing: {c}")
    output_class_path = os.path.join(final_data_path,c)
    os.makedirs(output_class_path,exist_ok=True)
    
    original_class_path = os.path.join(original_data_path,c)
    if not os.path.exists(original_class_path):
        print(f"Folder not found: {original_class_path}")
        continue
    
    original_images = [f for f in os.listdir(original_class_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Number of original images: {len(original_images)}")
    
    copied_images_count = 0 
    for imgName in original_images:
        src_path = os.path.join(original_class_path,imgName)
        img = cv2.imread(src_path)
        if img is None:
            print(f"Corrupted image:{imgName}")
            continue
        destination_path = os.path.join(output_class_path,imgName)
        cv2.imwrite(destination_path,img)
        copied_images_count+=1
        
        print(f"Copied {copied_images_count} images", end="\r")
        
    current_image_count = copied_images_count
    aug_count = 0
    
    while current_image_count + aug_count< target_images_per_class:   
        
        valid_image = random.choice(original_images)
        src_path = os.path.join(original_class_path,valid_image)
        img = cv2.imread(src_path)
        if img is None:
            print(f"Corrupted image:{valid_image}")
            continue    
    
        # 1- rotation by some angle between -20 and 20
        angle = random.randint(-20,20)
        img = transform.rotate(img,angle,resize=False , preserve_range=True).astype(np.uint8)
        
        # 2- flipping image horizontally
        if random.random() > 0.5:
            img = cv2.flip(img,1)
        
        # 3- adjust lighting
        gamma = random.uniform(0.7,1.3)
        img = exposure.adjust_gamma(img,gamma)
        
        # 4- change size (resize)
        scale = random.uniform(0.9,1.1)
        h, w = img.shape[:2]
        new_h, new_w = int(h*scale),int(w*scale)
        img = cv2.resize(img,(new_w, new_h))
        
        new_img_name = f"{c}_aug_{aug_count}.jpg"
        save_path = os.path.join(output_class_path, new_img_name)
        cv2.imwrite(save_path, img)
        aug_count+= 1
        
        print(f"done adding: {aug_count} augmented images, total number: {copied_images_count+aug_count}")
        
        
        