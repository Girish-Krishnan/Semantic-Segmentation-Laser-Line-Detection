import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def load_data(image_dir, mask_dir, img_height, img_width, test_size):
    image_files = sorted(os.listdir(image_dir))
    image_files = [f for f in image_files if f.endswith('.jpg')]
    
    images, masks = [], []
    
    for img_file in image_files:
        # Load and resize image
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_width, img_height))
        images.append(img)
        
        # Load and resize mask
        mask_file = img_file.replace('.jpg', '_mask.jpg')
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (img_width, img_height))
        masks.append(mask)
    
    images = np.array(images) / 255.0  # Normalize images
    masks = np.array(masks) / 255.0  # Normalize masks (0 to 1)
    masks = np.expand_dims(masks, axis=-1)  # Add channel dimension

    original_img_size = cv2.imread(os.path.join(image_dir, image_files[0])).shape[:2]
    
    return train_test_split(images, masks, test_size=test_size, random_state=42), original_img_size
