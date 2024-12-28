import matplotlib.pyplot as plt
import numpy as np
import cv2

def visualize_predictions(model, X_test, y_test, num_samples, original_img_size):
    for idx in range(num_samples):
        sample_image = X_test[idx]
        sample_mask = y_test[idx][:, :, 0]
        pred_mask = model.predict(np.expand_dims(sample_image, axis=0))[0][:, :, 0]
        
        pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Threshold prediction

        accuracy = np.mean(sample_mask == pred_mask)

        # Resize images to original size
        sample_image = cv2.resize(sample_image, (original_img_size[1], original_img_size[0]))
        sample_mask = cv2.resize(sample_mask, (original_img_size[1], original_img_size[0]))
        pred_mask = cv2.resize(pred_mask, (original_img_size[1], original_img_size[0]))
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title('Input Image')
        plt.imshow(sample_image)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title('True Mask')
        plt.imshow(sample_mask, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title('Predicted Mask')
        plt.imshow(pred_mask, cmap='gray')
        plt.axis('off')

        plt.suptitle(f'Accuracy: {accuracy:.3f}')
        
        plt.show()
