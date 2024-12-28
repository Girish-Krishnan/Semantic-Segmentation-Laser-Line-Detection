import argparse
from src.config_loader import load_config
from src.data_loader import load_data
from src.model import unet_model
from src.train import train_model
from src.predict import visualize_predictions
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Project")
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the images directory')
    parser.add_argument('--mask_dir', type=str, required=True, help='Path to the masks directory')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    parser.add_argument('--train', action='store_true', help='Flag to train a new model')
    parser.add_argument('--model_path', type=str, default='models/unet_model.keras', help='Path to save/load the model')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    img_height, img_width = config['img_height'], config['img_width']
    batch_size, epochs = config['batch_size'], config['epochs']
    test_size = config['test_size']
    
    # Load data
    (X_train, X_test, y_train, y_test), original_img_size = load_data(args.image_dir, args.mask_dir, img_height, img_width, test_size)
    
    if args.train:
        # Train model
        model = unet_model(input_size=(img_height, img_width, 3))
        train_model(model, X_train, y_train, X_test, y_test, batch_size, epochs, args.model_path)
    else:
        # Load existing model
        model = tf.keras.models.load_model(args.model_path)
    
    # Visualize predictions
    visualize_predictions(model, X_test, y_test, args.num_samples, original_img_size)

if __name__ == "__main__":
    main()
